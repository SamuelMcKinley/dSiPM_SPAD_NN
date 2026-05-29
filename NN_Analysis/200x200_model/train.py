#!/usr/bin/env python3
import csv, argparse
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from contextlib import nullcontext

from dataset import PhotonEnergyDataset
from model import EnergyRegressionCNN


def parse_args():
    p = argparse.ArgumentParser(description="Train CNN on UNNORMALIZED hit-maps (from .npz) (resumable, balanced).")
    p.add_argument("tensor_path", help="Root folder (recursive) OR single .npz")
    p.add_argument("--spad", required=True, help="SPAD size label (e.g. 20x20)")
    p.add_argument("--group", default=None, help="Optional group label (e.g. G7). If omitted, auto-assigns.")
    p.add_argument("--epochs", type=int, default=50, help="Epochs per run")
    p.add_argument("--bs", type=int, default=32, help="Batch size")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay")
    p.add_argument("--val-split", type=float, default=0.30, help="Validation fraction per-energy")
    p.add_argument("--workers", type=int, default=8, help="DataLoader workers")
    p.add_argument("--cpu-threads", type=int, default=0, help="torch.set_num_threads; 0=leave default")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--base-dir", default="NN_model", help="Base output directory (shared across runs)")
    p.add_argument("--early-stop", type=int, default=0, help="Patience in epochs (0=off)")
    p.add_argument("--recursive", action="store_true", help="Recursively search tensor_path for *.npz (recommended)")
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="Max gradient norm for clipping (0=off). Strongly recommended to prevent NaN.")

    # --- prediction-only mode ---
    p.add_argument("--predict-only", action="store_true",
                   help="Skip training, run inference on tensor_path and write predictions CSV.")
    p.add_argument("--checkpoint", default="",
                   help="Path to .ckpt or .pth to load for predict-only (or to override resume).")
    p.add_argument("--pred-csv", default="predictions.csv",
                   help="Prediction CSV filename (inside output dir) for predict-only.")
    return p.parse_args()


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_loss_history(loss_csv: Path) -> List[Dict[str, str]]:
    if not loss_csv.exists():
        return []
    with open(loss_csv, "r", newline="") as f:
        return list(csv.DictReader(f))


def determine_group(loss_csv: Path, requested_group: str | None) -> str:
    rows = read_loss_history(loss_csv)
    used = {r["group"] for r in rows} if rows else set()

    if requested_group:
        if requested_group not in used:
            return requested_group
        base = requested_group.rstrip("0123456789")
        i = 1
        while f"{base}{i}" in used:
            i += 1
        return f"{base}{i}"

    i = 0
    while f"G{i}" in used:
        i += 1
    return f"G{i}"


def next_cumulative_epoch(loss_csv: Path) -> int:
    rows = read_loss_history(loss_csv)
    if not rows:
        return 1
    mx = 0
    for r in rows:
        e = r.get("epoch", "")
        if str(e).isdigit():
            mx = max(mx, int(e))
    return mx + 1


def stratified_indices_by_energy(energies: List[float], val_frac: float, seed: int) -> Tuple[List[int], List[int]]:
    energies = np.asarray(energies, dtype=np.float32)
    energies = np.round(energies, 6)

    uniq = sorted(set(energies.tolist()))
    rng = np.random.default_rng(seed)

    tr, va = [], []
    for E in uniq:
        idx = np.where(energies == E)[0].tolist()
        rng.shuffle(idx)
        n = len(idx)
        n_val = max(1, int(round(n * val_frac))) if n > 1 else 0
        if n_val >= n and n > 1:
            n_val = n - 1
        va.extend(idx[:n_val])
        tr.extend(idx[n_val:])

    rng.shuffle(tr)
    rng.shuffle(va)
    return tr, va


# -------------------- NaN / inf diagnostic helpers --------------------

def check_tensor(t: torch.Tensor, name: str, logger=None) -> bool:
    """Returns True if tensor contains NaN or Inf. Logs details."""
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    if has_nan or has_inf:
        msg = (
            f"[NaN/Inf DETECTED] {name}: "
            f"nan={torch.isnan(t).sum().item()}, "
            f"inf={torch.isinf(t).sum().item()}, "
            f"min={t[~torch.isnan(t) & ~torch.isinf(t)].min().item() if (~torch.isnan(t) & ~torch.isinf(t)).any() else 'all-bad'}, "
            f"max={t[~torch.isnan(t) & ~torch.isinf(t)].max().item() if (~torch.isnan(t) & ~torch.isinf(t)).any() else 'all-bad'}"
        )
        print(msg)
        if logger:
            logger.write(msg + "\n")
            logger.flush()
    return bool(has_nan or has_inf)


def check_model_params(model: nn.Module, label: str, logger=None) -> bool:
    """Check all model parameters and gradients for NaN/Inf."""
    any_bad = False
    for name, p in model.named_parameters():
        if check_tensor(p.data, f"{label} param {name}", logger):
            any_bad = True
        if p.grad is not None:
            if check_tensor(p.grad, f"{label} grad {name}", logger):
                any_bad = True
    return any_bad


def audit_dataset(ds, n_samples: int = 20, logger=None):
    """
    Spot-check random samples from the dataset for NaN/Inf/zero energies.
    Call this once before training starts.
    """
    print(f"\n=== Dataset audit ({min(n_samples, len(ds))} samples) ===")
    rng = np.random.default_rng(0)
    indices = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False).tolist()
    bad_count = 0
    for i in indices:
        x, y, name = ds[i]
        x_bad = check_tensor(x, f"sample[{i}] x (file={name})", logger)
        # y is a scalar tensor
        y_val = float(y.item()) if hasattr(y, 'item') else float(y)
        if y_val <= 0 or np.isnan(y_val) or np.isinf(y_val):
            msg = f"[BAD ENERGY] sample[{i}] file={name} energy={y_val}"
            print(msg)
            if logger:
                logger.write(msg + "\n")
                logger.flush()
            bad_count += 1
        if x_bad:
            bad_count += 1
    print(f"Audit complete: {bad_count} bad samples found out of {min(n_samples, len(ds))} checked.\n")
    return bad_count


# -------------------- log-energy helpers --------------------

def safe_log_energy(y: torch.Tensor) -> torch.Tensor:
    # FIX: clamp to a more meaningful minimum (1e-6 eV) rather than 1e-12
    # which, after log, gives -27.6 — a valid float, but suspicious. More
    # importantly, if y contains 0 or negative from bad data, this masks it.
    clamped = torch.clamp(y, min=1e-6)
    result = torch.log(clamped)
    return result


def compute_target_norm(train_targets_log: torch.Tensor) -> Tuple[float, float]:
    # Guard against NaN in training targets before computing stats
    valid = train_targets_log[~torch.isnan(train_targets_log) & ~torch.isinf(train_targets_log)]
    if len(valid) == 0:
        raise RuntimeError(
            "All log-energies in the training set are NaN or Inf. "
            "Check that your energy labels are positive finite values."
        )
    if len(valid) < len(train_targets_log):
        print(f"WARNING: {len(train_targets_log) - len(valid)} NaN/Inf log-energy values excluded from norm computation.")

    mu = float(valid.mean().item())
    sigma = float(valid.std().item())
    if sigma <= 0 or np.isnan(sigma):
        print("WARNING: sigma_logE is 0 or NaN (all training energies identical?). Setting sigma=1.0.")
        sigma = 1.0
    return mu, sigma


def normalize_targets(y_log: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    return (y_log - mu) / sigma


def denormalize_targets(y_norm: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    return y_norm * sigma + mu


def load_checkpoint_into(model, optimizer, path: Path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        if optimizer is not None and "optim" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optim"])
            except Exception:
                pass
        meta = ckpt
    else:
        model.load_state_dict(ckpt)
        meta = {}
    return meta


def run_predict_only(model, loader, dev, use_cuda, mu_logE, sigma_logE, out_csv: Path, use_log1p: bool = False):
    model.eval()
    amp_ctx = torch.amp.autocast("cuda", enabled=use_cuda)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "true_energy", "pred_energy", "deviation", "abs_error", "squared_error"])

        with torch.no_grad(), (amp_ctx if use_cuda else nullcontext()):
            for x, y, names in loader:
                x = x.to(dev, non_blocking=use_cuda)
                y = y.to(dev, non_blocking=use_cuda)

                if use_log1p:
                    x = torch.log1p(x)

                y_log = safe_log_energy(y)
                y_log_norm = normalize_targets(y_log, mu_logE, sigma_logE)

                out_log_norm = model(x)

                out_log = denormalize_targets(out_log_norm, mu_logE, sigma_logE)
                out_log = torch.clamp(out_log, min=np.log(1e-3), max=np.log(1e4))
                out_linear = torch.exp(out_log).detach().cpu().float()

                y_cpu = y.detach().cpu().float()
                diffs = out_linear - y_cpu

                for name, t, p, d in zip(names, y_cpu.tolist(), out_linear.tolist(), diffs.tolist()):
                    w.writerow([name, f"{t:.6f}", f"{p:.6f}", f"{d:.6f}", f"{abs(d):.6f}", f"{(d*d):.6f}"])


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)

    outdir = Path(f"{args.base_dir}_{args.spad}").resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Open a persistent diagnostic log for this run
    diag_log_path = outdir / "nan_diagnostics.log"
    diag_log = open(diag_log_path, "a")
    diag_log.write(f"\n\n=== New run: epochs={args.epochs}, lr={args.lr}, bs={args.bs} ===\n")

    ds = PhotonEnergyDataset(args.tensor_path, recursive=args.recursive)
    all_E = ds.get_all_energies()
    print(f"Loaded {len(ds)} samples | Energies present: {sorted(set(all_E))}")

    # ---- CRITICAL: audit the dataset for bad values before anything else ----
    n_bad = audit_dataset(ds, n_samples=min(50, len(ds)), logger=diag_log)
    if n_bad > 0:
        print(f"WARNING: {n_bad} bad samples found in audit. Check {diag_log_path} for details.")

    # Check for zero/negative energies (these will produce -inf after log)
    zero_or_neg = [i for i, e in enumerate(all_E) if e <= 0 or np.isnan(e) or np.isinf(e)]
    if zero_or_neg:
        msg = (f"FATAL: {len(zero_or_neg)} samples have energy <= 0, NaN, or Inf. "
               f"Indices: {zero_or_neg[:20]}{'...' if len(zero_or_neg) > 20 else ''}. "
               f"This will cause NaN losses immediately. Fix your dataset.")
        print(msg)
        diag_log.write(msg + "\n")
        diag_log.flush()
        raise RuntimeError(msg)

    use_cuda = torch.cuda.is_available()
    dev = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    model = EnergyRegressionCNN(in_channels=ds.channels).to(dev)

    tr_idx, va_idx = stratified_indices_by_energy(all_E, args.val_split, args.seed)
    if len(tr_idx) == 0:
        raise RuntimeError("Need at least 1 training sample to compute normalization.")
    train_set = Subset(ds, tr_idx)
    val_set   = Subset(ds, va_idx) if len(va_idx) > 0 else None

    # loaders
    if args.predict_only:
        pred_loader = DataLoader(
            ds, batch_size=args.bs, shuffle=False,
            num_workers=max(0, args.workers), pin_memory=use_cuda, drop_last=False
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=args.bs, shuffle=True,
            num_workers=max(0, args.workers), pin_memory=use_cuda, drop_last=False
        )
        val_loader = DataLoader(
            val_set, batch_size=args.bs, shuffle=False,
            num_workers=max(0, args.workers), pin_memory=use_cuda, drop_last=False
        )

    # normalization defaults
    mu_logE = sigma_logE = None

    # checkpoint choice
    best_latest_ckpt = outdir / "best_latest.ckpt"
    last_ckpt        = outdir / "last.ckpt"

    ckpt_path = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    resume_path = ckpt_path if (ckpt_path and ckpt_path.exists()) else (last_ckpt if last_ckpt.exists() else best_latest_ckpt)

    meta = {}
    if resume_path.exists():
        print(f"Loading checkpoint: {resume_path}")
        meta = load_checkpoint_into(model, None, resume_path, map_location="cpu")
        mu_logE = meta.get("mu_logE", None)
        sigma_logE = meta.get("sigma_logE", None)

        # ---- Sanity-check the loaded model weights immediately ----
        print("Checking loaded model weights for NaN/Inf...")
        if check_model_params(model, "loaded-checkpoint", diag_log):
            print("WARNING: Loaded checkpoint contains NaN/Inf weights! Starting from scratch.")
            diag_log.write("WARNING: checkpoint weights had NaN — reinitializing model.\n")
            model = EnergyRegressionCNN(in_channels=ds.channels).to(dev)
            mu_logE = sigma_logE = None

    # compute mu/sigma if missing
    if mu_logE is None or sigma_logE is None:
        train_targets_log_list = []
        with torch.no_grad():
            for _x, y, _name in DataLoader(train_set, batch_size=512, shuffle=False, num_workers=0):
                y_log = safe_log_energy(y.float())
                train_targets_log_list.append(y_log)
        train_targets_log = torch.cat(train_targets_log_list).float()
        mu_logE, sigma_logE = compute_target_norm(train_targets_log)

    mu_logE = float(mu_logE); sigma_logE = float(sigma_logE)
    print(f"mu_logE={mu_logE:.6f}, sigma_logE={sigma_logE:.6f}")

    if np.isnan(mu_logE) or np.isnan(sigma_logE) or np.isinf(mu_logE) or np.isinf(sigma_logE):
        raise RuntimeError(
            f"mu_logE={mu_logE} or sigma_logE={sigma_logE} is NaN/Inf. "
            "Your energy labels are likely all zero, negative, or corrupt."
        )

    diag_log.write(f"mu_logE={mu_logE:.6f}, sigma_logE={sigma_logE:.6f}\n")
    diag_log.flush()

    # Determine whether to apply log1p input scaling (needed when raw pixel values are very large,
    # e.g. photon counts up to 330k, which causes BatchNorm to overflow in train mode).
    # Sample a few inputs to check — works for both training and predict-only paths.
    _sample_loader = DataLoader(Subset(ds, list(range(min(32, len(ds))))),
                                batch_size=32, shuffle=False, num_workers=0)
    _sx, _, _ = next(iter(_sample_loader))
    USE_LOG1P_INPUT = bool(_sx.max().item() > 1000)
    if USE_LOG1P_INPUT:
        print(f"INPUT SCALING: max raw value={_sx.max():.1f} — log1p scaling ENABLED for all splits.")
        diag_log.write(f"log1p scaling enabled (max={_sx.max():.1f})\n")
    else:
        print(f"INPUT SCALING: max raw value={_sx.max():.4f} — log1p scaling not needed.")
    del _sample_loader, _sx
    diag_log.flush()

    # ---------------- PREDICT ONLY ----------------
    if args.predict_only:
        out_csv = outdir / args.pred_csv
        print(f"Predict-only: writing {out_csv}")
        run_predict_only(model, pred_loader, dev, use_cuda, mu_logE, sigma_logE, out_csv, use_log1p=USE_LOG1P_INPUT)
        print("Done.")
        diag_log.close()
        return

    # ---------------- TRAINING ----------------
    loss_csv = outdir / "loss_history.csv"
    val_csv  = outdir / "val_predictions_all_epochs.csv"

    if not loss_csv.exists():
        with open(loss_csv, "w", newline="") as f:
            csv.writer(f).writerow([
                "group","epoch","train_loss","val_loss","val_mae","val_rmse",
                "val_mean_true","val_mean_pred","num_train","num_val",
                "mu_logE","sigma_logE","best_val_at_save"
            ])

    if not val_csv.exists():
        with open(val_csv, "w", newline="") as f:
            csv.writer(f).writerow([
                "group","epoch","filename","true_energy","pred_energy",
                "deviation","squared_error","abs_error"
            ])

    group = determine_group(loss_csv, args.group)
    start_epoch = next_cumulative_epoch(loss_csv)
    print(f"Group {group} | Cumulative training will start at epoch {start_epoch}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
        threshold=1e-4, cooldown=0, min_lr=1e-6
    )

    # resume optimizer + best_val if last/best exists
    best_val = float("inf")
    if resume_path.exists():
        ckpt = torch.load(resume_path, map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt:
            if "optim" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optim"])
                except Exception:
                    pass
            best_val = float(ckpt.get("best_val", best_val))

    amp_ctx = torch.amp.autocast("cuda", enabled=use_cuda)
    scaler = torch.amp.GradScaler("cuda") if use_cuda else None

    best_latest_ckpt = outdir / "best_latest.ckpt"
    best_latest_pth  = outdir / "best_latest.pth"
    last_ckpt        = outdir / "last.ckpt"

    no_improve = 0
    cur_epoch = start_epoch

    # ---- Check first batch manually before training loop ----
    print("Running pre-training sanity check on first batch...")
    first_x, first_y, first_names = next(iter(train_loader))
    first_x_dev = first_x.to(dev)
    if USE_LOG1P_INPUT:
        first_x_dev = torch.log1p(first_x_dev)

    print(f"  Input x (raw): shape={first_x.shape}, min={first_x.min():.4f}, max={first_x.max():.4f}")
    print(f"  Input x (scaled): min={first_x_dev.min():.4f}, max={first_x_dev.max():.4f}, "
          f"nan={torch.isnan(first_x_dev).any()}, inf={torch.isinf(first_x_dev).any()}")
    print(f"  Target y: min={first_y.min():.6f}, max={first_y.max():.6f}, "
          f"nan={torch.isnan(first_y).any()}, inf={torch.isinf(first_y).any()}")

    # Check per-channel input stats
    if first_x_dev.ndim == 4:
        for c in range(first_x_dev.shape[1]):
            ch = first_x_dev[:, c]
            msg = f"  Channel {c} (scaled): mean={ch.mean():.4f}, std={ch.std():.4f}, min={ch.min():.4f}, max={ch.max():.4f}"
            print(msg)
            diag_log.write(msg + "\n")

    first_y_log = safe_log_energy(first_y.float())
    first_y_norm = normalize_targets(first_y_log, mu_logE, sigma_logE)
    print(f"  y_log_norm: min={first_y_norm.min():.4f}, max={first_y_norm.max():.4f}, "
          f"nan={torch.isnan(first_y_norm).any()}")

    # Test eval mode vs train mode — BatchNorm difference is the key diagnostic
    model.eval()
    with torch.no_grad():
        out_eval = model(first_x_dev)
        print(f"  Model output (eval  mode): min={out_eval.min():.4f}, max={out_eval.max():.4f}, "
              f"nan={torch.isnan(out_eval).any()}, inf={torch.isinf(out_eval).any()}")

    model.train()
    with torch.no_grad():
        out_train = model(first_x_dev)
        print(f"  Model output (train mode): min={out_train.min():.4f}, max={out_train.max():.4f}, "
              f"nan={torch.isnan(out_train).any()}, inf={torch.isinf(out_train).any()}")

    if torch.isnan(out_train).any() and not torch.isnan(out_eval).any():
        msg = "  RESIDUAL ISSUE: NaN still in train mode after log1p scaling. BatchNorm may need eps increase or model has instability."
        print(msg); diag_log.write(msg + "\n")
    elif torch.isnan(out_train).any():
        msg = "  RESIDUAL ISSUE: NaN in both eval+train modes. Check model architecture (divide-by-zero, log of zero, etc)."
        print(msg); diag_log.write(msg + "\n")
    else:
        print("  Pre-training check: PASS ✓")

    diag_log.flush()

    for _ in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        nan_batch_count = 0

        for batch_idx, (x, y, _name) in enumerate(train_loader):
            x = x.to(dev, non_blocking=use_cuda)
            y = y.to(dev, non_blocking=use_cuda)

            # Reduce dynamic range so BatchNorm doesn't overflow (330k raw → ~12.7 log1p)
            if USE_LOG1P_INPUT:
                x = torch.log1p(x)

            # ---- Input guard ----
            if torch.isnan(x).any() or torch.isinf(x).any():
                msg = f"[Epoch {cur_epoch} Batch {batch_idx}] NaN/Inf in input x — skipping batch."
                print(msg)
                diag_log.write(msg + "\n")
                diag_log.flush()
                nan_batch_count += 1
                continue

            if torch.isnan(y).any() or torch.isinf(y).any() or (y <= 0).any():
                msg = (f"[Epoch {cur_epoch} Batch {batch_idx}] Bad target y "
                       f"(nan/inf/non-positive) — skipping batch. "
                       f"y range: [{y.min():.4f}, {y.max():.4f}]")
                print(msg)
                diag_log.write(msg + "\n")
                diag_log.flush()
                nan_batch_count += 1
                continue

            y_log = safe_log_energy(y)
            y_log_norm = normalize_targets(y_log, mu_logE, sigma_logE)

            if torch.isnan(y_log_norm).any() or torch.isinf(y_log_norm).any():
                msg = (f"[Epoch {cur_epoch} Batch {batch_idx}] NaN/Inf in y_log_norm — skipping. "
                       f"y_log range: [{y_log.min():.4f}, {y_log.max():.4f}]")
                print(msg)
                diag_log.write(msg + "\n")
                diag_log.flush()
                nan_batch_count += 1
                continue

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                out_log_norm = model(x)
                loss = criterion(out_log_norm, y_log_norm)

            if torch.isnan(loss) or torch.isinf(loss):
                msg = (f"[Epoch {cur_epoch} Batch {batch_idx}] NaN/Inf LOSS={loss.item()} — "
                       f"skipping backward. Checking model params...")
                print(msg)
                diag_log.write(msg + "\n")
                check_model_params(model, f"epoch{cur_epoch}_batch{batch_idx}", diag_log)
                diag_log.flush()
                nan_batch_count += 1
                continue

            if use_cuda:
                scaler.scale(loss).backward()
                # FIX: unscale before clipping so clip operates on true grad magnitudes
                scaler.unscale_(optimizer)
                if args.grad_clip > 0:
                    total_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    if total_norm > args.grad_clip * 10:
                        diag_log.write(
                            f"[Epoch {cur_epoch} Batch {batch_idx}] "
                            f"Large grad norm={total_norm:.4f} (clipped to {args.grad_clip})\n"
                        )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip > 0:
                    total_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    if total_norm > args.grad_clip * 10:
                        diag_log.write(
                            f"[Epoch {cur_epoch} Batch {batch_idx}] "
                            f"Large grad norm={total_norm:.4f} (clipped to {args.grad_clip})\n"
                        )
                optimizer.step()

            train_loss_sum += loss.item()

        if nan_batch_count > 0:
            print(f"[{group}] Epoch {cur_epoch:03d} — WARNING: {nan_batch_count} batches skipped due to NaN/Inf.")
            diag_log.write(f"Epoch {cur_epoch}: {nan_batch_count} batches skipped.\n")
            diag_log.flush()

        # Check model params after each epoch for early NaN detection
        if check_model_params(model, f"after-epoch-{cur_epoch}", diag_log):
            msg = (f"[FATAL] Model weights are NaN/Inf after epoch {cur_epoch}. "
                   "Training is diverged. Try: lower --lr, raise --grad-clip, check data.")
            print(msg)
            diag_log.write(msg + "\n")
            diag_log.flush()
            break

        n_train_batches = max(1, len(train_loader) - nan_batch_count)
        avg_train = train_loss_sum / n_train_batches

        model.eval()
        val_loss_sum = 0.0
        preds_linear, trues_linear, names_all = [], [], []

        with torch.no_grad(), (amp_ctx if use_cuda else nullcontext()):
            for x, y, names in val_loader:
                x = x.to(dev, non_blocking=use_cuda)
                y = y.to(dev, non_blocking=use_cuda)

                if USE_LOG1P_INPUT:
                    x = torch.log1p(x)

                y_log = safe_log_energy(y)
                y_log_norm = normalize_targets(y_log, mu_logE, sigma_logE)

                out_log_norm = model(x)
                val_loss_sum += criterion(out_log_norm, y_log_norm).item()

                out_log = denormalize_targets(out_log_norm, mu_logE, sigma_logE)
                out_log = torch.clamp(out_log, min=np.log(1e-3), max=np.log(1e4))
                out_linear = torch.exp(out_log)

                preds_linear.append(out_linear.detach().cpu())
                trues_linear.append(y.detach().cpu())
                names_all.extend(list(names))

        preds = torch.cat(preds_linear).float()
        trues = torch.cat(trues_linear).float()

        diffs = preds - trues
        val_mae = float(diffs.abs().mean().item())
        val_rmse = float(diffs.pow(2).mean().sqrt().item())

        avg_val = val_loss_sum / max(1, len(val_loader))
        mean_true = float(trues.mean().item())
        mean_pred = float(preds.mean().item())

        print(f"[{group}] Epoch {cur_epoch:03d} | Train {avg_train:.6f} | Val {avg_val:.6f} | "
              f"MAE {val_mae:.6f} | RMSE {val_rmse:.6f}")

        with open(loss_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                group, cur_epoch,
                f"{avg_train:.6f}", f"{avg_val:.6f}",
                f"{val_mae:.6f}", f"{val_rmse:.6f}",
                f"{mean_true:.6f}", f"{mean_pred:.6f}",
                len(train_set), len(val_set),
                f"{mu_logE:.6f}", f"{sigma_logE:.6f}",
                f"{best_val:.6f}"
            ])

        with open(val_csv, "a", newline="") as f:
            w = csv.writer(f)
            for name, t, p in zip(names_all, trues.tolist(), preds.tolist()):
                d = p - t
                w.writerow([
                    group, cur_epoch, name,
                    f"{t:.6f}", f"{p:.6f}", f"{d:.6f}",
                    f"{(d*d):.6f}", f"{abs(d):.6f}"
                ])

        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "best_val": best_val,
                    "mu_logE": mu_logE,
                    "sigma_logE": sigma_logE,
                },
                best_latest_ckpt
            )
            torch.save(model.state_dict(), best_latest_pth)
            torch.save(model.state_dict(), outdir / f"best_{group}.pth")
            print(f"💾 New best (val={best_val:.6f}) saved to {best_latest_ckpt}")
            no_improve = 0
        else:
            no_improve += 1

        torch.save(
            {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "best_val": best_val,
                "mu_logE": mu_logE,
                "sigma_logE": sigma_logE,
            },
            last_ckpt
        )

        if args.early_stop > 0 and no_improve >= args.early_stop:
            print(f"[{group}] Early stopping (patience={args.early_stop})")
            break

        cur_epoch += 1

    diag_log.close()
    print(f"[{group}] Finished. Best val={best_val:.6f}. Logs in {outdir}")
    print(f"NaN diagnostics written to {diag_log_path}")


if __name__ == "__main__":
    main()