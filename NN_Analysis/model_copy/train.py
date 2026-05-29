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


# -------------------- log-energy helpers --------------------

def safe_log_energy(y: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.clamp(y, min=1e-12))


def compute_target_norm(train_targets_log: torch.Tensor) -> Tuple[float, float]:
    mu = float(train_targets_log.mean().item())
    sigma = float(train_targets_log.std().item())
    if sigma <= 0:
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


def run_predict_only(model, loader, dev, use_cuda, mu_logE, sigma_logE, out_csv: Path):
    model.eval()
    amp_ctx = torch.amp.autocast("cuda", enabled=use_cuda)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "true_energy", "pred_energy", "deviation", "abs_error", "squared_error"])

        with torch.no_grad(), (amp_ctx if use_cuda else nullcontext()):
            for x, y, names in loader:
                x = x.to(dev, non_blocking=use_cuda)
                y = y.to(dev, non_blocking=use_cuda)

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

    ds = PhotonEnergyDataset(args.tensor_path, recursive=args.recursive)
    all_E = ds.get_all_energies()
    print(f"Loaded {len(ds)} samples | Energies present: {sorted(set(all_E))}")

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

    # compute mu/sigma if missing (needed for both train + predict-only)
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

    # ---------------- PREDICT ONLY ----------------
    if args.predict_only:
        out_csv = outdir / args.pred_csv
        print(f"Predict-only: writing {out_csv}")
        run_predict_only(model, pred_loader, dev, use_cuda, mu_logE, sigma_logE, out_csv)
        print("Done.")
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

    for _ in range(args.epochs):
        model.train()
        train_loss_sum = 0.0

        for x, y, _name in train_loader:
            x = x.to(dev, non_blocking=use_cuda)
            y = y.to(dev, non_blocking=use_cuda)

            y_log = safe_log_energy(y)
            y_log_norm = normalize_targets(y_log, mu_logE, sigma_logE)

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                out_log_norm = model(x)
                loss = criterion(out_log_norm, y_log_norm)

            if use_cuda:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss_sum += loss.item()

        avg_train = train_loss_sum / max(1, len(train_loader))

        model.eval()
        val_loss_sum = 0.0
        preds_linear, trues_linear, names_all = [], [], []

        with torch.no_grad(), (amp_ctx if use_cuda else nullcontext()):
            for x, y, names in val_loader:
                x = x.to(dev, non_blocking=use_cuda)
                y = y.to(dev, non_blocking=use_cuda)

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

    print(f"[{group}] Finished. Best val={best_val:.6f}. Logs in {outdir}")


if __name__ == "__main__":
    main()