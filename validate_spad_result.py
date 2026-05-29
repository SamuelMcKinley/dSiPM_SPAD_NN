#!/usr/bin/env python3
import os
import re
import csv
import math
import glob
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

MODEL_RE = re.compile(r"NN_model_(\d+)x\1")

def parse_spad_from_model_dir(dirname: str) -> int:
    m = MODEL_RE.search(dirname)
    if not m:
        raise ValueError(f"Cannot parse SPAD from model dir: {dirname}")
    return int(m.group(1))

def newest_pred_csv(model_dir: str, prefer: str | None):
    if prefer:
        p = os.path.join(model_dir, prefer)
        if os.path.exists(p):
            return p
    cands = glob.glob(os.path.join(model_dir, "pred_*.csv"))
    if not cands:
        return None
    return max(cands, key=lambda p: os.stat(p).st_mtime)

def read_pred_rows(pred_csv: str):
    """
    Reads prediction CSV expected columns:
      filename, lnN, true_energy, pred_energy
    Returns dict of arrays.
    """
    fn = []
    lnN = []
    tE = []
    pE = []
    with open(pred_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                fn.append(row["filename"])
                lnN.append(float(row["lnN"]))
                tE.append(float(row["true_energy"]))
                pE.append(float(row["pred_energy"]))
            except Exception:
                continue
    if not tE:
        raise ValueError(f"No usable rows in {pred_csv}")
    return {
        "filename": np.asarray(fn, dtype=object),
        "lnN": np.asarray(lnN, dtype=np.float64),
        "true": np.asarray(tE, dtype=np.float64),
        "pred": np.asarray(pE, dtype=np.float64),
    }

def per_energy_stats(trueE, predE, lnN):
    """
    Returns sorted energies and per-energy metrics:
      res_frac(E) = std((pred-true)/true)
      meanN(E) = mean(exp(lnN))
      std_lnN(E) = std(lnN)
      corr_res_lnN(E) = corr(residual, lnN) within that energy
    """
    byE = defaultdict(list)
    for t, p, l in zip(trueE, predE, lnN):
        byE[float(t)].append((float(p), float(l)))

    energies = np.array(sorted(byE.keys()), dtype=np.float64)
    res_frac = np.full_like(energies, np.nan, dtype=np.float64)
    meanN = np.full_like(energies, np.nan, dtype=np.float64)
    stdln = np.full_like(energies, np.nan, dtype=np.float64)
    corr = np.full_like(energies, np.nan, dtype=np.float64)
    n = np.zeros_like(energies, dtype=np.int64)

    for i, E in enumerate(energies):
        arr = byE[E]
        n[i] = len(arr)
        pred = np.array([x[0] for x in arr], dtype=np.float64)
        ln = np.array([x[1] for x in arr], dtype=np.float64)
        true = E

        frac_err = (pred - true) / max(true, 1e-12)
        res_frac[i] = np.std(frac_err, ddof=1) if frac_err.size > 1 else np.nan
        meanN[i] = float(np.mean(np.exp(ln)))
        stdln[i] = float(np.std(ln, ddof=1)) if ln.size > 1 else np.nan

        resid = (pred - true)
        if resid.size > 2 and np.std(resid) > 0 and np.std(ln) > 0:
            corr[i] = float(np.corrcoef(resid, ln)[0, 1])
        else:
            corr[i] = np.nan

    return energies, res_frac, meanN, stdln, corr, n

def safe_first_array_from_npz(path: str):
    """
    Loads an .npz and returns the first array it finds (by key order).
    """
    z = np.load(path, allow_pickle=False)
    keys = list(z.keys())
    if not keys:
        return None, None
    k = keys[0]
    return k, z[k]

def tensor_sparsity_sample(filenames, max_files=200, seed=0):
    """
    Samples up to max_files .npz tensors and returns:
      mean_sum, mean_nnz, mean_size, mean_nnz_frac
    This is the big check for 20x20 vs 200x200 sparsity.
    """
    rng = np.random.default_rng(seed)
    files = [f for f in filenames if isinstance(f, str) and f.endswith(".npz") and os.path.exists(f)]
    if not files:
        return None

    if len(files) > max_files:
        files = list(rng.choice(files, size=max_files, replace=False))

    sums = []
    nnz = []
    sizes = []
    nnz_frac = []

    for f in files:
        try:
            key, arr = safe_first_array_from_npz(f)
            if arr is None:
                continue
            a = np.asarray(arr)
            sizes.append(a.size)
            s = float(np.sum(a))
            nz = int(np.count_nonzero(a))
            sums.append(s)
            nnz.append(nz)
            nnz_frac.append(nz / a.size if a.size > 0 else np.nan)
        except Exception:
            continue

    if not sums:
        return None

    return {
        "n_files": len(sums),
        "mean_sum": float(np.mean(sums)),
        "mean_nnz": float(np.mean(nnz)),
        "mean_size": float(np.mean(sizes)),
        "mean_nnz_frac": float(np.mean(nnz_frac)),
    }

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def savefig(outdir, name):
    path = os.path.join(outdir, name)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/lustre/work/samumcki/final_dSiPM_run")
    ap.add_argument("--prefer_pred", default="pred_10k_10to100_step10.csv")
    ap.add_argument("--outdir", default="spad_validation")
    ap.add_argument("--spads", default="20,50,75,100,200")
    ap.add_argument("--max_npz_per_spad", type=int, default=200,
                    help="How many NPZ tensors to sample per SPAD for sparsity test (0 disables)")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    outdir = ensure_dir(os.path.join(root, args.outdir))
    spads = [int(x.strip()) for x in args.spads.split(",") if x.strip()]

    model_glob = os.path.join(root, "NN_Analysis", "*_model", "NN_model_*")
    model_dirs = [p for p in glob.glob(model_glob) if os.path.isdir(p)]

    # Load per-SPAD prediction data
    per_spad = {}
    for d in model_dirs:
        try:
            spad = parse_spad_from_model_dir(os.path.basename(d))
        except Exception:
            continue
        if spad not in spads:
            continue
        pred_csv = newest_pred_csv(d, args.prefer_pred)
        if pred_csv is None:
            print(f"[WARN] No pred csv in {d}")
            continue
        rows = read_pred_rows(pred_csv)
        per_spad[spad] = {"pred_csv": pred_csv, **rows}

    if not per_spad:
        raise SystemExit("No SPAD pred CSVs loaded. Check paths and --prefer_pred name.")

    # Compute per-energy metrics and (optional) sparsity
    summary = {}
    for spad in spads:
        if spad not in per_spad:
            continue
        r = per_spad[spad]
        E, resF, meanN, stdln, corr, n = per_energy_stats(r["true"], r["pred"], r["lnN"])
        summary[spad] = {
            "E": E, "resF": resF, "meanN": meanN, "stdln": stdln, "corr": corr, "n": n
        }

        if args.max_npz_per_spad > 0:
            s = tensor_sparsity_sample(r["filename"], max_files=args.max_npz_per_spad, seed=spad)
            summary[spad]["sparsity"] = s

    # --------- TEST 1: Resolution vs Energy for each SPAD ----------
    plt.figure()
    for spad in spads:
        if spad not in summary:
            continue
        plt.plot(summary[spad]["E"], summary[spad]["resF"], marker="o", label=f"{spad}x{spad}")
    plt.xlabel("Energy (GeV)")
    plt.ylabel("Fractional resolution std((pred-true)/true)")
    plt.title("Resolution vs Energy (all SPADs)")
    plt.legend()
    savefig(outdir, "check1_resolution_vs_energy_all_spads.png")

    # --------- TEST 2: mean exp(lnN) vs Energy ----------
    plt.figure()
    for spad in spads:
        if spad not in summary:
            continue
        plt.plot(summary[spad]["E"], summary[spad]["meanN"], marker="o", label=f"{spad}x{spad}")
    plt.xlabel("Energy (GeV)")
    plt.ylabel("Mean N = mean(exp(lnN))")
    plt.title("Mean photon proxy vs Energy (all SPADs)")
    plt.legend()
    savefig(outdir, "check2_meanN_vs_energy_all_spads.png")

    # --------- TEST 3: std(lnN) vs Energy ----------
    plt.figure()
    for spad in spads:
        if spad not in summary:
            continue
        plt.plot(summary[spad]["E"], summary[spad]["stdln"], marker="o", label=f"{spad}x{spad}")
    plt.xlabel("Energy (GeV)")
    plt.ylabel("std(lnN)")
    plt.title("Spread of lnN vs Energy (all SPADs)")
    plt.legend()
    savefig(outdir, "check3_stdlnN_vs_energy_all_spads.png")

    # --------- TEST 4: Does resolution follow ~1/sqrt(meanN)? ----------
    # For each SPAD, plot resF vs 1/sqrt(meanN) over energies.
    plt.figure()
    for spad in spads:
        if spad not in summary:
            continue
        meanN = summary[spad]["meanN"]
        resF = summary[spad]["resF"]
        m = np.isfinite(meanN) & np.isfinite(resF) & (meanN > 0)
        x = 1.0 / np.sqrt(meanN[m])
        y = resF[m]
        plt.plot(x, y, marker="o", linestyle="none", label=f"{spad}x{spad}")
    plt.xlabel("1 / sqrt(mean(exp(lnN)))")
    plt.ylabel("Fractional resolution")
    plt.title("Photon-statistics test: resolution vs 1/sqrt(N)")
    plt.legend()
    savefig(outdir, "check4_resolution_vs_inv_sqrt_meanN.png")

    # --------- TEST 5: Is residual correlated with lnN? ----------
    # Per energy, corr(residual, lnN). If large magnitude, NN depends strongly on lnN.
    plt.figure()
    for spad in spads:
        if spad not in summary:
            continue
        plt.plot(summary[spad]["E"], summary[spad]["corr"], marker="o", label=f"{spad}x{spad}")
    plt.axhline(0.0)
    plt.xlabel("Energy (GeV)")
    plt.ylabel("corr(residual, lnN) within energy bin")
    plt.title("lnN-dependence diagnostic")
    plt.legend()
    savefig(outdir, "check5_corr_residual_vs_lnN.png")

    # --------- TEST 6 (optional): Tensor sparsity report ----------
    report_path = os.path.join(outdir, "sparsity_report.txt")
    with open(report_path, "w") as f:
        f.write("Tensor sparsity sample (from filenames in pred CSV):\n\n")
        for spad in spads:
            if spad not in summary:
                continue
            s = summary[spad].get("sparsity", None)
            f.write(f"SPAD {spad}x{spad}\n")
            if s is None:
                f.write("  (no npz sampled / none found)\n\n")
                continue
            f.write(f"  n_files_sampled: {s['n_files']}\n")
            f.write(f"  mean_sum:        {s['mean_sum']:.3f}\n")
            f.write(f"  mean_nnz:        {s['mean_nnz']:.3f}\n")
            f.write(f"  mean_size:       {s['mean_size']:.3f}\n")
            f.write(f"  mean_nnz_frac:   {s['mean_nnz_frac']:.6f}\n\n")
    print(f"Wrote: {report_path}")

    # --------- Save a compact numeric summary CSV (no pandas) ----------
    out_csv = os.path.join(outdir, "per_spad_energy_metrics.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["spad_um", "energy_GeV", "n_events", "res_frac", "meanN", "std_lnN", "corr_resid_lnN"])
        for spad in spads:
            if spad not in summary:
                continue
            E = summary[spad]["E"]
            n = summary[spad]["n"]
            resF = summary[spad]["resF"]
            meanN = summary[spad]["meanN"]
            stdln = summary[spad]["stdln"]
            corr = summary[spad]["corr"]
            for i in range(E.size):
                w.writerow([spad, float(E[i]), int(n[i]), float(resF[i]), float(meanN[i]), float(stdln[i]), float(corr[i])])
    print(f"Wrote: {out_csv}")

    print("\nDone. Open the plots in:", outdir)

if __name__ == "__main__":
    main()