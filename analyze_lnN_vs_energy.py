#!/usr/bin/env python3
import os
import re
import csv
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

MODEL_RE = re.compile(r"NN_model_(\d+)x\1")

def parse_spad(dirname):
    m = MODEL_RE.search(dirname)
    if not m:
        return None
    return int(m.group(1))

def newest_pred_csv(model_dir, prefer):
    if prefer:
        p = os.path.join(model_dir, prefer)
        if os.path.exists(p):
            return p
    cands = glob.glob(os.path.join(model_dir, "pred_*.csv"))
    if not cands:
        return None
    return max(cands, key=lambda p: os.stat(p).st_mtime)

def read_csv(pred_csv):
    lnN = []
    E = []
    with open(pred_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                lnN.append(float(row["lnN"]))
                E.append(float(row["true_energy"]))
            except:
                continue
    return np.array(lnN), np.array(E)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/lustre/work/samumcki/final_dSiPM_run")
    ap.add_argument("--prefer_pred", default="pred_10k_10to100_step10.csv")
    ap.add_argument("--outdir", default="lnN_analysis")
    ap.add_argument("--spads", default="20,50,75,100,200")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    outdir = ensure_dir(os.path.join(root, args.outdir))
    spads = [int(x.strip()) for x in args.spads.split(",")]

    model_glob = os.path.join(root, "NN_Analysis", "*_model", "NN_model_*")
    model_dirs = [p for p in glob.glob(model_glob) if os.path.isdir(p)]

    summary_path = os.path.join(outdir, "lnN_linear_fit_summary.csv")
    with open(summary_path, "w", newline="") as fsum:
        writer = csv.writer(fsum)
        writer.writerow(["SPAD_um", "slope_a", "intercept_b", "R2"])

        for d in model_dirs:
            spad = parse_spad(os.path.basename(d))
            if spad not in spads:
                continue

            pred_csv = newest_pred_csv(d, args.prefer_pred)
            if not pred_csv:
                continue

            lnN, E = read_csv(pred_csv)
            if len(E) == 0:
                continue

            # Linear fit: E = a lnN + b
            a, b = np.polyfit(lnN, E, 1)
            E_fit = a * lnN + b

            # R^2
            ss_res = np.sum((E - E_fit) ** 2)
            ss_tot = np.sum((E - np.mean(E)) ** 2)
            R2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

            writer.writerow([spad, a, b, R2])

            # Plot
            plt.figure()
            plt.scatter(lnN, E, s=10, alpha=0.3)
            x_line = np.linspace(min(lnN), max(lnN), 200)
            plt.plot(x_line, a*x_line + b)
            plt.xlabel("lnN")
            plt.ylabel("True Energy (GeV)")
            plt.title(f"True Energy vs lnN (SPAD {spad}x{spad})\n"
                      f"E = {a:.3f} lnN + {b:.3f}   R²={R2:.4f}")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"lnN_vs_energy_SPAD{spad}.png"), dpi=200)
            plt.close()

            print(f"SPAD {spad}: slope={a:.4f}, intercept={b:.4f}, R2={R2:.4f}")

    print(f"\nSaved summary to {summary_path}")
    print("Done.")

if __name__ == "__main__":
    main()