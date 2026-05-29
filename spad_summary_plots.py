#!/usr/bin/env python3
"""
make_spad_plots_no_pandas.py  (NO pandas)

Uses:
  1) NN prediction CSVs (one per SPAD) from:
       <ROOT>/NN_Analysis/*_model/NN_model_*/pred_*.csv
     Expected columns: true_energy, pred_energy   (lnN optional)

  2) Photon summary CSV from:
       <ROOT>/batch_jobs/LOGDIR/photon_csvs/photon_summary_by_key_spad.csv
     Expected columns:
       spad_um,n_events_parsed,mean_reach,mean_kept,mean_lost,mean_lost_frac_of_reach,energy_GeV,index,key

Produces:
  - nPhoton vs SPAD size with Lost fraction overlay (dotted, right axis)
  - E_res (overall fractional resolution) vs SPAD size
  - E_res vs loss fraction (lost/reach)  [labels points by SPAD size]
  - A merged summary CSV

Notes:
  - Photon summary is filtered to energies 10..100 step 10 by default.
  - SPAD sizes are restricted to {20,50,75,100,200} by default.
  - NN side uses --prefer_pred if present, else newest pred_*.csv in each NN_model_* dir.
"""

import os
import re
import csv
import glob
import math
import argparse

import numpy as np
import matplotlib.pyplot as plt


MODEL_RE = re.compile(r"NN_model_(\d+)x\1")  # NN_model_20x20 etc.


def parse_spad_from_model_dir(dirname: str) -> int:
    m = MODEL_RE.search(dirname)
    if not m:
        raise ValueError(f"Cannot parse SPAD from model dir name: {dirname}")
    return int(m.group(1))


def newest_pred_csv(model_dir: str, prefer_name: str | None):
    if prefer_name:
        p = os.path.join(model_dir, prefer_name)
        if os.path.exists(p):
            return p

    cands = glob.glob(os.path.join(model_dir, "pred_*.csv"))
    if not cands:
        return None
    # newest by mtime
    return max(cands, key=lambda p: os.stat(p).st_mtime)


def read_pred_metrics(pred_csv: str):
    """
    Returns:
      res_overall_frac = std((pred-true)/true) over all rows
    """
    frac_err = []
    with open(pred_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                t = float(row["true_energy"])
                p = float(row["pred_energy"])
            except Exception:
                continue
            if t == 0:
                continue
            frac_err.append((p - t) / t)

    if not frac_err:
        raise ValueError(f"No usable rows in {pred_csv}")

    arr = np.asarray(frac_err, dtype=np.float64)
    return float(arr.std(ddof=1))


def load_photon_summary_weighted(
    photon_summary_csv: str,
    keep_spads=(20, 50, 75, 100, 200),
    energies=(10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
):
    """
    Reads photon_summary_by_key_spad.csv and returns weighted means per SPAD:
      mean_kept, mean_lost_frac_of_reach

    Weight = n_events_parsed (so runs with more parsed events matter more).
    Filters:
      - spad_um in keep_spads
      - energy_GeV in energies  (if the column exists)
    """
    keep_spads = set(int(x) for x in keep_spads)
    energies = set(float(x) for x in energies)

    # accumulators per spad: sum(w*x), sum(w)
    acc = {sp: {"w": 0.0,
                "kept_wsum": 0.0,
                "lostfrac_wsum": 0.0}
           for sp in keep_spads}

    with open(photon_summary_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        cols = set(r.fieldnames or [])

        # required
        for req in ("spad_um", "n_events_parsed", "mean_kept", "mean_lost_frac_of_reach"):
            if req not in cols:
                raise ValueError(f"{photon_summary_csv} missing '{req}'. Has: {sorted(cols)}")

        has_energy = ("energy_GeV" in cols)

        for row in r:
            try:
                spad = int(float(row["spad_um"]))
            except Exception:
                continue
            if spad not in keep_spads:
                continue

            if has_energy:
                try:
                    E = float(row["energy_GeV"])
                except Exception:
                    continue
                if E not in energies:
                    continue

            try:
                w = float(row["n_events_parsed"])
                kept = float(row["mean_kept"])
                lostfrac = float(row["mean_lost_frac_of_reach"])
            except Exception:
                continue

            if w <= 0:
                continue

            acc[spad]["w"] += w
            acc[spad]["kept_wsum"] += w * kept
            acc[spad]["lostfrac_wsum"] += w * lostfrac

    out = {}
    for spad in sorted(keep_spads):
        w = acc[spad]["w"]
        if w <= 0:
            out[spad] = {"mean_kept": math.nan, "mean_lost_frac": math.nan}
        else:
            out[spad] = {
                "mean_kept": acc[spad]["kept_wsum"] / w,
                "mean_lost_frac": acc[spad]["lostfrac_wsum"] / w,
            }
    return out


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def savefig(outdir: str, name: str):
    p = os.path.join(outdir, name)
    plt.tight_layout()
    plt.savefig(p, dpi=200)
    plt.close()
    print(f"Saved: {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/lustre/work/samumcki/final_dSiPM_run")
    ap.add_argument("--prefer_pred", default="pred_10k_10to100_step10.csv",
                    help="Pred CSV filename to use inside each NN_model_* dir if present; else newest pred_*.csv")
    ap.add_argument("--photon_summary",
                    default="/lustre/work/samumcki/final_dSiPM_run/batch_jobs/LOGDIR/photon_csvs/photon_summary_by_key_spad.csv")
    ap.add_argument("--outdir", default="spad_plots_full_no_pandas")
    ap.add_argument("--spads", default="20,50,75,100,200",
                    help="Comma list of SPAD sizes (um) to include")
    ap.add_argument("--energies", default="10,20,30,40,50,60,70,80,90,100",
                    help="Comma list of energies (GeV) to include for photon summary filtering")
    ap.add_argument(
        "--model_prefix",
        default="",
        help="Prefix for model directories (e.g. 'unnorm_' or '')",
    )
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    outdir = ensure_dir(os.path.join(root, args.outdir))

    spads = tuple(int(x.strip()) for x in args.spads.split(",") if x.strip())
    energies = tuple(float(x.strip()) for x in args.energies.split(",") if x.strip())

    # ---------------- Photon summary (weighted means) ----------------
    if os.path.exists(args.photon_summary):
        phot = load_photon_summary_weighted(
            args.photon_summary,
            keep_spads=spads,
            energies=energies,
        )
    else:
        print("Photon summary not found — skipping photon plots.")
        phot = {}
    # phot[spad] = {"mean_kept":..., "mean_lost_frac":...}

    # ---------------- NN prediction metrics per SPAD ----------------
    model_glob = os.path.join(root, "NN_Analysis", args.model_prefix + "*_model", "NN_model_*")
    model_dirs = [p for p in glob.glob(model_glob) if os.path.isdir(p)]
    if not model_dirs:
        raise SystemExit(f"No model dirs found: {model_glob}")

    nn = {}  # spad -> {"Eres_overall_frac":..., "pred_csv":...}
    for d in model_dirs:
        try:
            spad = parse_spad_from_model_dir(os.path.basename(d))
        except Exception:
            continue
        if spad not in spads:
            continue

        pred_csv = newest_pred_csv(d, args.prefer_pred if args.prefer_pred else None)
        if pred_csv is None:
            print(f"[WARN] No pred_*.csv in {d}")
            continue

        try:
            res = read_pred_metrics(pred_csv)
        except Exception as e:
            print(f"[WARN] Failed reading {pred_csv}: {e}")
            continue

        nn[spad] = {"Eres_overall_frac": res, "pred_csv": pred_csv}

    if not nn:
        raise SystemExit("No NN pred metrics loaded. Check your NN_Analysis folders and pred CSV names.")

    # ---------------- Merge + write summary CSV ----------------
    merged_path = os.path.join(outdir, "spad_merged_summary.csv")
    with open(merged_path, "w", newline="") as f:
        fieldnames = [
            "spad_um",
            "Eres_overall_frac",
            "mean_kept",
            "mean_lost_frac_of_reach",
            "pred_csv",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for spad in sorted(spads):
            if spad not in nn:
                continue
            row = {
                "spad_um": spad,
                "Eres_overall_frac": nn[spad]["Eres_overall_frac"],
                "mean_kept": phot.get(spad, {}).get("mean_kept", math.nan),
                "mean_lost_frac_of_reach": phot.get(spad, {}).get("mean_lost_frac", math.nan),
                "pred_csv": nn[spad]["pred_csv"],
            }
            w.writerow(row)
    print(f"Wrote: {merged_path}")

    # build arrays for plotting (only SPADs with NN metrics)
    spad_list = [sp for sp in sorted(spads) if sp in nn]
    x_spad = np.asarray(spad_list, dtype=np.float64)
    y_res = np.asarray([nn[sp]["Eres_overall_frac"] for sp in spad_list], dtype=np.float64)
    y_kept = np.asarray([phot.get(sp, {}).get("mean_kept", math.nan) for sp in spad_list], dtype=np.float64)
    y_lostfrac = np.asarray([phot.get(sp, {}).get("mean_lost_frac", math.nan) for sp in spad_list], dtype=np.float64)

    # ---------------- Plot 1: nPhoton vs SPAD with lost overlay ----------------
    if np.isfinite(y_kept).any():
        plt.figure()
        fig, ax1 = plt.subplots()
        ax1.plot(x_spad, y_kept, marker="o")
        ax1.set_xlabel("SPAD size (µm)")
        ax1.set_ylabel("Mean detected photons (mean_kept)")

        if np.isfinite(y_lostfrac).any():
            ax2 = ax1.twinx()
            ax2.plot(x_spad, y_lostfrac, marker="o", linestyle=":")
            ax2.set_ylabel("Lost fraction (lost/reach) from sim logs")

        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "nPhoton_vs_SPAD_with_lost_overlay.png"), dpi=200)
        plt.close(fig)
        print(f"Saved: {os.path.join(outdir, 'nPhoton_vs_SPAD_with_lost_overlay.png')}")

    # ---------------- Plot 2: E_res vs SPAD ----------------
    plt.figure()
    plt.plot(x_spad, y_res, marker="o")
    plt.xlabel("SPAD size (µm)")
    plt.ylabel("Overall fractional resolution std((pred-true)/true)")
    plt.title("E_res vs SPAD size")
    savefig(outdir, "Eres_vs_SPAD.png")

    # ---------------- Plot 3: E_res vs loss fraction ----------------
    m = np.isfinite(y_res) & np.isfinite(y_lostfrac)
    if np.any(m):
        plt.figure()
        plt.plot(y_lostfrac[m], y_res[m], marker="o", linestyle="none")
        for sp, xf, yr in zip(x_spad[m], y_lostfrac[m], y_res[m]):
            plt.annotate(str(int(sp)), (xf, yr), textcoords="offset points", xytext=(5, 5))
        plt.xlabel("Loss fraction (lost/reach) from sim logs")
        plt.ylabel("Overall fractional resolution std((pred-true)/true)")
        plt.title("Resolution vs loss fraction")
        savefig(outdir, "Eres_vs_loss_fraction.png")

    print("Done.")


if __name__ == "__main__":
    main()