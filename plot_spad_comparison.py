#
#   Many useful plots for NN output comparing SPADs
#   Input: python plot_spad_comparison.py --root <root directory, usually the one this code is in> \
#   --pred-name <name of prediction csv file> --spads <all spad sizes> --energies <all energies> \
#   --hist-energies <energies for gaussian histograms> --outdir <output directory>

#   Input list above non-comprehensive. Other optional parameters at start of main()

import os
import csv
import math
import glob
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def savefig(outdir, name):
    path = os.path.join(outdir, name)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def read_pred_csv(path):
    """
    Expected columns:
      filename,true_energy,pred_energy,deviation,abs_error,squared_error
    or with lnN present as extra column.
    """
    true_E = []
    pred_E = []
    resid = []

    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                t = float(row["true_energy"])
                p = float(row["pred_energy"])
                if "deviation" in row and row["deviation"] != "":
                    d = float(row["deviation"])
                else:
                    d = p - t
            except Exception:
                continue

            true_E.append(t)
            pred_E.append(p)
            resid.append(d)

    true_E = np.asarray(true_E, dtype=np.float64)
    pred_E = np.asarray(pred_E, dtype=np.float64)
    resid = np.asarray(resid, dtype=np.float64)

    if true_E.size == 0:
        raise ValueError(f"No usable rows in {path}")

    return true_E, pred_E, resid


def per_energy_stats(true_E, pred_E, resid):
    byE_resid = defaultdict(list)
    byE_pred = defaultdict(list)

    for t, p, d in zip(true_E, pred_E, resid):
        byE_resid[float(t)].append(float(d))
        byE_pred[float(t)].append(float(p))

    energies = np.array(sorted(byE_resid.keys()), dtype=np.float64)

    n = []
    bias = []
    std_resid = []
    rms_resid = []
    frac_resid_std = []
    frac_resid_rms = []
    ratio_mean = []

    for E in energies:
        r = np.asarray(byE_resid[E], dtype=np.float64)
        p = np.asarray(byE_pred[E], dtype=np.float64)

        n.append(r.size)
        bias.append(r.mean())
        std_resid.append(r.std(ddof=0))
        rms_resid.append(np.sqrt(np.mean(r**2)))
        frac_resid_std.append(r.std(ddof=0) / E)
        frac_resid_rms.append(np.sqrt(np.mean(r**2)) / E)
        ratio_mean.append(np.mean(p / E))

    return {
        "energy": energies,
        "n": np.asarray(n, dtype=np.int64),
        "bias": np.asarray(bias, dtype=np.float64),
        "std_resid": np.asarray(std_resid, dtype=np.float64),
        "rms_resid": np.asarray(rms_resid, dtype=np.float64),
        "frac_resid_std": np.asarray(frac_resid_std, dtype=np.float64),
        "frac_resid_rms": np.asarray(frac_resid_rms, dtype=np.float64),
        "ratio_mean": np.asarray(ratio_mean, dtype=np.float64),
    }


def parse_spad_from_dir(model_dir):
    # expects .../NN_model_20x20
    base = os.path.basename(model_dir.rstrip("/"))
    if not base.startswith("NN_model_"):
        raise ValueError(f"Cannot parse SPAD from {model_dir}")
    tag = base.replace("NN_model_", "")
    a, b = tag.split("x")
    if a != b:
        raise ValueError(f"Non-square SPAD tag: {tag}")
    return int(a)


def find_model_dirs(root, prefix):
    pattern = os.path.join(root, "NN_Analysis", f"{prefix}*_model", "NN_model_*")
    dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    return sorted(dirs)


def choose_pred_csv(model_dir, prefer_name=None):
    if prefer_name:
        p = os.path.join(model_dir, prefer_name)
        if os.path.exists(p):
            return p
    cands = glob.glob(os.path.join(model_dir, "pred_*.csv"))
    if not cands:
        return None
    return max(cands, key=lambda x: os.stat(x).st_mtime)


def write_summary_csv(outdir, summary):
    out_csv = os.path.join(outdir, "spad_energy_summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "spad_um",
            "energy_GeV",
            "n_events",
            "bias_GeV",
            "std_resid_GeV",
            "rms_resid_GeV",
            "frac_resid_std",
            "frac_resid_rms",
            "mean_pred_over_true",
            "pred_csv",
        ])
        for spad in sorted(summary.keys()):
            s = summary[spad]
            E = s["stats"]["energy"]
            for i in range(len(E)):
                w.writerow([
                    spad,
                    f"{E[i]:.6f}",
                    int(s["stats"]["n"][i]),
                    f"{s['stats']['bias'][i]:.6f}",
                    f"{s['stats']['std_resid'][i]:.6f}",
                    f"{s['stats']['rms_resid'][i]:.6f}",
                    f"{s['stats']['frac_resid_std'][i]:.6f}",
                    f"{s['stats']['frac_resid_rms'][i]:.6f}",
                    f"{s['stats']['ratio_mean'][i]:.6f}",
                    s["pred_csv"],
                ])
    print(f"Saved: {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/lustre/work/samumcki/final_dSiPM_run")
    ap.add_argument("--model-prefix", default="",  # Used for name prefixes such as A_20x20_model --> "A_". Leave blank in most cases                      
                    help="Set to 'unnorm_' for unnormalized models, or '' for normal models")
    ap.add_argument("--pred-name", default="pred_10k_10to100_step10.csv",
                    help="Preferred prediction CSV filename inside each NN_model_* dir")
    ap.add_argument("--spads", default="20,50,75,100,200")
    ap.add_argument("--energies", default="10,20,30,40,50,60,70,80,90,100")
    ap.add_argument("--hist-energies", default="10,20,80",
                    help="Comma-separated energies for residual histogram overlays")
    ap.add_argument("--outdir", default="spad_comparison_plots")
    ap.add_argument("--hist-bins", type=int, default=60)
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    outdir = ensure_dir(os.path.join(root, args.outdir))

    spads_keep = set(int(x.strip()) for x in args.spads.split(",") if x.strip())
    energies_keep = [float(x.strip()) for x in args.energies.split(",") if x.strip()]
    hist_energies = [float(x.strip()) for x in args.hist_energies.split(",") if x.strip()]

    model_dirs = find_model_dirs(root, args.model_prefix)
    if not model_dirs:
        raise SystemExit(f"No model dirs found under {root}/NN_Analysis with prefix '{args.model_prefix}'")

    summary = {}

    for d in model_dirs:
        try:
            spad = parse_spad_from_dir(d)
        except Exception:
            continue

        if spad not in spads_keep:
            continue

        pred_csv = choose_pred_csv(d, args.pred_name)
        if pred_csv is None:
            print(f"[WARN] No prediction CSV found in {d}")
            continue

        try:
            true_E, pred_E, resid = read_pred_csv(pred_csv)
        except Exception as e:
            print(f"[WARN] Failed reading {pred_csv}: {e}")
            continue

        stats = per_energy_stats(true_E, pred_E, resid)

        summary[spad] = {
            "pred_csv": pred_csv,
            "true_E": true_E,
            "pred_E": pred_E,
            "resid": resid,
            "stats": stats,
        }

        print(f"[OK] Loaded SPAD {spad} from {pred_csv}")

    if not summary:
        raise SystemExit("No usable SPAD prediction files loaded.")

    # -------------------------------
    # 1) Fractional resolution vs energy overlay
    # -------------------------------
    plt.figure()
    for spad in sorted(summary.keys()):
        s = summary[spad]["stats"]
        plt.plot(s["energy"], s["frac_resid_std"], marker="o", label=str(spad))
    plt.xlabel("Energy (GeV)")
    plt.ylabel("σ(E_pred - E_true) / E")
    plt.title("Fractional Resolution vs Energy")
    plt.legend(title="SPAD (µm)")
    savefig(outdir, "overlay_frac_resolution_vs_energy.png")

    # -------------------------------
    # 2) Bias vs energy overlay
    # -------------------------------
    plt.figure()
    for spad in sorted(summary.keys()):
        s = summary[spad]["stats"]
        plt.plot(s["energy"], s["bias"], marker="o", label=str(spad))
    plt.axhline(0.0)
    plt.xlabel("Energy (GeV)")
    plt.ylabel("Mean residual (GeV)")
    plt.title("Bias vs Energy")
    plt.legend(title="SPAD (µm)")
    savefig(outdir, "overlay_bias_vs_energy.png")

    # -------------------------------
    # 3) Mean pred/true ratio vs energy overlay
    # -------------------------------
    plt.figure()
    for spad in sorted(summary.keys()):
        s = summary[spad]["stats"]
        plt.plot(s["energy"], s["ratio_mean"], marker="o", label=str(spad))
    plt.axhline(1.0)
    plt.xlabel("Energy (GeV)")
    plt.ylabel("Mean(E_pred / E_true)")
    plt.title("Mean Pred/True vs Energy")
    plt.legend(title="SPAD (µm)")
    savefig(outdir, "overlay_mean_ratio_vs_energy.png")

    # -------------------------------
    # 4) Event counts per energy overlay
    # -------------------------------
    plt.figure()
    for spad in sorted(summary.keys()):
        s = summary[spad]["stats"]
        plt.plot(s["energy"], s["n"], marker="o", label=str(spad))
    plt.xlabel("Energy (GeV)")
    plt.ylabel("N events")
    plt.title("Event Counts per Energy Bin")
    plt.legend(title="SPAD (µm)")
    savefig(outdir, "overlay_event_counts_vs_energy.png")

    # -------------------------------
    # 5) Residual histograms at selected energies
    # -------------------------------
    for Esel in hist_energies:
        plt.figure()
        plotted = False
        for spad in sorted(summary.keys()):
            true_E = summary[spad]["true_E"]
            resid = summary[spad]["resid"]
            mask = np.isclose(true_E, Esel)
            arr = resid[mask]
            if arr.size == 0:
                continue
            plt.hist(arr, bins=args.hist_bins, density=True, histtype="step", label=str(spad))
            plotted = True

        if plotted:
            plt.xlabel(f"Residual at {Esel:.0f} GeV (GeV)")
            plt.ylabel("Density")
            plt.title(f"Residual Distribution at {Esel:.0f} GeV")
            plt.legend(title="SPAD (µm)")
            savefig(outdir, f"overlay_residual_hist_{int(Esel)}GeV.png")
        else:
            plt.close()

    # -------------------------------
    # 6) Overall collapsed fractional std vs SPAD
    # -------------------------------
    plt.figure()
    spad_vals = []
    overall_vals = []
    for spad in sorted(summary.keys()):
        true_E = summary[spad]["true_E"]
        pred_E = summary[spad]["pred_E"]
        frac = (pred_E - true_E) / true_E
        spad_vals.append(spad)
        overall_vals.append(np.std(frac, ddof=1))
    plt.plot(spad_vals, overall_vals, marker="o")
    plt.xlabel("SPAD size (µm)")
    plt.ylabel("Overall std((pred-true)/true)")
    plt.title("Overall Fractional Resolution vs SPAD")
    savefig(outdir, "overall_frac_resolution_vs_spad.png")

    # -------------------------------
    # 7) Write summary CSV
    # -------------------------------
    write_summary_csv(outdir, summary)

    # -------------------------------
    # 8) Simple text summary
    # -------------------------------
    txt = os.path.join(outdir, "notes.txt")
    with open(txt, "w") as f:
        f.write("Loaded SPAD sizes:\n")
        for spad in sorted(summary.keys()):
            f.write(f"  {spad} µm -> {summary[spad]['pred_csv']}\n")
        f.write("\nQuick checks:\n")
        f.write("1. overlay_frac_resolution_vs_energy.png\n")
        f.write("2. overlay_bias_vs_energy.png\n")
        f.write("3. overlay_mean_ratio_vs_energy.png\n")
        f.write("4. overlay_event_counts_vs_energy.png\n")
        f.write("5. overlay_residual_hist_10GeV.png, 20GeV.png, 80GeV.png\n")
        f.write("6. overall_frac_resolution_vs_spad.png\n")
    print(f"Saved: {txt}")

    print("Done.")


if __name__ == "__main__":
    main()