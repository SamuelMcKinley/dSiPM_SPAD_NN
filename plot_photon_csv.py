#   ***** outdated? *****
#   Code plots Deadtime loss fraction vs. Energy & lnN vs. Energy
#   Input: python plot_photon_csv.py <path_to_photon_tracking.csv> -o <outdir> --events-per-group <events each csv row summarizes>
#


import argparse
import csv
import math
import os
import matplotlib.pyplot as plt

def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")

def std_sample(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="Path to photon_tracking.csv (or similar)")
    ap.add_argument("-o", "--outdir", default="plots", help="Output directory for PNGs")
    ap.add_argument("--events-per-group", type=float, default=100.0,
                    help="How many events each CSV row summarizes (default: 100)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # energy -> list of values across repeated runs
    loss_frac = {}  # Lost/Total
    lnN_vals  = {}  # ln(Total/EventsPerGroup)

    with open(args.csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"Energy", "Total_Photons", "Lost_Photons"}
        if not required.issubset(reader.fieldnames or []):
            missing = sorted(required - set(reader.fieldnames or []))
            raise SystemExit(f"Missing required columns in CSV: {missing}")

        for row in reader:
            E = float(row["Energy"])
            total = float(row["Total_Photons"])
            lost = float(row["Lost_Photons"])

            if total <= 0:
                continue

            lf = lost / total
            N_per_event = total / float(args.events_per_group)
            lnN = math.log(max(N_per_event, 1e-12))

            loss_frac.setdefault(E, []).append(lf)
            lnN_vals.setdefault(E, []).append(lnN)

    energies = sorted(loss_frac.keys())

    loss_mean = [mean(loss_frac[E]) for E in energies]
    loss_std  = [std_sample(loss_frac[E]) for E in energies]

    lnN_mean = [mean(lnN_vals[E]) for E in energies]
    lnN_std  = [std_sample(lnN_vals[E]) for E in energies]

    # ---- Plot 1: Deadtime loss fraction vs energy ----
    plt.figure()
    plt.errorbar(energies, loss_mean, yerr=loss_std, fmt="o-", capsize=3)
    plt.xlabel("Energy (GeV)")
    plt.ylabel("Deadtime loss fraction (Lost/Total)")
    plt.title("Deadtime loss fraction vs energy")
    plt.tight_layout()
    out1 = os.path.join(args.outdir, "deadtime_loss_fraction_vs_energy.png")
    plt.savefig(out1, dpi=300)
    plt.close()

    # ---- Plot 2: lnN vs energy ----
    plt.figure()
    plt.errorbar(energies, lnN_mean, yerr=lnN_std, fmt="o-", capsize=3)
    plt.xlabel("Energy (GeV)")
    plt.ylabel("ln(N)  (N = Total_Photons / events_per_group)")
    plt.title("lnN vs energy")
    plt.tight_layout()
    out2 = os.path.join(args.outdir, "lnN_vs_energy.png")
    plt.savefig(out2, dpi=300)
    plt.close()

    print(f"Wrote:\n  {out1}\n  {out2}")
    print(f"(events_per_group = {args.events_per_group:g})")

if __name__ == "__main__":
    main()