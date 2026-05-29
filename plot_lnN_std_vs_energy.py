#!/usr/bin/env python3

import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path


def read_pred_csv(path):
    energies = []
    lnNs = []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            energies.append(float(row["true_energy"]))
            lnNs.append(float(row["lnN"]))

    return np.array(energies), np.array(lnNs)


def compute_std_vs_energy(energies, lnNs):
    bins = defaultdict(list)

    for E, l in zip(energies, lnNs):
        bins[E].append(l)

    Es = []
    stds = []
    means = []

    for E in sorted(bins.keys()):
        arr = np.array(bins[E])
        Es.append(E)
        stds.append(np.std(arr))
        means.append(np.mean(arr))

    return np.array(Es), np.array(stds), np.array(means)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", nargs="+", required=True,
                        help="Prediction CSV files (one per SPAD size)")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Labels for SPAD sizes (same order as --pred)")
    args = parser.parse_args()

    plt.figure(figsize=(7, 5))

    for file, label in zip(args.pred, args.labels):
        energies, lnNs = read_pred_csv(file)
        Es, stds, means = compute_std_vs_energy(energies, lnNs)

        plt.plot(Es, stds, marker="o", label=label)

    plt.xlabel("Energy (GeV)")
    plt.ylabel("STD of lnN")
    plt.title("σ(lnN) vs Energy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("std_lnN_vs_energy.png", dpi=200)
    plt.show()

    # ---- Also plot vs 1/sqrt(E) ----
    plt.figure(figsize=(7, 5))

    for file, label in zip(args.pred, args.labels):
        energies, lnNs = read_pred_csv(file)
        Es, stds, _ = compute_std_vs_energy(energies, lnNs)

        x = 1 / np.sqrt(Es)
        plt.plot(x, stds, marker="o", label=label)

    plt.xlabel("1 / sqrt(E)  [1/sqrt(GeV)]")
    plt.ylabel("STD of lnN")
    plt.title("σ(lnN) vs 1/√E")
    plt.legend()
    plt.tight_layout()
    plt.savefig("std_lnN_vs_inv_sqrtE.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()