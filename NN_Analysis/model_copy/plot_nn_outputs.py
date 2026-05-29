#!/usr/bin/env python3

import os
import csv
import math
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_pred_csv(path):
    xs = []
    lnN = []
    y_true = []
    y_pred = []
    resid = []

    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)

        has_lnn = "lnN" in r.fieldnames

        for row in r:
            try:
                t = float(row["true_energy"])
                p = float(row["pred_energy"])

                if "deviation" in row and row["deviation"] != "":
                    d = float(row["deviation"])
                else:
                    d = p - t

                if has_lnn:
                    ln = float(row["lnN"])
                else:
                    ln = np.nan

            except Exception:
                continue

            xs.append(row.get("filename", ""))
            lnN.append(ln)
            y_true.append(t)
            y_pred.append(p)
            resid.append(d)

    lnN = np.asarray(lnN, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    resid = np.asarray(resid, dtype=np.float64)

    return xs, lnN, y_true, y_pred, resid


def read_loss_history(path, group=None):

    rows = []

    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)

        for row in r:
            if group is not None and row.get("group", "") != group:
                continue
            rows.append(row)

    epochs = []
    train_loss = []
    val_loss = []

    for row in rows:
        try:
            e = int(float(row["epoch"]))
            tr = float(row["train_loss"])
            va = float(row["val_loss"])
        except Exception:
            continue

        epochs.append(e)
        train_loss.append(tr)
        val_loss.append(va)

    if not epochs:
        return None

    idx = np.argsort(np.asarray(epochs))

    return (
        np.asarray(epochs)[idx],
        np.asarray(train_loss)[idx],
        np.asarray(val_loss)[idx],
    )


def ensure_outdir(outdir):
    os.makedirs(outdir, exist_ok=True)
    return outdir


def savefig(outdir, name):
    path = os.path.join(outdir, name)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print("Saved:", path)


def gaussian(mu, sigma, x):

    if sigma <= 0:
        sigma = 1.0

    return (
        1.0
        / (sigma * math.sqrt(2.0 * math.pi))
        * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    )


def per_energy_stats(y_true, resid):

    byE = defaultdict(list)

    for t, d in zip(y_true, resid):
        byE[float(t)].append(float(d))

    energies = np.array(sorted(byE.keys()))

    mean = []
    rms = []
    std = []
    n = []

    for E in energies:

        arr = np.asarray(byE[E])

        n.append(arr.size)
        mean.append(arr.mean())
        rms.append(np.sqrt((arr ** 2).mean()))
        std.append(arr.std())

    return (
        np.asarray(energies),
        np.asarray(mean),
        np.asarray(rms),
        np.asarray(std),
        np.asarray(n),
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--pred", required=True)
    ap.add_argument("--loss", default="")
    ap.add_argument("--group", default=None)
    ap.add_argument("--outdir", default="plots_out")
    ap.add_argument("--max-scatter", type=int, default=50000)

    args = ap.parse_args()

    outdir = ensure_outdir(args.outdir)

    names, lnN, y_true, y_pred, resid = read_pred_csv(args.pred)

    if y_true.size == 0:
        raise SystemExit(f"No usable rows read from: {args.pred}")

    rng = np.random.default_rng(0)

    idx_scatter = np.arange(y_true.size)

    if y_true.size > args.max_scatter:
        idx_scatter = rng.choice(idx_scatter, args.max_scatter, replace=False)

    E, bias, rms, std, n = per_energy_stats(y_true, resid)

    frac_res = rms / E
    frac_bias = bias / E

    # -------------------------
    # Pred vs Truth
    # -------------------------

    plt.figure()

    plt.scatter(
        y_true[idx_scatter],
        y_pred[idx_scatter],
        alpha=0.3,
        s=12,
    )

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())

    plt.plot([lo, hi], [lo, hi])

    plt.xlabel("True Energy (GeV)")
    plt.ylabel("Pred Energy (GeV)")
    plt.title("Prediction vs Truth")

    savefig(outdir, "pred_vs_true_scatter.png")

    # -------------------------
    # Residual Distribution
    # -------------------------

    mu = resid.mean()
    sigma = resid.std()

    plt.figure()

    plt.hist(resid, bins=60, density=True)

    xg = np.linspace(resid.min(), resid.max(), 400)

    plt.plot(xg, gaussian(mu, sigma, xg))

    plt.xlabel("Residual (GeV)")
    plt.ylabel("Density")
    plt.title("Residual Distribution")

    savefig(outdir, "residual_distribution.png")

    # -------------------------
    # Bias vs Energy
    # -------------------------

    plt.figure()

    plt.plot(E, bias, marker="o")

    plt.axhline(0)

    plt.xlabel("Energy (GeV)")
    plt.ylabel("Bias (GeV)")
    plt.title("Bias vs Energy")

    savefig(outdir, "bias_vs_energy.png")

    # -------------------------
    # Resolution vs Energy
    # -------------------------

    plt.figure()

    plt.plot(E, rms, marker="o")

    plt.xlabel("Energy (GeV)")
    plt.ylabel("Resolution (GeV)")
    plt.title("E_res vs Energy")

    savefig(outdir, "Eres_vs_energy.png")

    # -------------------------
    # Fractional Resolution
    # -------------------------

    plt.figure()

    plt.plot(E, frac_res, marker="o")

    plt.xlabel("Energy (GeV)")
    plt.ylabel("σE / E")
    plt.title("Fractional Resolution vs Energy")

    savefig(outdir, "frac_resolution_vs_energy.png")

    # -------------------------
    # σ/E vs 1/sqrt(E)
    # -------------------------

    x = 1.0 / np.sqrt(E)

    plt.figure()

    plt.plot(x, frac_res, marker="o")

    if x.size >= 2:

        a, b = np.polyfit(x, frac_res, 1)

        xline = np.linspace(x.min(), x.max(), 200)

        plt.plot(xline, a * xline + b)

    plt.xlabel("1 / sqrt(E)")
    plt.ylabel("σE / E")

    plt.title("σE/E vs 1/sqrt(E)")

    savefig(outdir, "frac_resolution_vs_inv_sqrtE.png")

    # -------------------------
    # Residual vs Energy scatter
    # -------------------------

    plt.figure()

    plt.scatter(
        y_true[idx_scatter],
        resid[idx_scatter],
        alpha=0.3,
        s=12,
    )

    plt.axhline(0)

    plt.xlabel("True Energy")
    plt.ylabel("Residual")

    plt.title("Residual vs Energy")

    savefig(outdir, "residual_vs_true_energy_scatter.png")

    # -------------------------
    # Pull Distribution
    # -------------------------

    sigma_by_E = {float(e): float(s) for e, s in zip(E, std)}

    pulls = np.array(
        [
            d / max(sigma_by_E.get(float(t), 1.0), 1e-12)
            for t, d in zip(y_true, resid)
        ]
    )

    plt.figure()

    plt.hist(pulls, bins=60, density=True)

    mu = pulls.mean()
    sig = pulls.std()

    xg = np.linspace(pulls.min(), pulls.max(), 400)

    plt.plot(xg, gaussian(mu, sig, xg))

    plt.xlabel("Pull")
    plt.ylabel("Density")

    plt.title("Pull Distribution")

    savefig(outdir, "pull_distribution.png")

    # -------------------------
    # lnN plots only if lnN exists
    # -------------------------

    if np.isfinite(lnN).any():

        plt.figure()

        plt.scatter(
            y_true[idx_scatter],
            lnN[idx_scatter],
            alpha=0.3,
            s=12,
        )

        plt.xlabel("Energy")
        plt.ylabel("lnN")

        plt.title("lnN vs Energy")

        savefig(outdir, "lnN_vs_energy.png")

        plt.figure()

        plt.scatter(
            lnN[idx_scatter],
            np.abs(resid[idx_scatter]),
            alpha=0.3,
            s=12,
        )

        plt.xlabel("lnN")
        plt.ylabel("|Residual|")

        plt.title("|Residual| vs lnN")

        savefig(outdir, "abs_residual_vs_lnN.png")

    # -------------------------
    # Loss curves
    # -------------------------

    if args.loss:

        loss = read_loss_history(args.loss, group=args.group)

        if loss is not None:

            epochs, train_loss, val_loss = loss

            plt.figure()

            plt.plot(epochs, train_loss, marker="o", label="train")
            plt.plot(epochs, val_loss, marker="o", label="val")

            plt.xlabel("Epoch")
            plt.ylabel("Loss")

            plt.legend()

            savefig(outdir, "train_vs_val_loss.png")

            plt.figure()

            plt.plot(epochs, val_loss, marker="o")

            plt.xlabel("Epoch")
            plt.ylabel("Val loss")

            savefig(outdir, "val_loss_vs_epoch.png")


if __name__ == "__main__":
    main()