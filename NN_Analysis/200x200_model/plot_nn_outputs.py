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
from matplotlib.gridspec import GridSpec


# -------------------------
# I/O helpers
# -------------------------

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
                d = float(row["deviation"]) if row.get("deviation", "") != "" else p - t
                ln = float(row["lnN"]) if has_lnn else np.nan
            except Exception:
                continue
            xs.append(row.get("filename", ""))
            lnN.append(ln)
            y_true.append(t)
            y_pred.append(p)
            resid.append(d)

    return (
        xs,
        np.asarray(lnN, dtype=np.float64),
        np.asarray(y_true, dtype=np.float64),
        np.asarray(y_pred, dtype=np.float64),
        np.asarray(resid, dtype=np.float64),
    )


def read_loss_history(path, group=None):
    rows = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if group is not None and row.get("group", "") != group:
                continue
            rows.append(row)

    epochs, train_loss, val_loss, val_mae, val_rmse = [], [], [], [], []
    for row in rows:
        try:
            epochs.append(int(float(row["epoch"])))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            val_mae.append(float(row["val_mae"]))
            val_rmse.append(float(row["val_rmse"]))
        except Exception:
            continue

    if not epochs:
        return None

    idx = np.argsort(np.asarray(epochs))
    return (
        np.asarray(epochs)[idx],
        np.asarray(train_loss)[idx],
        np.asarray(val_loss)[idx],
        np.asarray(val_mae)[idx],
        np.asarray(val_rmse)[idx],
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
        sigma = 1e-12
    return (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# -------------------------
# Stats helpers
# -------------------------

def per_energy_stats(y_true, resid):
    byE = defaultdict(list)
    for t, d in zip(y_true, resid):
        byE[float(t)].append(float(d))

    energies = np.array(sorted(byE.keys()))
    mean, rms, std, n = [], [], [], []
    raw = []

    for E in energies:
        arr = np.asarray(byE[E])
        n.append(arr.size)
        mean.append(arr.mean())
        rms.append(np.sqrt((arr ** 2).mean()))
        std.append(arr.std())
        raw.append(arr)

    return (
        energies,
        np.asarray(mean),
        np.asarray(rms),
        np.asarray(std),
        np.asarray(n),
        raw,
    )


def stochastic_model(sqrtE_inv, a, b):
    """σE/E = a/√E + b  (stochastic + constant term)"""
    return a * sqrtE_inv + b


def fit_stochastic_linear(E, frac_res):
    """
    Fit σE/E = a/√E + b using pure numpy.
    Since this model is linear in x = 1/sqrt(E), we can use linear least squares.

    Returns:
        a, b, a_err, b_err
    """
    x = 1.0 / np.sqrt(E)
    y = frac_res

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < 2:
        raise RuntimeError("Not enough valid points for fit.")

    A = np.column_stack([x, np.ones_like(x)])
    coeffs, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
    a, b = coeffs

    n = len(y)
    p = 2

    if n > p:
        y_fit = A @ coeffs
        rss = np.sum((y - y_fit) ** 2)
        dof = n - p
        sigma2 = rss / dof if dof > 0 else 0.0
        cov = sigma2 * np.linalg.inv(A.T @ A)
        errs = np.sqrt(np.diag(cov))
        a_err, b_err = errs
    else:
        a_err, b_err = np.nan, np.nan

    return a, b, a_err, b_err


# -------------------------
# Existing plots (unchanged)
# -------------------------

def plot_pred_vs_true(y_true, y_pred, idx_scatter, outdir):
    plt.figure()
    plt.scatter(y_true[idx_scatter], y_pred[idx_scatter], alpha=0.3, s=12)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="ideal")
    plt.xlabel("True Energy (GeV)")
    plt.ylabel("Pred Energy (GeV)")
    plt.title("Prediction vs Truth")
    plt.legend()
    savefig(outdir, "pred_vs_true_scatter.png")


def plot_residual_distribution(resid, outdir):
    mu, sigma = resid.mean(), resid.std()
    plt.figure()
    plt.hist(resid, bins=60, density=True)
    xg = np.linspace(resid.min(), resid.max(), 400)
    plt.plot(xg, gaussian(mu, sigma, xg), label=f"μ={mu:.3f}, σ={sigma:.3f}")
    plt.xlabel("Residual (GeV)")
    plt.ylabel("Density")
    plt.title("Residual Distribution")
    plt.legend()
    savefig(outdir, "residual_distribution.png")


def plot_bias_vs_energy(E, bias, outdir):
    plt.figure()
    plt.plot(E, bias, marker="o")
    plt.axhline(0, color="k", lw=0.8, ls="--")
    plt.xlabel("Energy (GeV)")
    plt.ylabel("Bias (GeV)")
    plt.title("Bias vs Energy")
    savefig(outdir, "bias_vs_energy.png")


def plot_resolution_vs_energy(E, rms, outdir):
    plt.figure()
    plt.plot(E, rms, marker="o")
    plt.xlabel("Energy (GeV)")
    plt.ylabel("Resolution (GeV)")
    plt.title("Energy Resolution vs Energy")
    savefig(outdir, "Eres_vs_energy.png")


def plot_frac_resolution(E, frac_res, outdir):
    plt.figure()
    plt.plot(E, frac_res, marker="o")
    plt.xlabel("Energy (GeV)")
    plt.ylabel("σE / E")
    plt.title("Fractional Resolution vs Energy")
    savefig(outdir, "frac_resolution_vs_energy.png")


def plot_frac_vs_inv_sqrtE(E, frac_res, outdir):
    x = 1.0 / np.sqrt(E)
    plt.figure()
    plt.scatter(x, frac_res, zorder=3)
    if x.size >= 2:
        a, b = np.polyfit(x, frac_res, 1)
        xline = np.linspace(x.min(), x.max(), 200)
        plt.plot(xline, a * xline + b, label=f"a={a:.3f}, b={b:.3f}")
        plt.legend()
    plt.xlabel("1 / √E")
    plt.ylabel("σE / E")
    plt.title("σE/E vs 1/√E")
    savefig(outdir, "frac_resolution_vs_inv_sqrtE.png")


def plot_residual_scatter(y_true, resid, idx_scatter, outdir):
    plt.figure()
    plt.scatter(y_true[idx_scatter], resid[idx_scatter], alpha=0.3, s=12)
    plt.axhline(0, color="r", lw=1, ls="--")
    plt.xlabel("True Energy (GeV)")
    plt.ylabel("Residual (GeV)")
    plt.title("Residual vs True Energy")
    savefig(outdir, "residual_vs_true_energy_scatter.png")


def plot_pull_distribution(y_true, resid, E, std, outdir):
    sigma_by_E = {float(e): float(s) for e, s in zip(E, std)}
    pulls = np.array([
        d / max(sigma_by_E.get(float(t), 1.0), 1e-12)
        for t, d in zip(y_true, resid)
    ])
    mu, sig = pulls.mean(), pulls.std()
    plt.figure()
    plt.hist(pulls, bins=60, density=True)
    xg = np.linspace(pulls.min(), pulls.max(), 400)
    plt.plot(xg, gaussian(mu, sig, xg), label=f"μ={mu:.3f}, σ={sig:.3f}")
    plt.xlabel("Pull")
    plt.ylabel("Density")
    plt.title("Pull Distribution")
    plt.legend()
    savefig(outdir, "pull_distribution.png")


# -------------------------
# New plots
# -------------------------

def plot_per_energy_residuals(E, raw_resid_list, outdir):
    n = len(E)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).flatten()

    for i, (energy, arr) in enumerate(zip(E, raw_resid_list)):
        ax = axes[i]
        mu, sigma = arr.mean(), arr.std()
        bins = min(30, max(10, arr.size // 5))
        ax.hist(arr, bins=bins, density=True, alpha=0.7)
        if sigma > 1e-12:
            xg = np.linspace(arr.min(), arr.max(), 300)
            ax.plot(xg, gaussian(mu, sigma, xg), lw=1.5)
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_title(f"{energy:.0f} GeV  σ={sigma:.3f}", fontsize=9)
        ax.set_xlabel("Residual (GeV)", fontsize=8)
        ax.tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Residual Distribution per Energy Class", y=1.01)
    savefig(outdir, "per_energy_residual_distributions.png")


def plot_colored_pred_vs_true(y_true, y_pred, idx_scatter, outdir):
    energies_unique = sorted(set(y_true.tolist()))
    cmap = plt.cm.get_cmap("tab20", len(energies_unique))

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, energy in enumerate(energies_unique):
        mask_full = y_true == energy
        mask_scatter = np.zeros(len(y_true), dtype=bool)
        mask_scatter[idx_scatter] = True
        mask = mask_full & mask_scatter
        ax.scatter(y_true[mask], y_pred[mask], s=10, alpha=0.4,
                   color=cmap(i), label=f"{energy:.0f} GeV")

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="ideal")
    ax.set_xlabel("True Energy (GeV)")
    ax.set_ylabel("Pred Energy (GeV)")
    ax.set_title("Prediction vs Truth (coloured by class)")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    savefig(outdir, "pred_vs_true_by_energy_class.png")


def plot_cumulative_error(y_true, y_pred, outdir):
    frac_err = np.abs(y_pred - y_true) / np.maximum(y_true, 1e-12) * 100.0
    thresholds = np.linspace(0, 100, 500)
    cumfrac = np.array([(frac_err <= t).mean() for t in thresholds])

    idx90 = np.searchsorted(cumfrac, 0.90)
    t90 = thresholds[min(idx90, len(thresholds) - 1)]

    plt.figure()
    plt.plot(thresholds, cumfrac * 100)
    plt.axhline(90, color="r", ls="--", lw=1, label=f"90% → {t90:.1f}% err")
    if idx90 < len(thresholds):
        plt.axvline(t90, color="r", ls="--", lw=1)
    plt.xlabel("% Error  |pred − true| / true × 100")
    plt.ylabel("Cumulative fraction of events (%)")
    plt.title("Cumulative Error Containment")
    plt.legend()
    plt.xlim(0, 100)
    plt.ylim(0, 101)
    savefig(outdir, "cumulative_error_containment.png")


def plot_relative_bias(E, bias, outdir):
    rel_bias = bias / E
    plt.figure()
    plt.plot(E, rel_bias * 100, marker="o")
    plt.axhline(0, color="k", lw=0.8, ls="--")
    plt.xlabel("Energy (GeV)")
    plt.ylabel("Relative Bias  (bias / E) × 100%")
    plt.title("Relative Bias vs Energy")
    savefig(outdir, "relative_bias_vs_energy.png")


def plot_stochastic_fit(E, frac_res, outdir):
    plt.figure()
    plt.scatter(E, frac_res, zorder=3, label="data")

    if E.size >= 3:
        try:
            a, b, a_err, b_err = fit_stochastic_linear(E, frac_res)
            E_fit = np.linspace(E.min(), E.max(), 300)
            fit_curve = stochastic_model(1.0 / np.sqrt(E_fit), a, b)

            if np.isfinite(a_err) and np.isfinite(b_err):
                label = f"a/√E + b  a={a:.4f}±{a_err:.4f}  b={b:.4f}±{b_err:.4f}"
            else:
                label = f"a/√E + b  a={a:.4f}  b={b:.4f}"

            plt.plot(E_fit, fit_curve, lw=1.8, label=label)
        except Exception as exc:
            print(f"  Stochastic fit failed: {exc}")

    plt.xlabel("Energy (GeV)")
    plt.ylabel("σE / E")
    plt.title("Fractional Resolution with Stochastic Fit  σE/E = a/√E + b")
    plt.legend(fontsize=9)
    savefig(outdir, "frac_resolution_stochastic_fit.png")


def plot_boxplot_per_energy(E, raw_resid_list, outdir):
    fig, ax = plt.subplots(figsize=(max(8, len(E) * 0.8), 5))
    ax.boxplot(raw_resid_list, labels=[f"{e:.0f}" for e in E],
               showfliers=True, flierprops=dict(marker=".", markersize=3, alpha=0.4))
    ax.axhline(0, color="r", lw=0.9, ls="--")
    ax.set_xlabel("True Energy (GeV)")
    ax.set_ylabel("Residual (GeV)")
    ax.set_title("Residual Boxplot per Energy Class")
    savefig(outdir, "residual_boxplot_per_energy.png")


def plot_abs_error_vs_energy(E, rms, std, n, outdir):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(E, rms, marker="o", color="steelblue", label="RMS error")
    ax1.fill_between(E, rms - std, rms + std, alpha=0.2, color="steelblue", label="±1σ")
    ax2.bar(E, n, width=np.diff(E).min() * 0.6 if len(E) > 1 else 5,
            alpha=0.25, color="gray", label="N samples")

    ax1.set_xlabel("Energy (GeV)")
    ax1.set_ylabel("Absolute Error (GeV)", color="steelblue")
    ax2.set_ylabel("N samples", color="gray")
    ax1.set_title("Absolute Error ± σ and Sample Count per Class")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    savefig(outdir, "abs_error_and_sample_count.png")


def plot_2d_confusion(y_true, y_pred, outdir):
    energies = sorted(set(y_true.tolist()))
    bins = np.array(energies)
    if len(bins) > 1:
        half = np.diff(bins).min() / 2.0
    else:
        half = 1.0
    edges = np.concatenate([[bins[0] - half], (bins[:-1] + bins[1:]) / 2, [bins[-1] + half]])

    H, xedges, yedges = np.histogram2d(y_true, y_pred, bins=[edges, edges])

    row_sums = H.sum(axis=1, keepdims=True)
    H_norm = np.where(row_sums > 0, H / row_sums, 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(H_norm, origin="lower", aspect="auto",
                   extent=[edges[0], edges[-1], edges[0], edges[-1]],
                   cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Fraction (row-normalised)")
    ax.set_xlabel("Pred Energy (GeV)")
    ax.set_ylabel("True Energy (GeV)")
    ax.set_title("Row-Normalised Prediction Confusion Matrix")
    ax.set_xticks(energies)
    ax.set_xticklabels([f"{e:.0f}" for e in energies], rotation=45, fontsize=7)
    ax.set_yticks(energies)
    ax.set_yticklabels([f"{e:.0f}" for e in energies], fontsize=7)
    savefig(outdir, "confusion_matrix_2d.png")


def plot_error_correlation(y_true, resid, outdir):
    abs_res = np.abs(resid)
    byE = defaultdict(list)
    for t, d in zip(y_true, abs_res):
        byE[float(t)].append(d)
    E_vals = np.array(sorted(byE.keys()))
    med_abs = np.array([np.median(byE[e]) for e in E_vals])

    fig, ax = plt.subplots()
    ax.scatter(y_true, abs_res, s=5, alpha=0.2, label="all events")
    ax.plot(E_vals, med_abs, "ro-", lw=2, ms=6, label="median per class")

    if E_vals.size >= 3:
        try:
            log_coeff = np.polyfit(np.log(E_vals), np.log(np.maximum(med_abs, 1e-12)), 1)
            alpha = log_coeff[0]
            E_fit = np.linspace(E_vals.min(), E_vals.max(), 200)
            ax.plot(E_fit, np.exp(log_coeff[1]) * E_fit ** alpha, "k--",
                    label=f"power law  α={alpha:.2f}")
        except Exception:
            pass

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("True Energy (GeV)")
    ax.set_ylabel("|Residual| (GeV)")
    ax.set_title("Absolute Error Scaling (log-log)")
    ax.legend(fontsize=8)
    savefig(outdir, "abs_error_log_log_scaling.png")

def plot_mean_pred_with_errorbars(y_true, y_pred, outdir):
    """
    For each true-energy class, plot the mean predicted energy with ±1σ error bars.
    x = true energy class
    y = mean predicted energy in that class
    error bar = std(predicted energy within that class)
    """
    byE = defaultdict(list)
    for t, p in zip(y_true, y_pred):
        byE[float(t)].append(float(p))

    E = np.array(sorted(byE.keys()), dtype=np.float64)
    mean_pred = np.array([np.mean(byE[e]) for e in E], dtype=np.float64)
    std_pred = np.array([np.std(byE[e]) for e in E], dtype=np.float64)

    plt.figure()
    plt.errorbar(E, mean_pred, yerr=std_pred, fmt="o", capsize=4, label="mean pred ± 1σ")
    lo = min(E.min(), mean_pred.min())
    hi = max(E.max(), mean_pred.max())
    plt.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="ideal")
    plt.xlabel("True Energy (GeV)")
    plt.ylabel("Predicted Energy (GeV)")
    plt.title("Mean Predicted Energy vs True Energy")
    plt.legend()
    savefig(outdir, "mean_pred_vs_true_errorbars.png")


def plot_summary_panel(E, bias, rms, frac_res, frac_bias, outdir):
    fig = plt.figure(figsize=(10, 7))
    gs = GridSpec(2, 2, hspace=0.4, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(E, rms, marker="o")
    ax.set_xlabel("E (GeV)")
    ax.set_ylabel("σE (GeV)")
    ax.set_title("Resolution")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(E, frac_res, marker="o")
    ax.set_xlabel("E (GeV)")
    ax.set_ylabel("σE / E")
    ax.set_title("Fractional resolution")

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(E, bias, marker="o")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("E (GeV)")
    ax.set_ylabel("Bias (GeV)")
    ax.set_title("Bias")

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(E, frac_bias * 100, marker="o")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("E (GeV)")
    ax.set_ylabel("Bias / E (%)")
    ax.set_title("Relative bias")

    fig.suptitle("Summary: Energy Reconstruction Performance", fontsize=13)
    savefig(outdir, "summary_panel.png")


# -------------------------
# Loss curves
# -------------------------

def plot_loss_curves(epochs, train_loss, val_loss, outdir):
    plt.figure()
    plt.plot(epochs, train_loss, marker="o", label="train")
    plt.plot(epochs, val_loss, marker="o", label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Train vs Val Loss")
    savefig(outdir, "train_vs_val_loss.png")

    plt.figure()
    plt.plot(epochs, val_loss, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Val loss")
    plt.title("Val Loss vs Epoch")
    savefig(outdir, "val_loss_vs_epoch.png")


def plot_metric_convergence(epochs, val_mae, val_rmse, outdir):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    l1, = ax1.plot(epochs, val_mae, "b-o", ms=4, label="Val MAE (GeV)")
    l2, = ax2.plot(epochs, val_rmse, "r-s", ms=4, label="Val RMSE (GeV)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MAE (GeV)", color="b")
    ax2.set_ylabel("RMSE (GeV)", color="r")
    ax1.set_title("Validation MAE and RMSE vs Epoch")
    ax1.legend(handles=[l1, l2], fontsize=8)
    savefig(outdir, "val_mae_rmse_vs_epoch.png")


def plot_train_val_ratio(epochs, train_loss, val_loss, outdir):
    ratio = np.asarray(val_loss) / np.maximum(np.asarray(train_loss), 1e-12)
    plt.figure()
    plt.plot(epochs, ratio, marker="o")
    plt.axhline(1, color="k", lw=0.8, ls="--", label="ratio = 1")
    plt.xlabel("Epoch")
    plt.ylabel("Val loss / Train loss")
    plt.title("Generalisation Gap (val/train ratio)")
    plt.legend()
    savefig(outdir, "generalisation_gap.png")


# -------------------------
# lnN plots
# -------------------------

def plot_lnn(y_true, lnN, resid, idx_scatter, outdir):
    if not np.isfinite(lnN).any():
        return

    plt.figure()
    plt.scatter(y_true[idx_scatter], lnN[idx_scatter], alpha=0.3, s=12)
    plt.xlabel("Energy")
    plt.ylabel("lnN")
    plt.title("lnN vs Energy")
    savefig(outdir, "lnN_vs_energy.png")

    plt.figure()
    plt.scatter(lnN[idx_scatter], np.abs(resid[idx_scatter]), alpha=0.3, s=12)
    plt.xlabel("lnN")
    plt.ylabel("|Residual|")
    plt.title("|Residual| vs lnN")
    savefig(outdir, "abs_residual_vs_lnN.png")


# -------------------------
# Main
# -------------------------

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

    E, bias, rms, std, n, raw_resid_list = per_energy_stats(y_true, resid)
    frac_res = rms / E
    frac_bias = bias / E

    plot_pred_vs_true(y_true, y_pred, idx_scatter, outdir)
    plot_mean_pred_with_errorbars(y_true, y_pred, outdir)
    plot_residual_distribution(resid, outdir)
    plot_bias_vs_energy(E, bias, outdir)
    plot_resolution_vs_energy(E, rms, outdir)
    plot_frac_resolution(E, frac_res, outdir)
    plot_frac_vs_inv_sqrtE(E, frac_res, outdir)
    plot_residual_scatter(y_true, resid, idx_scatter, outdir)
    plot_pull_distribution(y_true, resid, E, std, outdir)
    plot_lnn(y_true, lnN, resid, idx_scatter, outdir)

    plot_per_energy_residuals(E, raw_resid_list, outdir)
    plot_colored_pred_vs_true(y_true, y_pred, idx_scatter, outdir)
    plot_cumulative_error(y_true, y_pred, outdir)
    plot_relative_bias(E, bias, outdir)
    plot_stochastic_fit(E, frac_res, outdir)
    plot_boxplot_per_energy(E, raw_resid_list, outdir)
    plot_abs_error_vs_energy(E, rms, std, n, outdir)
    plot_2d_confusion(y_true, y_pred, outdir)
    plot_error_correlation(y_true, resid, outdir)
    plot_summary_panel(E, bias, rms, frac_res, frac_bias, outdir)

    if args.loss:
        result = read_loss_history(args.loss, group=args.group)
        if result is not None:
            epochs, train_loss, val_loss, val_mae, val_rmse = result
            plot_loss_curves(epochs, train_loss, val_loss, outdir)
            plot_metric_convergence(epochs, val_mae, val_rmse, outdir)
            plot_train_val_ratio(epochs, train_loss, val_loss, outdir)

    print(f"\nAll plots written to: {outdir}")


if __name__ == "__main__":
    main()