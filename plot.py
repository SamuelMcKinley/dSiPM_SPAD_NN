#
#   Creates many plots useful for photon analysis. NO NN PLOTS
#


"""
plot.py  —  Photon study plots from photon_study_summary.csv
Usage:  python plot.py <output_folder>
        output_folder must contain photon_study_summary.csv and photon_study_master.csv
Plots are saved to <output_folder>/plots_mpl/
No pandas — uses only csv + numpy.
"""

import argparse
import glob
import math
import os
import re
import sys
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif", "Times New Roman", "Georgia"],
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.titleweight":   "bold",
    "axes.labelsize":     11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         "#e0e0e0",
    "grid.linewidth":     0.6,
    "grid.linestyle":     "--",
    "legend.frameon":     True,
    "legend.framealpha":  0.92,
    "legend.edgecolor":   "#cccccc",
    "legend.fontsize":    8.5,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "figure.dpi":         150,
    "savefig.dpi":        200,
    "savefig.bbox":       "tight",
})

SPAD_COLORS  = {20: "#1f77b4", 50: "#d62728", 100: "#2ca02c", 200: "#ff7f0e"}
SPAD_MARKERS = {20: "o",       50: "s",       100: "^",       200: "D"}
SPAD_LABELS  = {s: f"{s}\u00d7{s} \u00b5m\u00b2" for s in [20, 50, 75, 100, 200]}

_TAB20 = plt.cm.tab20.colors
def energy_color(i): return _TAB20[i % len(_TAB20)]
def spad_color(spad):
    return SPAD_COLORS.get(int(spad), _TAB20[int(spad) % len(_TAB20)])
def spad_marker(spad):
    return SPAD_MARKERS.get(int(spad), "o")
def spad_label(spad):
    return SPAD_LABELS.get(int(spad), f"{int(spad)}\u00d7{int(spad)} \u00b5m\u00b2")

def fmt_k(x, _):
    return f"{x/1000:.1f}k" if abs(x) >= 1000 else f"{x:.0f}"

def save(fig, plot_dir, name):
    os.makedirs(plot_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(plot_dir, f"{name}.{ext}"))
    plt.close(fig)

# ── CSV loading ────────────────────────────────────────────────────────────────

def read_csv(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        raw = {k: [] for k in reader.fieldnames}
        for row in reader:
            for k, v in row.items():
                raw[k].append(v)
    out = {}
    for k, vals in raw.items():
        converted = []
        for v in vals:
            if v == "" or v is None:
                converted.append(np.nan)
            else:
                try:
                    converted.append(float(v))
                except ValueError:
                    converted.append(v)
        out[k] = converted
    return out

def col(data, key):
    return np.array(data[key], dtype=float)

def unique_sorted(data, key):
    return sorted(set(float(v) for v in data[key] if v not in ("", None) and not (isinstance(v, float) and np.isnan(v))))

def filter_rows(data, **kwargs):
    n = len(next(iter(data.values())))
    mask = np.ones(n, dtype=bool)
    for k, v in kwargs.items():
        arr = np.array([float(x) if x not in ("", None) else np.nan for x in data[k]])
        mask &= (arr == float(v))
    return {k: [data[k][i] for i in range(n) if mask[i]] for k in data}

def sort_by(data, key):
    arr = np.array([float(x) for x in data[key]])
    idx = np.argsort(arr)
    return {k: [data[k][i] for i in idx] for k in data}

# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_photon_stage_vs_energy(summ, plot_dir):
    spad_sizes = unique_sorted(summ, "spad_um")
    stages = [
        ("mean_n_raw",      "After geometry mask",    "#555555", "-",  "o"),
        ("mean_n_after_qe", "After QE",               "#1f77b4", "--", "s"),
        ("mean_n_final",    "After deadtime (final)",  "#2ca02c", "-",  "^"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    axes = axes.flatten()
    for ax, spad in zip(axes, spad_sizes):
        df = sort_by(filter_rows(summ, spad_um=spad), "energy")
        en = col(df, "energy")
        for c_key, label, color, ls, mk in stages:
            ax.plot(en, col(df, c_key), color=color, linestyle=ls,
                    marker=mk, markersize=5, linewidth=1.8, label=label)
        ax.set_title(f"SPAD {int(spad)}\u00d7{int(spad)} \u00b5m\u00b2")
        ax.set_ylabel("Mean photons / event")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_k))
        ax.set_xlim(left=0)
        ax.legend(loc="upper left", fontsize=8)
    for ax in axes[-2:]:
        ax.set_xlabel("Beam energy (GeV)")
    fig.suptitle("Photon budget stages vs beam energy", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, plot_dir, "01_photon_stages_vs_energy")


def plot_loss_breakdown_vs_energy(summ, plot_dir):
    spad_sizes = unique_sorted(summ, "spad_um")
    for spad in spad_sizes:
        df = sort_by(filter_rows(summ, spad_um=spad), "energy")
        en = col(df, "energy")
        fig, ax = plt.subplots(figsize=(8, 5))
        raw   = col(df, "mean_n_raw")
        qe    = col(df, "mean_n_lost_qe")
        dt    = col(df, "mean_n_lost_dt")
        final = col(df, "mean_n_final")
        ax.plot(en, raw,   color="#222222", marker="o", lw=2, ms=5, label="Raw (post-geom)")
        ax.fill_between(en, raw, alpha=0.06, color="#222222")
        ax.plot(en, qe,    color="#d62728", marker="v", lw=2, ms=5, label="Lost to QE")
        ax.fill_between(en, qe, alpha=0.10, color="#d62728")
        ax.plot(en, dt,    color="#ff7f0e", marker="D", lw=2, ms=5, label="Lost to deadtime")
        ax.plot(en, final, color="#2ca02c", marker="^", lw=2.2, ms=6, label="Final")
        ax.fill_between(en, final, alpha=0.12, color="#2ca02c")
        ax.set_xlabel("Beam energy (GeV)")
        ax.set_ylabel("Mean photons / event")
        ax.set_title(f"Photon budget \u2014 SPAD {int(spad)}\u00d7{int(spad)} \u00b5m\u00b2")
        ax.set_xlim(left=0)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_k))
        ax.legend(loc="upper left")
        fig.tight_layout()
        save(fig, plot_dir, f"02_loss_breakdown_SPAD{int(spad)}")


def plot_loss_fractions_vs_energy(summ, plot_dir):
    spad_sizes = unique_sorted(summ, "spad_um")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, spad in zip(axes, spad_sizes):
        df = sort_by(filter_rows(summ, spad_um=spad), "energy")
        en   = col(df, "energy")
        fqe  = col(df, "mean_frac_lost_qe_pct")
        fdt  = col(df, "mean_frac_lost_dt_pct")
        ffin = col(df, "mean_frac_final_pct")
        ax.plot(en, fqe,  color="#d62728", marker="v", lw=2, ms=5, label="QE loss %")
        ax.fill_between(en, fqe, alpha=0.10, color="#d62728")
        ax.plot(en, fdt,  color="#ff7f0e", marker="D", lw=2, ms=5, label="Deadtime loss %")
        ax.plot(en, ffin, color="#2ca02c", marker="^", lw=2, ms=5, label="Final %")
        ax.fill_between(en, ffin, alpha=0.10, color="#2ca02c")
        ax.set_title(f"SPAD {int(spad)}\u00d7{int(spad)} \u00b5m\u00b2")
        ax.set_ylim(0, 105)
        ax.set_ylabel("Fraction of raw photons (%)")
        ax.legend(loc="center right", fontsize=8)
        ax.set_xlim(left=0)
    for ax in axes[-2:]:
        ax.set_xlabel("Beam energy (GeV)")
    fig.suptitle("Loss fractions vs beam energy", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, plot_dir, "03_loss_fractions_vs_energy")


def plot_final_vs_spad(summ, plot_dir):
    energies   = unique_sorted(summ, "energy")
    spad_sizes = unique_sorted(summ, "spad_um")
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, en in enumerate(energies):
        df = sort_by(filter_rows(summ, energy=en), "spad_um")
        ax.plot(col(df, "spad_um"), col(df, "mean_n_final"),
                color=energy_color(i), marker="o", lw=1.8, ms=5, label=f"{en:.0f} GeV")
    ax.set_xlabel("SPAD side length (\u00b5m)")
    ax.set_ylabel("Mean final photons / event")
    ax.set_title("Final photons vs SPAD size")
    ax.set_xticks(spad_sizes)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_k))
    ax.set_xlim(min(spad_sizes) - 5, max(spad_sizes) + 10)
    ax.legend(loc="upper right", ncol=2, fontsize=8, title="Energy")
    fig.tight_layout()
    save(fig, plot_dir, "04_final_photons_vs_spad")


def plot_deadtime_vs_spad(summ, plot_dir):
    energies   = unique_sorted(summ, "energy")
    spad_sizes = unique_sorted(summ, "spad_um")
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, en in enumerate(energies):
        df = sort_by(filter_rows(summ, energy=en), "spad_um")
        ax.plot(col(df, "spad_um"), col(df, "mean_n_lost_dt"),
                color=energy_color(i), marker="o", lw=1.8, ms=5, label=f"{en:.0f} GeV")
    ax.set_xlabel("SPAD side length (\u00b5m)")
    ax.set_ylabel("Mean photons lost to deadtime / event")
    ax.set_title("Deadtime loss vs SPAD size")
    ax.set_xticks(spad_sizes)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_k))
    ax.set_xlim(min(spad_sizes) - 5, max(spad_sizes) + 10)
    ax.legend(loc="upper left", ncol=2, fontsize=8, title="Energy")
    fig.tight_layout()
    save(fig, plot_dir, "05_deadtime_loss_vs_spad")


def plot_deadtime_fraction_vs_spad(summ, plot_dir):
    energies   = unique_sorted(summ, "energy")
    spad_sizes = unique_sorted(summ, "spad_um")
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, en in enumerate(energies):
        df = sort_by(filter_rows(summ, energy=en), "spad_um")
        ax.plot(col(df, "spad_um"), col(df, "mean_frac_lost_dt_pct"),
                color=energy_color(i), marker="o", lw=1.8, ms=5, label=f"{en:.0f} GeV")
    ax.set_xlabel("SPAD side length (\u00b5m)")
    ax.set_ylabel("Deadtime loss (%)")
    ax.set_title("Deadtime loss fraction vs SPAD size")
    ax.set_xticks(spad_sizes)
    ax.set_ylim(bottom=0)
    ax.set_xlim(min(spad_sizes) - 5, max(spad_sizes) + 10)
    ax.legend(loc="upper left", ncol=2, fontsize=8, title="Energy")
    fig.tight_layout()
    save(fig, plot_dir, "06_deadtime_fraction_vs_spad")


def plot_all_spad_comparison(summ, plot_dir):
    spad_sizes = unique_sorted(summ, "spad_um")
    fig, ax = plt.subplots(figsize=(9, 6))
    for spad in spad_sizes:
        df = sort_by(filter_rows(summ, spad_um=spad), "energy")
        ax.plot(col(df, "energy"), col(df, "mean_n_final"),
                color=spad_color(spad), marker=spad_marker(spad),
                lw=2, ms=6, label=spad_label(spad))
    ax.set_xlabel("Beam energy (GeV)")
    ax.set_ylabel("Mean final photons / event")
    ax.set_title("Final photons vs energy \u2014 all SPAD sizes")
    ax.set_xlim(left=0)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_k))
    ax.legend(title="SPAD size")
    fig.tight_layout()
    save(fig, plot_dir, "07_final_photons_all_spads")


def plot_qe_eta_vs_energy(summ, plot_dir):
    spad_sizes = unique_sorted(summ, "spad_um")
    fig, ax = plt.subplots(figsize=(8, 5))
    for spad in spad_sizes:
        df = sort_by(filter_rows(summ, spad_um=spad), "energy")
        if "mean_eta" not in df:
            continue
        ax.plot(col(df, "energy"), col(df, "mean_eta"),
                color=spad_color(spad), marker=spad_marker(spad),
                lw=1.8, ms=5, label=spad_label(spad))
    ax.set_xlabel("Beam energy (GeV)")
    ax.set_ylabel("Mean QE \u03b7")
    ax.set_title("Mean photon detection efficiency vs energy")
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.legend(title="SPAD size")
    fig.tight_layout()
    save(fig, plot_dir, "08_mean_eta_vs_energy")


def plot_wavelength_vs_energy(summ, plot_dir):
    spad_sizes = unique_sorted(summ, "spad_um")
    fig, ax = plt.subplots(figsize=(8, 5))
    for spad in spad_sizes:
        df = sort_by(filter_rows(summ, spad_um=spad), "energy")
        if "mean_lam_nm" not in df:
            continue
        ax.plot(col(df, "energy"), col(df, "mean_lam_nm"),
                color=spad_color(spad), marker=spad_marker(spad),
                lw=1.8, ms=5, label=spad_label(spad))
    ax.set_xlabel("Beam energy (GeV)")
    ax.set_ylabel("Mean wavelength (nm)")
    ax.set_title("Mean Cherenkov wavelength vs energy")
    ax.set_xlim(left=0)
    ax.legend(title="SPAD size")
    fig.tight_layout()
    save(fig, plot_dir, "09_mean_wavelength_vs_energy")


def plot_qe_survival_vs_energy(summ, plot_dir):
    spad_sizes = unique_sorted(summ, "spad_um")
    fig, ax = plt.subplots(figsize=(8, 5))
    for spad in spad_sizes:
        df = sort_by(filter_rows(summ, spad_um=spad), "energy")
        surv = 100.0 - col(df, "mean_frac_lost_qe_pct")
        ax.plot(col(df, "energy"), surv,
                color=spad_color(spad), marker=spad_marker(spad),
                lw=1.8, ms=5, label=spad_label(spad))
    ax.set_xlabel("Beam energy (GeV)")
    ax.set_ylabel("QE survival (%)")
    ax.set_title("Fraction of photons surviving QE vs energy")
    ax.set_xlim(left=0); ax.set_ylim(0, 50)
    ax.legend(title="SPAD size")
    fig.tight_layout()
    save(fig, plot_dir, "10_qe_survival_vs_energy")


def plot_ratio_vs_smallest(summ, plot_dir):
    spad_sizes = unique_sorted(summ, "spad_um")
    ref_spad   = spad_sizes[0]
    ref_df     = filter_rows(summ, spad_um=ref_spad)
    ref_map    = {float(e): float(v)
                  for e, v in zip(ref_df["energy"], ref_df["mean_n_final"])}
    fig, ax = plt.subplots(figsize=(8, 5))
    for spad in spad_sizes[1:]:
        df  = sort_by(filter_rows(summ, spad_um=spad), "energy")
        en  = col(df, "energy")
        fin = col(df, "mean_n_final")
        ratios = np.array([fin[i] / ref_map[en[i]] if ref_map.get(en[i], 0) > 0 else np.nan
                           for i in range(len(en))])
        ax.plot(en, ratios,
                color=spad_color(spad), marker=spad_marker(spad),
                lw=1.8, ms=5, label=spad_label(spad))
    ax.axhline(1.0, color=spad_color(ref_spad), lw=1.5, ls="--",
               label=f"Reference ({spad_label(ref_spad)})")
    ax.set_xlabel("Beam energy (GeV)")
    ax.set_ylabel(f"Yield / yield at {int(ref_spad)}\u00d7{int(ref_spad)} \u00b5m\u00b2")
    ax.set_title(f"Relative photon yield vs {int(ref_spad)}\u00d7{int(ref_spad)} \u00b5m\u00b2 reference")
    ax.set_xlim(left=0)
    ax.legend(title="SPAD size")
    fig.tight_layout()
    save(fig, plot_dir, "11_ratio_vs_smallest_spad")


def plot_stacked_budget(summ, plot_dir):
    spad_sizes = unique_sorted(summ, "spad_um")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    axes = axes.flatten()
    for ax, spad in zip(axes, spad_sizes):
        df    = sort_by(filter_rows(summ, spad_um=spad), "energy")
        en    = col(df, "energy")
        final = col(df, "mean_n_final")
        qe    = col(df, "mean_n_lost_qe")
        dt    = col(df, "mean_n_lost_dt")
        ax.stackplot(en, [final, qe, dt],
                     labels=["Final", "Lost to QE", "Lost to deadtime"],
                     colors=["#2ca02c", "#d62728", "#ff7f0e"], alpha=0.82)
        ax.set_title(f"SPAD {int(spad)}\u00d7{int(spad)} \u00b5m\u00b2")
        ax.set_ylabel("Mean photons / event")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_k))
        ax.set_xlim(left=0)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc="upper left", fontsize=8)
    for ax in axes[-2:]:
        ax.set_xlabel("Beam energy (GeV)")
    fig.suptitle("Photon budget (stacked) vs beam energy", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, plot_dir, "12_stacked_budget")


def plot_heatmap_final(summ, plot_dir):
    spad_sizes = unique_sorted(summ, "spad_um")
    energies   = unique_sorted(summ, "energy")
    mat = np.full((len(energies), len(spad_sizes)), np.nan)
    for i, en in enumerate(energies):
        for j, sp in enumerate(spad_sizes):
            df = filter_rows(summ, energy=en, spad_um=sp)
            if df["mean_n_final"]:
                mat[i, j] = float(df["mean_n_final"][0])
    fig, ax = plt.subplots(figsize=(7, max(5, len(energies) * 0.45)))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(range(len(spad_sizes)))
    ax.set_xticklabels([f"{int(s)}\u00d7{int(s)}" for s in spad_sizes])
    ax.set_yticks(range(len(energies)))
    ax.set_yticklabels([f"{e:.0f} GeV" for e in energies])
    ax.set_xlabel("SPAD size (\u00b5m\u00b2)")
    ax.set_ylabel("Beam energy")
    ax.set_title("Mean final photons / event")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Mean final photons")
    cbar.formatter = ticker.FuncFormatter(fmt_k)
    cbar.update_ticks()
    vmax = np.nanmax(mat)
    for i in range(len(energies)):
        for j in range(len(spad_sizes)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, fmt_k(v, None), ha="center", va="center",
                        color="white" if v < vmax * 0.6 else "black", fontsize=7.5)
    fig.tight_layout()
    save(fig, plot_dir, "13_heatmap_final_photons")


def plot_heatmap_dt_loss(summ, plot_dir):
    spad_sizes = unique_sorted(summ, "spad_um")
    energies   = unique_sorted(summ, "energy")
    mat = np.full((len(energies), len(spad_sizes)), np.nan)
    for i, en in enumerate(energies):
        for j, sp in enumerate(spad_sizes):
            df = filter_rows(summ, energy=en, spad_um=sp)
            if df["mean_frac_lost_dt_pct"]:
                mat[i, j] = float(df["mean_frac_lost_dt_pct"][0])
    fig, ax = plt.subplots(figsize=(7, max(5, len(energies) * 0.45)))
    im = ax.imshow(mat, aspect="auto", cmap="Reds", origin="lower", vmin=0)
    ax.set_xticks(range(len(spad_sizes)))
    ax.set_xticklabels([f"{int(s)}\u00d7{int(s)}" for s in spad_sizes])
    ax.set_yticks(range(len(energies)))
    ax.set_yticklabels([f"{e:.0f} GeV" for e in energies])
    ax.set_xlabel("SPAD size (\u00b5m\u00b2)")
    ax.set_ylabel("Beam energy")
    ax.set_title("Deadtime loss (%)")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Deadtime loss (%)")
    vmax = np.nanmax(mat)
    for i in range(len(energies)):
        for j in range(len(spad_sizes)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                        color="white" if v > vmax * 0.55 else "black", fontsize=7.5)
    fig.tight_layout()
    save(fig, plot_dir, "14_heatmap_dt_loss")


def plot_per_event_scatter(mast, plot_dir):
    if mast is None:
        return
    spad_sizes = unique_sorted(mast, "spad_um")
    fig, ax = plt.subplots(figsize=(10, 6))
    for spad in spad_sizes:
        df  = filter_rows(mast, spad_um=spad)
        en  = col(df, "energy")
        fin = col(df, "n_final")
        jitter = np.random.uniform(-0.4, 0.4, len(en))
        ax.scatter(en + jitter, fin, color=spad_color(spad),
                   alpha=0.25, s=8, label=spad_label(spad), rasterized=True)
    ax.set_xlabel("Beam energy (GeV)")
    ax.set_ylabel("Final photons (per event)")
    ax.set_title("Per-event photon count scatter")
    ax.set_xlim(left=0)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_k))
    ax.legend(title="SPAD size", markerscale=2)
    fig.tight_layout()
    save(fig, plot_dir, "15_per_event_scatter")


def plot_photon_distributions(mast, plot_dir):
    if mast is None:
        return
    spad_sizes = unique_sorted(mast, "spad_um")
    energies   = unique_sorted(mast, "energy")
    fig, axes  = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    for ax, spad in zip(axes, spad_sizes):
        data, labels = [], []
        for en in energies:
            df  = filter_rows(mast, spad_um=spad, energy=en)
            arr = col(df, "n_final")
            arr = arr[~np.isnan(arr)]
            if len(arr) > 1:
                data.append(arr); labels.append(f"{en:.0f}")
        if not data:
            ax.set_visible(False); continue
        parts = ax.violinplot(data, positions=range(len(data)),
                              showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(spad_color(spad)); pc.set_alpha(0.55)
        parts["cmedians"].set_color("#222222"); parts["cmedians"].set_linewidth(1.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, fontsize=7.5)
        ax.set_title(f"SPAD {int(spad)}\u00d7{int(spad)} \u00b5m\u00b2")
        ax.set_xlabel("Energy (GeV)")
        ax.set_ylabel("Final photons / event")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_k))
    fig.suptitle("Distribution of final photon counts per event",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, plot_dir, "16_photon_distributions_violin")


# -- Optional photon-summary fallback -------------------------------------------------

def parse_square_um(text):
    m = re.search(r"(\d+)x\1", str(text))
    return int(m.group(1)) if m else None


def build_summary_from_photon_energy_csvs(root_dir):
    paths = sorted(glob.glob(os.path.join(root_dir, "photon_energy_SPAD*_CH*.csv")))
    if not paths:
        return None, None

    records = []
    for path in paths:
        spad = parse_square_um(os.path.basename(path))
        if spad is None:
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    records.append({
                        "energy": float(row["Energy"]),
                        "spad_um": float(spad),
                        "n_final": float(row["nPhotons"]),
                    })
                except Exception:
                    continue

    if not records:
        return None, None

    grouped = {}
    for r in records:
        grouped.setdefault((r["energy"], r["spad_um"]), []).append(r["n_final"])

    summ_rows = []
    for (energy, spad), vals in sorted(grouped.items()):
        mean_final = float(np.mean(vals))
        summ_rows.append({
            "energy": energy,
            "spad_um": spad,
            "n_events": len(vals),
            "mean_n_raw": mean_final,
            "mean_n_lost_qe": 0.0,
            "mean_n_after_qe": mean_final,
            "mean_n_lost_dt": 0.0,
            "mean_n_final": mean_final,
            "mean_frac_lost_qe_pct": 0.0,
            "mean_frac_lost_dt_pct": 0.0,
            "mean_frac_final_pct": 100.0,
            "mean_lam_nm": np.nan,
            "mean_eta": np.nan,
        })

    mast_rows = []
    for i, r in enumerate(records):
        mast_rows.append({
            "energy": r["energy"],
            "event": i,
            "spad_um": r["spad_um"],
            "n_final": r["n_final"],
        })

    def rows_to_cols(rows):
        keys = list(rows[0].keys())
        return {k: [row.get(k, np.nan) for row in rows] for k in keys}

    return rows_to_cols(summ_rows), rows_to_cols(mast_rows)


# -- NN prediction plots --------------------------------------------------------------

def find_prediction_csvs(nn_root, pred_name="", spads_keep=None):
    if not nn_root or not os.path.exists(nn_root):
        return {}
    pattern = os.path.join(nn_root, "**", "NN_model_*", "*.csv")
    by_spad = {}
    for model_dir in sorted({os.path.dirname(p) for p in glob.glob(pattern, recursive=True)}):
        spad = parse_square_um(os.path.basename(model_dir))
        if spad is None or (spads_keep and spad not in spads_keep):
            continue
        preferred = os.path.join(model_dir, pred_name) if pred_name else ""
        if preferred and os.path.exists(preferred):
            pred = preferred
        else:
            cands = glob.glob(os.path.join(model_dir, "pred*.csv"))
            if not cands:
                continue
            pred = max(cands, key=lambda x: os.stat(x).st_mtime)
        by_spad[spad] = pred
    return by_spad


def read_pred_csv(path):
    true_e, pred_e, resid = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                t = float(row["true_energy"])
                p = float(row["pred_energy"])
                d = float(row.get("deviation", p - t) or (p - t))
            except Exception:
                continue
            true_e.append(t); pred_e.append(p); resid.append(d)
    if not true_e:
        raise ValueError(f"No usable prediction rows in {path}")
    return np.asarray(true_e), np.asarray(pred_e), np.asarray(resid)


def plot_nn_predictions(nn_root, plot_dir, pred_name="", spads_keep=None, hist_energies=None):
    pred_paths = find_prediction_csvs(nn_root, pred_name=pred_name, spads_keep=spads_keep)
    if not pred_paths:
        print("No NN prediction CSVs found; skipping NN plots.")
        return

    outdir = os.path.join(plot_dir, "nn")
    os.makedirs(outdir, exist_ok=True)
    hist_energies = hist_energies or []
    summary_rows = []
    loaded = {}

    for spad, path in sorted(pred_paths.items()):
        try:
            true_e, pred_e, resid = read_pred_csv(path)
        except Exception as e:
            print(f"  NN SPAD {spad} skipped: {e}")
            continue
        loaded[spad] = (true_e, pred_e, resid, path)
        for energy in sorted(set(true_e.tolist())):
            mask = np.isclose(true_e, energy)
            r = resid[mask]
            p = pred_e[mask]
            summary_rows.append({
                "spad_um": spad,
                "energy_GeV": energy,
                "n_events": int(mask.sum()),
                "bias_GeV": float(np.mean(r)),
                "std_resid_GeV": float(np.std(r, ddof=0)),
                "rms_resid_GeV": float(np.sqrt(np.mean(r ** 2))),
                "frac_resid_std": float(np.std(r, ddof=0) / energy) if energy else np.nan,
                "mean_pred_over_true": float(np.mean(p / energy)) if energy else np.nan,
                "pred_csv": path,
            })

    if not loaded:
        return

    with open(os.path.join(outdir, "nn_energy_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)

    for ykey, ylabel, title, name in [
        ("frac_resid_std", "std(E_pred - E_true) / E", "Fractional resolution vs energy", "nn_frac_resolution_vs_energy"),
        ("bias_GeV", "Mean residual (GeV)", "Bias vs energy", "nn_bias_vs_energy"),
        ("mean_pred_over_true", "Mean(E_pred / E_true)", "Mean prediction ratio vs energy", "nn_mean_ratio_vs_energy"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for spad in sorted(loaded):
            rows = [r for r in summary_rows if r["spad_um"] == spad]
            rows.sort(key=lambda r: r["energy_GeV"])
            ax.plot([r["energy_GeV"] for r in rows], [r[ykey] for r in rows],
                    color=spad_color(spad), marker=spad_marker(spad), label=spad_label(spad))
        if ykey in ("bias_GeV",):
            ax.axhline(0.0, color="#555555", lw=1)
        if ykey == "mean_pred_over_true":
            ax.axhline(1.0, color="#555555", lw=1)
        ax.set_xlabel("Beam energy (GeV)"); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(title="SPAD size")
        save(fig, outdir, name)

    fig, ax = plt.subplots(figsize=(8, 5))
    xs, ys = [], []
    for spad, (true_e, pred_e, _resid, _path) in sorted(loaded.items()):
        xs.append(spad)
        ys.append(float(np.std((pred_e - true_e) / true_e, ddof=1)))
    ax.plot(xs, ys, marker="o", color="#1f77b4")
    ax.set_xlabel("SPAD side length (um)"); ax.set_ylabel("Overall std((pred-true)/true)")
    ax.set_title("Overall NN fractional resolution vs SPAD")
    save(fig, outdir, "nn_overall_frac_resolution_vs_spad")

    for energy in hist_energies:
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False
        for spad, (true_e, _pred_e, resid, _path) in sorted(loaded.items()):
            arr = resid[np.isclose(true_e, energy)]
            if arr.size == 0:
                continue
            ax.hist(arr, bins=60, density=True, histtype="step", label=spad_label(spad), color=spad_color(spad))
            plotted = True
        if plotted:
            ax.set_xlabel(f"Residual at {energy:g} GeV (GeV)")
            ax.set_ylabel("Density"); ax.set_title(f"Residual distribution at {energy:g} GeV")
            ax.legend(title="SPAD size")
            save(fig, outdir, f"nn_residual_hist_{energy:g}GeV".replace(".", "p"))
        else:
            plt.close(fig)


# -- Cumulative hit-map plots ---------------------------------------------------------

DEFAULT_TIME_SLICES_SPEC = "0-8,8-9,9-9.1,9.1-9.2,9.2-9.3,9.3-9.4,9.4-9.5,9.5-9.6,9.6-9.7,9.7-9.8,9.8-9.9,9.9-10,10-10.2,10.2-10.4,10.4-10.6,10.6-10.8,10.8-11,11-12,12-13,13-14,14-15,15-16,16-17,17-18,18-19,19-20,20-21,21-22,22-23,23-24,24-25,25-40"

def parse_time_slices(spec):
    ranges = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        low, high = part.split("-", 1)
        low = float(low); high = float(high)
        if high <= low:
            raise ValueError(f"Bad time slice: {part}")
        ranges.append((low, high))
    return ranges


def npz_matches_time_slices(path, time_slices=None):
    if time_slices is None:
        return True
    expected = np.asarray(time_slices, dtype=np.float32)
    try:
        with np.load(path, allow_pickle=False) as z:
            x = np.asarray(z["x"])
            if x.shape[0] != expected.shape[0] or "time_slices" not in z:
                return False
            actual = np.asarray(z["time_slices"], dtype=np.float32)
    except Exception:
        return False
    return actual.shape == expected.shape and np.allclose(actual, expected, rtol=0.0, atol=1.0e-6)


def iter_npz(root_dir, max_files=0, time_slices=None):
    n = 0
    for path in sorted(glob.glob(os.path.join(root_dir, "**", "*.npz"), recursive=True)):
        if ".bad_" in path or "_dup" in os.path.basename(path):
            continue
        if not npz_matches_time_slices(path, time_slices=time_slices):
            continue
        yield path
        n += 1
        if max_files and n >= max_files:
            return


def load_npz_counts(path):
    with np.load(path, allow_pickle=False) as z:
        x = np.asarray(z["x"], dtype=np.float64)
        lnN = float(z["lnN"])
    return x * math.exp(lnN)


def plot_cumulative_hitmap(tensor_root, plot_dir, max_files=0, time_slices=None):
    if not tensor_root or not os.path.exists(tensor_root):
        print("No tensor root supplied/found; skipping cumulative hit maps.")
        return
    if time_slices is None:
        time_slices = parse_time_slices(DEFAULT_TIME_SLICES_SPEC)
    acc = None
    n = 0
    for path in iter_npz(tensor_root, max_files=max_files, time_slices=time_slices):
        x = load_npz_counts(path)
        if acc is None:
            acc = np.zeros_like(x, dtype=np.float64)
        acc += x
        n += 1
    if acc is None:
        print("No current TIME_SLICES .npz tensors found; skipping cumulative hit maps.")
        return

    outdir = os.path.join(plot_dir, "hitmaps")
    os.makedirs(outdir, exist_ok=True)
    base = os.path.basename(os.path.abspath(tensor_root.rstrip(os.sep))) or "tensor_root"

    def draw(img, name, title):
        fig, ax = plt.subplots(figsize=(7, 6))
        masked = np.ma.masked_equal(img, 0.0)
        cmap = plt.cm.viridis.copy(); cmap.set_bad(color="white")
        im = ax.imshow(masked, origin="lower", cmap=cmap, interpolation="nearest", aspect="equal")
        ax.set_xlabel("x bin"); ax.set_ylabel("y bin"); ax.set_title(title)
        fig.colorbar(im, ax=ax, label="counts")
        save(fig, outdir, name)

    for i in range(acc.shape[0]):
        if i < len(time_slices):
            t0, t1 = time_slices[i]
            name = f"hitmap_{base}_slice{i}_{t0:g}-{t1:g}ns"
            title = f"Cumulative hit map, {t0:g}-{t1:g} ns ({n} events)"
        else:
            name = f"hitmap_{base}_slice{i}"
            title = f"Cumulative hit map, slice {i} ({n} events)"
        draw(acc[i], name, title)
    draw(np.sum(acc, axis=0), f"hitmap_{base}_sum", f"Cumulative hit map, all time slices ({n} events)")



def plot_single_event_hitmap(tensor_root, plot_dir, event_index=0, time_slices=None):
    if not tensor_root or not os.path.exists(tensor_root):
        print("No tensor root supplied/found; skipping single-event hit maps.")
        return
    if time_slices is None:
        time_slices = parse_time_slices(DEFAULT_TIME_SLICES_SPEC)

    paths = list(iter_npz(tensor_root, max_files=0, time_slices=time_slices))
    if not paths:
        print("No current TIME_SLICES .npz tensors found; skipping single-event hit maps.")
        return

    event_index = max(0, min(int(event_index), len(paths) - 1))
    path = paths[event_index]
    counts = load_npz_counts(path)

    tensor_base = os.path.basename(os.path.abspath(tensor_root.rstrip(os.sep))) or "tensor_root"
    event_base = os.path.splitext(os.path.basename(path))[0]
    outdir = os.path.join(plot_dir, "single_event_hitmaps", tensor_base, event_base)
    os.makedirs(outdir, exist_ok=True)

    def draw(img, name, title):
        fig, ax = plt.subplots(figsize=(7, 6))
        masked = np.ma.masked_equal(img, 0.0)
        cmap = plt.cm.magma.copy(); cmap.set_bad(color="white")
        im = ax.imshow(masked, origin="lower", cmap=cmap, interpolation="nearest", aspect="equal")
        ax.set_xlabel("x bin"); ax.set_ylabel("y bin"); ax.set_title(title)
        fig.colorbar(im, ax=ax, label="counts")
        save(fig, outdir, name)

    print(f"Plotting one NN input event: {path}")
    for i in range(counts.shape[0]):
        if i < len(time_slices):
            t0, t1 = time_slices[i]
            name = f"event_slice{i:02d}_{t0:g}-{t1:g}ns"
            title = f"Single event NN input, {t0:g}-{t1:g} ns"
        else:
            name = f"event_slice{i:02d}"
            title = f"Single event NN input, slice {i}"
        draw(counts[i], name, title)
    draw(np.sum(counts, axis=0), "event_sum_all_slices", "Single event NN input, all time slices")



# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Single plotting entry point for photon, NN, and cumulative hit-map analyses.")
    ap.add_argument("output_folder", nargs="?", default=".",
                    help="Folder containing photon_study_summary.csv, or the repo folder containing photon_energy_SPAD*.csv")
    ap.add_argument("--nn-root", default="NN_Analysis",
                    help="Root to search recursively for copied *_model/NN_model_* prediction CSVs")
    ap.add_argument("--pred-name", default="", help="Preferred prediction CSV filename; newest pred*.csv is used if omitted")
    ap.add_argument("--tensor-root", default="", help="Root containing .npz tensors for cumulative hit-map plots")
    ap.add_argument("--max-hitmap-files", type=int, default=0, help="Limit .npz files for quick hit-map tests; 0 means all")
    ap.add_argument("--event-index", type=int, default=0, help="Which individual tensor event to plot as one NN input example")
    ap.add_argument("--spads", default="", help="Comma-separated SPAD sizes to keep, e.g. 20,50,100,200")
    ap.add_argument("--hist-energies", default="10,20,80", help="Comma-separated energies for NN residual histograms")
    ap.add_argument("--time-slices", default=os.environ.get("TIME_SLICES", DEFAULT_TIME_SLICES_SPEC),
                    help="Comma-separated time slices like 0-8,8-9,9-9.1 for cumulative hit-map labels")
    args = ap.parse_args()

    output_folder = os.path.abspath(args.output_folder)
    plot_dir = os.path.join(output_folder, "plots_mpl")
    os.makedirs(plot_dir, exist_ok=True)

    spads_keep = set(int(x.strip()) for x in args.spads.split(",") if x.strip()) if args.spads else None
    hist_energies = [float(x.strip()) for x in args.hist_energies.split(",") if x.strip()]
    time_slices = parse_time_slices(args.time_slices)

    summary_path = os.path.join(output_folder, "photon_study_summary.csv")
    master_path = os.path.join(output_folder, "photon_study_master.csv")

    if os.path.exists(summary_path):
        print(f"Reading {summary_path} ...")
        summ = read_csv(summary_path)
        mast = read_csv(master_path) if os.path.exists(master_path) else None
    else:
        print(f"No photon_study_summary.csv in {output_folder}; trying photon_energy_SPAD*.csv fallback ...")
        summ, mast = build_summary_from_photon_energy_csvs(output_folder)

    if summ:
        n_en = len(unique_sorted(summ, "energy"))
        n_sp = len(unique_sorted(summ, "spad_um"))
        n_rows = len(next(iter(summ.values())))
        print(f"  Photon data: {n_en} energies x {n_sp} SPAD sizes ({n_rows} summary rows)")
        if mast:
            print(f"  {len(next(iter(mast.values())))} event-level photon records")

        plots = [
            ("Photon stages vs energy",              plot_photon_stage_vs_energy,    (summ,)),
            ("Loss breakdown per SPAD",              plot_loss_breakdown_vs_energy,  (summ,)),
            ("Loss fractions vs energy",             plot_loss_fractions_vs_energy,  (summ,)),
            ("Final photons vs SPAD size",           plot_final_vs_spad,             (summ,)),
            ("Deadtime loss vs SPAD size",           plot_deadtime_vs_spad,          (summ,)),
            ("Deadtime fraction vs SPAD size",       plot_deadtime_fraction_vs_spad, (summ,)),
            ("All SPAD sizes final vs energy",       plot_all_spad_comparison,       (summ,)),
            ("Mean QE eta vs energy",                plot_qe_eta_vs_energy,          (summ,)),
            ("Mean wavelength vs energy",            plot_wavelength_vs_energy,      (summ,)),
            ("QE survival vs energy",                plot_qe_survival_vs_energy,     (summ,)),
            ("Ratio vs smallest SPAD",               plot_ratio_vs_smallest,         (summ,)),
            ("Stacked budget",                       plot_stacked_budget,            (summ,)),
            ("Heatmap final photons",                plot_heatmap_final,             (summ,)),
            ("Heatmap deadtime loss",                plot_heatmap_dt_loss,           (summ,)),
            ("Per-event scatter",                    plot_per_event_scatter,         (mast,)),
            ("Violin photon distributions",          plot_photon_distributions,      (mast,)),
        ]
        for i, (desc, fn, fargs) in enumerate(plots, 1):
            print(f"  [{i:02d}/{len(plots)}] {desc} ...", end=" ", flush=True)
            try:
                fn(*fargs, plot_dir)
                print("done")
            except Exception as e:
                print(f"SKIPPED ({e})")
    else:
        print("No photon CSVs found; skipping photon plots.")

    nn_root = args.nn_root
    if nn_root and not os.path.isabs(nn_root):
        nn_root = os.path.join(output_folder, nn_root)
    plot_nn_predictions(nn_root, plot_dir, pred_name=args.pred_name, spads_keep=spads_keep, hist_energies=hist_energies)

    plot_cumulative_hitmap(args.tensor_root, plot_dir, max_files=args.max_hitmap_files, time_slices=time_slices)
    plot_single_event_hitmap(args.tensor_root, plot_dir, event_index=args.event_index, time_slices=time_slices)

    print(f"\nAll available plots saved to {plot_dir}/")

if __name__ == "__main__":
    main()