"""
plot.py  —  Photon study plots from photon_study_summary.csv
Usage:  python plot.py <output_folder>
        output_folder must contain photon_study_summary.csv and photon_study_master.csv
Plots are saved to <output_folder>/plots_mpl/
No pandas — uses only csv + numpy.
"""

import os
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
SPAD_LABELS  = {s: f"{s}\u00d7{s} \u00b5m\u00b2" for s in [20, 50, 100, 200]}

_TAB20 = plt.cm.tab20.colors
def energy_color(i): return _TAB20[i % len(_TAB20)]

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
                color=SPAD_COLORS[int(spad)], marker=SPAD_MARKERS[int(spad)],
                lw=2, ms=6, label=SPAD_LABELS[int(spad)])
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
                color=SPAD_COLORS[int(spad)], marker=SPAD_MARKERS[int(spad)],
                lw=1.8, ms=5, label=SPAD_LABELS[int(spad)])
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
                color=SPAD_COLORS[int(spad)], marker=SPAD_MARKERS[int(spad)],
                lw=1.8, ms=5, label=SPAD_LABELS[int(spad)])
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
                color=SPAD_COLORS[int(spad)], marker=SPAD_MARKERS[int(spad)],
                lw=1.8, ms=5, label=SPAD_LABELS[int(spad)])
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
                color=SPAD_COLORS[int(spad)], marker=SPAD_MARKERS[int(spad)],
                lw=1.8, ms=5, label=SPAD_LABELS[int(spad)])
    ax.axhline(1.0, color=SPAD_COLORS[int(ref_spad)], lw=1.5, ls="--",
               label=f"Reference ({SPAD_LABELS[int(ref_spad)]})")
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
        ax.scatter(en + jitter, fin, color=SPAD_COLORS[int(spad)],
                   alpha=0.25, s=8, label=SPAD_LABELS[int(spad)], rasterized=True)
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
            pc.set_facecolor(SPAD_COLORS[int(spad)]); pc.set_alpha(0.55)
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot.py <output_folder>")
        sys.exit(1)

    output_folder = sys.argv[1]
    plot_dir      = os.path.join(output_folder, "plots_mpl")

    summary_path = os.path.join(output_folder, "photon_study_summary.csv")
    master_path  = os.path.join(output_folder, "photon_study_master.csv")

    if not os.path.exists(summary_path):
        sys.exit(f"Cannot find {summary_path}")

    print(f"Reading {summary_path} ...")
    summ = read_csv(summary_path)
    mast = read_csv(master_path) if os.path.exists(master_path) else None

    n_en   = len(unique_sorted(summ, "energy"))
    n_sp   = len(unique_sorted(summ, "spad_um"))
    n_rows = len(next(iter(summ.values())))
    print(f"  {n_en} energies x {n_sp} SPAD sizes  ({n_rows} summary rows)")
    if mast:
        print(f"  {len(next(iter(mast.values())))} event-level records in master CSV")

    plots = [
        ("Photon stages vs energy (4-panel)",       plot_photon_stage_vs_energy,    (summ,)),
        ("Loss breakdown per SPAD",                  plot_loss_breakdown_vs_energy,  (summ,)),
        ("Loss fractions vs energy (4-panel)",       plot_loss_fractions_vs_energy,  (summ,)),
        ("Final photons vs SPAD size",               plot_final_vs_spad,             (summ,)),
        ("Deadtime loss vs SPAD size",               plot_deadtime_vs_spad,          (summ,)),
        ("Deadtime fraction vs SPAD size",           plot_deadtime_fraction_vs_spad, (summ,)),
        ("All SPAD sizes: final photons vs energy",  plot_all_spad_comparison,       (summ,)),
        ("Mean QE eta vs energy",                    plot_qe_eta_vs_energy,          (summ,)),
        ("Mean wavelength vs energy",                plot_wavelength_vs_energy,      (summ,)),
        ("QE survival % vs energy",                  plot_qe_survival_vs_energy,     (summ,)),
        ("Ratio vs smallest SPAD",                   plot_ratio_vs_smallest,         (summ,)),
        ("Stacked budget (4-panel)",                 plot_stacked_budget,            (summ,)),
        ("Heatmap: final photons",                   plot_heatmap_final,             (summ,)),
        ("Heatmap: deadtime loss %",                 plot_heatmap_dt_loss,           (summ,)),
        ("Per-event scatter",                        plot_per_event_scatter,         (mast,)),
        ("Violin: photon distributions",             plot_photon_distributions,      (mast,)),
    ]

    for i, (desc, fn, args) in enumerate(plots, 1):
        print(f"  [{i:02d}/{len(plots)}] {desc} ...", end=" ", flush=True)
        try:
            fn(*args, plot_dir)
            print("done")
        except Exception as e:
            print(f"SKIPPED ({e})")

    print(f"\nAll plots saved to {plot_dir}/")

if __name__ == "__main__":
    main()