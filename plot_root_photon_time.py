#!/usr/bin/env python3
"""Plot optical photon arrival time for a single DREAMSim ROOT file.

Example:
  python3 plot_root_photon_time.py /lustre/work/$USER/pi_predict/file.root
  python3 plot_root_photon_time.py file.root --time-range 0 40 --zoom-range 8 11 --bin-width 0.05

No pandas required.
"""

import argparse
import csv
import os
import sys

import numpy as np
import ROOT

ROOT.gROOT.SetBatch(True)


def parse_args():
    ap = argparse.ArgumentParser(description="Plot nPhotons vs arrival time for one ROOT file.")
    ap.add_argument("root_file", help="Input ROOT file containing a TTree named 'tree'.")
    ap.add_argument("--output-dir", default="photon_time_plots", help="Directory for PNG/PDF/CSV outputs.")
    ap.add_argument("--tree", default="tree", help="TTree name. Default: tree")
    ap.add_argument("--time-range", nargs=2, type=float, default=(0.0, 40.0), metavar=("TMIN", "TMAX"), help="Histogram time range in ns.")
    ap.add_argument("--zoom-range", nargs=2, type=float, default=(8.0, 11.0), metavar=("TMIN", "TMAX"), help="Zoom plot time range in ns.")
    ap.add_argument("--bin-width", type=float, default=0.05, help="Histogram bin width in ns. Default: 0.05")
    ap.add_argument("--max-events", type=int, default=0, help="Limit number of events; 0 means all events.")
    ap.add_argument("--raw", action="store_true", help="Use all OP_time_final photons. Default applies isCoreC and pos_final_z > 0 detector mask.")
    ap.add_argument("--per-event", action="store_true", help="Plot photons per event per bin instead of total photons per bin.")
    return ap.parse_args()


def as_array(branch):
    return np.asarray(branch)


def iter_times(tree, max_events=0, raw=False):
    used_events = 0
    total_photons = 0
    selected_photons = 0

    for i, event in enumerate(tree):
        if max_events and i >= max_events:
            break

        t = as_array(event.OP_time_final).astype(np.float64, copy=False)
        total_photons += int(t.size)

        if raw:
            selected = t
        else:
            is_core = as_array(event.OP_isCoreC).astype(bool, copy=False)
            z_final = as_array(event.OP_pos_final_z).astype(np.float64, copy=False)
            selected = t[is_core & (z_final > 0)]

        selected_photons += int(selected.size)
        used_events += 1
        if selected.size:
            yield selected, used_events, total_photons, selected_photons

    if used_events == 0:
        return


def collect_times(tree, max_events=0, raw=False):
    chunks = []
    used_events = 0
    total_photons = 0
    selected_photons = 0

    for selected, used_events, total_photons, selected_photons in iter_times(tree, max_events=max_events, raw=raw):
        chunks.append(selected)

    if not chunks:
        return np.asarray([], dtype=np.float64), used_events, total_photons, selected_photons
    return np.concatenate(chunks), used_events, total_photons, selected_photons


def make_hist(times, time_range, bin_width, per_event_denominator=1):
    t_min, t_max = time_range
    if t_max <= t_min:
        raise ValueError("time range max must be greater than min")
    if bin_width <= 0:
        raise ValueError("bin width must be positive")

    n_bins = int(np.ceil((t_max - t_min) / bin_width))
    edges = np.linspace(t_min, t_min + n_bins * bin_width, n_bins + 1)
    counts, edges = np.histogram(times, bins=edges)
    y = counts.astype(np.float64) / max(float(per_event_denominator), 1.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, edges, counts, y


def save_hist_csv(path, centers, edges, counts, y, y_name):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_low_ns", "time_high_ns", "time_center_ns", "count", y_name])
        for i, c in enumerate(centers):
            writer.writerow([edges[i], edges[i + 1], c, int(counts[i]), y[i]])


def plot_hist(path_png, path_pdf, centers, y, peak_time, title, ylabel, xlim=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.2))
    if len(centers) > 1:
        width = centers[1] - centers[0]
    else:
        width = 0.05
    ax.step(centers, y, where="mid", color="#1f77b4", linewidth=1.7)
    ax.fill_between(centers, y, step="mid", color="#1f77b4", alpha=0.18)
    ax.axvline(peak_time, color="#d62728", linestyle="--", linewidth=1.4, label=f"Peak ~ {peak_time:.3g} ns")
    ax.set_xlabel("Photon arrival time OP_time_final (ns)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xlim:
        ax.set_xlim(*xlim)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path_png, dpi=180)
    fig.savefig(path_pdf)
    plt.close(fig)


def main():
    args = parse_args()
    root_path = os.path.abspath(args.root_file)
    if not os.path.exists(root_path):
        raise SystemExit(f"Input ROOT file not found: {root_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    f = ROOT.TFile(root_path, "READ")
    if not f or f.IsZombie():
        raise SystemExit(f"Could not open ROOT file: {root_path}")
    tree = f.Get(args.tree)
    if not tree:
        f.Close()
        raise SystemExit(f"Could not find TTree '{args.tree}' in {root_path}")

    times, n_events, total_photons, selected_photons = collect_times(tree, max_events=args.max_events, raw=args.raw)
    f.Close()

    if times.size == 0:
        raise SystemExit("No photon times found after selection.")

    denom = n_events if args.per_event else 1
    centers, edges, counts, y = make_hist(times, tuple(args.time_range), args.bin_width, denom)
    if counts.sum() == 0:
        raise SystemExit("No photons fell inside the requested time range.")

    peak_idx = int(np.argmax(counts))
    peak_time = float(centers[peak_idx])
    y_name = "photons_per_event_per_bin" if args.per_event else "photons_per_bin"
    ylabel = "Photons / event / bin" if args.per_event else "Photons / bin"

    stem = os.path.splitext(os.path.basename(root_path))[0]
    suffix = "raw" if args.raw else "detector_mask"
    csv_path = os.path.join(args.output_dir, f"{stem}_nphotons_vs_time_{suffix}.csv")
    full_png = os.path.join(args.output_dir, f"{stem}_nphotons_vs_time_{suffix}.png")
    full_pdf = os.path.join(args.output_dir, f"{stem}_nphotons_vs_time_{suffix}.pdf")
    zoom_png = os.path.join(args.output_dir, f"{stem}_nphotons_vs_time_{suffix}_zoom.png")
    zoom_pdf = os.path.join(args.output_dir, f"{stem}_nphotons_vs_time_{suffix}_zoom.pdf")

    save_hist_csv(csv_path, centers, edges, counts, y, y_name)
    title = f"Photon arrival-time distribution ({'raw' if args.raw else 'detector mask'})"
    plot_hist(full_png, full_pdf, centers, y, peak_time, title, ylabel, xlim=tuple(args.time_range))
    plot_hist(zoom_png, zoom_pdf, centers, y, peak_time, title + " - zoom", ylabel, xlim=tuple(args.zoom_range))

    print(f"Input: {root_path}")
    print(f"Events used: {n_events}")
    print(f"Raw photons: {total_photons}")
    print(f"Selected photons: {selected_photons} ({'raw/no mask' if args.raw else 'isCoreC && pos_final_z > 0'})")
    print(f"Histogram range: {args.time_range[0]}-{args.time_range[1]} ns, bin width {args.bin_width} ns")
    print(f"Peak bin center: {peak_time:.4g} ns")
    print(f"CSV: {csv_path}")
    print(f"Plots: {full_png}, {zoom_png}")


if __name__ == "__main__":
    main()
