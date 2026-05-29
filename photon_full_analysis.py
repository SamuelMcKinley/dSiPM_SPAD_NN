import os
import sys
import re
import numpy as np
import ROOT
import csv
import fcntl
from datetime import datetime
from collections import defaultdict

# ── Configuration ─────────────────────────────────────────────────────────────

Deadtime = True
CHANNEL_SIZE = "1000x1000"   # fixed channel size for tensors

# SPAD sizes to loop over (um, square)
SPAD_SIZES_UM = [20, 50, 100, 200]

# ROOT setup
ROOT.gROOT.SetBatch(True)
ROOT.TH1.SetDefaultSumw2()
ROOT.TH1.AddDirectory(False)

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

# Geometry
xShift       = np.array([3.7308, 3.5768, 3.8008, 3.6468])
yShift       = np.array([-3.7293, -3.6878, -3.6878, -3.6488])
shrink_rules = [(0.1 + 0.4 * i, round(0.23 * i, 2)) for i in range(40)]
limits       = np.array([rule[0] for rule in shrink_rules])
shift_amounts= np.array([rule[1] for rule in shrink_rules])
DET_MIN      = -100.0
DET_MAX      =  100.0
DET_WIDTH    = DET_MAX - DET_MIN

# ── Wavelength-dependent QE ───────────────────────────────────────────────────
# Degree-5 polynomial fit to SiPM eta(lambda), onsemi MICRO-FC-30035 class,
# 10 um microcells (fill factor 0.28). Valid 300-620 nm.
_QE_COEFFS = [
     8.3089805235e-14,
    -1.7718571478e-10,
     1.4724264010e-07,
    -6.0692493107e-05,
     1.3083151701e-02,
    -1.0675291442e+00,
]

def eta_lambda(lam_nm: np.ndarray) -> np.ndarray:
    eta = np.polyval(_QE_COEFFS, lam_nm)
    eta = np.where((lam_nm >= 300) & (lam_nm <= 620), eta, 0.0)
    return np.clip(eta, 0.0, 1.0)

# ── Helpers ───────────────────────────────────────────────────────────────────

def shrink_toward_center_array(vals):
    abs_vals = np.abs(vals)
    idx = np.searchsorted(limits, abs_vals, side="right")
    idx = np.clip(idx, 0, len(shift_amounts) - 1)
    return vals - shift_amounts[idx] * np.sign(vals)

def unique_path(path):
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    k = 1
    while True:
        c = f"{base}_dup{k}{ext}"
        if not os.path.exists(c):
            return c
        k += 1

def getNBins(lo, hi, step):
    return int((hi - lo) / step)

# ── Classes ───────────────────────────────────────────────────────────────────

class Photons:
    def __init__(self, event):
        self.time_final     = np.array(event.OP_time_final)
        self.array4Sorting  = np.argsort(self.time_final)
        self.time_final     = self.time_final[self.array4Sorting]
        self.productionFiber= self._arr(event.OP_productionFiber)
        self.isCoreC        = self._arr(event.OP_isCoreC)
        self.pos_final_x    = self._arr(event.OP_pos_final_x)
        self.pos_final_y    = self._arr(event.OP_pos_final_y)
        self.pos_final_z    = self._arr(event.OP_pos_final_z)
        self.pos_produced_z = self._arr(event.OP_pos_produced_z)
        mom_x = self._arr(event.OP_mom_produced_x)
        mom_y = self._arr(event.OP_mom_produced_y)
        mom_z = self._arr(event.OP_mom_produced_z)
        pmag  = np.sqrt(mom_x**2 + mom_y**2 + mom_z**2)
        pmag  = np.where(pmag > 0, pmag, np.nan)
        self.wavelength_nm  = 1.2398e-6 / pmag
        self.w              = np.ones(self.nPhotons(), dtype=np.float32)

    def _arr(self, var): return np.array(var)[self.array4Sorting]
    def nPhotons(self):  return len(self.pos_final_x)

class ChannelInfo:
    def __init__(self, size_um, nBins):
        self.channelSize = size_um
        self.nBins       = nBins
        self.name        = f"{int(size_um)}x{int(size_um)}"

# ── CSV helpers ───────────────────────────────────────────────────────────────

def append_photon_energy_row(spad_str, ch_str, nPhotons, energy):
    csv_path = os.path.join(SCRIPT_DIR, f"photon_energy_SPAD{spad_str}_CH{ch_str}.csv")
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a+", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["nPhotons", "Energy"])
            writer.writerow([int(nPhotons), float(energy)])
            f.flush(); os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

# ── Per-file, per-SPAD processing ────────────────────────────────────────────

def process_file(input_file_path, energy, output_folder, spad_um, ch):
    """
    Process one ROOT file for one SPAD size.
    Returns list of per-event dicts with photon counts.
    """
    spad_str  = f"{spad_um}x{spad_um}"
    spad_spacing = spad_um / 1000.0        # um -> mm
    spad_nBins   = getNBins(DET_MIN, DET_MAX, spad_spacing)

    input_file = ROOT.TFile(input_file_path, "READ")
    tree       = input_file.Get("tree")
    root_stem  = os.path.splitext(os.path.basename(input_file_path))[0]

    os.makedirs(os.path.join(output_folder, "npy"), exist_ok=True)
    csv_path     = os.path.join(output_folder, "labels.csv")
    write_header = not os.path.exists(csv_path)

    time_slice_ranges = [(0,9),(9,9.5),(9.5,10),(10,15),(15,40)]
    event_records = []

    with open(csv_path, "a", newline="") as csvfile:
        lbl_writer = csv.writer(csvfile)
        if write_header:
            lbl_writer.writerow(["filename", "energy"])

        for ev_idx, event in enumerate(tree):
            g = Photons(event)

            x_raw = g.pos_final_x + np.take(xShift, g.productionFiber)
            y_raw = g.pos_final_y + np.take(yShift, g.productionFiber)
            x_s   = shrink_toward_center_array(x_raw)
            y_s   = shrink_toward_center_array(y_raw)

            mask = (
                g.isCoreC.astype(bool) &
                (g.pos_final_z > 0) &
                (0.0 < g.time_final) &
                (g.time_final < 40.0)
            )
            x_vals   = 10 * x_s[mask]
            y_vals   = 10 * y_s[mask]
            t_vals   = g.time_final[mask]
            w_vals   = g.w[mask]
            lam_vals = g.wavelength_nm[mask]

            n_raw = len(x_vals)

            # QE
            qe_per_photon  = eta_lambda(lam_vals)
            qe_mask        = np.random.random(n_raw) < qe_per_photon
            n_after_qe     = int(np.count_nonzero(qe_mask))
            n_lost_qe      = n_raw - n_after_qe

            valid = (lam_vals >= 300) & (lam_vals <= 620)
            mean_lam = float(lam_vals[valid].mean()) if valid.any() else float('nan')
            mean_eta = float(qe_per_photon[valid].mean()) if valid.any() else float('nan')
            n_oob    = int((~valid).sum())

            x_vals   = x_vals[qe_mask];   y_vals   = y_vals[qe_mask]
            t_vals   = t_vals[qe_mask];   w_vals   = w_vals[qe_mask]

            # Deadtime
            n_lost_dt = 0
            if Deadtime:
                ix = ((x_vals - DET_MIN) / (DET_WIDTH / spad_nBins)).astype(int)
                iy = ((y_vals - DET_MIN) / (DET_WIDTH / spad_nBins)).astype(int)
                ix = np.clip(ix, 0, spad_nBins-1)
                iy = np.clip(iy, 0, spad_nBins-1)
                ids = iy * spad_nBins + ix
                _, first = np.unique(ids, return_index=True)
                accepted = np.zeros(len(t_vals), dtype=bool)
                accepted[first] = True
                n_lost_dt = int(len(t_vals) - np.count_nonzero(accepted))
                x_vals = x_vals[accepted]; y_vals = y_vals[accepted]
                t_vals = t_vals[accepted]; w_vals = w_vals[accepted]

            nPhotons_used = int(len(t_vals))

            print(f"    [SPAD {spad_str}] Ev {ev_idx:03d}: "
                  f"raw={n_raw}  post-QE={n_after_qe} (-{n_lost_qe})  "
                  f"post-DT={nPhotons_used} (-{n_lost_dt})  "
                  f"<λ>={mean_lam:.0f}nm  <η>={mean_eta:.3f}  "
                  f"OOB={n_oob}")

            append_photon_energy_row(spad_str, ch.name, nPhotons_used, energy)

            # Tensor
            lnN = float(np.log(max(nPhotons_used, 1)))
            hist_tensor = []
            for t_lo, t_hi in time_slice_ranges:
                mt = (t_vals >= t_lo) & (t_vals < t_hi)
                H, _ = np.histogramdd(
                    np.stack((y_vals[mt], x_vals[mt]), axis=-1),
                    bins=(ch.nBins, ch.nBins),
                    range=[[DET_MIN,DET_MAX],[DET_MIN,DET_MAX]],
                    weights=w_vals[mt]
                )
                hist_tensor.append(H.astype(np.float32))

            event_tensor = np.stack(hist_tensor, axis=0)
            fname  = f"{root_stem}_ev{ev_idx:04d}_SPAD{spad_str}_CH{ch.name}.npz"
            outpath= unique_path(os.path.join(output_folder, "npy", fname))
            np.savez(outpath, x=event_tensor, lnN=np.float32(lnN))
            lbl_writer.writerow([os.path.basename(outpath), energy])

            event_records.append({
                "energy":       energy,
                "event":        ev_idx,
                "spad_um":      spad_um,
                "n_raw":        n_raw,
                "n_lost_qe":    n_lost_qe,
                "n_after_qe":   n_after_qe,
                "n_lost_dt":    n_lost_dt,
                "n_final":      nPhotons_used,
                "mean_lam_nm":  mean_lam,
                "mean_eta":     mean_eta,
                "n_oob":        n_oob,
            })

    input_file.Close()
    return event_records

# ── Plotting ──────────────────────────────────────────────────────────────────

def make_plots(all_records, output_folder):
    """Generate all comparison plots using ROOT."""
    import ROOT as R

    plot_dir = os.path.join(output_folder, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    energies  = sorted(set(r["energy"]  for r in all_records))
    spad_sizes= sorted(set(r["spad_um"] for r in all_records))

    # color palette per SPAD size
    spad_colors = {20: R.kBlue+1, 50: R.kRed+1, 100: R.kGreen+2, 200: R.kOrange+1}
    # marker styles per SPAD size
    spad_markers= {20: 20, 50: 21, 100: 22, 200: 23}
    # line styles per energy index
    energy_colors = [R.kBlue+1, R.kRed+1, R.kGreen+2, R.kOrange+1,
                     R.kMagenta+1, R.kCyan+2, R.kViolet+1, R.kTeal+2,
                     R.kPink+1, R.kSpring+5, R.kAzure+2, R.kYellow+2,
                     R.kGray+2, R.kBlack]

    def avg_by(records, key_x, key_y):
        """Average key_y grouped by key_x."""
        groups = defaultdict(list)
        for r in records:
            groups[r[key_x]].append(r[key_y])
        xs = sorted(groups.keys())
        ys = [np.mean(groups[x]) for x in xs]
        es = [np.std(groups[x]) / np.sqrt(len(groups[x])) for x in xs]
        return xs, ys, es

    def make_tgraph(xs, ys, es=None):
        n = len(xs)
        if es:
            g = R.TGraphErrors(n,
                np.array(xs,  dtype=float),
                np.array(ys,  dtype=float),
                np.zeros(n,   dtype=float),
                np.array(es,  dtype=float))
        else:
            g = R.TGraph(n,
                np.array(xs, dtype=float),
                np.array(ys, dtype=float))
        return g

    def save_canvas(c, name):
        c.SaveAs(os.path.join(plot_dir, name + ".pdf"))
        c.SaveAs(os.path.join(plot_dir, name + ".png"))

    def legend(c, graphs, labels, x1=0.55, y1=0.55, x2=0.88, y2=0.88):
        leg = R.TLegend(x1, y1, x2, y2)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        leg.SetTextSize(0.032)
        for g, l in zip(graphs, labels):
            leg.AddEntry(g, l, "lp")
        leg.Draw()
        return leg   # keep reference alive

    R.gStyle.SetOptStat(0)
    R.gStyle.SetOptTitle(1)
    R.gStyle.SetTitleSize(0.045, "XY")
    R.gStyle.SetLabelSize(0.038, "XY")

    # ── 1. Mean photons per stage vs energy, one line per SPAD ────────────────
    stages = [
        ("n_raw",      "Photons after geometry mask"),
        ("n_after_qe", "Photons after QE"),
        ("n_final",    "Photons after deadtime"),
    ]
    for key, stage_label in stages:
        c = R.TCanvas(f"c_{key}", stage_label, 900, 650)
        c.SetGrid()
        frame = c.DrawFrame(
            min(energies)*0.8, 0,
            max(energies)*1.1,
            max(np.mean([r[key] for r in all_records if r["spad_um"]==s])
                for s in spad_sizes) * 1.25
        )
        frame.GetXaxis().SetTitle("Beam energy (GeV)")
        frame.GetYaxis().SetTitle(f"Mean {stage_label} / event")
        frame.SetTitle(f"{stage_label} vs Energy")
        graphs, labels = [], []
        for spad in spad_sizes:
            recs = [r for r in all_records if r["spad_um"] == spad]
            xs, ys, es = avg_by(recs, "energy", key)
            g = make_tgraph(xs, ys, es)
            g.SetMarkerColor(spad_colors[spad])
            g.SetLineColor(spad_colors[spad])
            g.SetMarkerStyle(spad_markers[spad])
            g.SetMarkerSize(1.1)
            g.SetLineWidth(2)
            g.Draw("PL SAME")
            graphs.append(g); labels.append(f"SPAD {spad}x{spad} µm²")
        leg = legend(c, graphs, labels)
        save_canvas(c, f"photons_vs_energy_{key}")

    # ── 2. Photon loss breakdown vs energy, one line per SPAD ─────────────────
    for spad in spad_sizes:
        recs = [r for r in all_records if r["spad_um"] == spad]
        xs_raw, ys_raw, es_raw   = avg_by(recs, "energy", "n_raw")
        xs_qe,  ys_qe,  es_qe   = avg_by(recs, "energy", "n_lost_qe")
        xs_dt,  ys_dt,  es_dt   = avg_by(recs, "energy", "n_lost_dt")
        xs_fin, ys_fin, es_fin   = avg_by(recs, "energy", "n_final")

        ymax = max(ys_raw) * 1.25
        c = R.TCanvas(f"c_loss_{spad}", f"Loss breakdown SPAD {spad}µm", 900, 650)
        c.SetGrid()
        frame = c.DrawFrame(min(energies)*0.8, 0, max(energies)*1.1, ymax)
        frame.GetXaxis().SetTitle("Beam energy (GeV)")
        frame.GetYaxis().SetTitle("Mean photons / event")
        frame.SetTitle(f"Photon budget — SPAD {spad}x{spad} µm²")

        def styled_graph(xs, ys, es, col, style, width=2):
            g = make_tgraph(xs, ys, es)
            g.SetMarkerColor(col); g.SetLineColor(col)
            g.SetMarkerStyle(style); g.SetMarkerSize(1.0)
            g.SetLineWidth(width)
            g.Draw("PL SAME")
            return g

        g1 = styled_graph(xs_raw, ys_raw, es_raw, R.kBlack,     20)
        g2 = styled_graph(xs_qe,  ys_qe,  es_qe,  R.kRed+1,    22)
        g3 = styled_graph(xs_dt,  ys_dt,  es_dt,  R.kOrange+1, 23)
        g4 = styled_graph(xs_fin, ys_fin, es_fin, R.kGreen+2,  21)
        leg = legend(c, [g1,g2,g3,g4],
                     ["Raw (post-geom)","Lost to QE","Lost to deadtime","Final"],
                     0.52, 0.15, 0.88, 0.42)
        save_canvas(c, f"loss_breakdown_SPAD{spad}")

    # ── 3. Loss fractions vs energy per SPAD ──────────────────────────────────
    for spad in spad_sizes:
        recs = [r for r in all_records if r["spad_um"] == spad]
        xs_raw, ys_raw, _ = avg_by(recs, "energy", "n_raw")
        xs_qe,  ys_qe,  _ = avg_by(recs, "energy", "n_lost_qe")
        xs_dt,  ys_dt,  _ = avg_by(recs, "energy", "n_lost_dt")

        frac_qe = [lq/rw*100 if rw>0 else 0 for lq,rw in zip(ys_qe, ys_raw)]
        frac_dt = [ld/rw*100 if rw>0 else 0 for ld,rw in zip(ys_dt, ys_raw)]

        c = R.TCanvas(f"c_frac_{spad}", f"Loss fractions SPAD {spad}µm", 900, 650)
        c.SetGrid()
        ymax = max(max(frac_qe), max(frac_dt)) * 1.3
        frame = c.DrawFrame(min(energies)*0.8, 0, max(energies)*1.1, ymax)
        frame.GetXaxis().SetTitle("Beam energy (GeV)")
        frame.GetYaxis().SetTitle("Loss fraction (%)")
        frame.SetTitle(f"Loss fractions — SPAD {spad}x{spad} µm²")

        gq = make_tgraph(xs_qe, frac_qe)
        gq.SetMarkerColor(R.kRed+1); gq.SetLineColor(R.kRed+1)
        gq.SetMarkerStyle(22); gq.SetMarkerSize(1.1); gq.SetLineWidth(2)
        gq.Draw("PL SAME")

        gd = make_tgraph(xs_dt, frac_dt)
        gd.SetMarkerColor(R.kOrange+1); gd.SetLineColor(R.kOrange+1)
        gd.SetMarkerStyle(23); gd.SetMarkerSize(1.1); gd.SetLineWidth(2)
        gd.Draw("PL SAME")

        leg = legend(c, [gq, gd], ["QE loss %", "Deadtime loss %"])
        save_canvas(c, f"loss_fractions_SPAD{spad}")

    # ── 4. Final photons vs SPAD size at each energy ──────────────────────────
    c = R.TCanvas("c_vs_spad", "Final photons vs SPAD", 900, 650)
    c.SetGrid()
    ymax = max(r["n_final"] for r in all_records) * 1.25
    frame = c.DrawFrame(10, 0, 250, ymax)
    frame.GetXaxis().SetTitle("SPAD side length (µm)")
    frame.GetYaxis().SetTitle("Mean final photons / event")
    frame.SetTitle("Final photons vs SPAD size")
    graphs, labels = [], []
    for i, en in enumerate(energies):
        recs = [r for r in all_records if r["energy"] == en]
        xs, ys, es = avg_by(recs, "spad_um", "n_final")
        g = make_tgraph(xs, ys, es)
        col = energy_colors[i % len(energy_colors)]
        g.SetMarkerColor(col); g.SetLineColor(col)
        g.SetMarkerStyle(20); g.SetMarkerSize(0.9); g.SetLineWidth(2)
        g.Draw("PL SAME")
        graphs.append(g); labels.append(f"{en:.0f} GeV")
    leg = legend(c, graphs, labels, 0.12, 0.55, 0.45, 0.88)
    save_canvas(c, "final_photons_vs_spad")

    # ── 5. Deadtime loss vs SPAD size at each energy ──────────────────────────
    c = R.TCanvas("c_dt_vs_spad", "Deadtime loss vs SPAD", 900, 650)
    c.SetGrid()
    ymax = max(r["n_lost_dt"] for r in all_records) * 1.25 + 1
    frame = c.DrawFrame(10, 0, 250, ymax)
    frame.GetXaxis().SetTitle("SPAD side length (µm)")
    frame.GetYaxis().SetTitle("Mean photons lost to deadtime / event")
    frame.SetTitle("Deadtime loss vs SPAD size")
    graphs, labels = [], []
    for i, en in enumerate(energies):
        recs = [r for r in all_records if r["energy"] == en]
        xs, ys, es = avg_by(recs, "spad_um", "n_lost_dt")
        g = make_tgraph(xs, ys, es)
        col = energy_colors[i % len(energy_colors)]
        g.SetMarkerColor(col); g.SetLineColor(col)
        g.SetMarkerStyle(20); g.SetMarkerSize(0.9); g.SetLineWidth(2)
        g.Draw("PL SAME")
        graphs.append(g); labels.append(f"{en:.0f} GeV")
    leg = legend(c, graphs, labels, 0.12, 0.55, 0.45, 0.88)
    save_canvas(c, "deadtime_loss_vs_spad")

    # ── 6. Mean wavelength vs energy per SPAD ─────────────────────────────────
    c = R.TCanvas("c_lam", "Mean wavelength vs energy", 900, 650)
    c.SetGrid()
    all_lam = [r["mean_lam_nm"] for r in all_records if not np.isnan(r["mean_lam_nm"])]
    frame = c.DrawFrame(
        min(energies)*0.8, min(all_lam)*0.97,
        max(energies)*1.1, max(all_lam)*1.03
    )
    frame.GetXaxis().SetTitle("Beam energy (GeV)")
    frame.GetYaxis().SetTitle("Mean photon wavelength (nm)")
    frame.SetTitle("Mean Cherenkov wavelength vs Energy")
    graphs, labels = [], []
    for spad in spad_sizes:
        recs = [r for r in all_records if r["spad_um"] == spad]
        xs, ys, es = avg_by(recs, "energy", "mean_lam_nm")
        g = make_tgraph(xs, ys, es)
        g.SetMarkerColor(spad_colors[spad]); g.SetLineColor(spad_colors[spad])
        g.SetMarkerStyle(spad_markers[spad]); g.SetMarkerSize(1.0); g.SetLineWidth(2)
        g.Draw("PL SAME")
        graphs.append(g); labels.append(f"SPAD {spad}x{spad} µm²")
    leg = legend(c, graphs, labels)
    save_canvas(c, "mean_wavelength_vs_energy")

    # ── 7. Mean eta vs energy per SPAD ────────────────────────────────────────
    c = R.TCanvas("c_eta", "Mean eta vs energy", 900, 650)
    c.SetGrid()
    all_eta = [r["mean_eta"] for r in all_records if not np.isnan(r["mean_eta"])]
    frame = c.DrawFrame(
        min(energies)*0.8, min(all_eta)*0.97,
        max(energies)*1.1, max(all_eta)*1.03
    )
    frame.GetXaxis().SetTitle("Beam energy (GeV)")
    frame.GetYaxis().SetTitle("Mean QE η")
    frame.SetTitle("Mean QE η vs Energy")
    graphs, labels = [], []
    for spad in spad_sizes:
        recs = [r for r in all_records if r["spad_um"] == spad]
        xs, ys, es = avg_by(recs, "energy", "mean_eta")
        g = make_tgraph(xs, ys, es)
        g.SetMarkerColor(spad_colors[spad]); g.SetLineColor(spad_colors[spad])
        g.SetMarkerStyle(spad_markers[spad]); g.SetMarkerSize(1.0); g.SetLineWidth(2)
        g.Draw("PL SAME")
        graphs.append(g); labels.append(f"SPAD {spad}x{spad} µm²")
    leg = legend(c, graphs, labels)
    save_canvas(c, "mean_eta_vs_energy")

    # ── 8. QE survival fraction vs energy per SPAD ────────────────────────────
    c = R.TCanvas("c_qe_frac", "QE survival vs energy", 900, 650)
    c.SetGrid()
    frame = c.DrawFrame(min(energies)*0.8, 0, max(energies)*1.1, 100)
    frame.GetXaxis().SetTitle("Beam energy (GeV)")
    frame.GetYaxis().SetTitle("QE survival (%)")
    frame.SetTitle("Fraction of photons surviving QE vs Energy")
    graphs, labels = [], []
    for spad in spad_sizes:
        recs = [r for r in all_records if r["spad_um"] == spad]
        xs_raw, ys_raw, _ = avg_by(recs, "energy", "n_raw")
        xs_aqe, ys_aqe, _ = avg_by(recs, "energy", "n_after_qe")
        fracs = [aq/rw*100 if rw>0 else 0 for aq,rw in zip(ys_aqe, ys_raw)]
        g = make_tgraph(xs_raw, fracs)
        g.SetMarkerColor(spad_colors[spad]); g.SetLineColor(spad_colors[spad])
        g.SetMarkerStyle(spad_markers[spad]); g.SetMarkerSize(1.0); g.SetLineWidth(2)
        g.Draw("PL SAME")
        graphs.append(g); labels.append(f"SPAD {spad}x{spad} µm²")
    leg = legend(c, graphs, labels, 0.12, 0.15, 0.48, 0.42)
    save_canvas(c, "qe_survival_vs_energy")

    # ── 9. Final photon ratio: each SPAD vs smallest SPAD, vs energy ──────────
    if len(spad_sizes) > 1:
        ref_spad = min(spad_sizes)
        c = R.TCanvas("c_ratio", f"Photon ratio vs SPAD {ref_spad}µm", 900, 650)
        c.SetGrid()
        frame = c.DrawFrame(min(energies)*0.8, 0, max(energies)*1.1, 1.1)
        frame.GetXaxis().SetTitle("Beam energy (GeV)")
        frame.GetYaxis().SetTitle(f"Final photons / photons at SPAD {ref_spad}µm")
        frame.SetTitle(f"Relative yield vs reference SPAD {ref_spad}µm")
        graphs, labels = [], []

        ref_map = {}
        for r in all_records:
            if r["spad_um"] == ref_spad:
                ref_map[(r["energy"], r["event"])] = r["n_final"]

        for spad in spad_sizes:
            if spad == ref_spad:
                continue
            recs = [r for r in all_records if r["spad_um"] == spad]
            by_energy = defaultdict(list)
            for r in recs:
                ref_val = ref_map.get((r["energy"], r["event"]), None)
                if ref_val and ref_val > 0:
                    by_energy[r["energy"]].append(r["n_final"] / ref_val)
            xs = sorted(by_energy.keys())
            ys = [np.mean(by_energy[x]) for x in xs]
            es = [np.std(by_energy[x])/np.sqrt(len(by_energy[x])) for x in xs]
            g = make_tgraph(xs, ys, es)
            g.SetMarkerColor(spad_colors[spad]); g.SetLineColor(spad_colors[spad])
            g.SetMarkerStyle(spad_markers[spad]); g.SetMarkerSize(1.0); g.SetLineWidth(2)
            g.Draw("PL SAME")
            graphs.append(g); labels.append(f"SPAD {spad}x{spad} µm²")
        if graphs:
            leg = legend(c, graphs, labels)
        save_canvas(c, "photon_ratio_vs_ref_spad")

    # ── 10. Stacked bar: photon budget at each energy for each SPAD ───────────
    # Implemented as a TH1 multi-pad canvas
    n_spad = len(spad_sizes)
    c = R.TCanvas("c_budget", "Photon budget overview", 1200, 300 * n_spad)
    c.Divide(1, n_spad)
    for pi, spad in enumerate(spad_sizes):
        c.cd(pi + 1)
        R.gPad.SetGrid()
        recs = [r for r in all_records if r["spad_um"] == spad]
        xs_e, ys_fin, _ = avg_by(recs, "energy", "n_final")
        xs_e, ys_qe,  _ = avg_by(recs, "energy", "n_lost_qe")
        xs_e, ys_dt,  _ = avg_by(recs, "energy", "n_lost_dt")

        n = len(xs_e)
        h_fin = R.TH1F(f"hfin_{spad}", f"Photon budget SPAD {spad}µm;Energy (GeV);Mean photons/event",
                        n, 0.5, n + 0.5)
        h_qe  = R.TH1F(f"hqe_{spad}",  "", n, 0.5, n+0.5)
        h_dt  = R.TH1F(f"hdt_{spad}",  "", n, 0.5, n+0.5)
        for bi, (fin, qe, dt) in enumerate(zip(ys_fin, ys_qe, ys_dt)):
            h_fin.SetBinContent(bi+1, fin)
            h_qe.SetBinContent(bi+1,  qe)
            h_dt.SetBinContent(bi+1,  dt)
            h_fin.GetXaxis().SetBinLabel(bi+1, f"{xs_e[bi]:.0f}")

        h_fin.SetFillColor(R.kGreen+2);  h_fin.SetLineColor(R.kGreen+2)
        h_qe.SetFillColor(R.kRed+1);     h_qe.SetLineColor(R.kRed+1)
        h_dt.SetFillColor(R.kOrange+1);  h_dt.SetLineColor(R.kOrange+1)

        hs = R.THStack(f"hs_{spad}", f"SPAD {spad}x{spad} µm²  (green=final, red=QE loss, orange=DT loss)")
        hs.Add(h_fin); hs.Add(h_qe); hs.Add(h_dt)
        hs.Draw("HIST")
        hs.GetXaxis().SetTitle("Energy (GeV)")
        hs.GetYaxis().SetTitle("Mean photons / event")
        R.gPad.Update()

    save_canvas(c, "photon_budget_stacked")

    print(f"\nPlots saved to {plot_dir}/")


# ── Master CSV ────────────────────────────────────────────────────────────────

def write_master_csv(all_records, output_folder):
    path = os.path.join(output_folder, "photon_study_master.csv")
    fieldnames = [
        "energy", "event", "spad_um",
        "n_raw", "n_lost_qe", "n_after_qe",
        "n_lost_dt", "n_final",
        "frac_lost_qe_pct", "frac_lost_dt_pct", "frac_final_pct",
        "mean_lam_nm", "mean_eta", "n_oob"
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_records:
            raw = r["n_raw"]
            row = dict(r)
            row["frac_lost_qe_pct"] = round(r["n_lost_qe"]/raw*100, 3) if raw>0 else 0
            row["frac_lost_dt_pct"] = round(r["n_lost_dt"]/raw*100, 3) if raw>0 else 0
            row["frac_final_pct"]   = round(r["n_final"]  /raw*100, 3) if raw>0 else 0
            row["mean_lam_nm"]      = round(r["mean_lam_nm"], 2) if not np.isnan(r["mean_lam_nm"]) else ""
            row["mean_eta"]         = round(r["mean_eta"],    5) if not np.isnan(r["mean_eta"])    else ""
            w.writerow(row)
    print(f"Master CSV written to {path}")

    # Summary CSV: per (energy, spad) averages
    sum_path = os.path.join(output_folder, "photon_study_summary.csv")
    sum_fields = [
        "energy", "spad_um", "n_events",
        "mean_n_raw", "mean_n_lost_qe", "mean_n_after_qe",
        "mean_n_lost_dt", "mean_n_final",
        "mean_frac_lost_qe_pct", "mean_frac_lost_dt_pct", "mean_frac_final_pct",
        "mean_lam_nm", "mean_eta"
    ]
    energies   = sorted(set(r["energy"]  for r in all_records))
    spad_sizes = sorted(set(r["spad_um"] for r in all_records))
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sum_fields)
        w.writeheader()
        for en in energies:
            for spad in spad_sizes:
                recs = [r for r in all_records if r["energy"]==en and r["spad_um"]==spad]
                if not recs:
                    continue
                def m(k): return round(np.mean([r[k] for r in recs]), 3)
                def mnan(k):
                    v = [r[k] for r in recs if not np.isnan(r[k])]
                    return round(np.mean(v), 4) if v else ""
                raw = m("n_raw")
                w.writerow({
                    "energy":               en,
                    "spad_um":              spad,
                    "n_events":             len(recs),
                    "mean_n_raw":           raw,
                    "mean_n_lost_qe":       m("n_lost_qe"),
                    "mean_n_after_qe":      m("n_after_qe"),
                    "mean_n_lost_dt":       m("n_lost_dt"),
                    "mean_n_final":         m("n_final"),
                    "mean_frac_lost_qe_pct":round(m("n_lost_qe")/raw*100,2) if raw>0 else 0,
                    "mean_frac_lost_dt_pct":round(m("n_lost_dt")/raw*100,2) if raw>0 else 0,
                    "mean_frac_final_pct":  round(m("n_final")  /raw*100,2) if raw>0 else 0,
                    "mean_lam_nm":          mnan("mean_lam_nm"),
                    "mean_eta":             mnan("mean_eta"),
                })
    print(f"Summary CSV written to {sum_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Usage: python tensorMaker_multi.py <output_folder> <file1.root:energy1> <file2.root:energy2> ...
    # Example: python tensorMaker_multi.py ./out f1.root:10 f2.root:20 f3.root:50
    if len(sys.argv) < 3:
        print("Usage: python tensorMaker_multi.py <output_folder> <file.root:energy_GeV> [...]")
        print("Example: python tensorMaker_multi.py ./output file1.root:10 file2.root:30 file3.root:100")
        sys.exit(1)

    output_folder = sys.argv[1]
    os.makedirs(output_folder, exist_ok=True)

    # Parse file:energy pairs
    file_energy_pairs = []
    for arg in sys.argv[2:]:
        if ":" not in arg:
            print(f"Bad argument '{arg}'. Expected format: file.root:energy_GeV")
            sys.exit(1)
        parts = arg.rsplit(":", 1)
        fpath, en_str = parts[0], parts[1]
        if not os.path.exists(fpath):
            print(f"File not found: {fpath}")
            sys.exit(1)
        file_energy_pairs.append((fpath, float(en_str)))

    # Parse fixed channel size
    if not re.match(r"^\d+x\d+$", CHANNEL_SIZE):
        print(f"Bad CHANNEL_SIZE '{CHANNEL_SIZE}'")
        sys.exit(1)
    ch_side_um = int(CHANNEL_SIZE.split("x")[0])
    ch_spacing = ch_side_um / 1000.0
    ch_nBins   = getNBins(DET_MIN, DET_MAX, ch_spacing)
    ch         = ChannelInfo(ch_side_um, ch_nBins)

    print("=" * 70)
    print(f"Files      : {len(file_energy_pairs)}")
    print(f"SPAD sizes : {SPAD_SIZES_UM} µm")
    print(f"Channel    : {CHANNEL_SIZE} µm")
    print(f"Output     : {output_folder}")
    print(f"Deadtime   : {'ON' if Deadtime else 'OFF'}")
    print(f"QE model   : wavelength-dependent polynomial (300-620 nm)")
    print("=" * 70)

    all_records = []

    for fpath, energy in file_energy_pairs:
        print(f"\n{'─'*70}")
        print(f"File: {os.path.basename(fpath)}  |  Energy: {energy} GeV")
        print(f"{'─'*70}")
        for spad_um in SPAD_SIZES_UM:
            print(f"\n  → SPAD {spad_um}x{spad_um} µm²")
            records = process_file(fpath, energy, output_folder, spad_um, ch)
            all_records.extend(records)

    print(f"\n{'='*70}")
    print(f"All files done. Total event-records: {len(all_records)}")

    write_master_csv(all_records, output_folder)
    make_plots(all_records, output_folder)

    # Metadata
    meta_path = os.path.join(output_folder, "run_meta.txt")
    with open(meta_path, "w") as m:
        m.write(f"Run timestamp : {datetime.now()}\n")
        m.write(f"Files processed: {len(file_energy_pairs)}\n")
        for fp, en in file_energy_pairs:
            m.write(f"  {fp}  ({en} GeV)\n")
        m.write(f"SPAD sizes    : {SPAD_SIZES_UM} µm\n")
        m.write(f"Channel size  : {CHANNEL_SIZE} µm\n")
        m.write(f"Deadtime      : {Deadtime}\n")
        m.write(f"QE model      : wavelength-dependent 5th-order polynomial\n")
        m.write(f"  onsemi MICRO-FC-30035 class, 10 µm microcells, fill=0.28\n")
        m.write(f"  Valid: 300-620 nm. Outside -> eta=0.\n")
        m.write(f"Total records : {len(all_records)}\n")
    print(f"Metadata written to {meta_path}")

if __name__ == "__main__":
    main()