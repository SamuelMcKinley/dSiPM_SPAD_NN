#
#   Code to simulate dSiPM SPADs given the photon outputs of the GEANT4 sim
#   Outputs tensors to be trained to CNN
#


import os
import sys
import re
import numpy as np
import ROOT
import csv
import fcntl
import shutil

# Deadtime Boolean
Deadtime = True

# Set True only when you want the original "Mickey" fiber placement:
# one fixed fiber shift and no shrink/distortion correction.
RESET_FIBER_LAYOUT_TO_ORIGINAL = False

# ROOT setup
ROOT.gROOT.SetBatch(True)
ROOT.TH1.SetDefaultSumw2()
ROOT.TH1.AddDirectory(False)

# Script & repo paths
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

# Adjustable geometry settings
DEFAULT_X_SHIFT = np.array([3.7308, 3.5768, 3.8008, 3.6468])
DEFAULT_Y_SHIFT = np.array([-3.7293, -3.6878, -3.6878, -3.6488])
ORIGINAL_X_SHIFT = np.array([3.8523, 3.8523, 3.8523, 3.8523], dtype=np.float32)
ORIGINAL_Y_SHIFT = np.array([-3.8588, -3.8588, -3.8588, -3.8588], dtype=np.float32)
shrink_rules = [(0.1 + 0.4 * i, round(0.23 * i, 2)) for i in range(40)]
limits = np.array([rule[0] for rule in shrink_rules])
shift_amounts = np.array([rule[1] for rule in shrink_rules])
DET_MIN = -100.0
DET_MAX =  100.0
DET_WIDTH = DET_MAX - DET_MIN
DEFAULT_TIME_SLICES_SPEC = "0-8,8-9,9-9.1,9.1-9.2,9.2-9.3,9.3-9.4,9.4-9.5,9.5-9.6,9.6-9.7,9.7-9.8,9.8-9.9,9.9-10,10-10.2,10.2-10.4,10.4-10.6,10.6-10.8,10.8-11,11-12,12-13,13-14,14-15,15-16,16-17,17-18,18-19,19-20,20-21,21-22,22-23,23-24,24-25,25-40"


# Wavelength-dependent QE
# Degree-5 polynomial fit to SiPM eta(lambda)
# \eta = c_5 \lambda ^5 + c_4 \lambda ^4 + ...

_QE_COEFFS = [
     8.3089805235e-14,   # c5
    -1.7718571478e-10,   # c4
     1.4724264010e-07,   # c3
    -6.0692493107e-05,   # c2
     1.3083151701e-02,   # c1
    -1.0675291442e+00,   # c0
]

def parse_time_slices(spec: str):
    ranges = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            raise ValueError(f"Bad time slice '{part}'. Expected low-high, e.g. 9.1-9.2")
        low_s, high_s = part.split("-", 1)
        low = float(low_s)
        high = float(high_s)
        if high <= low:
            raise ValueError(f"Bad time slice '{part}': high must be greater than low")
        ranges.append((low, high))
    if not ranges:
        raise ValueError("TIME_SLICES produced no ranges")
    return ranges

def eta_lambda(lam_nm: np.ndarray) -> np.ndarray:

    eta = np.polyval(_QE_COEFFS, lam_nm)
    # polynomial is only valid 300-620 nm
    eta = np.where((lam_nm >= 300) & (lam_nm <= 620), eta, 0.0)
    return np.clip(eta, 0.0, 1.0)


def shrink_toward_center_array(vals: np.ndarray) -> np.ndarray:
    abs_vals = np.abs(vals)
    idx = np.searchsorted(limits, abs_vals, side="right")
    idx = np.clip(idx, 0, len(shift_amounts) - 1)
    return vals - shift_amounts[idx] * np.sign(vals)

def apply_fiber_layout(g):
    if RESET_FIBER_LAYOUT_TO_ORIGINAL:
        x_shifted = g.pos_final_x + np.take(ORIGINAL_X_SHIFT, g.productionFiber)
        y_shifted = g.pos_final_y + np.take(ORIGINAL_Y_SHIFT, g.productionFiber)
        return x_shifted, y_shifted

    x_raw = g.pos_final_x + np.take(DEFAULT_X_SHIFT, g.productionFiber)
    y_raw = g.pos_final_y + np.take(DEFAULT_Y_SHIFT, g.productionFiber)
    return shrink_toward_center_array(x_raw), shrink_toward_center_array(y_raw)

class Photons:
    def __init__(self, event):
        self.time_final = np.array(event.OP_time_final)
        self.array4Sorting = np.argsort(self.time_final)
        self.time_final = self.time_final[self.array4Sorting]
        self.productionFiber = self._arr(event.OP_productionFiber)
        self.isCoreC = self._arr(event.OP_isCoreC)
        self.pos_final_x = self._arr(event.OP_pos_final_x)
        self.pos_final_y = self._arr(event.OP_pos_final_y)
        self.pos_final_z = self._arr(event.OP_pos_final_z)
        self.pos_produced_z = self._arr(event.OP_pos_produced_z)
        # Momentum in GeV/c -> wavelength in nm via hc = 1.2398e-6 GeV·nm
        mom_x = self._arr(event.OP_mom_produced_x)
        mom_y = self._arr(event.OP_mom_produced_y)
        mom_z = self._arr(event.OP_mom_produced_z)
        pmag = np.sqrt(mom_x**2 + mom_y**2 + mom_z**2)
        # avoid divide-by-zero
        pmag = np.where(pmag > 0, pmag, np.nan)
        self.wavelength_nm = 1.2398e-6 / pmag
        self.w = np.ones(self.nPhotons(), dtype=np.float32)

    def _arr(self, var): return np.array(var)[self.array4Sorting]
    def nPhotons(self): return len(self.pos_final_x)

class ChannelInfo:
    def __init__(self, channelSize, nBins):
        self.channelSize = channelSize
        self.nBins = nBins
        self.name = f"{int(channelSize)}x{int(channelSize)}"

def getNBins(l, h, s): return int((h - l) / s)

# CSV Updating
def update_photon_tracking(spad_size, channel_size, energy, total_photons, lost_photons):
    csv_path = os.path.join(SCRIPT_DIR, "photon_tracking.csv")
    header = ["SPAD_Size", "Channel_Size", "Energy", "Total_Photons", "Lost_Photons"]

    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(header)
        writer.writerow([
            spad_size,
            channel_size,
            f"{float(energy):g}",
            str(int(total_photons)),
            str(int(lost_photons))
        ])

    print(f"Appended entry to {csv_path} for SPAD={spad_size}, CH={channel_size}, {energy} GeV")


def append_photon_energy_row(spad_size: str, channel_size: str, nPhotons: int, energy: float):
    csv_path = os.path.join(SCRIPT_DIR, f"photon_energy_SPAD{spad_size}_CH{channel_size}.csv")
    header = ["nPhotons", "Energy"]

    with open(csv_path, "a+", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0, os.SEEK_END)
            is_empty = (f.tell() == 0)

            writer = csv.writer(f)
            if is_empty:
                writer.writerow(header)

            writer.writerow([int(nPhotons), float(energy)])

            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def main():
    if len(sys.argv) < 6:
        print("Usage: python simSPADs.py <root_file> <energy> <output_folder> <SPAD_size> <channel_size>")
        sys.exit(1)

    input_file_path = sys.argv[1]
    energy = float(sys.argv[2])
    output_folder = sys.argv[3]
    spad_size = sys.argv[4].strip()
    channel_size = sys.argv[5].strip()

    # Validate sizes strictly
    if not re.match(r"^\d+x\d+$", spad_size):
        print(f"Invalid SPAD size '{spad_size}'. Expected like '250x250'.")
        sys.exit(1)
    if not re.match(r"^\d+x\d+$", channel_size):
        print(f"Invalid channel size '{channel_size}'. Expected like '1000x1000'.")
        sys.exit(1)

    try:
        spad_side_um = int(spad_size.split("x")[0])
        ch_side_um = int(channel_size.split("x")[0])

        spad_spacing = spad_side_um / 1000.0      # um -> mm
        ch_spacing = ch_side_um / 1000.0          # um -> mm
    except Exception:
        print("Invalid size format. Use '250x250' / '1000x1000' style.")
        sys.exit(1)

    spad_nBins = getNBins(DET_MIN, DET_MAX, spad_spacing)
    ch_nBins   = getNBins(DET_MIN, DET_MAX, ch_spacing)

    ch = ChannelInfo(ch_side_um, ch_nBins)

    input_file = ROOT.TFile(input_file_path, "READ")
    tree = input_file.Get("tree")

    npy_dir = os.path.join(output_folder, "npy")
    csv_path = os.path.join(output_folder, "labels.csv")
    photon_stats_path = os.path.join(output_folder, "photon_stats.csv")
    time_slices_path = os.path.join(output_folder, "time_slices.txt")

    # One batch job owns one output folder. Clean it so reruns replace the
    # previous attempt instead of appending duplicate event tensors.
    if os.path.isdir(npy_dir):
        shutil.rmtree(npy_dir)
    for stale_path in (csv_path, photon_stats_path, time_slices_path):
        if os.path.exists(stale_path):
            os.remove(stale_path)
    os.makedirs(npy_dir, exist_ok=True)
    photon_stat_rows = []

    total_photons_cumulative, total_lost_cumulative, nEvents = 0, 0, -1

    root_stem = os.path.splitext(os.path.basename(input_file_path))[0]

    print(f"QE mode: wavelength-dependent polynomial (300-620 nm, 10 um microcells)")
    print(f"Deadtime: {'ON' if Deadtime else 'OFF'}")
    print(f"Fiber layout: {'ORIGINAL_MICKEY' if RESET_FIBER_LAYOUT_TO_ORIGINAL else 'DEFAULT_SHIFT_AND_SHRINK'}")
    print("-" * 60)

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "energy"])

        time_slices_spec = os.environ.get("TIME_SLICES", DEFAULT_TIME_SLICES_SPEC)
        try:
            time_slice_ranges = parse_time_slices(time_slices_spec)
        except ValueError as exc:
            print(f"Invalid TIME_SLICES: {exc}")
            sys.exit(1)
        with open(time_slices_path, "w") as meta_file:
            meta_file.write(time_slices_spec.strip() + "\n")
        print(f"Time slices ({len(time_slice_ranges)}): " + ", ".join(f"{a:g}-{b:g} ns" for a, b in time_slice_ranges))

        for event in tree:
            nEvents += 1
            g = Photons(event)
            print(f"\nEvent {nEvents}: {g.nPhotons()} raw photons")

            x_shifted, y_shifted = apply_fiber_layout(g)

            mask = (g.isCoreC.astype(bool)) & (g.pos_final_z > 0) & (0.0 < g.time_final) & (g.time_final < 40.0)
            x_vals  = 10 * x_shifted[mask]
            y_vals  = 10 * y_shifted[mask]
            t_vals  = g.time_final[mask]
            w_vals  = g.w[mask]
            lam_vals = g.wavelength_nm[mask]

            print(f"  After geometry/timing mask: {len(x_vals)} photons")

            # --- QUANTUM EFFICIENCY: wavelength-dependent ---
            n_before_qe = len(x_vals)
            qe_per_photon = eta_lambda(lam_vals)
            qe_mask = np.random.random(n_before_qe) < qe_per_photon
            n_after_qe = int(np.count_nonzero(qe_mask))
            n_discarded_qe = n_before_qe - n_after_qe

            # valid wavelength stats for reporting
            valid = (lam_vals >= 300) & (lam_vals <= 620)
            mean_lam  = lam_vals[valid].mean() if valid.any() else float('nan')
            mean_eta  = qe_per_photon[valid].mean() if valid.any() else float('nan')
            n_oob     = int((~valid).sum())

            print(f"  QE (wavelength-dep): {n_after_qe}/{n_before_qe} survive "
                  f"({n_discarded_qe} discarded)")
            print(f"    mean lambda = {mean_lam:.1f} nm  |  mean eta = {mean_eta:.4f}  "
                  f"|  {n_oob} photons outside 300-620 nm (eta=0)")

            x_vals   = x_vals[qe_mask]
            y_vals   = y_vals[qe_mask]
            t_vals   = t_vals[qe_mask]
            w_vals   = w_vals[qe_mask]
            lam_vals = lam_vals[qe_mask]

            total_photons_cumulative += n_after_qe

            # --- deadtime applied on SPAD grid ---
            photons_lost = 0
            if Deadtime:
                ix_spad = ((x_vals - DET_MIN) / (DET_WIDTH / spad_nBins)).astype(int)
                iy_spad = ((y_vals - DET_MIN) / (DET_WIDTH / spad_nBins)).astype(int)
                ix_spad = np.clip(ix_spad, 0, spad_nBins - 1)
                iy_spad = np.clip(iy_spad, 0, spad_nBins - 1)

                spad_ids = iy_spad * spad_nBins + ix_spad
                _, first_indices = np.unique(spad_ids, return_index=True)

                accepted = np.zeros_like(t_vals, dtype=bool)
                accepted[first_indices] = True

                photons_after = int(np.count_nonzero(accepted))
                photons_lost  = int(len(t_vals) - photons_after)
                total_lost_cumulative += photons_lost

                print(f"  Deadtime: in={len(x_vals)}, kept={photons_after}, lost={photons_lost}")

                x_vals, y_vals, t_vals, w_vals = (
                    x_vals[accepted], y_vals[accepted],
                    t_vals[accepted], w_vals[accepted]
                )
                nPhotons_used = photons_after
            else:
                print(f"  Deadtime OFF: {len(t_vals)} photons used")
                nPhotons_used = int(len(t_vals))

            print(f"  Final photons used for tensor: {nPhotons_used}")

            append_photon_energy_row(spad_size, channel_size, nPhotons_used, energy)

            denom = float(max(nPhotons_used, 1))
            lnN = float(np.log(denom))

            # --- Tensor binning for channel grid ---
            hist_tensor = []
            for t_low, t_high in time_slice_ranges:
                mask_t = (t_vals >= t_low) & (t_vals < t_high)
                H, _ = np.histogramdd(
                    np.stack((y_vals[mask_t], x_vals[mask_t]), axis=-1),
                    bins=(ch_nBins, ch_nBins),
                    range=[[DET_MIN, DET_MAX], [DET_MIN, DET_MAX]],
                    weights=w_vals[mask_t]
                )
                hist_tensor.append(H.astype(np.float32))

            event_tensor = (np.stack(hist_tensor, axis=0) / denom).astype(np.float32)

            filename = f"{root_stem}_event_{nEvents:04d}_SPAD{spad_size}_CH{ch.name}.npz"
            out_path = os.path.join(npy_dir, filename)

            np.savez(
                out_path,
                x=event_tensor,
                lnN=np.float32(lnN),
                time_slices=np.asarray(time_slice_ranges, dtype=np.float32),
            )

            writer.writerow([os.path.basename(out_path), energy])

            raw = int(n_before_qe)
            photon_stat_rows.append({
                "source_root": os.path.basename(input_file_path),
                "tensor_file": os.path.basename(out_path),
                "energy": energy,
                "event": nEvents,
                "spad_um": spad_side_um,
                "channel_um": ch_side_um,
                "n_raw": raw,
                "n_lost_qe": int(n_discarded_qe),
                "n_after_qe": int(n_after_qe),
                "n_lost_dt": int(photons_lost),
                "n_final": int(nPhotons_used),
                "mean_lam_nm": float(mean_lam) if not np.isnan(mean_lam) else "",
                "mean_eta": float(mean_eta) if not np.isnan(mean_eta) else "",
                "n_oob": int(n_oob),
            })

    stat_fields = [
        "source_root", "tensor_file", "energy", "event", "spad_um", "channel_um",
        "n_raw", "n_lost_qe", "n_after_qe", "n_lost_dt", "n_final",
        "mean_lam_nm", "mean_eta", "n_oob",
    ]
    with open(photon_stats_path, "w", newline="") as stat_file:
        stat_writer = csv.DictWriter(stat_file, fieldnames=stat_fields)
        stat_writer.writeheader()
        stat_writer.writerows(photon_stat_rows)
    print(f"Photon stats written to {photon_stats_path}")

    print("\n" + "=" * 60)
    print(f"Done. {nEvents + 1} events processed.")
    print(f"Total photons post-QE:      {total_photons_cumulative}")
    print(f"Total photons lost deadtime: {total_lost_cumulative}")

    update_photon_tracking(spad_size, channel_size, energy,
                           total_photons_cumulative, total_lost_cumulative)

if __name__ == "__main__":
    main()