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

# Deadtime Boolean
Deadtime = True

# ROOT setup
ROOT.gROOT.SetBatch(True)
ROOT.TH1.SetDefaultSumw2()
ROOT.TH1.AddDirectory(False)

# Script & repo paths
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

# Adjustable geometry settings
xShift = np.array([3.7308, 3.5768, 3.8008, 3.6468])
yShift = np.array([-3.7293, -3.6878, -3.6878, -3.6488])
shrink_rules = [(0.1 + 0.4 * i, round(0.23 * i, 2)) for i in range(40)]
limits = np.array([rule[0] for rule in shrink_rules])
shift_amounts = np.array([rule[1] for rule in shrink_rules])
DET_MIN = -100.0
DET_MAX =  100.0
DET_WIDTH = DET_MAX - DET_MIN

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

def unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    k = 1
    while True:
        candidate = f"{base}_dup{k}{ext}"
        if not os.path.exists(candidate):
            return candidate
        k += 1

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

    os.makedirs(os.path.join(output_folder, "npy"), exist_ok=True)
    csv_path = os.path.join(output_folder, "labels.csv")
    write_header = not os.path.exists(csv_path)

    total_photons_cumulative, total_lost_cumulative, nEvents = 0, 0, -1

    root_stem = os.path.splitext(os.path.basename(input_file_path))[0]

    print(f"QE mode: wavelength-dependent polynomial (300-620 nm, 10 um microcells)")
    print(f"Deadtime: {'ON' if Deadtime else 'OFF'}")
    print("-" * 60)

    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["filename", "energy"])

        time_slice_ranges = [(0, 9), (9, 9.5), (9.5, 10), (10, 15), (15, 40)]

        for event in tree:
            nEvents += 1
            g = Photons(event)
            print(f"\nEvent {nEvents}: {g.nPhotons()} raw photons")

            x_raw = g.pos_final_x + np.take(xShift, g.productionFiber)
            y_raw = g.pos_final_y + np.take(yShift, g.productionFiber)
            x_shifted = shrink_toward_center_array(x_raw)
            y_shifted = shrink_toward_center_array(y_raw)

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

            event_tensor = np.stack(hist_tensor, axis=0).astype(np.float32)

            filename = f"{root_stem}_event_{nEvents:04d}_SPAD{spad_size}_CH{ch.name}.npz"
            out_path = unique_path(os.path.join(output_folder, "npy", filename))

            np.savez(
                out_path,
                x=event_tensor,
                lnN=np.float32(lnN),
            )

            writer.writerow([os.path.basename(out_path), energy])

    print("\n" + "=" * 60)
    print(f"Done. {nEvents + 1} events processed.")
    print(f"Total photons post-QE:      {total_photons_cumulative}")
    print(f"Total photons lost deadtime: {total_lost_cumulative}")

    update_photon_tracking(spad_size, channel_size, energy,
                           total_photons_cumulative, total_lost_cumulative)

if __name__ == "__main__":
    main()