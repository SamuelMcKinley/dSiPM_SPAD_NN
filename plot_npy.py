#
#   Plotting code that turns .npy into .png histograms
#   Input: python plot_npy.py <input_npy> -o <output_directory>
#


import os
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt

# Must match tensorMaker time slices
TIME_SLICES = [(0, 9), (9, 9.5), (9.5, 10), (10, 15), (15, 40)]

def parse_metadata_from_filename(fname):
    """
    Extract SPAD size, channel size, and energy from filename.
    Returns None if not found.
    """
    spad = re.search(r"SPAD(\d+x\d+)", fname)
    ch   = re.search(r"CH(\d+x\d+)", fname)
    gev  = re.search(r"(\d+(?:\.\d+)?)GeV", fname)

    return (
        spad.group(1) if spad else None,
        ch.group(1) if ch else None,
        gev.group(1) if gev else None,
    )


def build_title(spad=None, ch=None, gev=None):
    parts = []

    if spad:
        parts.append(f"SPAD {spad}")
    if ch:
        parts.append(f"CH {ch}")
    if gev:
        parts.append(f"{gev} GeV π+")

    if parts:
        return "Detector hit map — " + ", ".join(parts)
    else:
        return "Detector hit map"


def plot_slice(img, title, outpath):
    # Mask zero bins → white
    img_masked = np.ma.masked_equal(img, 0.0)

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")

    plt.figure()
    plt.imshow(
        img_masked,
        origin="lower",
        cmap=cmap,
        vmin=img_masked.min(),
        interpolation="nearest",
        aspect="equal",
    )
    plt.xlabel("x bin")
    plt.ylabel("y bin")
    plt.title(title)
    plt.colorbar(label="counts")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def load_npz_counts(path):
    """
    Load event-level NPZ and un-normalize using lnN.
    """
    data = np.load(path)
    x = data["x"]          # (5, H, W), normalized
    lnN = float(data["lnN"])
    denom = float(np.exp(lnN))
    x_counts = x * denom
    return x_counts, lnN


def load_npy_counts(path):
    """
    Load summed NPY already in counts.
    """
    return np.load(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_file", help="Path to event_*.npz OR summed_*.npy")
    ap.add_argument("-o", "--outdir", default="png_out", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    in_path = args.input_file
    base = os.path.splitext(os.path.basename(in_path))[0]
    ext = os.path.splitext(in_path)[1].lower()

    spad, ch, gev = parse_metadata_from_filename(base)
    title_base = build_title(spad, ch, gev)

    if ext == ".npz":
        x_counts, lnN = load_npz_counts(in_path)
        print(f"Loaded NPZ: lnN = {lnN:.6f}  → denom ≈ {np.exp(lnN):.0f}")
    elif ext == ".npy":
        x_counts = load_npy_counts(in_path)
        print("Loaded NPY: using counts tensor as-is (already un-normalized).")
    else:
        raise SystemExit(f"Unsupported input type: {ext} (expected .npz or .npy)")

    if x_counts.ndim != 3 or x_counts.shape[0] != 5:
        raise SystemExit(
            f"Unexpected tensor shape {x_counts.shape}; expected (5, H, W)"
        )

    # Per time slice
    for i, (t0, t1) in enumerate(TIME_SLICES):
        img = x_counts[i]
        outpath = os.path.join(
            args.outdir, f"{base}_slice{i}_{t0:g}-{t1:g}ns.png"
        )
        plot_slice(img, title=title_base, outpath=outpath)

    # Sum over slices
    img_sum = np.sum(x_counts, axis=0)
    outpath = os.path.join(args.outdir, f"{base}_sum.png")
    plot_slice(img_sum, title=title_base, outpath=outpath)

    print(f"Wrote PNGs to {args.outdir}")
    print(f"Tensor shape = {x_counts.shape}")


if __name__ == "__main__":
    main()