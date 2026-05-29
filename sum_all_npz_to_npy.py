#!/usr/bin/env python3
import argparse
import os
import math
import numpy as np

def iter_npz_files(root_dir: str):
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.endswith(".npz"):
                yield os.path.join(dirpath, fn)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root_dir", help="Root directory containing many subfolders with *.npz (recursive)")
    ap.add_argument("-o", "--out", default="summed_tensor.npy", help="Output .npy path")
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"],
                    help="Accumulator dtype (float64 safest; float32 faster/smaller)")
    ap.add_argument("--max-files", type=int, default=0,
                    help="Debug: only process first N files (0 = all)")
    ap.add_argument("--progress-every", type=int, default=100,
                    help="Print progress every N files")
    args = ap.parse_args()

    npz_paths = list(iter_npz_files(args.root_dir))
    npz_paths.sort()

    if args.max_files and args.max_files > 0:
        npz_paths = npz_paths[:args.max_files]

    if not npz_paths:
        raise SystemExit(f"No .npz files found under: {args.root_dir}")

    acc = None
    n_files = 0
    total_bytes = 0

    for p in npz_paths:
        total_bytes += os.path.getsize(p)
        with np.load(p) as data:
            x = data["x"]          # (T,H,W), normalized by denom
            lnN = float(data["lnN"])
            denom = math.exp(lnN)

            # un-normalize to (approx) counts
            x_un = x * denom

            if acc is None:
                acc = np.zeros_like(x_un, dtype=np.float64 if args.dtype == "float64" else np.float32)

            # accumulate
            acc += x_un.astype(acc.dtype, copy=False)

        n_files += 1
        if args.progress_every > 0 and (n_files % args.progress_every == 0):
            gb = total_bytes / (1024**3)
            print(f"[{n_files}/{len(npz_paths)}] processed, scanned ~{gb:.2f} GB of .npz")

    # Save summed tensor
    np.save(args.out, acc)
    print(f"\nDone. Summed {n_files} files.")
    print(f"Saved: {args.out}")
    print(f"Accumulator shape: {acc.shape}, dtype: {acc.dtype}")
    gb_total = total_bytes / (1024**3)
    print(f"Total .npz bytes scanned (stat): ~{gb_total:.2f} GB")

if __name__ == "__main__":
    main()