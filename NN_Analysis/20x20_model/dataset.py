#!/usr/bin/env python3
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# Matches any "...30GeV..." anywhere in the *full path string*
GEV_ANYWHERE_RE = re.compile(r"(?P<energy>\d+(?:\.\d+)?)GeV")

def _parse_energy_from_path(p: Path) -> float:
    s = str(p)
    m = GEV_ANYWHERE_RE.search(s)
    if not m:
        raise ValueError(f"Could not find '<energy>GeV' in path: {s}")
    return float(m.group("energy"))

def _to_chw(arr: np.ndarray) -> np.ndarray:
    """
    Ensure output is (C, H, W).
    Accepts (H, W) -> (1, H, W)
            (C, H, W) -> unchanged
            (H, W, C) -> transpose to (C, H, W) when C is small-ish
    """
    if arr.ndim == 2:
        return arr[np.newaxis, ...]
    if arr.ndim == 3:
        if arr.shape[-1] <= 16 and arr.shape[0] != arr.shape[-1]:
            return np.transpose(arr, (2, 0, 1))
        return arr
    raise ValueError(f"Unsupported tensor shape {arr.shape}; expected 2D or 3D")

class PhotonEnergyDataset(Dataset):
    """
    Loads MANY .npz files and reconstructs UNNORMALIZED tensors.

    Your .npz format:
      - x   : normalized tensor = counts / N    (C,H,W) float32
      - lnN : float32 = ln(N)

    We reconstruct:
      x_unnorm = x * exp(lnN)  (counts-like tensor)

    Energy is parsed from any '<energy>GeV' substring in the *path*.
    """

    def __init__(self, tensor_path: str, dtype: np.dtype = np.float32, recursive: bool = True):
        path = Path(tensor_path)

        if path.is_file() and path.suffix.lower() == ".npz":
            files = [path]
        elif path.is_dir():
            if recursive:
                files = sorted([p for p in path.rglob("*.npz") if p.is_file()])
            else:
                files = sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".npz"])
        else:
            raise FileNotFoundError(f"tensor_path not found: {tensor_path}")

        if len(files) == 0:
            raise RuntimeError(f"No .npz files found under {tensor_path} (recursive={recursive})")

        self.files: List[Path] = files
        self.dtype = dtype

        # Peek first file to infer channels/shape (C,H,W)
        p0 = self.files[0]
        with np.load(p0, allow_pickle=False) as z:
            arr0 = np.asarray(z["x"], dtype=self.dtype)
        arr0 = _to_chw(arr0)
        self.channels, self.height, self.width = arr0.shape

        # Pre-extract energies and names
        self._energies: List[float] = []
        self._names: List[str] = []
        for p in self.files:
            self._energies.append(_parse_energy_from_path(p))
            self._names.append(str(p))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        p = self.files[idx]
        e = self._energies[idx]
        name = self._names[idx]

        with np.load(p, allow_pickle=False) as z:
            # Force float32 immediately
            x_norm = np.asarray(z["x"], dtype=np.float32)
            lnN = np.asarray(z["lnN"], dtype=np.float32).reshape(()).item()

        # Ensure CHW and float32
        x_norm = _to_chw(x_norm).astype(np.float32, copy=False)

        # Unnormalize: counts = (counts/N) * N
        N = np.exp(np.float32(lnN)).astype(np.float32)
        x_unnorm = (x_norm * N).astype(np.float32, copy=False)

        # Convert to torch (explicit float32 safety)
        x = torch.from_numpy(x_unnorm).float()          # (C,H,W) float32
        y = torch.tensor(e, dtype=torch.float32)        # scalar (GeV)

        return x, y, name

    def get_all_energies(self) -> List[float]:
        return list(self._energies)

    def get_all_names(self) -> List[str]:
        return list(self._names)