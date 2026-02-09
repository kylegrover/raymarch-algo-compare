"""Simple image exporters for iteration and depth heatmaps."""

import os
import numpy as np
from typing import Optional

# Pillow is optional — fall back to saving raw numpy arrays if unavailable
try:
    from PIL import Image
    _HAVE_PIL = True
except Exception:
    Image = None  # type: ignore
    _HAVE_PIL = False


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _save_or_warn(arr: np.ndarray, path_png: str, path_npy: str) -> None:
    """Try to save PNG (if PIL available), otherwise save .npy and warn."""
    _ensure_dir(path_png)
    if _HAVE_PIL:
        Image.fromarray(arr).save(path_png)
    else:
        np.save(path_npy, arr)
        print(f"[visualization] Pillow not installed — wrote raw data to {path_npy}."
              " Install Pillow (`uv add pillow`) to get PNG output.")


def save_iteration_heatmap(iter_map: np.ndarray, max_iters: Optional[int], path: str) -> None:
    """Save iterations-per-pixel as an 8-bit heatmap (normalized).

    If Pillow is missing, writes a `.npy` instead and warns.
    """
    h, w = iter_map.shape
    mx = int(max_iters) if max_iters is not None else int(iter_map.max() if iter_map.size else 1)
    mx = max(mx, 1)
    norm = np.clip(iter_map.astype(np.float32) / mx, 0.0, 1.0)
    img = (255.0 * norm).astype(np.uint8)
    npy_path = path + '.npy'
    _save_or_warn(img, path, npy_path)


def save_hit_map(hit_map: np.ndarray, path: str) -> None:
    """Save boolean hit map as black/white PNG (or .npy fallback)."""
    img = (hit_map.astype(np.uint8) * 255)
    npy_path = path + '.npy'
    _save_or_warn(img, path, npy_path)


def save_inverse_depth_map(depth_map: np.ndarray, hit_map: np.ndarray, path: str,
                            clip_percentile: float = 99.0) -> None:
    """Save 1.0/depth for hit pixels as a normalized grayscale image.

    If Pillow is missing, writes a `.npy` instead and warns.
    """
    inv = np.zeros_like(depth_map, dtype=np.float32)
    mask = (hit_map.astype(bool)) & (depth_map > 1e-12)
    inv[mask] = 1.0 / depth_map[mask]

    if mask.any():
        vmax = float(np.percentile(inv[mask], clip_percentile))
        vmax = max(vmax, 1e-12)
        norm = np.clip(inv / vmax, 0.0, 1.0)
    else:
        norm = inv

    img = (255.0 * norm).astype(np.uint8)
    npy_path = path + '.npy'
    _save_or_warn(img, path, npy_path)
