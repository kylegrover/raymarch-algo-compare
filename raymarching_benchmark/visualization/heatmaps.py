"""Simple image exporters for iteration and depth heatmaps."""

import os
import numpy as np
from typing import Optional, List

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


def save_tiled_comparison(maps: List[np.ndarray], labels: List[str], path: str,
                           max_val: Optional[int] = None) -> None:
    """Save a grid of heatmaps side-by-side for visual comparison.
    Requires Pillow.
    """
    if not _HAVE_PIL or not maps:
        return

    n = len(maps)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    
    h, w = maps[0].shape
    
    # Calculate grid size with some padding for labels
    label_h = 30
    combined_w = w * cols
    combined_h = (h + label_h) * rows
    
    res = Image.new('RGB', (combined_w, combined_h), (255, 255, 255))
    
    # We use a simple normalization for all maps
    mx = float(max_val or max(m.max() for m in maps))
    
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(res)
    
    for i, (m, label) in enumerate(zip(maps, labels)):
        r, c = divmod(i, cols)
        
        # Normalize and colorize (simple grayscale to red)
        norm = np.clip(m.astype(np.float32) / mx, 0.0, 1.0)
        # Create a basic heatmap: Black -> Red -> Yellow
        # For now, let's just do grayscale for simplicity or a simple red ramp
        pixels = (norm * 255).astype(np.uint8)
        img = Image.fromarray(pixels).convert('RGB')
        
        # Paste into grid
        x = c * w
        y = r * (h + label_h)
        res.paste(img, (x, y + label_h))
        
        # Draw label
        draw.text((x + 5, y + 5), label, fill=(0, 0, 0))

    res.save(path)
