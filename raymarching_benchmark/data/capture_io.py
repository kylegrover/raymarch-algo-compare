"""Save / load a GPU capture (geom + normal + depth + color + hit) and encode
viewable PNGs. Shared by ground-truth reference generation and per-method
snapshots so both sides are stored identically and stay directly comparable.

Raw arrays are persisted as a single compressed ``.npz`` (the source of truth
for scoring); PNGs are written alongside purely for eyeballing.
"""
from __future__ import annotations
import os
from typing import Optional, Tuple, Dict

import numpy as np
from PIL import Image


def _to_u8(x: np.ndarray) -> np.ndarray:
    # nan_to_num first: a NaN/inf (e.g. a degenerate normal or an all-miss frame)
    # otherwise casts to garbage and warns "invalid value encountered in cast".
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def normal_to_image(normal: np.ndarray, hit: np.ndarray) -> np.ndarray:
    """Map world normal [-1,1] -> RGB [0,255]; misses are black."""
    img = normal * 0.5 + 0.5
    img = np.where(hit[..., None], img, 0.0)
    return _to_u8(img)


def color_to_image(color: np.ndarray) -> np.ndarray:
    """Shaded color is already gamma-encoded [0,1]."""
    return _to_u8(color)


def depth_to_image(depth: np.ndarray, hit: np.ndarray,
                   depth_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Grayscale depth, near = bright. ``depth_range`` lets the reference and a
    method share the same normalization so the images line up visually."""
    if depth_range is None:
        if hit.any():
            d = depth[hit]
            lo, hi = float(d.min()), float(d.max())
        else:
            lo, hi = 0.0, 1.0
    else:
        lo, hi = depth_range
    rng = max(hi - lo, 1e-6)
    norm = np.clip((depth - lo) / rng, 0.0, 1.0)
    gray = np.where(hit, 1.0 - norm, 0.0)
    return _to_u8(gray)


def depth_range_of(capture: Dict[str, np.ndarray]) -> Tuple[float, float]:
    hit = capture["hit"]
    if hit.any():
        d = capture["depth"][hit]
        return float(d.min()), float(d.max())
    return 0.0, 1.0


def save_capture(capture: Dict[str, np.ndarray], out_dir: str, name: str,
                 *, depth_range: Optional[Tuple[float, float]] = None,
                 write_pngs: bool = True) -> str:
    """Persist raw arrays as ``<name>.npz`` and (optionally) viewable PNGs.

    Returns the path to the ``.npz``.
    """
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, f"{name}.npz")
    np.savez_compressed(
        npz_path,
        geom=capture["geom"].astype(np.float32),
        normal=capture["normal"].astype(np.float32),
        depth=capture["depth"].astype(np.float32),
        color=capture["color"].astype(np.float32),
        hit=capture["hit"].astype(np.bool_),
    )

    if write_pngs:
        if depth_range is None:
            depth_range = depth_range_of(capture)
        Image.fromarray(depth_to_image(capture["depth"], capture["hit"], depth_range)).save(
            os.path.join(out_dir, f"{name}_depth.png"))
        Image.fromarray(normal_to_image(capture["normal"], capture["hit"])).save(
            os.path.join(out_dir, f"{name}_normal.png"))
        Image.fromarray(color_to_image(capture["color"])).save(
            os.path.join(out_dir, f"{name}_color.png"))

    return npz_path


def load_capture(npz_path: str) -> Dict[str, np.ndarray]:
    z = np.load(npz_path)
    return {k: z[k] for k in ("geom", "normal", "depth", "color", "hit")}
