"""Score a method capture against a ground-truth reference capture.

Metric hierarchy (post-review, SWEEP_PLAN §D):

  * PRIMARY (objective geometric) — **hit-mask IoU**, plus false-hit / false-miss.
    A skipped thin feature or a ragged silhouette tanks IoU with no arbitrary
    weighting, un-confounded by the large smooth regions everyone gets right.
    This is the headline accuracy number used to pick winners.
  * SECONDARY (objective, descriptive) — raw depth error (|t - t_ref| over
    co-hit pixels) and normal angular error. Quantitative but only defined where
    both agree there's a surface, so they can't see silhouette/thin-feature loss.
  * TERTIARY (structural, descriptive only) — SSIM on the depth image, normal
    map and lit color. Whole-image SSIM is dominated by the flat regions and
    dilutes the silhouette errors that matter, so it is **demoted to descriptive**
    and never used as the ranking objective. (FLIP, the peer-reviewed perceptual
    metric meant to replace it as the primary perceptual measure, is deferred.)

``score_capture`` returns the explicit ``primary``/``secondary``/``tertiary``
grouping and ALSO keeps the legacy flat keys (``hit``/``depth``/``normal``/
``ssim``) so existing report/sweep consumers keep working.
"""
from __future__ import annotations
from typing import Dict

import numpy as np
from skimage.metrics import structural_similarity as ssim

from ..data.capture_io import depth_to_image, normal_to_image, color_to_image


def _hit_metrics(m_hit: np.ndarray, r_hit: np.ndarray) -> Dict[str, float]:
    total = float(m_hit.size)
    false_hit = float(np.logical_and(m_hit, ~r_hit).sum()) / total
    false_miss = float(np.logical_and(~m_hit, r_hit).sum()) / total
    inter = float(np.logical_and(m_hit, r_hit).sum())
    union = float(np.logical_or(m_hit, r_hit).sum())
    iou = inter / union if union > 0 else 1.0
    return {"false_hit_rate": false_hit, "false_miss_rate": false_miss, "iou": iou,
            "agreement": float((m_hit == r_hit).mean())}


def _depth_metrics(method: Dict, reference: Dict) -> Dict[str, float]:
    both = np.logical_and(method["hit"], reference["hit"])
    if not both.any():
        return {"rmse": float("nan"), "mae": float("nan"), "p95": float("nan"),
                "n_pixels": 0}
    err = np.abs(method["depth"][both] - reference["depth"][both]).astype(np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(err)),
        "p95": float(np.percentile(err, 95)),
        "n_pixels": int(both.sum()),
    }


def _normal_angle_error(method: Dict, reference: Dict) -> Dict[str, float]:
    both = np.logical_and(method["hit"], reference["hit"])
    if not both.any():
        return {"mean_deg": float("nan"), "p95_deg": float("nan")}
    a = method["normal"][both].astype(np.float64)
    b = reference["normal"][both].astype(np.float64)
    # normals are already unit; clamp dot for numerical safety
    dot = np.clip(np.sum(a * b, axis=1), -1.0, 1.0)
    ang = np.degrees(np.arccos(dot))
    return {"mean_deg": float(np.mean(ang)), "p95_deg": float(np.percentile(ang, 95))}


def _ssim_scores(method: Dict, reference: Dict) -> Dict[str, float]:
    # Use a SHARED depth normalization (reference range) so the depth images are
    # directly comparable.
    r_hit = reference["hit"]
    if r_hit.any():
        d = reference["depth"][r_hit]
        drange = (float(d.min()), float(d.max()))
    else:
        drange = (0.0, 1.0)

    depth_m = depth_to_image(method["depth"], method["hit"], drange)
    depth_r = depth_to_image(reference["depth"], reference["hit"], drange)
    depth_ssim = float(ssim(depth_r, depth_m, data_range=255))

    norm_m = normal_to_image(method["normal"], method["hit"])
    norm_r = normal_to_image(reference["normal"], reference["hit"])
    normal_ssim = float(ssim(norm_r, norm_m, data_range=255, channel_axis=2))

    col_m = color_to_image(method["color"])
    col_r = color_to_image(reference["color"])
    color_ssim = float(ssim(col_r, col_m, data_range=255, channel_axis=2))
    color_rmse = float(np.sqrt(np.mean((col_r.astype(np.float64) - col_m) ** 2)))

    return {"depth_ssim": depth_ssim, "normal_ssim": normal_ssim,
            "color_ssim": color_ssim, "color_rmse": color_rmse}


#: The single objective metric used to rank strategies. Kept as a name so
#: reports/sweeps don't hard-code "iou" in selection logic.
PRIMARY_METRIC = "iou"


def score_capture(method: Dict, reference: Dict) -> Dict[str, Dict]:
    """Full accuracy report of a method capture vs a reference capture.

    Both dicts must come from ``GPURunner.capture`` (same resolution/camera).

    Returns the primary/secondary/tertiary hierarchy (see module docstring) plus
    legacy flat keys for back-compat.
    """
    if method["hit"].shape != reference["hit"].shape:
        raise ValueError(
            f"shape mismatch: method {method['hit'].shape} vs reference {reference['hit'].shape}")
    hit = _hit_metrics(method["hit"], reference["hit"])
    depth = _depth_metrics(method, reference)
    normal = _normal_angle_error(method, reference)
    ssim_scores = _ssim_scores(method, reference)
    return {
        # explicit hierarchy
        "primary": hit,                                   # IoU + false hit/miss
        "secondary": {"depth": depth, "normal": normal},  # objective, co-hit only
        "tertiary": ssim_scores,                          # descriptive SSIM
        # legacy flat keys (existing consumers)
        "hit": hit,
        "depth": depth,
        "normal": normal,
        "ssim": ssim_scores,
    }
