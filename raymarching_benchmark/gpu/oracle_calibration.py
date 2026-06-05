"""Oracle calibration — measure the reference march's residual vs closed-form.

This answers SWEEP_PLAN validation gate #1: *on analytic scenes, what is the
reference oracle's error vs exact ray/surface depth?* It turns "we think the
oracle is good" into a reported error bar, and sweeps the dense march's
``minStep`` so we can pick the floor where the residual stops improving.

For every scene with a closed form (see gpu/analytic.py) it compares, against
analytic truth:

  * the **dense march** (id 9) at several ``minStep`` values, and
  * the **understep oracle** (id 0 understepped ×0.6), the previous reference,

reporting hit-mask IoU, false-hit / false-miss, and depth error over the
co-hit pixels. Because the only *irreducible* sampling error lives in the thin
silhouette band (feature chords < step), we also report a **core** IoU that
excludes a few-pixel band around the analytic silhouette — that separates
"systematic bias" (must be ~0) from "silhouette tunneling" (bounded by minStep).

Run:

    uv run python -m raymarching_benchmark.gpu.oracle_calibration --res 512
"""
from __future__ import annotations
import os
import sys
import json
import argparse
from typing import Dict, List, Optional

import numpy as np

from .runner import GPURunner
from .groundtruth import (
    reference_render_config, reference_march_config, oracle_params,
    dense_march_config, dense_march_params, DENSE_MARCH_STRATEGY_ID,
    force_utf8_stdout,
)
from . import analytic
from ..scenes.catalog import get_all_scenes


def _dilate(mask: np.ndarray, k: int) -> np.ndarray:
    """Binary dilation by k (4-connected), numpy-only."""
    out = mask.copy()
    for _ in range(k):
        s = np.zeros_like(out)
        s[1:, :] |= out[:-1, :]
        s[:-1, :] |= out[1:, :]
        s[:, 1:] |= out[:, :-1]
        s[:, :-1] |= out[:, 1:]
        out = out | s
    return out


def silhouette_band(hit: np.ndarray, k: int = 2) -> np.ndarray:
    """Pixels within k of the analytic hit-mask boundary."""
    # boundary = hit pixels adjacent to a miss (or vice versa)
    nb_miss = _dilate(~hit, 1) & hit
    nb_hit = _dilate(hit, 1) & ~hit
    boundary = nb_miss | nb_hit
    return _dilate(boundary, k)


def residual(method_hit, method_depth, an_hit, an_depth, band) -> Dict[str, float]:
    inter = np.logical_and(method_hit, an_hit)
    union = np.logical_or(method_hit, an_hit)
    iou = float(inter.sum()) / float(max(union.sum(), 1))

    # core = exclude the silhouette band, where tunneling is irreducible
    keep = ~band
    ch = np.logical_and(method_hit, an_hit) & keep
    cu = np.logical_or(method_hit, an_hit) & keep
    core_iou = float(ch.sum()) / float(max(cu.sum(), 1))

    n_an = int(an_hit.sum())
    n_m = int(method_hit.sum())
    false_hit = float((method_hit & ~an_hit).sum()) / float(max(n_m, 1))
    false_miss = float((~method_hit & an_hit).sum()) / float(max(n_an, 1))

    both = inter
    if both.sum() > 0:
        de = method_depth[both] - an_depth[both]
        ad = np.abs(de)
        depth_rmse = float(np.sqrt(np.mean(de * de)))
        depth_med = float(np.median(ad))
        depth_p95 = float(np.percentile(ad, 95))
        depth_signed = float(np.mean(de))
    else:
        depth_rmse = depth_med = depth_p95 = depth_signed = float("nan")

    return {
        "iou": iou,
        "core_iou": core_iou,
        "false_hit": false_hit,
        "false_miss": false_miss,
        "depth_rmse": depth_rmse,
        "depth_med": depth_med,
        "depth_p95": depth_p95,
        "depth_signed": depth_signed,
        "n_analytic_hit": n_an,
        "n_method_hit": n_m,
        "n_co_hit": int(both.sum()),
    }


def calibrate(scene_names: List[str], *, width: int = 512, height: int = 512,
              min_steps: Optional[List[float]] = None) -> Dict:
    if min_steps is None:
        min_steps = [0.02, 0.01, 0.005, 0.002, 0.001]
    runner = GPURunner()
    report: Dict = {"resolution": [width, height], "min_steps": min_steps, "scenes": {}}

    all_scenes = get_all_scenes()
    for name in scene_names:
        scene_id = next((i for i, s in enumerate(all_scenes) if s.name == name), None)
        if scene_id is None or not analytic.has_analytic(name):
            print(f"  [skip] {name} — no closed form")
            continue
        scene = all_scenes[scene_id]
        rc = reference_render_config(scene, width, height)

        an = analytic.analytic_capture(name, rc)
        band = silhouette_band(an["hit"], k=2)
        band_frac = float(band.sum()) / band.size

        rows: Dict[str, Dict] = {}

        # understep oracle (previous reference): standard id 0, ×0.6
        ucap = runner.capture(scene_id, 0, rc, reference_march_config(),
                              lipschitz=scene.known_lipschitz_bound(),
                              params=oracle_params())
        rows["understep_x0.6"] = residual(ucap["hit"], ucap["depth"],
                                          an["hit"], an["depth"], band)

        # dense march at each minStep
        dmc = dense_march_config()
        for ms in min_steps:
            dcap = runner.capture(scene_id, DENSE_MARCH_STRATEGY_ID, rc, dmc,
                                  lipschitz=scene.known_lipschitz_bound(),
                                  params=dense_march_params(ms))
            rows[f"dense_minStep={ms}"] = residual(dcap["hit"], dcap["depth"],
                                                   an["hit"], an["depth"], band)

        report["scenes"][name] = {
            "analytic_hit_rate": float(an["hit"].mean()),
            "silhouette_band_frac": band_frac,
            "methods": rows,
        }
        _print_scene(name, band_frac, rows)

    return report


def _print_scene(name: str, band_frac: float, rows: Dict[str, Dict]) -> None:
    print(f"\n=== {name}  (silhouette band = {band_frac:.2%} of pixels) ===")
    hdr = (f"  {'method':<20} {'IoU':>7} {'coreIoU':>8} {'fHit':>7} {'fMiss':>7} "
           f"{'dRMSE':>9} {'dMed':>9} {'dSign':>10}")
    print(hdr)
    for k, m in rows.items():
        print(f"  {k:<20} {m['iou']:>7.4f} {m['core_iou']:>8.4f} "
              f"{m['false_hit']:>7.3f} {m['false_miss']:>7.3f} "
              f"{m['depth_rmse']:>9.2e} {m['depth_med']:>9.2e} {m['depth_signed']:>10.2e}")


def main(argv: Optional[List[str]] = None) -> int:
    force_utf8_stdout()
    p = argparse.ArgumentParser(description="Calibrate the reference oracle vs closed-form depth.")
    p.add_argument("--scenes", type=str, default="Sphere,Grazing Plane,Cube,Thin Torus",
                   help="Comma-separated scene names (must have closed forms), or 'all'.")
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--min-steps", type=str, default="0.02,0.01,0.005,0.002,0.001")
    p.add_argument("--out", type=str, default="references/oracle_calibration.json")
    args = p.parse_args(argv)

    if args.scenes.strip().lower() == "all":
        names = list(analytic.ANALYTIC_SCENES.keys())
    else:
        names = [s.strip() for s in args.scenes.split(",") if s.strip()]
    min_steps = [float(x) for x in args.min_steps.split(",") if x.strip()]

    print(f"Calibrating {len(names)} analytic scene(s) at {args.res}x{args.res} "
          f"vs closed-form depth; minStep sweep = {min_steps}")
    report = calibrate(names, width=args.res, height=args.res, min_steps=min_steps)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved calibration report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
