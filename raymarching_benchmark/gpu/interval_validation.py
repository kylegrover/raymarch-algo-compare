"""Validate the interval-arithmetic gold oracle and bound the dense-march oracle.

This is the trust gate for `gpu/interval_oracle.py`. The interval march is *sound*
by construction (it can only prove emptiness, never hide a surface), but we still
have to show its implementation reproduces ground truth at image scale before
trusting it where no closed form exists. For each metric scene with a closed form
it reports three residuals against the analytic truth and each other:

  * **interval vs analytic** — the trust gate. If the interval oracle is correct
    this is ~tol depth error and IoU ≈ 1; that licenses using it on the
    non-analytic metric scenes (cylinder / CSG / onion / blend) where it is the
    *only* gold standard.
  * **dense-march vs analytic** — the existing reference's error bar (context).
  * **dense-march vs interval** — do the two oracles agree? This is the headline:
    the measured error of the oracle the whole grid was scored against, now
    cross-checked against a sound reference rather than only closed form.

Run:

    uv run python -m raymarching_benchmark.gpu.interval_validation --res 384
"""
from __future__ import annotations
import os
import json
import argparse
from typing import Dict, List, Optional

from .runner import GPURunner
from .groundtruth import (
    reference_render_config, dense_march_config, dense_march_params,
    DENSE_MARCH_STRATEGY_ID, force_utf8_stdout,
)
from .oracle_calibration import residual, silhouette_band
from . import analytic
from .interval_oracle import interval_capture, DEFAULT_TOL
from .interval import has_interval
from ..scenes.catalog import get_all_scenes


def validate(scene_names: List[str], *, width: int = 384, height: int = 384,
             tol: float = DEFAULT_TOL) -> Dict:
    runner = GPURunner()
    report: Dict = {"resolution": [width, height], "tol": tol, "scenes": {}}
    all_scenes = get_all_scenes()

    for name in scene_names:
        scene_id = next((i for i, s in enumerate(all_scenes) if s.name == name), None)
        if scene_id is None or not has_interval(name):
            print(f"  [skip] {name} — no interval extension")
            continue
        scene = all_scenes[scene_id]
        rc = reference_render_config(scene, width, height)

        iv = interval_capture(name, rc, tol=tol)
        an = analytic.analytic_capture(name, rc)
        dcap = runner.capture(scene_id, DENSE_MARCH_STRATEGY_ID, rc, dense_march_config(),
                              lipschitz=scene.known_lipschitz_bound(),
                              params=dense_march_params())

        band = silhouette_band(an["hit"], k=2)
        band_frac = float(band.sum()) / band.size

        rows = {
            "interval_vs_analytic": residual(iv["hit"], iv["depth"], an["hit"], an["depth"], band),
            "dense_vs_analytic": residual(dcap["hit"], dcap["depth"], an["hit"], an["depth"], band),
            "dense_vs_interval": residual(dcap["hit"], dcap["depth"], iv["hit"], iv["depth"], band),
        }
        report["scenes"][name] = {
            "analytic_hit_rate": float(an["hit"].mean()),
            "interval_hit_rate": float(iv["hit"].mean()),
            "silhouette_band_frac": band_frac,
            "comparisons": rows,
        }
        _print_scene(name, band_frac, rows)

    return report


def _print_scene(name: str, band_frac: float, rows: Dict[str, Dict]) -> None:
    print(f"\n=== {name}  (silhouette band = {band_frac:.2%} of pixels) ===")
    print(f"  {'comparison':<22} {'IoU':>7} {'coreIoU':>8} {'fHit':>7} {'fMiss':>7} "
          f"{'dRMSE':>9} {'dMed':>9} {'dSign':>10}")
    for k, m in rows.items():
        print(f"  {k:<22} {m['iou']:>7.4f} {m['core_iou']:>8.4f} "
              f"{m['false_hit']:>7.3f} {m['false_miss']:>7.3f} "
              f"{m['depth_rmse']:>9.2e} {m['depth_med']:>9.2e} {m['depth_signed']:>10.2e}")


def main(argv: Optional[List[str]] = None) -> int:
    force_utf8_stdout()
    p = argparse.ArgumentParser(description="Validate the interval oracle vs analytic + dense-march.")
    p.add_argument("--scenes", type=str, default="Sphere,Grazing Plane,Cube,Thin Torus")
    p.add_argument("--res", type=int, default=384)
    p.add_argument("--tol", type=float, default=DEFAULT_TOL)
    p.add_argument("--out", type=str, default="references/interval_validation.json")
    args = p.parse_args(argv)

    names = [s.strip() for s in args.scenes.split(",") if s.strip()]
    print(f"Validating interval oracle on {len(names)} scene(s) at {args.res}x{args.res} "
          f"(tol={args.tol})")
    report = validate(names, width=args.res, height=args.res, tol=args.tol)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved validation report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
