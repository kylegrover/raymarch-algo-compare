"""Faithful, sound Galin-style segment tracer (offline accuracy/cost ceiling).

This is the honest counterpart to the GLSL ``segment`` strawman, which faked the
Lipschitz bound with 3-point sampling and consequently tunnelled (Thin-Torus
macro IoU 0.003 in FINDINGS.md). Here the local directional-Lipschitz bound K is
computed *soundly* by interval automatic differentiation (gpu/interval_autodiff),
so the safe step ``|f|/K`` provably never passes the surface:

    at cursor τ (a point),  f = sdf(point)
    K = sound bound of |g'| over [τ, τ+h]        (interval autodiff)
    step = min(|f|/K, h)                          (≤ h keeps K valid)
    advance τ += step ;  grow h

It is **not** wall-clock-comparable to the GPU strategies (it runs offline on the
CPU). Its purpose is the ceiling: a faithful sound segment tracer reaches the
surface (IoU ≈ 1) on every metric scene — proving "Segment is broken" was an
implementation artifact — and we report the step cost a sound version pays
(notably *fewer* steps than sphere tracing at grazing angles, the real Galin win).
"""
from __future__ import annotations
import os
import json
import argparse
from typing import Dict, List, Optional

import numpy as np

from ..config import RenderConfig
from .groundtruth import reference_render_config, force_utf8_stdout
from .analytic import generate_rays
from .oracle_calibration import residual, silhouette_band
from . import analytic
from .interval import COMPONENT_SCENES, has_interval
from .interval_autodiff import seed_segment
from .interval_oracle import (
    interval_capture, _scalar_sdf, _prune_candidates, SCENE_BOUND, DEFAULT_T_MAX,
)

_TOL = 1e-4
_H0 = 0.1
_KAPPA = 1.5
_HMIN = 1e-5
_HMAX = 10.0
_KMIN = 1e-6
_BUDGET = 4096


def segment_trace(ro: np.ndarray, rd: np.ndarray, scene_name: str,
                  t_max: float = DEFAULT_T_MAX, tol: float = _TOL,
                  l_global: float = 1.0):
    """Sound directional-Lipschitz segment trace. Returns (t_hit (M,), iters (M,));
    t_hit is inf for misses. ``rd`` is (M, 3).

    ``l_global`` is the known global Lipschitz bound (1.0 for these exact metric
    SDFs). The autodiff directional bound is *clamped* to it: ``|g'| ≤ |∇f| ≤
    l_global`` always holds, while the autodiff value is tighter at grazing
    angles. This keeps the safe step sound where the interval autodiff
    overestimates K near the SDF's ``sqrt(0)`` non-smoothness (faces/edges)."""
    comp = COMPONENT_SCENES[scene_name]
    M = rd.shape[0]
    t_hit = np.full(M, np.inf)
    iters = np.zeros(M, dtype=np.int32)
    if M == 0:
        return t_hit, iters

    t = np.zeros(M)
    h = np.full(M, _H0)
    active = np.ones(M, dtype=bool)

    for _ in range(_BUDGET):
        idx = np.nonzero(active)[0]
        if idx.size == 0:
            break
        iters[idx] += 1
        P = ro[None, :] + t[idx, None] * rd[idx]
        f = _scalar_sdf(scene_name, P)                  # scalar SDF at cursor

        hit = np.abs(f) < tol
        hit_i = idx[hit]
        t_hit[hit_i] = t[hit_i]
        active[hit_i] = False

        run = ~hit
        ridx = idx[run]
        if ridx.size == 0:
            continue
        fr = f[run]
        hr = h[ridx]
        # K over [t, t+h] via interval autodiff
        X, Y, Z = seed_segment(ro, rd[ridx], t[ridx], t[ridx] + hr)
        d = comp(X, Y, Z)
        K = np.maximum(np.abs(d.der.lo), np.abs(d.der.hi))
        K = np.clip(K, _KMIN, l_global)          # |g'| <= |grad f| <= l_global

        safe = np.abs(fr) / K
        step = np.minimum(safe, hr)
        t[ridx] = t[ridx] + step
        # grow the probe; rays that ran past the horizon are misses
        h[ridx] = np.clip(np.maximum(step, safe) * _KAPPA, _HMIN, _HMAX)
        active[ridx[t[ridx] > t_max]] = False

    return t_hit, iters


def faithful_capture(scene_name: str, rc: RenderConfig, tol: float = _TOL) -> Dict:
    ro, rd = generate_rays(rc)
    H, W, _ = rd.shape
    rd_flat = rd.reshape(-1, 3)
    cand = _prune_candidates(ro, rd_flat, SCENE_BOUND.get(scene_name))
    t_hit, iters = segment_trace(ro, rd_flat[cand], scene_name, tol=tol)

    depth = np.zeros(H * W)
    hit = np.zeros(H * W, dtype=bool)
    cand_idx = np.nonzero(cand)[0]
    got = np.isfinite(t_hit)
    depth[cand_idx[got]] = t_hit[got]
    hit[cand_idx[got]] = True
    return {"depth": depth.reshape(H, W), "hit": hit.reshape(H, W),
            "iters_hit": iters[got]}


def evaluate(scene_names: List[str], *, width: int = 384, height: int = 384) -> Dict:
    report: Dict = {"resolution": [width, height], "scenes": {}}
    from .interval_oracle import interval_capture as _oracle
    from .groundtruth import _resolve_scene  # noqa: F401  (kept for parity)
    from ..scenes.catalog import get_all_scenes
    all_scenes = get_all_scenes()

    for name in scene_names:
        scene = next((s for s in all_scenes if s.name == name), None)
        if scene is None or not has_interval(name):
            print(f"  [skip] {name} — no interval extension")
            continue
        rc = reference_render_config(scene, width, height)
        gold = _oracle(name, rc)                     # sound oracle = truth
        fc = faithful_capture(name, rc)
        band = silhouette_band(gold["hit"], k=2)
        res = residual(fc["hit"], fc["depth"], gold["hit"], gold["depth"], band)
        cost = {
            "iters_median": float(np.median(fc["iters_hit"])) if fc["iters_hit"].size else 0.0,
            "iters_p95": float(np.percentile(fc["iters_hit"], 95)) if fc["iters_hit"].size else 0.0,
            "iters_max": int(fc["iters_hit"].max()) if fc["iters_hit"].size else 0,
        }
        report["scenes"][name] = {"accuracy_vs_oracle": res, "cost": cost}
        print(f"\n=== {name} ===")
        print(f"  IoU vs sound oracle : {res['iou']:.4f}  (core {res['core_iou']:.4f})  "
              f"depth med {res['depth_med']:.2e}")
        print(f"  cost (steps to hit) : median {cost['iters_median']:.0f}  "
              f"p95 {cost['iters_p95']:.0f}  max {cost['iters_max']}")
    return report


def main(argv: Optional[List[str]] = None) -> int:
    force_utf8_stdout()
    p = argparse.ArgumentParser(description="Faithful sound segment tracer — accuracy/cost ceiling.")
    p.add_argument("--scenes", type=str, default="Sphere,Grazing Plane,Cube,Thin Torus")
    p.add_argument("--res", type=int, default=384)
    p.add_argument("--out", type=str, default="references/faithful_segment.json")
    args = p.parse_args(argv)
    names = [s.strip() for s in args.scenes.split(",") if s.strip()]
    print(f"Faithful segment tracer ceiling on {len(names)} scene(s) at {args.res}x{args.res}")
    report = evaluate(names, width=args.res, height=args.res)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
