"""Phase 6.1 — strategy-independent scene/ray features per (scene, viewpoint).

The discovery question (SWEEP_PLAN §G / 6.2) is: *does a small set of cheap
scene/ray features predict which raymarching method is Pareto-optimal?* The
features must therefore describe the **scene+camera**, never the strategy — so
each is computed once per (scene, viewpoint) and later joined to every grid row
sharing that key. The expensive GPU grid is **not** re-run.

Per (scene, viewpoint) we emit:

  view-dependent (from one dense-march capture + reconstructed rays)
    hit_rate            fraction of pixels that hit a surface (coverage)
    grazing_frac        fraction of hits with |cos(ray, normal)| < cos(78.5°)
                        — the grazing-angle stress that murders naive step sizes
    silhouette_cplx     hit-mask perimeter / sqrt(area): shape complexity / how
                        much of the image is silhouette (high = filigree)
    hardness_mean       mean dense-march SDF evals over hit rays (ray difficulty)
    hardness_cv         coeff. of variation of those evals (warp-divergence proxy:
                        neighbouring rays taking very different path lengths)

  intrinsic (CPU SDF sampling in the visible bounding box)
    lipschitz_p99       p99 of |∇f| sampled in-frustum — >1 ⇒ the SDF lies about
                        distance (over-relaxation/segment hazard; fractals, bad-L)
    thin_slab           median solid-slab thickness along random chords, in units
                        of the bbox diagonal — small ⇒ thin features that tunnel

Run:
    uv run python -m raymarching_benchmark.report.features --out features.jsonl
"""
from __future__ import annotations
import sys
import json
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import RenderConfig
from ..scenes.base import SDFScene
from ..scenes.catalog import get_all_scenes
from ..viewpoints import viewpoints_for, Viewpoint
from ..core.vec3 import Vec3
from ..gpu.runner import GPURunner
from ..gpu.analytic import generate_rays
from ..gpu.groundtruth import (
    dense_march_config, dense_march_params, DENSE_MARCH_STRATEGY_ID, force_utf8_stdout,
)

GRAZING_COS = 0.20          # |cos(ray,n)| below this ⇒ grazing (angle > ~78.5°)
_LIP_SAMPLES = 1500         # frustum points for the Lipschitz estimate
_LIP_EPS = 1e-4             # central-difference step for |∇f|
_CHORDS = 240               # random chords for the thin-slab measurement
_CHORD_STEPS = 160          # samples per chord


def _sdf_at(scene: SDFScene, pts: np.ndarray) -> np.ndarray:
    """Evaluate the scalar Python SDF at points ``pts`` (N,3). Looped (the SDFs
    are scalar Vec3 functions) but N is small and this runs once per scene-view."""
    out = np.empty(pts.shape[0], dtype=np.float64)
    sdf = scene.sdf
    for i in range(pts.shape[0]):
        out[i] = sdf(Vec3(pts[i, 0], pts[i, 1], pts[i, 2]))
    return out


# ── view-dependent features ──────────────────────────────────────────────────

def _view_features(cap: Dict[str, np.ndarray], rc: RenderConfig
                   ) -> Tuple[Dict[str, float], np.ndarray]:
    """Features from the dense-march capture; also returns the world-space hit
    points (for deriving the intrinsic sampling box)."""
    hit = cap["hit"]
    n_hit = int(hit.sum())
    ro, rd = generate_rays(rc)                       # rd: (H,W,3) normalized
    feats: Dict[str, float] = {"hit_rate": float(hit.mean())}

    if n_hit == 0:
        feats.update(grazing_frac=0.0, silhouette_cplx=0.0,
                     hardness_mean=0.0, hardness_cv=0.0)
        return feats, np.zeros((0, 3))

    # Grazing: incidence angle between ray and outward normal at hit pixels.
    cosang = np.abs(np.sum(rd * cap["normal"], axis=-1))
    feats["grazing_frac"] = float((hit & (cosang < GRAZING_COS)).sum()) / n_hit

    # Silhouette complexity: perimeter / sqrt(area). A 4-neighbour boundary pixel
    # is a hit pixel touching a non-hit (image edge counts as non-hit).
    pad = np.pad(hit, 1, constant_values=False)
    interior = pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    perim = int((hit & ~interior).sum())
    feats["silhouette_cplx"] = perim / float(np.sqrt(n_hit))

    # Ray hardness from dense-march SDF eval counts on hits.
    ev = cap["evals"][hit].astype(np.float64)
    m = float(ev.mean())
    feats["hardness_mean"] = m
    feats["hardness_cv"] = float(ev.std() / m) if m > 0 else 0.0

    P = ro[None, None, :] + cap["depth"][..., None] * rd
    return feats, P[hit]


# ── intrinsic features (CPU SDF sampling) ────────────────────────────────────

def _intrinsic_features(scene: SDFScene, hit_pts: np.ndarray,
                        rng: np.random.Generator) -> Dict[str, float]:
    """Lipschitz estimate + thin-slab thickness, sampled in the box that bounds
    the visible geometry (so unbounded scenes get a finite, relevant region)."""
    if hit_pts.shape[0] < 8:
        return {"lipschitz_p99": float("nan"), "thin_slab": float("nan")}
    lo = hit_pts.min(axis=0)
    hi = hit_pts.max(axis=0)
    span = hi - lo
    diag = float(np.linalg.norm(span)) or 1.0
    pad = 0.05 * span + 1e-3
    lo = lo - pad
    hi = hi + pad

    # Lipschitz: p99 of |∇f| by central differences at uniform-random points.
    pts = lo + rng.random((_LIP_SAMPLES, 3)) * (hi - lo)
    grad = np.empty((_LIP_SAMPLES, 3))
    for ax in range(3):
        e = np.zeros(3); e[ax] = _LIP_EPS
        grad[:, ax] = (_sdf_at(scene, pts + e) - _sdf_at(scene, pts - e)) / (2 * _LIP_EPS)
    gmag = np.linalg.norm(grad, axis=1)
    gmag = gmag[np.isfinite(gmag)]
    lipschitz = float(np.percentile(gmag, 99)) if gmag.size else float("nan")

    # Thin-slab: along random chords through the box, measure solid (f<0) run
    # lengths. The median solid thickness (in bbox-diagonal units) is the thin-
    # feature scale; small ⇒ geometry a step can tunnel through.
    ts = np.linspace(0.0, 1.0, _CHORD_STEPS)
    slab_lengths: List[float] = []
    for _ in range(_CHORDS):
        a = lo + rng.random(3) * (hi - lo)
        b = lo + rng.random(3) * (hi - lo)
        seg = a[None, :] + ts[:, None] * (b - a)[None, :]
        seglen = float(np.linalg.norm(b - a))
        f = _sdf_at(scene, seg)
        inside = f < 0.0
        if not inside.any():
            continue
        # contiguous True runs → their physical lengths
        d = np.diff(inside.astype(np.int8))
        starts = np.where(d == 1)[0] + 1
        ends = np.where(d == -1)[0] + 1
        if inside[0]:
            starts = np.r_[0, starts]
        if inside[-1]:
            ends = np.r_[ends, _CHORD_STEPS]
        step_len = seglen / (_CHORD_STEPS - 1)
        for s, e in zip(starts, ends):
            slab_lengths.append((e - s) * step_len)
    thin = float(np.median(slab_lengths) / diag) if slab_lengths else float("nan")

    return {"lipschitz_p99": lipschitz, "thin_slab": thin}


# ── per scene-view driver ────────────────────────────────────────────────────

def scene_view_features(scene: SDFScene, vp: Viewpoint, runner: GPURunner,
                        res: int, rng: np.random.Generator) -> Dict[str, float]:
    sid = next(i for i, s in enumerate(get_all_scenes()) if s.name == scene.name)
    rc = vp.render_config(res, res)
    cap = runner.capture(sid, DENSE_MARCH_STRATEGY_ID, rc, dense_march_config(),
                         lipschitz=scene.known_lipschitz_bound(),
                         params=dense_march_params())
    vfeat, hit_pts = _view_features(cap, rc)
    ifeat = _intrinsic_features(scene, hit_pts, rng)
    return {**vfeat, **ifeat}


def build(res: int, seed: int = 0) -> List[Dict]:
    runner = GPURunner()
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []
    for scene in get_all_scenes():
        for vp in viewpoints_for(scene):
            print(f"  features: {scene.name} / {vp.name} ...")
            f = scene_view_features(scene, vp, runner, res, rng)
            rows.append({
                "scene": scene.name,
                "scene_category": scene.category,
                "viewpoint": vp.name,
                "view_category": vp.category,
                "features": f,
            })
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    force_utf8_stdout()
    p = argparse.ArgumentParser(description="Compute per scene-view discovery features (6.1).")
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--out", type=str, default="features.jsonl")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    print(f"Computing scene/ray features at {args.res}x{args.res} ...")
    rows = build(args.res, args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {args.out} ({len(rows)} scene-views).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
