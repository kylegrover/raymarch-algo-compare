"""Gold-standard offline oracle: first ray/surface intersection by interval
root isolation.

Soundness argument (why this can't tunnel): the interval extension in
``interval.py`` returns ``[lo, hi]`` enclosing every SDF value on a ray segment.
``lo > 0`` therefore *proves* the segment contains no surface and is discarded
with zero risk. We branch-and-bound the parameter ``t``: keep only segments that
could contain a root (``lo <= 0``), split them, and report the nearest segment
that has narrowed below ``tol``. Because empty segments are eliminated by proof
(not by sampling), the nearest reported root is the true first intersection — the
property the dense-march oracle cannot guarantee on L>1 SDFs.

Vectorized across rays via a segment worklist (one branch-and-bound for every
candidate ray simultaneously). Returns the same ``{depth, hit, normal}`` contract
as ``gpu/analytic.analytic_capture`` and is pixel-aligned with ``GPURunner.capture``
(it reuses ``analytic.generate_rays``).
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Dict, Optional

from ..config import RenderConfig
from .analytic import generate_rays, _normalize
from .interval import Interval, IVec3, INTERVAL_SCENES

# Bounding-sphere radius per scene (origin-centered) for cheap pre-pruning of
# rays that provably miss. ``None`` => unbounded (e.g. infinite plane): skip the
# prune and let the interval march resolve every ray.
SCENE_BOUND: Dict[str, Optional[float]] = {
    "Sphere": 1.05,
    "Cube": 1.7421,        # sqrt(3) + slack
    "Thin Torus": 1.65,    # R + r + slack
    "Grazing Plane": None,
}

DEFAULT_T_MAX = 100.0      # matches reference/dense-march maxDistance
DEFAULT_TOL = 1e-5
_H0 = 0.25                 # initial probe length
_GROWTH = 1.5              # probe growth across proven-empty space
_HMAX = 10.0               # probe ceiling
_MAX_ITERS = 20000         # per-ray march cap (grazing safety)


def _scalar_sdf(scene_name: str, P: np.ndarray) -> np.ndarray:
    """Evaluate the scene SDF at points ``P`` (..., 3) via degenerate intervals
    (lo == hi == f(P)). Reuses the exact interval construction so it stays in
    lock-step with the oracle's notion of the surface."""
    fn = INTERVAL_SCENES[scene_name]
    p = IVec3(Interval.point(P[..., 0]),
              Interval.point(P[..., 1]),
              Interval.point(P[..., 2]))
    return np.asarray(fn(p).lo, dtype=np.float64)


def _prune_candidates(ro: np.ndarray, rd: np.ndarray, bound: Optional[float]) -> np.ndarray:
    """Boolean mask over flattened rays that *could* hit the bounding sphere."""
    if bound is None:
        return np.ones(rd.shape[0], dtype=bool)
    oc = ro                                   # sphere centered at origin
    proj = rd @ oc                            # = O·d
    dist2 = float(oc @ oc) - proj * proj      # squared distance line→origin
    t_close = -proj                           # parameter of closest approach
    return (dist2 <= bound * bound) & (t_close + bound > 1e-6)


def first_hit(ro: np.ndarray, rd: np.ndarray,
              sdf_i: Callable[[IVec3], Interval],
              t_max: float = DEFAULT_T_MAX,
              tol: float = DEFAULT_TOL) -> np.ndarray:
    """Nearest provable intersection ``t`` per ray (inf if none), via a sound
    forward interval march. ``rd`` is (M, 3).

    Each step probes the segment ``[t, t+h]``:
      * interval ``lo > 0`` ⇒ the segment is *proven* empty ⇒ jump ``t += h`` and
        grow ``h`` (this is the no-tunnel guarantee — a true enclosure can only
        prove emptiness, never hide a surface);
      * ``lo <= 0`` with ``h <= tol`` ⇒ first contact is bracketed ⇒ report ``t``;
      * ``lo <= 0`` with ``h > tol`` ⇒ shrink ``h`` and re-probe the same ``t``.
    Interval wrapping (the loose AABB enclosure of a slanted segment) can only
    make a step *more* conservative — extra shrink steps, never a missed surface.
    """
    M = rd.shape[0]
    result = np.full(M, np.inf, dtype=np.float64)
    if M == 0:
        return result

    t = np.zeros(M, dtype=np.float64)
    h = np.full(M, _H0, dtype=np.float64)
    active = np.ones(M, dtype=bool)

    for _ in range(_MAX_ITERS):
        idx = np.nonzero(active)[0]
        if idx.size == 0:
            break
        t0 = t[idx]
        hh = h[idx]
        t1 = t0 + hh

        def _pos(comp_dir, o):
            a = comp_dir * t0
            b = comp_dir * t1
            return Interval(np.minimum(a, b), np.maximum(a, b)) + o

        pos = IVec3(_pos(rd[idx, 0], ro[0]),
                    _pos(rd[idx, 1], ro[1]),
                    _pos(rd[idx, 2], ro[2]))
        d = sdf_i(pos)

        empty = d.lo > 0.0
        hit = (~empty) & (hh <= tol)
        shrink = (~empty) & (~hit)

        # Confirmed first hits.
        hit_i = idx[hit]
        result[hit_i] = t0[hit]
        active[hit_i] = False

        # Jump across proven-empty segments and grow the probe.
        emp_i = idx[empty]
        t[emp_i] = t1[empty]
        h[emp_i] = np.minimum(hh[empty] * _GROWTH, _HMAX)
        # Rays that ran past the far plane without a hit are misses.
        active[emp_i[t[emp_i] > t_max]] = False

        # Shrink and re-probe the same cursor.
        h[idx[shrink]] = hh[shrink] * 0.5

    return result


def _normals_fd(scene_name: str, P: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Outward SDF-gradient normals by central differences at hit points P (K,3)."""
    if P.shape[0] == 0:
        return np.zeros((0, 3))
    g = np.empty_like(P)
    for ax in range(3):
        e = np.zeros(3); e[ax] = eps
        g[:, ax] = _scalar_sdf(scene_name, P + e) - _scalar_sdf(scene_name, P - e)
    return _normalize(g, axis=-1)


def interval_capture(scene_name: str, rc: RenderConfig,
                     t_max: float = DEFAULT_T_MAX,
                     tol: float = DEFAULT_TOL) -> Optional[Dict[str, np.ndarray]]:
    """Sound first-hit depth/hit/normal for a metric scene, pixel-aligned with
    ``GPURunner.capture``. Returns None for scenes without an interval extension."""
    if scene_name not in INTERVAL_SCENES:
        return None
    ro, rd = generate_rays(rc)
    H, W, _ = rd.shape
    rd_flat = rd.reshape(-1, 3)

    cand = _prune_candidates(ro, rd_flat, SCENE_BOUND.get(scene_name))
    t_cand = first_hit(ro, rd_flat[cand], INTERVAL_SCENES[scene_name], t_max, tol)

    depth = np.zeros(H * W, dtype=np.float64)
    hit = np.zeros(H * W, dtype=bool)
    cand_idx = np.nonzero(cand)[0]
    got = np.isfinite(t_cand)
    hit_global = cand_idx[got]
    depth[hit_global] = t_cand[got]
    hit[hit_global] = True

    normal = np.zeros((H * W, 3), dtype=np.float64)
    P = ro[None, :] + depth[hit_global][:, None] * rd_flat[hit_global]
    normal[hit_global] = _normals_fd(scene_name, P)

    return {"depth": depth.reshape(H, W),
            "hit": hit.reshape(H, W),
            "normal": normal.reshape(H, W, 3)}
