"""Revised affine arithmetic — the real "RevAA", and its tightness win over plain
interval arithmetic.

The GLSL ``rev_affine`` strategy does no affine arithmetic at all (it min/maxes
two endpoint samples), so "RevAA ≡ Standard" was a tautology. Affine arithmetic
is a genuinely different, *tighter* inclusion: a quantity is tracked as

    x = x0 + x1·ε + e·[-1, 1]

where ε is the shared noise symbol of the march parameter τ (so the linear
dependence on τ is preserved exactly through additions and the linear part of
products — the correlation interval arithmetic throws away) and ``e ≥ 0`` is the
accumulated nonlinear remainder. Both AA and IA are *sound*, so as a first-hit
root isolator both reach the surface (IoU ≈ 1) — neither tunnels, unlike the
strawman. AA's payoff is efficiency: tighter enclosures ⇒ larger provably-empty
jumps ⇒ fewer SDF segment-evaluations. This module measures exactly that, running
one march driver with the IA range and the AA range and comparing eval counts.

AA implements the same op names as ``interval.Interval`` (``+ - *``, ``square``,
``sqrt``, ``abs``, ``max0/min0``, ``maximum/minimum``) so it evaluates the shared
``COMPONENT_SCENES`` SDFs unchanged. Non-smooth ops (abs/min/max) fall back to a
sound interval hull (a fresh remainder, no τ-correlation); the smooth chain
(square+sqrt, e.g. the sphere/torus radial distance) keeps full correlation, which
is where AA beats IA.
"""
from __future__ import annotations
import os
import json
import argparse
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..config import RenderConfig
from .groundtruth import reference_render_config, force_utf8_stdout
from .analytic import generate_rays
from .oracle_calibration import residual, silhouette_band
from .interval import Interval, IVec3, INTERVAL_SCENES, COMPONENT_SCENES, has_interval
from .interval_oracle import (
    interval_capture, _prune_candidates, SCENE_BOUND, DEFAULT_T_MAX, DEFAULT_TOL,
    _H0, _GROWTH, _HMAX, _MAX_ITERS,
)

_EPS = 1e-12


class AAForm:
    """Revised affine form with one shared symbol (the march parameter) + an
    accumulated nonlinear remainder. lo/hi via ``range()``."""

    __slots__ = ("x0", "x1", "e")

    def __init__(self, x0, x1, e):
        self.x0 = np.asarray(x0, dtype=np.float64)   # center
        self.x1 = np.asarray(x1, dtype=np.float64)   # coeff of the τ symbol
        self.e = np.asarray(e, dtype=np.float64)     # nonneg remainder radius

    def range(self) -> Tuple[np.ndarray, np.ndarray]:
        rad = np.abs(self.x1) + self.e
        return self.x0 - rad, self.x0 + rad

    # ── linear ops (exact, correlation preserved) ────────────────────────────
    def __add__(self, o):
        if isinstance(o, AAForm):
            return AAForm(self.x0 + o.x0, self.x1 + o.x1, self.e + o.e)
        return AAForm(self.x0 + o, self.x1, self.e)

    __radd__ = __add__

    def __sub__(self, o):
        if isinstance(o, AAForm):
            return AAForm(self.x0 - o.x0, self.x1 - o.x1, self.e + o.e)
        return AAForm(self.x0 - o, self.x1, self.e)

    def __rsub__(self, o):
        return AAForm(o - self.x0, -self.x1, self.e)

    def __neg__(self):
        return AAForm(-self.x0, -self.x1, self.e)

    def __mul__(self, o):
        if isinstance(o, AAForm):
            # linear part exact; bound every nonlinear/cross term into e.
            e_new = (np.abs(self.x1) * o.e + np.abs(o.x1) * self.e + self.e * o.e
                     + np.abs(self.x1 * o.x1))
            return AAForm(self.x0 * o.x0, self.x0 * o.x1 + self.x1 * o.x0, e_new)
        return AAForm(self.x0 * o, self.x1 * o, self.e * np.abs(o))

    __rmul__ = __mul__

    # ── nonlinear smooth ops (keep correlation) ──────────────────────────────
    def square(self):
        # x^2 = x0^2 + 2 x0 x1 ε + x1^2 ε^2 + (remainder); ε^2 ∈ [0,1]
        return AAForm(self.x0 * self.x0 + 0.5 * self.x1 * self.x1,
                      2.0 * self.x0 * self.x1,
                      0.5 * self.x1 * self.x1
                      + 2.0 * (np.abs(self.x0) + np.abs(self.x1)) * self.e
                      + self.e * self.e)

    def sqrt(self):
        lo, hi = self.range()
        a = np.maximum(lo, 0.0)
        b = np.maximum(hi, 0.0)
        sa, sb = np.sqrt(a), np.sqrt(b)
        wide = (b - a) > 1e-12
        # Chebyshev (minimax) affine approx of a concave increasing function.
        alpha = np.where(wide, (sb - sa) / np.where(wide, b - a, 1.0),
                         0.5 / np.sqrt(np.maximum(a, _EPS)))
        # tangent point u where f'(u)=alpha ⇒ u = 1/(4 alpha^2)
        u = 1.0 / (4.0 * np.maximum(alpha * alpha, _EPS))
        u = np.clip(u, a, b)
        zeta_chord = sa - alpha * a
        r = np.maximum(np.sqrt(u) - (alpha * u + zeta_chord), 0.0)
        zeta = zeta_chord + 0.5 * r
        delta = 0.5 * r
        return AAForm(alpha * self.x0 + zeta, alpha * self.x1,
                      np.abs(alpha) * self.e + delta)

    # ── non-smooth ops → sound interval hull (drops τ-correlation) ────────────
    def _from_range(self, lo, hi):
        return AAForm(0.5 * (lo + hi), np.zeros_like(self.x0), 0.5 * (hi - lo))

    def abs(self):
        lo, hi = self.range()
        new_lo = np.where(lo >= 0.0, lo, np.where(hi <= 0.0, -hi, 0.0))
        new_hi = np.maximum(np.abs(lo), np.abs(hi))
        return self._from_range(new_lo, new_hi)

    def max0(self):
        lo, hi = self.range()
        return self._from_range(np.maximum(lo, 0.0), np.maximum(hi, 0.0))

    def min0(self):
        lo, hi = self.range()
        return self._from_range(np.minimum(lo, 0.0), np.minimum(hi, 0.0))

    def maximum(self, o: "AAForm"):
        alo, ahi = self.range(); blo, bhi = o.range()
        return self._from_range(np.maximum(alo, blo), np.maximum(ahi, bhi))

    def minimum(self, o: "AAForm"):
        alo, ahi = self.range(); blo, bhi = o.range()
        return self._from_range(np.minimum(alo, blo), np.minimum(ahi, bhi))


def _aff_positions(ro, rd, t0, t1):
    """Affine positions for segments τ∈[t0,t1]: τ = c + r·ε (one symbol)."""
    c = 0.5 * (t0 + t1)
    r = 0.5 * (t1 - t0)
    return tuple(AAForm(ro[k] + rd[:, k] * c, rd[:, k] * r, np.zeros_like(c))
                 for k in range(3))


def _affine_range(scene_name, ro, rd, t0, t1):
    X, Y, Z = _aff_positions(ro, rd, t0, t1)
    return COMPONENT_SCENES[scene_name](X, Y, Z).range()


def _interval_range(scene_name, ro, rd, t0, t1):
    def comp(cdir, o):
        a, b = cdir * t0, cdir * t1
        return Interval(np.minimum(a, b), np.maximum(a, b)) + o
    pos = IVec3(comp(rd[:, 0], ro[0]), comp(rd[:, 1], ro[1]), comp(rd[:, 2], ro[2]))
    d = INTERVAL_SCENES[scene_name](pos)
    return d.lo, d.hi


def march_count(ro, rd, range_fn: Callable, t_max=DEFAULT_T_MAX, tol=DEFAULT_TOL):
    """Forward march identical to interval_oracle.first_hit but with a pluggable
    range function; returns (t_hit (M,), total_evals). Same soundness logic
    (lo>0 ⇒ proven empty). Eval count is the tightness metric: a tighter
    enclosure proves more empty space per probe ⇒ fewer evals."""
    M = rd.shape[0]
    t_hit = np.full(M, np.inf)
    if M == 0:
        return t_hit, 0
    t = np.zeros(M)
    h = np.full(M, _H0)
    active = np.ones(M, dtype=bool)
    total = 0
    for _ in range(_MAX_ITERS):
        idx = np.nonzero(active)[0]
        if idx.size == 0:
            break
        total += idx.size
        t0 = t[idx]; hh = h[idx]; t1 = t0 + hh
        lo, hi = range_fn(ro, rd[idx], t0, t1)
        empty = lo > 0.0
        hit = (~empty) & (hh <= tol)
        shrink = (~empty) & (~hit)
        hit_i = idx[hit]
        t_hit[hit_i] = t0[hit]
        active[hit_i] = False
        emp_i = idx[empty]
        t[emp_i] = t1[empty]
        h[emp_i] = np.minimum(hh[empty] * _GROWTH, _HMAX)
        active[emp_i[t[emp_i] > t_max]] = False
        h[idx[shrink]] = hh[shrink] * 0.5
    return t_hit, total


def _capture(scene_name, rc, range_fn):
    ro, rd = generate_rays(rc)
    H, W, _ = rd.shape
    rd_flat = rd.reshape(-1, 3)
    cand = _prune_candidates(ro, rd_flat, SCENE_BOUND.get(scene_name))
    t_hit, evals = march_count(ro, rd_flat[cand],
                               lambda o, d, a, b: range_fn(scene_name, o, d, a, b))
    depth = np.zeros(H * W); hit = np.zeros(H * W, dtype=bool)
    ci = np.nonzero(cand)[0]; got = np.isfinite(t_hit)
    depth[ci[got]] = t_hit[got]; hit[ci[got]] = True
    return {"depth": depth.reshape(H, W), "hit": hit.reshape(H, W)}, evals


def evaluate(scene_names: List[str], *, width=384, height=384) -> Dict:
    report: Dict = {"resolution": [width, height], "scenes": {}}
    from ..scenes.catalog import get_all_scenes
    all_scenes = get_all_scenes()
    for name in scene_names:
        scene = next((s for s in all_scenes if s.name == name), None)
        if scene is None or not has_interval(name):
            print(f"  [skip] {name} — no interval extension")
            continue
        rc = reference_render_config(scene, width, height)
        gold = interval_capture(name, rc)
        aa_cap, aa_evals = _capture(name, rc, _affine_range)
        ia_cap, ia_evals = _capture(name, rc, _interval_range)
        band = silhouette_band(gold["hit"], k=2)
        aa_res = residual(aa_cap["hit"], aa_cap["depth"], gold["hit"], gold["depth"], band)
        ia_res = residual(ia_cap["hit"], ia_cap["depth"], gold["hit"], gold["depth"], band)
        speedup = ia_evals / max(aa_evals, 1)
        report["scenes"][name] = {
            "affine": {"iou": aa_res["iou"], "core_iou": aa_res["core_iou"], "evals": aa_evals},
            "interval": {"iou": ia_res["iou"], "core_iou": ia_res["core_iou"], "evals": ia_evals},
            "eval_speedup_aa_over_ia": speedup,
        }
        print(f"\n=== {name} ===")
        print(f"  affine (RevAA): IoU {aa_res['iou']:.4f} (core {aa_res['core_iou']:.4f})  "
              f"evals {aa_evals:,}")
        print(f"  interval (IA) : IoU {ia_res['iou']:.4f} (core {ia_res['core_iou']:.4f})  "
              f"evals {ia_evals:,}")
        print(f"  → affine is {speedup:.2f}x fewer SDF segment-evals (tightness win)")
    return report


def main(argv: Optional[List[str]] = None) -> int:
    force_utf8_stdout()
    p = argparse.ArgumentParser(description="Real revised affine arithmetic vs interval arithmetic.")
    p.add_argument("--scenes", type=str, default="Sphere,Grazing Plane,Cube,Thin Torus")
    p.add_argument("--res", type=int, default=384)
    p.add_argument("--out", type=str, default="references/affine_revaa.json")
    args = p.parse_args(argv)
    names = [s.strip() for s in args.scenes.split(",") if s.strip()]
    print(f"RevAA (affine) vs IA tightness on {len(names)} scene(s) at {args.res}x{args.res}")
    report = evaluate(names, width=args.res, height=args.res)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
