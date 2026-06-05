"""Interval automatic differentiation — sound bounds on the *directional*
derivative of the SDF along a ray, for the faithful Galin segment tracer.

A ``DInterval`` carries two enclosures: ``val`` (the SDF value over a ray segment,
exactly as ``interval.py``) and ``der`` (an enclosure of the derivative w.r.t. the
march parameter τ along the ray, where the point is ``ro + rd·τ``). Pushing the
seed ``pos_c = ro_c + rd_c·τ`` (so ``d pos_c/dτ = rd_c``) through the shared
``COMPONENT_SCENES`` SDFs yields a guaranteed enclosure of ``g'(τ)`` over the whole
segment. Its magnitude bound

    K = max(|der.lo|, |der.hi|)

is a **sound local directional-Lipschitz constant**: ``|g(τ+s) − g(τ)| ≤ K·s`` for
``s`` in the segment, so a step of ``|f|/K`` provably cannot reach the surface.

This is the mechanism the GLSL ``segment`` strawman faked with 3-point sampling.
Crucially K is the *directional* derivative: at grazing angles ``rd`` is nearly
parallel to the surface so ``K ≪ 1`` and the safe step is far larger than a sphere
trace — the genuine Galin et al. 2020 advantage, here computed soundly.
"""
from __future__ import annotations

import numpy as np

from .interval import Interval

_BIG = 1.0e12
_EPS = 1e-15


def _recip_pos(s: Interval) -> Interval:
    """Reciprocal of a nonnegative interval (= sqrt(...) result), zero-guarded.
    1/[a,b] = [1/b, 1/a] for 0 < a <= b; near 0 the upper end is capped at _BIG."""
    a = np.maximum(s.lo, 0.0)
    b = np.maximum(s.hi, 0.0)
    rlo = np.where(b > _EPS, 1.0 / np.maximum(b, _EPS), _BIG)
    rhi = np.where(a > _EPS, 1.0 / np.maximum(a, _EPS), _BIG)
    return Interval(rlo, rhi)


def _hull(a: Interval, b: Interval) -> Interval:
    return Interval(np.minimum(a.lo, b.lo), np.maximum(a.hi, b.hi))


class DInterval:
    __slots__ = ("val", "der")

    def __init__(self, val: Interval, der: Interval):
        self.val = val
        self.der = der

    # ── linear ops ───────────────────────────────────────────────────────────
    def __add__(self, o):
        if isinstance(o, DInterval):
            return DInterval(self.val + o.val, self.der + o.der)
        return DInterval(self.val + o, self.der)             # scalar/array const

    __radd__ = __add__

    def __sub__(self, o):
        if isinstance(o, DInterval):
            return DInterval(self.val - o.val, self.der - o.der)
        return DInterval(self.val - o, self.der)

    def __rsub__(self, o):                                   # o - self, const o
        return DInterval(o - self.val, -self.der)

    def __neg__(self):
        return DInterval(-self.val, -self.der)

    def __mul__(self, o):
        if isinstance(o, DInterval):                          # product rule
            return DInterval(self.val * o.val,
                             self.der * o.val + self.val * o.der)
        return DInterval(self.val * o, self.der * o)          # const scale

    __rmul__ = __mul__

    # ── nonlinear ops (chain rule) ────────────────────────────────────────────
    def square(self):
        return DInterval(self.val.square(), 2.0 * (self.val * self.der))

    def sqrt(self):
        s = self.val.sqrt()
        # d/dτ sqrt(v) = v' / (2 sqrt(v))
        return DInterval(s, self.der * _recip_pos(s) * 0.5)

    def abs(self):
        lo, hi = self.val.lo, self.val.hi
        m = np.maximum(np.abs(self.der.lo), np.abs(self.der.hi))
        # where val>0: der ; where val<0: -der ; spanning 0: [-m, m]
        der_lo = np.where(lo >= 0.0, self.der.lo, np.where(hi <= 0.0, -self.der.hi, -m))
        der_hi = np.where(lo >= 0.0, self.der.hi, np.where(hi <= 0.0, -self.der.lo, m))
        return DInterval(self.val.abs(), Interval(der_lo, der_hi))

    def _clamp_der(self, active: np.ndarray, zero: np.ndarray) -> Interval:
        """Derivative when value is clamped: take self.der where `active`, 0 where
        `zero`, else the hull with 0 (ambiguous boundary)."""
        amb = ~active & ~zero
        der_lo = np.where(active, self.der.lo, np.where(zero, 0.0, np.minimum(self.der.lo, 0.0)))
        der_hi = np.where(active, self.der.hi, np.where(zero, 0.0, np.maximum(self.der.hi, 0.0)))
        return Interval(der_lo, der_hi)

    def max0(self):
        lo, hi = self.val.lo, self.val.hi
        return DInterval(self.val.max0(), self._clamp_der(lo > 0.0, hi < 0.0))

    def min0(self):
        lo, hi = self.val.lo, self.val.hi
        return DInterval(self.val.min0(), self._clamp_der(hi < 0.0, lo > 0.0))

    def maximum(self, o: "DInterval"):
        self_wins = self.val.lo > o.val.hi
        o_wins = o.val.lo > self.val.hi
        der = _hull(self.der, o.der)
        der_lo = np.where(self_wins, self.der.lo, np.where(o_wins, o.der.lo, der.lo))
        der_hi = np.where(self_wins, self.der.hi, np.where(o_wins, o.der.hi, der.hi))
        return DInterval(self.val.maximum(o.val), Interval(der_lo, der_hi))

    def minimum(self, o: "DInterval"):
        self_wins = self.val.hi < o.val.lo
        o_wins = o.val.hi < self.val.lo
        der = _hull(self.der, o.der)
        der_lo = np.where(self_wins, self.der.lo, np.where(o_wins, o.der.lo, der.lo))
        der_hi = np.where(self_wins, self.der.hi, np.where(o_wins, o.der.hi, der.hi))
        return DInterval(self.val.minimum(o.val), Interval(der_lo, der_hi))


def seed_segment(ro: np.ndarray, rd: np.ndarray, t0: np.ndarray, t1: np.ndarray):
    """Build the three position DIntervals for ray segments τ∈[t0, t1].
    ``rd`` is (M,3); ``t0,t1`` are (M,). Returns (X, Y, Z) DIntervals."""
    out = []
    for c in range(3):
        a = rd[:, c] * t0
        b = rd[:, c] * t1
        val = Interval(np.minimum(a, b), np.maximum(a, b)) + ro[c]
        der = Interval.point(rd[:, c])           # d(ro_c + rd_c·τ)/dτ = rd_c
        out.append(DInterval(val, der))
    return out[0], out[1], out[2]
