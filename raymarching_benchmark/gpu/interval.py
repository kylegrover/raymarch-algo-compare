"""Interval arithmetic over SDFs — the sound inclusion functions behind the
gold-standard oracle (gpu/interval_oracle.py).

This is the *trustworthy* leg of the oracle chain. Unlike sphere/dense marching
(which sample the SDF at points and can step past a surface) an interval
extension evaluates the SDF over a whole ray segment at once and returns a
**guaranteed enclosure** ``[lo, hi]`` of every value the SDF takes on that
segment. The contract that makes the oracle sound:

    for all p in the box,  lo <= f(p) <= hi.

Therefore ``lo > 0`` *proves* the segment is empty (no surface) and can be
skipped with zero risk of tunneling — the property the GLSL ``segment`` /
``rev_affine`` strawmen lack because they only sample endpoints.

Each ``i_*`` primitive mirrors ``scenes/primitives.py`` op-for-op so the natural
interval extension is, by construction, the same function — validated pointwise
and for inclusion in tests/test_interval.py. Everything is NumPy-vectorized:
``Interval.lo/.hi`` are arrays so a whole worklist of ray segments evaluates in
one pass.

Scope (per plan): the four metric grid scenes (Sphere, Grazing Plane, Cube, Thin
Torus). Mandelbulb/fractals are deferred (need interval escape-time, not IA on
the distance estimator).
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Dict, Union

Number = Union[float, np.ndarray]


class Interval:
    """A guaranteed enclosure ``[lo, hi]``. ``lo``/``hi`` are floats or NumPy
    arrays of identical shape (one entry per ray-segment in a batch)."""

    __slots__ = ("lo", "hi")

    def __init__(self, lo: Number, hi: Number):
        self.lo = np.asarray(lo, dtype=np.float64)
        self.hi = np.asarray(hi, dtype=np.float64)

    # ── constructors ────────────────────────────────────────────────────────
    @staticmethod
    def point(x: Number) -> "Interval":
        """Degenerate interval [x, x] — used for the pointwise-equivalence gate."""
        x = np.asarray(x, dtype=np.float64)
        return Interval(x, x)

    # ── arithmetic ──────────────────────────────────────────────────────────
    def __add__(self, o) -> "Interval":
        if isinstance(o, Interval):
            return Interval(self.lo + o.lo, self.hi + o.hi)
        return Interval(self.lo + o, self.hi + o)

    __radd__ = __add__

    def __sub__(self, o) -> "Interval":
        if isinstance(o, Interval):
            return Interval(self.lo - o.hi, self.hi - o.lo)
        return Interval(self.lo - o, self.hi - o)

    def __rsub__(self, o) -> "Interval":  # o - self,  o scalar/array
        return Interval(o - self.hi, o - self.lo)

    def __neg__(self) -> "Interval":
        return Interval(-self.hi, -self.lo)

    def __mul__(self, o) -> "Interval":
        if isinstance(o, Interval):
            p = (self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi)
            return Interval(np.minimum(np.minimum(p[0], p[1]), np.minimum(p[2], p[3])),
                            np.maximum(np.maximum(p[0], p[1]), np.maximum(p[2], p[3])))
        a, b = self.lo * o, self.hi * o          # o may be a signed array
        return Interval(np.minimum(a, b), np.maximum(a, b))

    __rmul__ = __mul__

    # ── unary ───────────────────────────────────────────────────────────────
    def abs(self) -> "Interval":
        lo, hi = self.lo, self.hi
        new_lo = np.where(lo >= 0.0, lo, np.where(hi <= 0.0, -hi, 0.0))
        new_hi = np.maximum(np.abs(lo), np.abs(hi))
        return Interval(new_lo, new_hi)

    def square(self) -> "Interval":
        lo, hi = self.lo, self.hi
        a, b = lo * lo, hi * hi
        new_lo = np.where(lo >= 0.0, a, np.where(hi <= 0.0, b, 0.0))
        new_hi = np.maximum(a, b)
        return Interval(new_lo, new_hi)

    def sqrt(self) -> "Interval":
        # domain is clamped at 0: callers only sqrt provably-nonneg quantities
        # (squared lengths), but rounding can dip lo slightly below 0.
        return Interval(np.sqrt(np.maximum(self.lo, 0.0)),
                        np.sqrt(np.maximum(self.hi, 0.0)))

    # ── lattice ops (monotone increasing in both args ⇒ exact extension) ─────
    def minimum(self, o: "Interval") -> "Interval":
        return Interval(np.minimum(self.lo, o.lo), np.minimum(self.hi, o.hi))

    def maximum(self, o: "Interval") -> "Interval":
        return Interval(np.maximum(self.lo, o.lo), np.maximum(self.hi, o.hi))

    def max0(self) -> "Interval":
        """max(self, 0) — the clamp used in the box SDF."""
        return Interval(np.maximum(self.lo, 0.0), np.maximum(self.hi, 0.0))

    def min0(self) -> "Interval":
        """min(self, 0)."""
        return Interval(np.minimum(self.lo, 0.0), np.minimum(self.hi, 0.0))

    def width(self) -> np.ndarray:
        return self.hi - self.lo

    def __repr__(self):
        return f"Interval({self.lo}, {self.hi})"


class IVec3:
    """Three component intervals — an axis-aligned box enclosure of a position."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x: Interval, y: Interval, z: Interval):
        self.x, self.y, self.z = x, y, z

    def __add__(self, o: "IVec3") -> "IVec3":
        return IVec3(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o: "IVec3") -> "IVec3":
        return IVec3(self.x - o.x, self.y - o.y, self.z - o.z)

    def abs(self) -> "IVec3":
        return IVec3(self.x.abs(), self.y.abs(), self.z.abs())

    def dot(self, n) -> Interval:
        """Dot with a constant 3-vector ``n`` (the only form the scenes need)."""
        return self.x * float(n[0]) + self.y * float(n[1]) + self.z * float(n[2])

    def length(self) -> Interval:
        return (self.x.square() + self.y.square() + self.z.square()).sqrt()

    @staticmethod
    def constant(v) -> "IVec3":
        return IVec3(Interval.point(float(v[0])),
                     Interval.point(float(v[1])),
                     Interval.point(float(v[2])))


# ──────────────────────────────────────────────────────────────────────────────
# Metric primitive SDFs — structurally identical to scenes/primitives.py so the
# extension *is* the natural extension. Written against component objects (any
# type exposing square/sqrt/abs/max0/min0/maximum/minimum and +,-,* with a
# scalar): this single source is shared by the interval oracle (Interval
# components) and the interval-autodiff segment tracer (dual-interval components).
# ──────────────────────────────────────────────────────────────────────────────

def _length3(x, y, z):
    return (x.square() + y.square() + z.square()).sqrt()


def _sd_sphere(x, y, z, radius):
    return _length3(x, y, z) - radius


def _sd_plane(x, y, z, normal, offset):
    return x * float(normal[0]) + y * float(normal[1]) + z * float(normal[2]) - offset


def _sd_box(x, y, z, half):
    qx = x.abs() - float(half[0])
    qy = y.abs() - float(half[1])
    qz = z.abs() - float(half[2])
    outside = _length3(qx.max0(), qy.max0(), qz.max0())
    inside = qx.maximum(qy).maximum(qz).min0()
    return outside + inside


def _sd_torus(x, y, z, major_radius, minor_radius):
    q_xz = (x.square() + z.square()).sqrt() - major_radius
    return (q_xz.square() + y.square()).sqrt() - minor_radius


# Interval-typed wrappers (the oracle calls these via IVec3).

def i_sphere(p: IVec3, radius: float) -> Interval:
    return _sd_sphere(p.x, p.y, p.z, radius)


def i_plane(p: IVec3, normal, offset: float) -> Interval:
    return _sd_plane(p.x, p.y, p.z, normal, offset)


def i_box(p: IVec3, half) -> Interval:
    return _sd_box(p.x, p.y, p.z, half)


def i_torus(p: IVec3, major_radius: float, minor_radius: float) -> Interval:
    return _sd_torus(p.x, p.y, p.z, major_radius, minor_radius)


# ── registry (mirrors gpu/analytic.ANALYTIC_SCENES) ──────────────────────────

INTERVAL_SCENES: Dict[str, Callable[[IVec3], Interval]] = {
    "Sphere": lambda p: i_sphere(p, 1.0),
    "Grazing Plane": lambda p: i_plane(p, (0.0, 1.0, 0.0), -0.5),
    "Cube": lambda p: i_box(p, (1.0, 1.0, 1.0)),
    "Thin Torus": lambda p: i_torus(p, 1.5, 0.05),
}

#: Type-agnostic component SDFs (x, y, z component objects -> value). Shared by
#: the interval-autodiff segment tracer so it traces the *identical* surface.
COMPONENT_SCENES: Dict[str, Callable] = {
    "Sphere": lambda x, y, z: _sd_sphere(x, y, z, 1.0),
    "Grazing Plane": lambda x, y, z: _sd_plane(x, y, z, (0.0, 1.0, 0.0), -0.5),
    "Cube": lambda x, y, z: _sd_box(x, y, z, (1.0, 1.0, 1.0)),
    "Thin Torus": lambda x, y, z: _sd_torus(x, y, z, 1.5, 0.05),
}


def has_interval(scene_name: str) -> bool:
    return scene_name in INTERVAL_SCENES
