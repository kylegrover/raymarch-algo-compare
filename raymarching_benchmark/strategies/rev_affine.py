"""Lightweight RevAA approximation using Interval Arithmetic (IA).

Full Revised Affine Arithmetic is heavy; this approximation tracks an SDF interval
over a ray segment by sampling endpoints and bisection when the interval
indicates a possible zero-crossing. The implementation mirrors the GLSL
`rev_affine` shader and is intentionally conservative.
"""
from .base import MarchStrategy
from ..core.ray import Ray
from ..core.types import MarchResult
from ..config import MarchConfig
from typing import Callable, NamedTuple


class Interval(NamedTuple):
    lo: float
    hi: float


def interval_from_samples(a: float, b: float) -> Interval:
    return Interval(min(a, b), max(a, b))


def interval_spans_zero(iv: Interval) -> bool:
    return iv.lo <= 0.0 and iv.hi >= 0.0


class RevAAApproxTracing(MarchStrategy):
    @property
    def name(self) -> str:
        return "RevAA (Interval Approx)"

    @property
    def short_name(self) -> str:
        return "RevAA"

    def march(self, ray: Ray, sdf_func: Callable, config: MarchConfig) -> MarchResult:
        t = 0.0
        iterations = 0
        final_sdf = 0.0

        for i in range(config.max_iterations):
            iterations = i + 1
            pos = ray.at(t)
            d = sdf_func(pos)
            final_sdf = d
            if abs(d) < config.hit_threshold:
                return MarchResult(
                    hit=True, t=t, iterations=iterations, final_sdf=d,
                    position_x=pos.x, position_y=pos.y, position_z=pos.z
                )

            # conservative step
            if d < 0.0:
                t = max(0.0, t + d)
                continue

            next_t = t + max(d, config.hit_threshold)
            # sample endpoints to form an interval over the segment
            d_lo = d
            d_hi = sdf_func(ray.at(next_t))
            dr = interval_from_samples(d_lo, d_hi)

            if interval_spans_zero(dr):
                # possible root in [t, next_t] -> bisection
                a = t
                b = next_t
                for j in range(8):
                    mid = 0.5 * (a + b)
                    dm = sdf_func(ray.at(mid))
                    iterations += 1
                    if abs(dm) < config.hit_threshold:
                        return MarchResult(
                            hit=True, t=mid, iterations=iterations, final_sdf=dm,
                            position_x=ray.at(mid).x, position_y=ray.at(mid).y, position_z=ray.at(mid).z
                        )
                    if dm > 0.0:
                        a = mid
                    else:
                        b = mid
                tm = 0.5 * (a + b)
                dm = sdf_func(ray.at(tm))
                return MarchResult(
                    hit=True, t=tm, iterations=iterations, final_sdf=dm,
                    position_x=ray.at(tm).x, position_y=ray.at(tm).y, position_z=ray.at(tm).z
                )

            t = next_t
            if t > config.max_distance:
                break

        pos = ray.at(t)
        d = sdf_func(pos)
        return MarchResult(
            hit=False, t=t, iterations=iterations, final_sdf=d,
            position_x=pos.x, position_y=pos.y, position_z=pos.z
        )
