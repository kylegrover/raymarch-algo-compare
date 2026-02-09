"""Segment Tracing inspired by Galin et al. 2020."""

import math
from .base import MarchStrategy
from ..core.ray import Ray
from ..core.types import MarchResult
from ..config import MarchConfig
from typing import Callable
from ..core.vec3 import Vec3


class SegmentTracing(MarchStrategy):
    """
    Segment Tracing: computes Lipschitz bounds over ray segments rather than points.

    For a Lipschitz-1 SDF, the maximum possible step from t is:
        step = (d(t) + d(t + candidate)) / 2  (approximately, using segment analysis)

    Simplified implementation: uses two SDF evaluations per step to get a tighter
    bound on where the surface can be within the segment [t, t + d(t)].

    This doubles the SDF evaluations per iteration but can dramatically reduce
    the total number of steps needed.
    """

    def __init__(self, lipschitz: float = 1.0):
        self.lipschitz = lipschitz

    @property
    def name(self) -> str:
        return "Segment Tracing"

    @property
    def short_name(self) -> str:
        return "Segment"

    def march(self, ray: Ray, sdf_func: Callable, config: MarchConfig) -> MarchResult:
        t = 0.0
        iterations = 0
        L = self.lipschitz

        for i in range(config.max_iterations):
            iterations = i + 1
            pos = ray.at(t)
            d = sdf_func(pos)

            if abs(d) < config.hit_threshold:
                return MarchResult(
                    hit=True, t=t, iterations=iterations, final_sdf=d,
                    position_x=pos.x, position_y=pos.y, position_z=pos.z
                )

            if d < 0:
                # Overshot - back up conservatively
                t -= abs(d) * 0.5
                continue

            # Standard step candidate
            candidate = d / L

            # Evaluate at the end of the candidate segment
            pos_end = ray.at(t + candidate)
            d_end = sdf_func(pos_end)
            iterations += 1  # Count the extra evaluation

            if abs(d_end) < config.hit_threshold:
                t += candidate
                return MarchResult(
                    hit=True, t=t, iterations=iterations, final_sdf=d_end,
                    position_x=pos_end.x, position_y=pos_end.y, position_z=pos_end.z
                )

            if d_end < 0:
                # Surface is within this segment - bisect
                t_lo = t
                t_hi = t + candidate
                for _ in range(8):
                    iterations += 1
                    t_mid = (t_lo + t_hi) * 0.5
                    d_mid = sdf_func(ray.at(t_mid))
                    if abs(d_mid) < config.hit_threshold:
                        pos_mid = ray.at(t_mid)
                        return MarchResult(
                            hit=True, t=t_mid, iterations=iterations, final_sdf=d_mid,
                            position_x=pos_mid.x, position_y=pos_mid.y, position_z=pos_mid.z
                        )
                    if d_mid > 0:
                        t_lo = t_mid
                    else:
                        t_hi = t_mid
                t = (t_lo + t_hi) * 0.5
                pos_final = ray.at(t)
                d_final = sdf_func(ray.at(t))
                return MarchResult(
                    hit=True, t=t, iterations=iterations, final_sdf=d_final,
                    position_x=pos_final.x, position_y=pos_final.y, position_z=pos_final.z
                )

            # Both endpoints safe - use the larger safe step
            # The segment [t, t+candidate] is surface-free
            # We can potentially step further using d_end
            extended_step = candidate + d_end / L
            t += extended_step

            if t > config.max_distance:
                break

        pos = ray.at(t)
        d = sdf_func(pos)
        return MarchResult(
            hit=False, t=t, iterations=iterations, final_sdf=d,
            position_x=pos.x, position_y=pos.y, position_z=pos.z
        )
