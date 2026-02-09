"""Overstep-Bisect ray marching strategy."""

from .base import MarchStrategy
from ..core.ray import Ray
from ..core.types import MarchResult
from ..config import MarchConfig
from typing import Callable


class OverstepBisectTracing(MarchStrategy):
    """
    Two-phase strategy:
    Phase 1 - Sphere trace with enforced minimum step size to avoid getting stuck.
              Track t_near (last SDF > 0) and t_far (first SDF < 0).
    Phase 2 - Once overshoot detected (sign change), bisect between t_near and t_far.
    """

    def __init__(self, min_step_factor: float = 0.01, bisection_steps: int = 16):
        self._min_step_factor = min_step_factor
        self._bisection_steps = bisection_steps

    @property
    def name(self) -> str:
        return "Overstep-Bisect"

    @property
    def short_name(self) -> str:
        return "Overstep-Bisect"

    def march(self, ray: Ray, sdf_func: Callable, config: MarchConfig) -> MarchResult:
        t = 0.0
        iterations = 0
        t_near = 0.0  # Last t where SDF > 0
        t_far = -1.0  # First t where SDF < 0 (invalid until set)
        bisection_steps_used = 0

        min_step = self._min_step_factor  # Minimum step size

        # Phase 1: Forward march with minimum step enforcement
        phase1_budget = config.max_iterations - self._bisection_steps
        for i in range(phase1_budget):
            iterations = i + 1
            pos = ray.at(t)
            d = sdf_func(pos)

            if abs(d) < config.hit_threshold:
                return MarchResult(
                    hit=True, t=t, iterations=iterations, final_sdf=d,
                    position_x=pos.x, position_y=pos.y, position_z=pos.z,
                    bisection_steps=0
                )

            if d > 0:
                t_near = t
                # Take the larger of SDF distance and minimum step
                step = max(d, min_step)
                t += step
            else:
                # Overshot! Record far bound and enter bisection
                t_far = t
                break

            if t > config.max_distance:
                pos = ray.at(t)
                d = sdf_func(pos)
                return MarchResult(
                    hit=False, t=t, iterations=iterations, final_sdf=d,
                    position_x=pos.x, position_y=pos.y, position_z=pos.z,
                    bisection_steps=0
                )

        # Phase 2: Bisection between t_near and t_far
        if t_far > 0:
            for j in range(self._bisection_steps):
                iterations += 1
                bisection_steps_used = j + 1
                t_mid = (t_near + t_far) * 0.5
                pos = ray.at(t_mid)
                d = sdf_func(pos)

                if abs(d) < config.hit_threshold:
                    return MarchResult(
                        hit=True, t=t_mid, iterations=iterations, final_sdf=d,
                        position_x=pos.x, position_y=pos.y, position_z=pos.z,
                        bisection_steps=bisection_steps_used
                    )

                if d > 0:
                    t_near = t_mid
                else:
                    t_far = t_mid

                if (t_far - t_near) < config.hit_threshold:
                    t_mid = (t_near + t_far) * 0.5
                    pos = ray.at(t_mid)
                    d = sdf_func(pos)
                    return MarchResult(
                        hit=True, t=t_mid, iterations=iterations, final_sdf=d,
                        position_x=pos.x, position_y=pos.y, position_z=pos.z,
                        bisection_steps=bisection_steps_used
                    )

            # Bisection converged to interval but not threshold
            t_final = (t_near + t_far) * 0.5
            pos = ray.at(t_final)
            d = sdf_func(pos)
            return MarchResult(
                hit=abs(d) < config.hit_threshold * 10,  # Slightly relaxed
                t=t_final, iterations=iterations, final_sdf=d,
                position_x=pos.x, position_y=pos.y, position_z=pos.z,
                bisection_steps=bisection_steps_used
            )

        # No overshoot detected - miss
        pos = ray.at(t)
        d = sdf_func(pos)
        return MarchResult(
            hit=False, t=t, iterations=iterations, final_sdf=d,
            position_x=pos.x, position_y=pos.y, position_z=pos.z,
            bisection_steps=0
        )
