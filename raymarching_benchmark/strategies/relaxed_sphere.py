"""Over-relaxed sphere tracing with fixed omega."""

from .base import MarchStrategy
from ..core.ray import Ray
from ..core.types import MarchResult
from ..config import MarchConfig
from typing import Callable


class RelaxedSphereTracing(MarchStrategy):
    """
    Over-relaxed sphere tracing (Keinert et al., 2014).
    Steps by omega * sdf(pos), where omega > 1.
    If overshoot detected (step > |sdf| at new position), fallback to safe step.
    """

    def __init__(self, omega: float = 1.6):
        self._omega = omega

    @property
    def name(self) -> str:
        return f"Relaxed Sphere Tracing (ω={self._omega})"

    @property
    def short_name(self) -> str:
        return f"Relaxed(ω={self._omega})"

    def march(self, ray: Ray, sdf_func: Callable, config: MarchConfig) -> MarchResult:
        t = 0.0
        iterations = 0
        prev_d = 0.0
        omega = self._omega

        for i in range(config.max_iterations):
            iterations = i + 1
            pos = ray.at(t)
            d = sdf_func(pos)

            if abs(d) < config.hit_threshold:
                return MarchResult(
                    hit=True, t=t, iterations=iterations, final_sdf=d,
                    position_x=pos.x, position_y=pos.y, position_z=pos.z
                )

            # Check if the previous relaxed step overshot
            # If |prev_step| + |current_d| < |prev_step * omega|, overshoot occurred
            step = d * omega

            # Safety check: if we took an over-relaxed step and the new SDF
            # is small enough that we might have overshot, fall back
            if i > 0 and (prev_d + d) < prev_d * omega:
                # Overshoot detected: fall back to conservative step
                step = d  # Use unrelaxed step
                # Could also reduce omega here

            t += step
            prev_d = d

            if t > config.max_distance:
                break

        pos = ray.at(t)
        d = sdf_func(pos)
        return MarchResult(
            hit=False, t=t, iterations=iterations, final_sdf=d,
            position_x=pos.x, position_y=pos.y, position_z=pos.z
        )
