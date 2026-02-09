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

    def __init__(self, omega: float = 1.2):
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

            # Check for overshoot (inside the surface)
            if d < 0:
                # Back up by the amount we're inside, and switch to conservative
                t += d
                omega = 1.0  # Go safe for one step
                prev_d = abs(d)
                continue

            # Fallback for "grazing" overshoot (still outside but took too large a step)
            step = d * omega
            if i > 0 and (prev_d + d) < prev_d * omega:
                step = d
                omega = 1.0

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
