"""Standard sphere tracing (naive)."""

from .base import MarchStrategy
from ..core.ray import Ray
from ..core.types import MarchResult
from ..config import MarchConfig
from typing import Callable


class StandardSphereTracing(MarchStrategy):
    """
    Classic sphere tracing: step by SDF distance each iteration.
    t += sdf(pos) until |sdf| < threshold or max_iterations reached.
    """

    @property
    def name(self) -> str:
        return "Standard Sphere Tracing"

    @property
    def short_name(self) -> str:
        return "Standard"

    def march(self, ray: Ray, sdf_func: Callable, config: MarchConfig) -> MarchResult:
        t = 0.0
        iterations = 0

        for i in range(config.max_iterations):
            iterations = i + 1
            pos = ray.at(t)
            d = sdf_func(pos)

            if abs(d) < config.hit_threshold:
                return MarchResult(
                    hit=True, t=t, iterations=iterations, final_sdf=d,
                    position_x=pos.x, position_y=pos.y, position_z=pos.z
                )

            t += d

            if t > config.max_distance:
                break

        pos = ray.at(t)
        d = sdf_func(pos)
        return MarchResult(
            hit=False, t=t, iterations=iterations, final_sdf=d,
            position_x=pos.x, position_y=pos.y, position_z=pos.z
        )
