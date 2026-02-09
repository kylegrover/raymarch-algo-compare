"""Enhanced Sphere Tracing with planar surface assumption."""

from .base import MarchStrategy
from ..core.ray import Ray
from ..core.types import MarchResult
from ..config import MarchConfig
from typing import Callable


class EnhancedSphereTracing(MarchStrategy):
    """
    Enhanced Sphere Tracing (Keinert et al., 2014 / Hart improvements).

    Uses the history of previous SDF evaluations to estimate the surface
    as locally planar and predicts the intersection point geometrically.

    Key idea: if we have two consecutive SDF samples d_prev at t_prev and d_curr at t_curr,
    and assume the surface is a plane, the intersection is approximately at:
        t_intersection ≈ t_curr + d_curr * (t_curr - t_prev) / (d_prev - d_curr)

    Falls back to standard sphere tracing when the prediction is unreliable.
    """

    @property
    def name(self) -> str:
        return "Enhanced Sphere Tracing"

    @property
    def short_name(self) -> str:
        return "Enhanced"

    def march(self, ray: Ray, sdf_func: Callable, config: MarchConfig) -> MarchResult:
        t = 0.0
        iterations = 0
        prev_t = 0.0
        prev_d = float('inf')
        use_prediction = False

        for i in range(config.max_iterations):
            iterations = i + 1
            pos = ray.at(t)
            d = sdf_func(pos)

            if abs(d) < config.hit_threshold:
                return MarchResult(
                    hit=True, t=t, iterations=iterations, final_sdf=d,
                    position_x=pos.x, position_y=pos.y, position_z=pos.z
                )

            # Try planar prediction if we have history
            step = d  # Default: standard sphere trace step

            if i > 0 and prev_d > d and d > 0 and (prev_d - d) > 1e-10:
                dt = t - prev_t
                # Linear extrapolation: when will SDF reach 0?
                predicted_step = d * dt / (prev_d - d)

                # Sanity check: prediction should be positive and not too large
                if 0 < predicted_step < d * 3.0:
                    step = predicted_step
                    use_prediction = True
                else:
                    use_prediction = False
            else:
                use_prediction = False

            # Safety: if we overshot (d < 0), back up
            if d < 0:
                # Bisect between prev_t and t
                t = (prev_t + t) * 0.5
                prev_d = abs(d)
                continue

            prev_t = t
            prev_d = d
            t += step

            if t > config.max_distance:
                break

        pos = ray.at(t)
        d = sdf_func(pos)
        return MarchResult(
            hit=False, t=t, iterations=iterations, final_sdf=d,
            position_x=pos.x, position_y=pos.y, position_z=pos.z
        )
