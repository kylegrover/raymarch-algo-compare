"""Auto-Relaxed Sphere Tracing (AR-ST) with adaptive omega."""

from .base import MarchStrategy
from ..core.ray import Ray
from ..core.types import MarchResult
from ..config import MarchConfig
from typing import Callable


class AutoRelaxedSphereTracing(MarchStrategy):
    """
    Auto-Relaxed Sphere Tracing.
    Uses exponential smoothing to estimate the local SDF slope along the ray
    and dynamically adjusts the relaxation parameter omega.

    Based on the idea that if SDF values are decreasing rapidly (approaching surface),
    omega should decrease toward 1.0 (conservative), and if SDF values are
    steady/increasing (open space), omega can increase (aggressive).
    """

    def __init__(self, omega_min: float = 1.0, omega_max: float = 2.0,
                 smoothing: float = 0.7, growth_rate: float = 1.05,
                 decay_rate: float = 0.7):
        self._omega_min = omega_min
        self._omega_max = omega_max
        self._smoothing = smoothing
        self._growth_rate = growth_rate
        self._decay_rate = decay_rate

    @property
    def name(self) -> str:
        return "Auto-Relaxed Sphere Tracing"

    @property
    def short_name(self) -> str:
        return "AR-ST"

    def march(self, ray: Ray, sdf_func: Callable, config: MarchConfig) -> MarchResult:
        t = 0.0
        iterations = 0
        omega = 1.2  # Start mildly aggressive
        prev_d = float('inf')
        ema_ratio = 1.0  # Exponential moving average of d[i]/d[i-1]

        for i in range(config.max_iterations):
            iterations = i + 1
            pos = ray.at(t)
            d = sdf_func(pos)

            if abs(d) < config.hit_threshold:
                return MarchResult(
                    hit=True, t=t, iterations=iterations, final_sdf=d,
                    position_x=pos.x, position_y=pos.y, position_z=pos.z
                )

            # Compute ratio of consecutive SDF values
            if prev_d > 1e-10 and i > 0:
                ratio = d / prev_d
                ema_ratio = self._smoothing * ema_ratio + (1.0 - self._smoothing) * ratio

                # If ratio < 1, we're approaching surface -> decrease omega
                # If ratio >= 1, we're in open space -> increase omega
                if ema_ratio < 0.8:
                    omega = max(self._omega_min, omega * self._decay_rate)
                elif ema_ratio > 1.0:
                    omega = min(self._omega_max, omega * self._growth_rate)
                # else: maintain current omega

            step = d * omega

            # Safety: detect overshoot (previous relaxed step was too large)
            if d < 0:
                # We've overshot - back up and go conservative
                t += d  # d is negative, so this backs up
                omega = self._omega_min
                prev_d = abs(d)
                continue

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
