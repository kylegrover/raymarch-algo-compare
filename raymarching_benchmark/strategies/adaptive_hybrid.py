"""Adaptive hybrid: starts with sphere tracing, detects stuck, switches to overstep-bisect."""

from .base import MarchStrategy
from ..core.ray import Ray
from ..core.types import MarchResult
from ..config import MarchConfig
from typing import Callable


class AdaptiveHybridTracing(MarchStrategy):
    """
    Starts with standard sphere tracing for efficiency.
    If it detects it's "stuck" (many consecutive tiny steps), switches to
    the overstep-bisect strategy to force progress.
    """

    def __init__(self, stuck_threshold: int = 5, stuck_step_ratio: float = 0.001,
                 min_step_factor: float = 0.005, bisection_steps: int = 12,
                 fallback_to_segment_after: int | None = None):
        self._stuck_threshold = stuck_threshold
        self._stuck_step_ratio = stuck_step_ratio
        self._min_step_factor = min_step_factor
        self._bisection_steps = bisection_steps
        # Experimental: fallback to Segment tracing after N iterations (per-ray)
        self.fallback_to_segment_after = fallback_to_segment_after

    @property
    def name(self) -> str:
        return "Adaptive Hybrid"

    @property
    def short_name(self) -> str:
        return "Hybrid"

    def march(self, ray: Ray, sdf_func: Callable, config: MarchConfig) -> MarchResult:
        t = 0.0
        iterations = 0
        consecutive_small = 0
        mode = "sphere"  # "sphere" or "overstep"
        strategy_switches = 0
        stuck_count = 0

        # Overstep mode state
        t_near = 0.0
        t_far = -1.0
        min_step = self._min_step_factor

        for i in range(config.max_iterations):
            iterations = i + 1
            pos = ray.at(t)
            d = sdf_func(pos)

            if abs(d) < config.hit_threshold:
                return MarchResult(
                    hit=True, t=t, iterations=iterations, final_sdf=d,
                    position_x=pos.x, position_y=pos.y, position_z=pos.z,
                    stuck_count=stuck_count,
                    strategy_switches=strategy_switches
                )

            if mode == "sphere":
                # Standard sphere tracing
                step = d

                # Detect stuck condition
                if d > 0 and d < self._stuck_step_ratio * max(t, 1.0):
                    consecutive_small += 1
                else:
                    consecutive_small = 0

                if consecutive_small >= self._stuck_threshold:
                    mode = "overstep"
                    strategy_switches += 1
                    stuck_count += 1
                    t_near = t
                    t_far = -1.0
                    consecutive_small = 0
                    continue  # Re-evaluate in new mode

                if d < 0:
                    # Unexpected overshoot in sphere mode - switch to bisection
                    t_far = t
                    t_near = max(0.0, t + d)  # Back up by |d|
                    mode = "bisect"
                    strategy_switches += 1
                    continue

                t += step

            elif mode == "overstep":
                # Overstep with minimum step enforcement
                if d > 0:
                    t_near = t
                    step = max(d, min_step)
                    t += step
                else:
                    # Overshot - enter bisection
                    t_far = t
                    mode = "bisect"
                    continue

            elif mode == "bisect":
                # Binary search between t_near and t_far
                if t_far < 0:
                    mode = "sphere"
                    continue

                t_mid = (t_near + t_far) * 0.5
                pos = ray.at(t_mid)
                d = sdf_func(pos)

                if abs(d) < config.hit_threshold:
                    return MarchResult(
                        hit=True, t=t_mid, iterations=iterations, final_sdf=d,
                        position_x=pos.x, position_y=pos.y, position_z=pos.z,
                        stuck_count=stuck_count,
                        strategy_switches=strategy_switches
                    )

                if d > 0:
                    t_near = t_mid
                else:
                    t_far = t_mid

                if (t_far - t_near) < config.hit_threshold:
                    t = (t_near + t_far) * 0.5
                    pos = ray.at(t)
                    d = sdf_func(pos)
                    return MarchResult(
                        hit=True, t=t, iterations=iterations, final_sdf=d,
                        position_x=pos.x, position_y=pos.y, position_z=pos.z,
                        stuck_count=stuck_count,
                        strategy_switches=strategy_switches
                    )

                t = t_mid
                continue

            if t > config.max_distance:
                break

        pos = ray.at(t)
        d = sdf_func(pos)
        return MarchResult(
            hit=False, t=t, iterations=iterations, final_sdf=d,
            position_x=pos.x, position_y=pos.y, position_z=pos.z,
            stuck_count=stuck_count,
            strategy_switches=strategy_switches
        )
