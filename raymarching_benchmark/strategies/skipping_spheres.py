"""Coarse-to-fine "Skipping Spheres" approximation.

Implements a two-pass march:
 - coarse pass uses a scaled (fattened) SDF to catch thin/grazing features
 - fine pass refines using the true SDF

This mirrors the GLSL `skipping_spheres` implementation used by the GPU renderer.
"""
from .base import MarchStrategy
from ..core.ray import Ray
from ..core.types import MarchResult
from ..config import MarchConfig
from typing import Callable


class SkippingSpheresTracing(MarchStrategy):
    @property
    def name(self) -> str:
        return "Skipping Spheres (Coarse->Fine)"

    @property
    def short_name(self) -> str:
        return "Skipping-Spheres"

    def march(self, ray: Ray, sdf_func: Callable, config: MarchConfig) -> MarchResult:
        t = 0.0
        iterations = 0
        final_sdf = 0.0

        margin = 0.05
        # Split the SHARED iteration budget (~2/3 coarse, 1/3 fine) so this
        # method gets the same total chances to converge as every other strategy.
        coarse_budget = (config.max_iterations * 2) // 3
        fine_budget = config.max_iterations - coarse_budget

        # --- Coarse (scaled) pass ---
        for i in range(coarse_budget):
            iterations = i + 1
            pos = ray.at(t)
            d = sdf_func(pos) - margin
            final_sdf = d + margin
            if d < config.hit_threshold:
                break
            t += d
            if t > config.max_distance:
                break

        # --- Fine (true SDF) pass ---
        for j in range(fine_budget):
            iterations += 1
            pos = ray.at(t)
            d = sdf_func(pos)
            final_sdf = d
            if abs(d) < config.hit_threshold:
                return MarchResult(
                    hit=True, t=t, iterations=iterations, final_sdf=d,
                    position_x=pos.x, position_y=pos.y, position_z=pos.z
                )
            if d < 0.0:
                # conservative back-up
                t = max(0.0, t + d)
                continue
            t += d
            if t > config.max_distance:
                break

        pos = ray.at(t)
        d = sdf_func(pos)
        return MarchResult(
            hit=False, t=t, iterations=iterations, final_sdf=d,
            position_x=pos.x, position_y=pos.y, position_z=pos.z
        )
