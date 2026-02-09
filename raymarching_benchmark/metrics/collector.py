"""Metrics collection for ray marching benchmarks."""

import time
import numpy as np
from typing import List, Callable
from raymarching_benchmark.core.camera import Camera
from raymarching_benchmark.core.types import MarchResult, RayMarchStats
from raymarching_benchmark.strategies.base import MarchStrategy
from raymarching_benchmark.scenes.base import SDFScene
from raymarching_benchmark.config import MarchConfig


class MetricsCollector:
    """Collects per-ray metrics and computes aggregate statistics."""

    def __init__(self, config: MarchConfig):
        self.config = config

    def benchmark_strategy(self, strategy: MarchStrategy, scene: SDFScene,
                           camera: Camera, verbose: bool = True) -> RayMarchStats:
        """
        Run a complete benchmark of one strategy on one scene.

        Returns aggregate statistics.
        """
        width = camera.width
        height = camera.height
        total_rays = width * height

        sdf_func = scene.sdf

        if verbose:
            print(f"  Benchmarking: {strategy.short_name} on {scene.name} "
                  f"({width}x{height} = {total_rays} rays)...")

        results: List[MarchResult] = []

        start_time = time.perf_counter()

        for py in range(height):
            for px in range(width):
                ray = camera.get_ray(px, py)
                result = strategy.march(ray, sdf_func, self.config)
                results.append(result)

            if verbose and (py + 1) % max(1, height // 10) == 0:
                pct = (py + 1) / height * 100
                elapsed = time.perf_counter() - start_time
                eta = elapsed / (py + 1) * (height - py - 1)
                print(f"    {pct:5.1f}% complete, ETA: {eta:.1f}s")

        elapsed = time.perf_counter() - start_time

        stats = RayMarchStats(
            strategy_name=strategy.short_name,
            scene_name=scene.name
        )
        stats.compute(results, width, height, elapsed)

        if verbose:
            print(f"    Done in {elapsed:.2f}s. "
                  f"Hit rate: {stats.hit_rate:.1%}, "
                  f"Mean iters: {stats.iteration_mean:.1f}, "
                  f"Max iters: {stats.iteration_max}")

        return stats
