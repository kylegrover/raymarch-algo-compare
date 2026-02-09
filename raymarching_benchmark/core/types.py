"""Shared data types for march results and statistics."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np


@dataclass
class MarchResult:
    """Result of marching a single ray."""
    hit: bool  # Did we find a surface intersection?
    t: float  # Parameter along ray at termination
    iterations: int  # Total iterations used
    final_sdf: float  # SDF value at termination point
    position_x: float = 0.0
    position_y: float = 0.0
    position_z: float = 0.0
    bisection_steps: int = 0  # For methods that use bisection
    stuck_count: int = 0  # Times detected as "stuck"
    strategy_switches: int = 0  # For hybrid methods


@dataclass
class RayMarchStats:
    """Aggregate statistics for a complete render pass."""
    strategy_name: str
    scene_name: str
    total_rays: int = 0
    hit_count: int = 0
    miss_count: int = 0

    # Iteration statistics
    iteration_counts: List[int] = field(default_factory=list)
    iteration_mean: float = 0.0
    iteration_median: float = 0.0
    iteration_std: float = 0.0
    iteration_min: int = 0
    iteration_max: int = 0
    iteration_p95: float = 0.0
    iteration_p99: float = 0.0

    # Accuracy statistics (for hits only)
    final_sdf_values: List[float] = field(default_factory=list)
    accuracy_mean: float = 0.0
    accuracy_max: float = 0.0
    accuracy_std: float = 0.0

    # Convergence
    hit_rate: float = 0.0

    # Timing
    total_time_seconds: float = 0.0
    time_per_ray_us: float = 0.0

    # Warp divergence proxy (std of iterations in local neighborhoods)
    warp_divergence_proxy: float = 0.0

    # Heatmap data (iteration count per pixel)
    iteration_heatmap: Optional[np.ndarray] = None
    hit_map: Optional[np.ndarray] = None
    depth_map: Optional[np.ndarray] = None  # per-pixel t value (0 for miss)

    def compute(self, results: List[MarchResult], width: int, height: int,
                elapsed_seconds: float):
        """Compute all aggregate statistics from raw results."""
        self.total_rays = len(results)
        self.total_time_seconds = elapsed_seconds

        iterations = []
        sdf_values_hits = []
        hit_map = np.zeros((height, width), dtype=bool)
        iter_map = np.zeros((height, width), dtype=np.int32)
        depth_map = np.zeros((height, width), dtype=np.float64)

        for i, r in enumerate(results):
            py, px = divmod(i, width)
            iterations.append(r.iterations)
            iter_map[py, px] = r.iterations
            depth_map[py, px] = float(r.t) if r.hit else 0.0

            if r.hit:
                self.hit_count += 1
                hit_map[py, px] = True
                sdf_values_hits.append(abs(r.final_sdf))
            else:
                self.miss_count += 1

        self.iteration_counts = iterations
        iter_arr = np.array(iterations, dtype=np.float64)

        self.iteration_mean = float(np.mean(iter_arr))
        self.iteration_median = float(np.median(iter_arr))
        self.iteration_std = float(np.std(iter_arr))
        self.iteration_min = int(np.min(iter_arr))
        self.iteration_max = int(np.max(iter_arr))
        self.iteration_p95 = float(np.percentile(iter_arr, 95))
        self.iteration_p99 = float(np.percentile(iter_arr, 99))

        if sdf_values_hits:
            sdf_arr = np.array(sdf_values_hits, dtype=np.float64)
            self.accuracy_mean = float(np.mean(sdf_arr))
            self.accuracy_max = float(np.max(sdf_arr))
            self.accuracy_std = float(np.std(sdf_arr))

        self.hit_rate = self.hit_count / max(self.total_rays, 1)
        self.time_per_ray_us = (elapsed_seconds * 1e6) / max(self.total_rays, 1)

        # Warp divergence proxy: std of iteration counts in 8x4 blocks (simulating 32-thread warps)
        warp_h, warp_w = 4, 8
        block_stds = []
        for by in range(0, height - warp_h + 1, warp_h):
            for bx in range(0, width - warp_w + 1, warp_w):
                block = iter_map[by:by+warp_h, bx:bx+warp_w].flatten().astype(np.float64)
                if len(block) == warp_h * warp_w:
                    block_stds.append(float(np.std(block)))
        self.warp_divergence_proxy = float(np.mean(block_stds)) if block_stds else 0.0

        self.iteration_heatmap = iter_map
        self.depth_map = depth_map
