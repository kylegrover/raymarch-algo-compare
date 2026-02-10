"""Statistical analysis and cross-comparison of benchmark results."""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional
from raymarching_benchmark.core.types import RayMarchStats
from dataclasses import dataclass


@dataclass
class ComparisonResult:
    """Result of comparing strategies across scenes."""
    scene_name: str
    rankings: List[Tuple[str, float]]  # (strategy_name, score) sorted best-first
    best_strategy: str
    best_mean_iterations: float
    worst_strategy: str
    worst_mean_iterations: float
    speedup_ratio: float  # worst/best


class MetricsAnalyzer:
    """Analyzes and compares benchmark results across strategies and scenes."""

    def __init__(self):
        self.all_stats: List[RayMarchStats] = []

    def add_result(self, stats: RayMarchStats):
        self.all_stats.append(stats)

    def add_results(self, stats_list: List[RayMarchStats]):
        self.all_stats.extend(stats_list)

    def get_strategies(self) -> List[str]:
        return sorted(set(s.strategy_name for s in self.all_stats))

    def get_scenes(self) -> List[str]:
        seen = []
        for s in self.all_stats:
            if s.scene_name not in seen:
                seen.append(s.scene_name)
        return seen

    def get_stat(self, strategy: str, scene: str) -> Optional[RayMarchStats]:
        for s in self.all_stats:
            if s.strategy_name == strategy and s.scene_name == scene:
                return s
        return None

    def compare_by_iterations(self) -> List[ComparisonResult]:
        """Compare strategies by mean iteration count per scene."""
        results = []
        for scene in self.get_scenes():
            rankings = []
            for strategy in self.get_strategies():
                stat = self.get_stat(strategy, scene)
                if stat:
                    rankings.append((strategy, stat.iteration_mean))

            rankings.sort(key=lambda x: x[1])

            if rankings:
                best = rankings[0]
                worst = rankings[-1]
                speedup = worst[1] / max(best[1], 0.01)

                results.append(ComparisonResult(
                    scene_name=scene,
                    rankings=rankings,
                    best_strategy=best[0],
                    best_mean_iterations=best[1],
                    worst_strategy=worst[0],
                    worst_mean_iterations=worst[1],
                    speedup_ratio=speedup
                ))

        return results

    def strategy_summary(self) -> Dict[str, Dict[str, float]]:
        """
        For each strategy, compute summary metrics across all scenes:
        - avg_mean_iterations
        - avg_hit_rate
        - avg_accuracy
        - num_wins (scenes where it had lowest mean iterations)
        - avg_warp_divergence
        """
        comparisons = self.compare_by_iterations()
        strategies = self.get_strategies()

        summary = {}
        for strat in strategies:
            stats_for_strat = [s for s in self.all_stats if s.strategy_name == strat]
            if not stats_for_strat:
                continue

            wins = sum(1 for c in comparisons if c.best_strategy == strat)
            avg_iters = np.mean([s.iteration_mean for s in stats_for_strat])
            avg_hit_rate = np.mean([s.hit_rate for s in stats_for_strat])
            avg_accuracy = np.mean([s.accuracy_mean for s in stats_for_strat if s.accuracy_mean > 0])
            avg_warp_div = np.mean([s.warp_divergence_proxy for s in stats_for_strat])
            avg_p99 = np.mean([s.iteration_p99 for s in stats_for_strat])
            avg_time = np.mean([s.time_per_ray_us for s in stats_for_strat])
            # GPU fields: ignore None values
            # Prefer median GPU time if available (more robust to outliers/repeats)
            gpu_medians = [getattr(s, 'gpu_time_per_ray_median_us', None) for s in stats_for_strat if getattr(s, 'gpu_time_per_ray_median_us', None) is not None]
            gpu_legacy = [s.gpu_time_per_ray_us for s in stats_for_strat if getattr(s, 'gpu_time_per_ray_us', None) is not None and getattr(s, 'gpu_time_per_ray_median_us', None) is None]
            gpu_times = gpu_medians + gpu_legacy

            gpu_wds = [s.gpu_warp_divergence_proxy for s in stats_for_strat if getattr(s, 'gpu_warp_divergence_proxy', None) is not None]
            avg_gpu_time = float(np.mean(gpu_times)) if gpu_times else None
            avg_gpu_wd = float(np.mean(gpu_wds)) if gpu_wds else None

            # ms/frame median where available
            frame_ms = [getattr(s, 'gpu_frame_ms_median', None) for s in stats_for_strat if getattr(s, 'gpu_frame_ms_median', None) is not None]
            avg_gpu_frame_ms = float(np.mean(frame_ms)) if frame_ms else None

            summary[strat] = {
                'avg_mean_iterations': float(avg_iters),
                'avg_hit_rate': float(avg_hit_rate),
                'avg_accuracy': float(avg_accuracy) if not np.isnan(avg_accuracy) else 0.0,
                'num_wins': wins,
                'total_scenes': len(stats_for_strat),
                'avg_warp_divergence': float(avg_warp_div),
                'avg_p99_iterations': float(avg_p99),
                'avg_time_per_ray_us': float(avg_time),
                'avg_gpu_time_per_ray_us': avg_gpu_time,
                'avg_gpu_frame_ms': avg_gpu_frame_ms,
                'avg_gpu_sample_count': float(np.mean([s.gpu_time_sample_count for s in stats_for_strat if getattr(s, 'gpu_time_sample_count', None) is not None])) if any(getattr(s, 'gpu_time_sample_count', None) is not None for s in stats_for_strat) else None,
                'avg_gpu_warp_divergence': avg_gpu_wd,
            }

        return summary

    def per_scene_matrix(self, metric: str = 'iteration_mean') -> Tuple[List[str], List[str], np.ndarray]:
        """
        Build a matrix of [scenes x strategies] for a given metric.
        Returns (scene_names, strategy_names, matrix).
        """
        scenes = self.get_scenes()
        strategies = self.get_strategies()

        matrix = np.full((len(scenes), len(strategies)), np.nan)

        for si, scene in enumerate(scenes):
            for sti, strategy in enumerate(strategies):
                stat = self.get_stat(strategy, scene)
                if stat:
                    matrix[si, sti] = getattr(stat, metric, np.nan)

        return scenes, strategies, matrix

    def save_csv_matrices(self, output_dir: str):
        """Save iteration_mean and time_per_ray matrices as CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        
        metrics = [
            'iteration_mean', 'time_per_ray_us', 'hit_rate', 'warp_divergence_proxy',
            'gpu_time_per_ray_us', 'gpu_time_per_ray_median_us', 'gpu_frame_ms_median', 'gpu_time_sample_count',
            'gpu_warp_divergence_proxy'
        ]
        
        import pandas as pd
        
        for metric in metrics:
            scenes, strategies, matrix = self.per_scene_matrix(metric)
            df = pd.DataFrame(matrix, index=scenes, columns=strategies)
            csv_path = os.path.join(output_dir, f'matrix_{metric}.csv')
            df.to_csv(csv_path)
