"""Console + file tables for benchmark results (lightweight, human-readable)."""

import os
from typing import List, Dict, Optional
from ..metrics.analyzer import MetricsAnalyzer, ComparisonResult
from ..core.types import RayMarchStats


def _emit_lines(lines: List[str], out_path: Optional[str] = None) -> None:
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')

    for L in lines:
        print(L)


def print_comparison_tables(analyzer: MetricsAnalyzer, output_dir: Optional[str] = None):
    """Print comparison tables and optionally save to `output_dir/tables.txt`."""
    lines: List[str] = []

    def emit(text=""):
        lines.append(text)

    emit("=" * 100)
    emit("RAY MARCHING STRATEGY BENCHMARK RESULTS")
    emit("=" * 100)
    emit()

    scenes = analyzer.get_scenes()
    strategies = analyzer.get_strategies()

    # Header
    col_width = 14
    header = f"{'Scene':<30}"
    for strat in strategies:
        header += f" {strat:>{col_width}}"
    emit(header)
    emit("-" * len(header))

    # Mean iterations per scene
    emit("\n── Mean Iterations per Ray ──")
    emit(header)
    emit("-" * len(header))

    for scene in scenes:
        row = f"{scene:<30}"
        for strat in strategies:
            stat = analyzer.get_stat(strat, scene)
            row += f" {stat.iteration_mean:>{col_width}.2f}" if stat else f" {'-':>{col_width}}"
        emit(row)

    # Summary per-strategy
    emit('\n── Strategy Summary (aggregated) ──')
    summary = analyzer.strategy_summary()
    emit(f"{'Strategy':<24}{'avg_iters':>12}{'hit_rate':>12}{'wins':>8}{'time(us/ray)':>16}")
    emit('-' * 72)
    for strat, vals in summary.items():
        emit(f"{strat:<24}{vals['avg_mean_iterations']:12.2f}{vals['avg_hit_rate']:12.2%}{vals['num_wins']:8d}{vals['avg_time_per_ray_us']:16.2f}")

    # Per-scene best/worst
    emit('\n── Per-scene best/worst (by mean iterations) ──')
    comps = analyzer.compare_by_iterations()
    emit(f"{'Scene':<30}{'best':>14}{'best_iters':>14}{'worst':>14}{'worst_iters':>14}{'speedup':>10}")
    emit('-' * 100)
    for c in comps:
        emit(f"{c.scene_name:<30}{c.best_strategy:>14}{c.best_mean_iterations:14.2f}{c.worst_strategy:>14}{c.worst_mean_iterations:14.2f}{c.speedup_ratio:10.2f}x")

    # Finalize: print + optional file
    out_path = None
    if output_dir:
        out_path = os.path.join(output_dir, 'tables.txt')
    _emit_lines(lines, out_path)
