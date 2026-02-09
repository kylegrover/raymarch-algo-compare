"""Markdown report generator for benchmark results."""

import os
from typing import List, Optional
from ..metrics.analyzer import MetricsAnalyzer


def _find_latest_gpu_csv(output_dir: str) -> Optional[str]:
    import glob
    pattern = os.path.join(output_dir, 'gpu_validation__*.csv')
    files = sorted(glob.glob(pattern), reverse=True)
    return files[0] if files else None


def generate_markdown_report(analyzer: MetricsAnalyzer, output_dir: str):
    """Create a comprehensive Markdown report of the benchmark results.

    If a GPU validation CSV (produced by `gpu_confirm.py`) exists in `output_dir`,
    merge GPU timing and warp-divergence columns into the tables.
    """
    import pandas as _pd

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'REPORT.md')

    # try to load latest GPU validation (optional)
    gpu_csv = _find_latest_gpu_csv(output_dir)
    gpu_df = None
    if gpu_csv:
        try:
            gpu_df = _pd.read_csv(gpu_csv)
            # Normalize column names for robust merging
            gpu_df.columns = [c.strip() for c in gpu_df.columns]
        except Exception:
            gpu_df = None

    # Also consider per-run GPU fields present in the analyzer (we may have GPU timings
    # even if `gpu_confirm.py` wasn't executed to produce a `gpu_validation__*.csv`).
    has_gpu = gpu_df is not None or any(
        getattr(s, 'gpu_time_per_ray_us', None) is not None for s in analyzer.all_stats
    )

    lines = []
    lines.append("# Ray Marching Algorithm Benchmark Report")
    lines.append("")
    lines.append("## Overview")
    lines.append(f"Tested **{len(analyzer.get_strategies())}** strategies across **{len(analyzer.get_scenes())}** scenes.")
    lines.append("")

    # GPU note
    if has_gpu:
        lines.append("> Note: GPU timings (where shown) are measured by executing the GLSL shader via ModernGL and")
        lines.append("> dividing the synchronous GPU render time by the number of pixels. Readback is excluded; driver/submit")
        lines.append("> overhead may still be included. Use the `gpu_validation__*.csv` in the `results/` folder for raw data.")
        lines.append("")

    # 1. Charts
    lines.append("## Visual Comparisons")
    lines.append("### Performance overview")
    lines.append("![Iteration Count](chart_iterations.png)")
    lines.append("")
    lines.append("### Workload Divergence (Warp Divergence)")
    lines.append("![Divergence](chart_divergence.png)")
    lines.append("")
    lines.append("### Speed vs Accuracy")
    lines.append("![Speed vs Accuracy](chart_speed_accuracy.png)")
    lines.append("")

    # 2. Results Table (merge GPU time if available)
    lines.append("## Aggregated Statistics")
    lines.append("")
    summary = analyzer.strategy_summary()

    # Always include GPU columns now (values will be '-' if missing)
    header = "| Strategy | Iterations (avg) | Hit Rate | Wins | Warp Div | CPU Time (us/ray) | GPU Time (us/ray) | GPU WD |"
    sep = "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"

    lines.append(header)
    lines.append(sep)

    for strat, vals in summary.items():
        gpu_time = vals.get('avg_gpu_time_per_ray_us')
        gpu_wd = vals.get('avg_gpu_warp_divergence')
        gpu_time_str = f"{gpu_time:.2f}" if gpu_time is not None else "-"
        gpu_wd_str = f"{gpu_wd:.2f}" if gpu_wd is not None else "-"

        row = (f"| {strat} | {vals['avg_mean_iterations']:.2f} | {vals['avg_hit_rate']:.1%} | "
               f"{vals['num_wins']} | {vals['avg_warp_divergence']:.2f} | {vals['avg_time_per_ray_us']:.2f} | {gpu_time_str} | {gpu_wd_str} |")
        lines.append(row)
    lines.append("")

    # 3. Scene Breakdown
    lines.append("## Per-Scene Analysis")
    for scene in analyzer.get_scenes():
        lines.append(f"### {scene}")

        # Check if we have a comparison image for this scene
        compare_img = f"compare__{scene.replace(' ', '_').replace('/', '_')}.png"
        if os.path.exists(os.path.join(output_dir, compare_img)):
            lines.append(f"![{scene} Comparison]({compare_img})")
            lines.append("")

        # Build header with optional GPU columns
        row_hdr = "| Strategy | Iterations | Hit Rate | P95 | Warp Div | CPU Time (us/ray)"
        if has_gpu:
            row_hdr += " | GPU Time (us/ray) | GPU WD"
        row_hdr += " |"
        lines.append(row_hdr)

        sep = "| :--- | :---: | :---: | :---: | :---: | :---:"
        if has_gpu:
            sep += " | :---: | :---:"
        sep += " |"
        lines.append(sep)

        for strat in analyzer.get_strategies():
            stat = analyzer.get_stat(strat, scene)
            if not stat:
                # stat missing: emit an empty row so the table shape is stable
                if has_gpu:
                    lines.append(f"| {strat} | {'-':>8} | {'-':>6} | {'-':>4} | {'-':>8} | {'-':>12} | {'-':>12} | {'-':>6} |")
                else:
                    lines.append(f"| {strat} | {'-':>8} | {'-':>6} | {'-':>4} | {'-':>8} | {'-':>12} |")
                continue

            gpu_time = getattr(stat, 'gpu_time_per_ray_us', None)
            gpu_wd = getattr(stat, 'gpu_warp_divergence_proxy', None)
            gpu_time_str = f"{gpu_time:.2f}" if gpu_time is not None else "-"
            gpu_wd_str = f"{gpu_wd:.2f}" if gpu_wd is not None else "-"

            if has_gpu:
                base = (f"| {strat} | {stat.iteration_mean:.2f} | {stat.hit_rate:.1%} | "
                        f"{stat.iteration_p95:.1f} | {stat.warp_divergence_proxy:.2f} | {stat.time_per_ray_us:.2f} | {gpu_time_str} | {gpu_wd_str} |")
            else:
                base = (f"| {strat} | {stat.iteration_mean:.2f} | {stat.hit_rate:.1%} | "
                        f"{stat.iteration_p95:.1f} | {stat.warp_divergence_proxy:.2f} | {stat.time_per_ray_us:.2f} |")

            lines.append(base)

        lines.append("")

    # write out the report
    with open(report_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')

    return report_path
