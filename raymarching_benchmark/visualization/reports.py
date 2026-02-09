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

    lines = []
    lines.append("# Ray Marching Algorithm Benchmark Report")
    lines.append("")
    lines.append("## Overview")
    lines.append(f"Tested **{len(analyzer.get_strategies())}** strategies across **{len(analyzer.get_scenes())}** scenes.")
    lines.append("")

    # GPU note
    if gpu_df is not None:
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

    header = "| Strategy | Iterations (avg) | Hit Rate | Wins | Warp Div | CPU Time (us/ray)"
    if gpu_df is not None:
        header += " | GPU Time (us/ray) | GPU WD"
    header += " |"
    sep = "| :--- | :---: | :---: | :---: | :---: | :---:"
    if gpu_df is not None:
        sep += " | :---: | :---:"
    sep += " |"

    lines.append(header)
    lines.append(sep)

    for strat, vals in summary.items():
        row = (f"| {strat} | {vals['avg_mean_iterations']:.2f} | {vals['avg_hit_rate']:.1%} | "
               f"{vals['num_wins']} | {vals['avg_warp_divergence']:.2f} | {vals['avg_time_per_ray_us']:.2f}")
        if gpu_df is not None:
            # try to find a GPU aggregate for this strategy
            g = gpu_df[gpu_df['Strategy'] == strat]
            if not g.empty:
                gpu_mean = float(g['Time_us_per_ray'].mean()) if 'Time_us_per_ray' in g.columns else float('nan')
                gpu_wd = float(g['Warp_divergence'].mean()) if 'Warp_divergence' in g.columns else float('nan')
                row += f" | {gpu_mean:.2f} | {gpu_wd:.2f}"
            else:
                row += " | - | -"
        row += " |"
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
        if gpu_df is not None:
            row_hdr += " | GPU Time (us/ray) | GPU WD"
        row_hdr += " |"
        lines.append(row_hdr)

        sep = "| :--- | :---: | :---: | :---: | :---: | :---:"
        if gpu_df is not None:
            sep += " | :---: | :---:"
        sep += " |"
        lines.append(sep)

        for strat in analyzer.get_strategies():
            stat = analyzer.get_stat(strat, scene)
            if not stat:
                continue
            base = (f"| {strat} | {stat.iteration_mean:.2f} | {stat.hit_rate:.1%} | "
                    f"{stat.iteration_p95:.1f} | {stat.warp_divergence_proxy:.2f} | {stat.time_per_ray_us:.2f}")
            if gpu_df is not None:
                g = gpu_df[(gpu_df['Scene'] == scene) & (gpu_df['Strategy'] == strat)]
                if not g.empty:
                    gpu_mean = float(g['Time_us_per_ray'].mean()) if 'Time_us_per_ray' in g.columns else float('nan')
                    gpu_wd = float(g['Warp_divergence'].mean()) if 'Warp_divergence' in g.columns else float('nan')
                    base += f" | {gpu_mean:.2f} | {gpu_wd:.2f}"
                else:
                    base += " | - | -"
            base += " |"
            lines.append(base)
        lines.append("")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return report_path
