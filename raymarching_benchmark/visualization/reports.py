"""Markdown report generator for benchmark results."""

import os
from typing import List, Optional
from ..metrics.analyzer import MetricsAnalyzer


def generate_markdown_report(analyzer: MetricsAnalyzer, output_dir: str):
    """Create a comprehensive Markdown report of the benchmark results."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'REPORT.md')
    
    lines = []
    lines.append("# Ray Marching Algorithm Benchmark Report")
    lines.append("")
    lines.append("## Overview")
    lines.append(f"Tested **{len(analyzer.get_strategies())}** strategies across **{len(analyzer.get_scenes())}** scenes.")
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
    
    # 2. Results Table
    lines.append("## Aggregated Statistics")
    lines.append("")
    summary = analyzer.strategy_summary()
    
    header = "| Strategy | Iterations (avg) | Hit Rate | Wins | Warp Div | Time (us/ray) |"
    sep = "| :--- | :---: | :---: | :---: | :---: | :---: |"
    lines.append(header)
    lines.append(sep)
    
    for strat, vals in summary.items():
        row = (f"| {strat} | {vals['avg_mean_iterations']:.2f} | {vals['avg_hit_rate']:.1%} | "
               f"{vals['num_wins']} | {vals['avg_warp_divergence']:.2f} | {vals['avg_time_per_ray_us']:.2f} |")
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
            
        lines.append("| Strategy | Iterations | Hit Rate | P95 | Warp Div |")
        lines.append("| :--- | :---: | :---: | :---: | :---: |")
        
        for strat in analyzer.get_strategies():
            stat = analyzer.get_stat(strat, scene)
            if stat:
                lines.append(f"| {strat} | {stat.iteration_mean:.2f} | {stat.hit_rate:.1%} | "
                             f"{stat.iteration_p95:.1f} | {stat.warp_divergence_proxy:.2f} |")
        lines.append("")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    return report_path
