"""Comparative charts for benchmark results using matplotlib."""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List
from ..metrics.analyzer import MetricsAnalyzer


def render_charts(analyzer: MetricsAnalyzer, output_dir: str):
    """Generate all comparative charts and save to `output_dir`."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Mean Iterations Bar Chart (all strategies, all scenes)
    _plot_iterations_by_scene(analyzer, output_dir)
    
    # 2. Warp Divergence Comparison
    _plot_warp_divergence(analyzer, output_dir)
    
    # 3. Hit Rate Comparison
    _plot_hit_rate(analyzer, output_dir)
    
    # 4. Accuracy (SDF error) vs Iterations Scatter Plot
    _plot_accuracy_vs_speed(analyzer, output_dir)


def _plot_iterations_by_scene(analyzer: MetricsAnalyzer, output_dir: str):
    scenes, strategies, matrix = analyzer.per_scene_matrix('iteration_mean')
    if matrix.size == 0: return

    x = np.arange(len(scenes))
    width = 0.8 / len(strategies)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, strat in enumerate(strategies):
        ax.bar(x + i * width - (len(strategies)-1)*width/2, matrix[:, i], width, label=strat)
    
    ax.set_ylabel('Mean Iterations per Ray')
    ax.set_title('Iteration Count Comparison by Scene')
    ax.set_xticks(x)
    ax.set_xticklabels(scenes, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'chart_iterations.png'), dpi=150)
    plt.close()


def _plot_warp_divergence(analyzer: MetricsAnalyzer, output_dir: str):
    scenes, strategies, matrix = analyzer.per_scene_matrix('warp_divergence_proxy')
    if matrix.size == 0: return

    x = np.arange(len(scenes))
    width = 0.8 / len(strategies)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, strat in enumerate(strategies):
        ax.bar(x + i * width - (len(strategies)-1)*width/2, matrix[:, i], width, label=strat)
    
    ax.set_ylabel('Warp Divergence Proxy (Std Dev)')
    ax.set_title('Iteration Variance (Workload Divergence) by Scene')
    ax.set_xticks(x)
    ax.set_xticklabels(scenes, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'chart_divergence.png'), dpi=150)
    plt.close()


def _plot_hit_rate(analyzer: MetricsAnalyzer, output_dir: str):
    scenes, strategies, matrix = analyzer.per_scene_matrix('hit_rate')
    if matrix.size == 0: return

    x = np.arange(len(scenes))
    width = 0.8 / len(strategies)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, strat in enumerate(strategies):
        ax.bar(x + i * width - (len(strategies)-1)*width/2, matrix[:, i] * 100.0, width, label=strat)
    
    ax.set_ylabel('Hit Rate (%)')
    ax.set_title('Success Rate Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(scenes, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'chart_hit_rate.png'), dpi=150)
    plt.close()


def _plot_accuracy_vs_speed(analyzer: MetricsAnalyzer, output_dir: str):
    fig, ax = plt.subplots(figsize=(10, 7))
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for i, strat in enumerate(analyzer.get_strategies()):
        iters = []
        accuracy = []
        for s in analyzer.all_stats:
            if s.strategy_name == strat:
                iters.append(s.iteration_mean)
                accuracy.append(s.accuracy_mean)
        
        if iters:
            ax.scatter(iters, accuracy, label=strat, alpha=0.8, marker=markers[i % len(markers)], s=80)
    
    ax.set_xlabel('Mean Iterations (Speed)')
    ax.set_ylabel('Mean Accuracy (SDF Error)')
    ax.set_yscale('log')
    ax.set_title('Speed vs Accuracy Tradeoff')
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'chart_speed_accuracy.png'), dpi=150)
    plt.close()
