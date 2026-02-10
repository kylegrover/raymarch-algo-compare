"""Minimal CLI / entrypoint for MVP benchmarking (version 0).

Runs a single strategy (default: StandardSphereTracing) on a single scene
(default: Sphere) at a small resolution and prints a compact summary.

This is intentionally tiny: enough to run, test, and iterate from.
"""

from __future__ import annotations
import argparse
import json
import os
from typing import Optional

from .config import RenderConfig, MarchConfig
from .core.camera import Camera
from .core.vec3 import Vec3
from .scenes.catalog import get_scene_by_name, get_all_scenes, SphereScene
from .strategies import get_strategy_by_name, list_strategies, StandardSphereTracing
from .metrics.collector import MetricsCollector
from .metrics.analyzer import MetricsAnalyzer
from .core.types import RayMarchStats
from .visualization.tables import print_comparison_tables
from .visualization.charts import render_charts
from .visualization.heatmaps import save_tiled_comparison
from .visualization.reports import generate_markdown_report


def run_once(render: Optional[RenderConfig] = None,
             march: Optional[MarchConfig] = None,
             scene_name: str = "Sphere",
             strategy_name: str = "Standard",
             hybrid_fallback_iter: int | None = None) -> RayMarchStats:
    """Run a single small benchmark and return the stats (callable from tests)."""
    render = render or RenderConfig(width=64, height=48)
    march = march or MarchConfig()

    # Resolve scene & strategy
    scene = get_scene_by_name(scene_name) or SphereScene()
    strategy = get_strategy_by_name(strategy_name) or StandardSphereTracing()

    # If caller requested an experimental hybrid->segment fallback, apply it
    if hybrid_fallback_iter is not None and hasattr(strategy, 'fallback_to_segment_after'):
        try:
            strategy.fallback_to_segment_after = int(hybrid_fallback_iter)
        except Exception:
            pass

    # Wire suggested camera
    suggestion = scene.suggested_camera()
    if suggestion:
        render.camera_position = suggestion.camera_position
        render.camera_target = suggestion.camera_target
        render.camera_up = suggestion.camera_up
        render.fov_degrees = suggestion.fov_degrees

    # Wire Lipschitz bound if strategy supports it
    if hasattr(strategy, 'lipschitz'):
        bound = scene.known_lipschitz_bound()
        if bound is not None:
            strategy.lipschitz = bound

    # Ensure Camera receives Vec3 instances (RenderConfig stores tuples)
    pos_tuple = getattr(render, 'camera_position', (0.0, 0.0, 5.0))
    tgt_tuple = getattr(render, 'camera_target', (0.0, 0.0, 0.0))
    up_tuple = getattr(render, 'camera_up', (0.0, 1.0, 0.0))
    cam = Camera(
        position=Vec3(*pos_tuple),
        target=Vec3(*tgt_tuple),
        up=Vec3(*up_tuple),
        fov_degrees=render.fov_degrees,
        width=render.width,
        height=render.height,
    )

    collector = MetricsCollector(march)
    stats = collector.benchmark_strategy(strategy, scene, cam, verbose=False)
    return stats


def _print_stats(s: RayMarchStats) -> None:
    print(f"Strategy: {s.strategy_name} | Scene: {s.scene_name}")
    print(f"Rays: {s.total_rays}  Hits: {s.hit_count}  Hit rate: {s.hit_rate:.2%}")
    print(f"Iter mean: {s.iteration_mean:.2f}  p95: {s.iteration_p95:.1f}  max: {s.iteration_max}")
    print(f"Time: {s.time_per_ray_us:.1f} us/ray  Total: {s.total_time_seconds:.2f}s")


def _safe_name(s: str) -> str:
    return s.replace(' ', '_').replace('/', '_')


def _save_outputs(stats: RayMarchStats, results_dir: str, max_iters: Optional[int] = None) -> None:
    """Save images and a small JSON log for one stats object."""
    from .visualization.heatmaps import save_iteration_heatmap, save_inverse_depth_map, save_hit_map
    import numpy as _np
    timestamp = _np.datetime64(_np.datetime64('now'), 's').astype(str).replace(':', '-')
    scene_name = _safe_name(stats.scene_name)
    strat_name = _safe_name(stats.strategy_name)
    out_dir = os.path.join(results_dir, f"{scene_name}__{strat_name}__{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    # iteration heatmap
    if stats.iteration_heatmap is not None:
        path = os.path.join(out_dir, 'iterations.png')
        save_iteration_heatmap(stats.iteration_heatmap, max_iters or stats.iteration_max, path)

    # hit map
    if stats.hit_map is not None:
        path = os.path.join(out_dir, 'hit_map.png')
        save_hit_map(stats.hit_map, path)

    # inverse depth map
    if stats.depth_map is not None and stats.hit_map is not None:
        path = os.path.join(out_dir, 'inv_depth.png')
        save_inverse_depth_map(stats.depth_map, stats.hit_map, path)
        # also save raw depth as npy for debugging
        _np.save(os.path.join(out_dir, 'depth_map.npy'), stats.depth_map)

    # stats json
    json_path = os.path.join(out_dir, 'stats.json')
    compact = {
        'strategy': stats.strategy_name,
        'scene': stats.scene_name,
        'total_rays': int(stats.total_rays),
        'hit_count': int(stats.hit_count),
        'hit_rate': float(stats.hit_rate),
        'iteration_mean': float(stats.iteration_mean),
        'iteration_p95': float(stats.iteration_p95),
        'iteration_max': int(stats.iteration_max),
        'warp_divergence': float(stats.warp_divergence_proxy),
        'time_us_per_ray': float(stats.time_per_ray_us),
        'gpu_time_us_per_ray': float(stats.gpu_time_per_ray_us) if stats.gpu_time_per_ray_us is not None else None,
        'gpu_time_us_per_ray_median': float(stats.gpu_time_per_ray_median_us) if stats.gpu_time_per_ray_median_us is not None else None,
        'gpu_time_sample_count': int(stats.gpu_time_sample_count) if stats.gpu_time_sample_count is not None else None,
        'gpu_warp_divergence': float(stats.gpu_warp_divergence_proxy) if stats.gpu_warp_divergence_proxy is not None else None,
        'gpu_width': int(stats.gpu_width) if getattr(stats, 'gpu_width', None) is not None else None,
        'gpu_height': int(stats.gpu_height) if getattr(stats, 'gpu_height', None) is not None else None,
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(compact, f, indent=2)

    print(f"  Saved outputs to: {out_dir}")


def cli(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="raymarch-bench", description="Ray marching benchmark runner")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--height", type=int, default=48)
    p.add_argument("--gpu-width", type=int, default=None, help="GPU render width (overrides CPU width)")
    p.add_argument("--gpu-height", type=int, default=None, help="GPU render height (overrides CPU height)")
    p.add_argument("--gpu-1080p", action='store_true', help="Shortcut: run GPU measurements at 1920x1080")
    p.add_argument("--gpu-warmup", type=int, default=5, help="Number of warmup GPU frames to discard")
    p.add_argument("--gpu-repeats", type=int, default=10, help="Number of GPU timed repeats (median reported)")
    p.add_argument("--hybrid-fallback-iter", type=int, default=None, help="(experimental) Hybrid: fallback to Segment after N iterations")
    p.add_argument("--scene", type=str, default="Sphere",
                   help="Scene name (comma-separated, or 'all')")
    p.add_argument("--strategy", type=str, default="Standard",
                   help="Strategy name (comma-separated, or 'all')")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Directory to write images and logs (default: results)")
    p.add_argument("--no-save-images", action='store_true', help="Disable image/log output")
    p.add_argument("--json", type=str, default=None, help="Write stats JSON summary to path")
    p.add_argument("--kappa", type=float, default=None, help="Override segment-tracing kappa (growth factor)")
    p.add_argument("--min-step-fraction", type=float, default=None, help="Override march.min_step_fraction")
    args = p.parse_args(argv)

    rc = RenderConfig(width=args.width, height=args.height)
    # Determine GPU render config: explicit flags > --gpu-1080p shortcut > CPU resolution
    if args.gpu_width is not None or args.gpu_height is not None:
        gw = args.gpu_width or args.width
        gh = args.gpu_height or args.height
        gpu_rc = RenderConfig(width=int(gw), height=int(gh))
    elif args.gpu_1080p:
        gpu_rc = RenderConfig(width=1920, height=1080)
    else:
        gpu_rc = rc

    mc = MarchConfig()
    if args.kappa is not None:
        mc.kappa = float(args.kappa)
    if args.min_step_fraction is not None:
        mc.min_step_fraction = float(args.min_step_fraction)

    results_dir = args.output_dir or 'results'

    # Resolve scenes
    if args.scene.lower() == 'all':
        scene_names = [s.name for s in get_all_scenes()]
    else:
        scene_names = [s.strip() for s in args.scene.split(',') if s.strip()]

    # Resolve strategies
    if args.strategy.lower() == 'all':
        strategy_names = list_strategies()
    else:
        strategy_names = [s.strip() for s in args.strategy.split(',') if s.strip()]

    analyzer = MetricsAnalyzer()
    
    print(f"Running benchmark: {len(strategy_names)} strategies x {len(scene_names)} scenes "
          f"at {args.width}x{args.height}")

    for scene_name in scene_names:
        for strat_name in strategy_names:
            print(f"\n>> Scene: {scene_name} | Strategy: {strat_name}")
            stats = run_once(render=rc, march=mc, scene_name=scene_name, strategy_name=strat_name, hybrid_fallback_iter=args.hybrid_fallback_iter)
            _print_stats(stats)

            # Always attempt a GPU validation run for the same scene/strategy so reports
            # consistently contain CPU+GPU columns. Fail gracefully if GPU is not available.
            try:
                from .gpu.runner import run_gpu_benchmark
                gpu_res = run_gpu_benchmark(scene_name, strat_name, gpu_rc, mc,
                                            gpu_warmup=int(args.gpu_warmup), gpu_repeats=int(args.gpu_repeats))
                if gpu_res is not None:
                    # pixels[...,1] encodes iterations/maxIterations
                    pixels = gpu_res['pixels']

                    # record resolution
                    stats.gpu_width = int(gpu_rc.width)
                    stats.gpu_height = int(gpu_rc.height)

                    # timing: prefer median across repeats
                    median_s = float(gpu_res.get('render_time_s_median', gpu_res['render_times_s'][0]))
                    stats.gpu_time_per_ray_median_us = (median_s / (gpu_rc.width * gpu_rc.height)) * 1e6
                    stats.gpu_time_sample_count = int(gpu_res.get('sample_count', 1))
                    stats.gpu_time_per_ray_us = float((gpu_res.get('render_time_s_mean', median_s) / (gpu_rc.width * gpu_rc.height)) * 1e6)

                    # report ms/frame median (useful for real-time comparisons)
                    stats.gpu_frame_ms_median = (stats.gpu_time_per_ray_median_us * gpu_rc.width * gpu_rc.height) / 1000.0

                    # compute warp-divergence proxy on the median iteration map
                    gpu_iters = pixels[..., 1] * mc.max_iterations
                    h, w = gpu_iters.shape
                    bx, by = 8, 4
                    stds = []
                    for yy in range(0, h, by):
                        for xx in range(0, w, bx):
                            block = gpu_iters[yy: min(yy+by, h), xx: min(xx+bx, w)].ravel()
                            if block.size == 0:
                                continue
                            stds.append(float(block.std()))
                    stats.gpu_warp_divergence_proxy = float(sum(stds) / len(stds)) if stds else 0.0
                else:
                    stats.gpu_time_per_ray_us = None
                    stats.gpu_time_per_ray_median_us = None
                    stats.gpu_time_sample_count = None
                    stats.gpu_frame_ms_median = None
                    stats.gpu_warp_divergence_proxy = None
            except Exception:
                # GPU unavailable or failed; mark fields as None and continue
                stats.gpu_time_per_ray_us = None
                stats.gpu_warp_divergence_proxy = None

            analyzer.add_result(stats)

            if not args.no_save_images:
                _save_outputs(stats, results_dir, max_iters=mc.max_iterations)

    # Comparison tables
    if len(analyzer.all_stats) > 1:
        print("\n" + "="*80)
        print_comparison_tables(analyzer, results_dir)
        
        # Generation charts (non-fatal)
        if not args.no_save_images:
            try:
                render_charts(analyzer, results_dir)
                print(f"  Saved comparative charts to: {results_dir}")

                # Tiled comparisons per scene
                for scene_name in analyzer.get_scenes():
                    strat_stats = [s for s in analyzer.all_stats if s.scene_name == scene_name]
                    if len(strat_stats) > 1:
                        maps = [s.iteration_heatmap for s in strat_stats if s.iteration_heatmap is not None]
                        labels = [s.strategy_name for s in strat_stats if s.iteration_heatmap is not None]
                        if maps:
                            fname = f"compare__{_safe_name(scene_name)}.png"
                            save_tiled_comparison(maps, labels, os.path.join(results_dir, fname))
            except Exception as e:
                print(f"  [Warning] Chart generation failed: {e}")

        # REPORT + CSVs — always attempt these (don't let charts break report generation)
        try:
            report_path = generate_markdown_report(analyzer, results_dir)
            print(f"  Generated Markdown report: {report_path}")
        except Exception as e:
            print(f"  [Error] Failed to generate Markdown report: {e}")

        try:
            analyzer.save_csv_matrices(results_dir)
            print(f"  Saved CSV matrices to: {results_dir}")
        except Exception as e:
            print(f"  [Error] Failed to save CSV matrices: {e}")

    if args.json and analyzer.all_stats:
        summary = []
        for s in analyzer.all_stats:
            summary.append({
                'strategy': s.strategy_name,
                'scene': s.scene_name,
                'total_rays': int(s.total_rays),
                'hit_count': int(s.hit_count),
                'hit_rate': float(s.hit_rate),
                'iteration_mean': float(s.iteration_mean),
                'iteration_p95': float(s.iteration_p95),
                'iteration_max': int(s.iteration_max),
                'time_per_ray_us': float(s.time_per_ray_us),
                'warp_divergence': float(s.warp_divergence_proxy),
            })
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Saved summary JSON to: {args.json}")

    return 0


if __name__ == '__main__':
    raise SystemExit(cli())
