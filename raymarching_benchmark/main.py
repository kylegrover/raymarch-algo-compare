"""Minimal CLI / entrypoint for MVP benchmarking (version 0).

Runs a single strategy (default: StandardSphereTracing) on a single scene
(default: Sphere) at a small resolution and prints a compact summary.

This is intentionally tiny: enough to run, test, and iterate from.
"""

from __future__ import annotations
import argparse
import json
from typing import Optional

from .config import RenderConfig, MarchConfig
from .core.camera import Camera
from .core.vec3 import Vec3
from .scenes.catalog import get_scene_by_name, SphereScene
from .strategies.standard_sphere import StandardSphereTracing
from .metrics.collector import MetricsCollector
from .core.types import RayMarchStats


def run_once(render: Optional[RenderConfig] = None,
             march: Optional[MarchConfig] = None,
             scene_name: str = "Sphere",
             strategy_name: str = "Standard") -> RayMarchStats:
    """Run a single small benchmark and return the stats (callable from tests)."""
    render = render or RenderConfig(width=64, height=48)
    march = march or MarchConfig()

    # Resolve scene & strategy (only minimal mapping for MVP)
    scene = get_scene_by_name(scene_name) or SphereScene()
    strategy = StandardSphereTracing()

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
    import os as _os
    import json as _json
    timestamp = _np.datetime64(_np.datetime64('now'), 's').astype(str).replace(':', '-')
    scene_name = _safe_name(stats.scene_name)
    strat_name = _safe_name(stats.strategy_name)
    out_dir = _os.path.join(results_dir, f"{scene_name}__{strat_name}__{timestamp}")
    _os.makedirs(out_dir, exist_ok=True)

    # iteration heatmap
    if stats.iteration_heatmap is not None:
        path = _os.path.join(out_dir, 'iterations.png')
        save_iteration_heatmap(stats.iteration_heatmap, max_iters or stats.iteration_max, path)

    # hit map
    if stats.hit_map is not None:
        path = _os.path.join(out_dir, 'hit_map.png')
        save_hit_map(stats.hit_map, path)

    # inverse depth map
    if stats.depth_map is not None and stats.hit_map is not None:
        path = _os.path.join(out_dir, 'inv_depth.png')
        save_inverse_depth_map(stats.depth_map, stats.hit_map, path)
        # also save raw depth as npy for debugging
        _np.save(_os.path.join(out_dir, 'depth_map.npy'), stats.depth_map)

    # stats json
    json_path = _os.path.join(out_dir, 'stats.json')
    compact = {
        'strategy': stats.strategy_name,
        'scene': stats.scene_name,
        'total_rays': int(stats.total_rays),
        'hit_count': int(stats.hit_count),
        'hit_rate': float(stats.hit_rate),
        'iteration_mean': float(stats.iteration_mean),
        'iteration_p95': float(stats.iteration_p95),
        'iteration_max': int(stats.iteration_max),
    }
    with open(json_path, 'w', encoding='utf-8') as _f:
        _json.dump(compact, _f, indent=2)

    print(f"  Saved outputs to: {out_dir}")


def cli(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="raymarch-bench", description="Tiny MVP benchmark runner")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--height", type=int, default=48)
    p.add_argument("--scene", type=str, default="Sphere",
                   help="Scene name (comma-separated to run multiple)")
    p.add_argument("--strategy", type=str, default="Standard")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Directory to write images and logs (default: results)")
    p.add_argument("--no-save-images", action='store_true', help="Disable image/log output")
    p.add_argument("--json", type=str, default=None, help="Write stats JSON to path")
    args = p.parse_args(argv)

    rc = RenderConfig(width=args.width, height=args.height)
    mc = MarchConfig()

    results_dir = args.output_dir or 'results'

    # allow multiple comma-separated scenes
    scenes = [s.strip() for s in args.scene.split(',') if s.strip()]
    for scene_name in scenes:
        stats = run_once(render=rc, march=mc, scene_name=scene_name, strategy_name=args.strategy)
        _print_stats(stats)

        if not args.no_save_images:
            _save_outputs(stats, results_dir, max_iters=mc.max_iterations)

        if args.json:
            # serialize a compact subset
            out = {
                'strategy': stats.strategy_name,
                'scene': stats.scene_name,
                'total_rays': stats.total_rays,
                'hit_count': stats.hit_count,
                'hit_rate': stats.hit_rate,
                'iteration_mean': stats.iteration_mean,
            }
            with open(args.json, 'w', encoding='utf-8') as f:
                json.dump(out, f, indent=2)

    return 0


if __name__ == '__main__':
    raise SystemExit(cli())
