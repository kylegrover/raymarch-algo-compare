import argparse
import csv
import math
import time
import os
from datetime import datetime

import numpy as np
import pandas as pd
from raymarching_benchmark.config import RenderConfig, MarchConfig
from raymarching_benchmark.gpu.runner import run_gpu_benchmark
from raymarching_benchmark.scenes.catalog import get_all_scenes


def compute_warp_divergence(iter_map: np.ndarray, block_size=(8, 4)) -> float:
    """Proxy for warp divergence: mean stddev of iteration counts across blocks."""
    h, w = iter_map.shape
    bx, by = block_size
    stds = []
    for y in range(0, h, by):
        for x in range(0, w, bx):
            block = iter_map[y : min(y + by, h), x : min(x + bx, w)].ravel()
            if block.size == 0:
                continue
            stds.append(float(np.std(block)))
    return float(np.mean(stds)) if stds else 0.0


def run_single_gpu(scene_name: str, strat_name: str, rc: RenderConfig, mc: MarchConfig):
    res = run_gpu_benchmark(scene_name, strat_name, rc, mc)
    if res is None:
        return None
    pixels = res['pixels']
    render_time_s = res['render_time_s']

    # iterations encoded in G channel as fraction of maxIterations
    gpu_iters = pixels[..., 1] * mc.max_iterations
    mean_iters = float(np.mean(gpu_iters))
    p95 = float(np.percentile(gpu_iters, 95))
    warp_div = compute_warp_divergence(gpu_iters)
    time_per_ray_us = (render_time_s / (rc.width * rc.height)) * 1e6

    return {
        'mean_iters': mean_iters,
        'p95_iters': p95,
        'warp_divergence': warp_div,
        'time_per_ray_us': time_per_ray_us,
        'render_time_s': render_time_s,
        'pixels': pixels,
    }


def tune_segment_grid(scenes: list[str], rc: RenderConfig, mc: MarchConfig,
                      kappa_values: list[float], min_step_fracs: list[float], out_dir: str):
    rows = []
    combos = [(k, m) for k in kappa_values for m in min_step_fracs]
    total = len(scenes) * len(combos)
    idx = 0
    for scene in scenes:
        for kappa, min_frac in combos:
            idx += 1
            mc.kappa = float(kappa)
            mc.min_step_fraction = float(min_frac)
            print(f"Tuning [{idx}/{total}] {scene} kappa={kappa} min_frac={min_frac}")
            out = run_single_gpu(scene, 'Segment', rc, mc)
            if out is None:
                continue
            rows.append({
                'scene': scene,
                'kappa': kappa,
                'min_step_fraction': min_frac,
                'mean_iters': out['mean_iters'],
                'p95_iters': out['p95_iters'],
                'warp_divergence': out['warp_divergence'],
                'time_per_ray_us': out['time_per_ray_us'],
            })
    df = pd.DataFrame(rows)
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    csv_path = os.path.join(out_dir, f'segment_tune__{ts}.csv')
    df.to_csv(csv_path, index=False)

    # Choose best by mean_iters then warp_divergence as tie-breaker
    best = df.sort_values(['mean_iters', 'warp_divergence']).iloc[0]
    print('\nTUNING RESULT — best combo:')
    print(best.to_string())

    # Append short summary to REPORT.md if present
    report_path = os.path.join(out_dir, 'REPORT.md')
    summary = f"\n### GPU segment tuning ({ts})\n- scene set: {', '.join(scenes)}\n- best: kappa={best.kappa}, min_step_fraction={best.min_step_fraction}\n- mean_iters={best.mean_iters:.2f}, warp_div={best.warp_divergence:.2f}\n\n"
    if os.path.exists(report_path):
        with open(report_path, 'a', encoding='utf-8') as f:
            f.write(summary)

    return csv_path, best


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser(prog='gpu-confirm', description='GPU validation & tuning for raymarch-bench')
    p.add_argument('--width', type=int, default=160)
    p.add_argument('--height', type=int, default=120)
    p.add_argument('--tune', action='store_true', help='Run segment tuning grid-search')
    p.add_argument('--out-dir', type=str, default='results')
    args = p.parse_args(argv)

    rc = RenderConfig(width=args.width, height=args.height)
    mc = MarchConfig()

    # Load CPU results for comparison
    cpu_results_path = os.path.join(args.out_dir, 'matrix_iteration_mean.csv')
    if not os.path.exists(cpu_results_path):
        print(f"Error: CPU results not found at {cpu_results_path}. Run CPU benchmark first.")
        return
    cpu_df = pd.read_csv(cpu_results_path, index_col=0)

    scenes = get_all_scenes()
    strategies = ["Standard", "Overstep-Bisect", "Relaxed(ω=1.2)", "Segment", "Enhanced", "AR-ST"]

    comparison_data = []

    if args.tune:
        tune_scenes = ["Thin Planes Stack", "Pillar Forest", "Bad Lipschitz Sphere"]
        kappa_values = [1.2, 1.5, 2.0, 3.0]
        min_step_fracs = [1e-4, 1e-3, 1e-2]
        csv_path, best = tune_segment_grid(tune_scenes, rc, mc, kappa_values, min_step_fracs, args.out_dir)
        print(f"Saved tuning CSV: {csv_path}")
        return

    print("="*60)
    print("GPU Ray Marching Benchmark Confirmation")
    print("="*60)

    for scene in scenes:
        scene_name = scene.name
        if scene_name not in cpu_df.index:
            continue

        print(f"\n>> Validating Scene: {scene_name}")

        for strat_name in strategies:
            if strat_name not in cpu_df.columns:
                continue

            cpu_val = cpu_df.loc[scene_name, strat_name]

            try:
                out = run_single_gpu(scene_name, strat_name, rc, mc)
                if out is None:
                    continue

                gpu_mean = out['mean_iters']
                diff = gpu_mean - cpu_val
                pct_diff = (diff / cpu_val * 100) if cpu_val != 0 else 0
                time_us = out['time_per_ray_us']
                warp_div = out['warp_divergence']

                match_status = 'OK' if abs(pct_diff) < 5.0 else 'DRIFT'
                if abs(diff) < 0.1:
                    match_status = 'EXACT'

                print(f"  {strat_name:15} | CPU: {cpu_val:6.2f} | GPU: {gpu_mean:6.2f} | Diff: {diff:+6.2f} ({pct_diff:+.1f}%) | {time_us:6.2f} us/ray | WD: {warp_div:.2f} | [{match_status}]")

                comparison_data.append({
                    'Scene': scene_name,
                    'Strategy': strat_name,
                    'CPU_Mean': cpu_val,
                    'GPU_Mean': gpu_mean,
                    'Diff': diff,
                    'Time_us_per_ray': time_us,
                    'Warp_divergence': warp_div,
                    'Status': match_status,
                })
            except Exception as e:
                print(f"  {strat_name:15} | FAILED: {e}")

    comp_df = pd.DataFrame(comparison_data)

    # Save GPU validation CSV for later analysis
    if comp_df.empty:
        print("No data collected.")
        return

    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    out_csv = os.path.join(args.out_dir, f'gpu_validation__{ts}.csv')
    comp_df.to_csv(out_csv, index=False)
    print(f"\nSaved GPU validation CSV: {out_csv}")

    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    status_counts = comp_df['Status'].value_counts()
    print(status_counts)
    avg_abs_diff = comp_df['Diff'].abs().mean()
    print(f"\nAverage Absolute Iteration Drift: {avg_abs_diff:.4f}")
    if avg_abs_diff < 1.0:
        print("\nRESULT: SUCCESS! GPU results statistically match CPU results.")
    else:
        print("\nRESULT: WARNING! Significant drift detected. Check GLSL implementation details.")

    # Append short summary to REPORT.md
    report_path = os.path.join(args.out_dir, 'REPORT.md')
    summary = f"\n### GPU validation ({ts})\n- mean absolute iteration drift: {avg_abs_diff:.3f}\n- CSV: {os.path.basename(out_csv)}\n\n"
    if os.path.exists(report_path):
        with open(report_path, 'a', encoding='utf-8') as f:
            f.write(summary)
    else:
        print(f"Note: {report_path} not found; GPU summary not appended to report.")


if __name__ == "__main__":
    main()
