"""GPU-primary sweep engine.

Runs scene x strategy x iteration-budget, measuring three fairness axes per run:
  * runtime   (median GPU ms)
  * steps     (iteration count)
  * SDF evals (true map() work)
and scoring accuracy against a cached high-budget ground-truth reference
(raw depth + SSIM on depth/normal/lit color + hit-mask agreement).

Appends one JSONL row per run; resumable (skips combos already present) and
crash-tolerant (a failed combo writes a status="error" row and the sweep
continues). Sweeping the budget is what lets the analysis layer compare methods
at equal steps, equal evals, OR equal runtime.

    uv run python -m raymarching_benchmark.sweep --out sweep.jsonl --res 384
"""
from __future__ import annotations
import sys
import time
import argparse
import statistics
from typing import Dict, List, Tuple

import numpy as np

from .config import RenderConfig, MarchConfig
from .gpu.runner import GPURunner
from .gpu.groundtruth import _resolve_scene, reference_render_config, reference_march_config
from .metrics.scoring import score_capture
from .data.dataset import JsonlDataset
from .data.provenance import provenance, config_hash


DEFAULT_SCENES = ["Sphere", "Grazing Plane", "Thin Torus", "Mandelbulb"]
DEFAULT_BUDGETS = [32, 64, 128, 256, 512]
STRATEGIES: List[Tuple[int, str]] = [
    (0, "Standard"), (1, "Overstep-Bisect"), (2, "Relaxed"), (3, "Segment"),
    (4, "Enhanced"), (5, "Heuristic-Auto-Relaxed"), (6, "Skipping-Spheres"), (7, "RevAA"),
]
REFERENCE_TAG = "Standard@4096/1e-6"


def _force_utf8():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _dist(a: np.ndarray) -> Dict[str, float]:
    a = a.astype(np.float64).ravel()
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def measure_ms(runner: GPURunner, scene_id: int, sid: int, rc: RenderConfig,
               mc: MarchConfig, lip: float, warmup: int, repeats: int) -> Dict[str, float]:
    for _ in range(warmup):
        runner.render(scene_id, sid, rc, mc, lipschitz=lip)
    ts = []
    for _ in range(repeats):
        _, t = runner.render(scene_id, sid, rc, mc, lipschitz=lip)
        ts.append(t * 1000.0)
    ts.sort()
    return {"ms_median": float(statistics.median(ts)),
            "ms_min": float(ts[0]), "ms_max": float(ts[-1])}


def run_sweep(out_path: str, scene_names: List[str], budgets: List[int],
              res: int, hit_threshold: float, warmup: int, repeats: int) -> None:
    runner = GPURunner()
    prov = provenance(runner.gpu_info())
    ds = JsonlDataset(out_path)

    # Pre-capture references once per scene (cached for all budgets/strategies).
    refs: Dict[str, Dict] = {}
    scene_meta: Dict[str, Tuple[int, object, RenderConfig, float]] = {}
    print(f"Capturing {len(scene_names)} references @ {res}²...")
    for name in scene_names:
        scene_id, scene = _resolve_scene(name)
        if scene is None:
            print(f"  [skip] unknown scene {name!r}")
            continue
        rc = reference_render_config(scene, res, res)
        lip = scene.known_lipschitz_bound()
        refs[name] = runner.capture(scene_id, 0, rc, reference_march_config(), lipschitz=lip)
        scene_meta[name] = (scene_id, scene, rc, lip)

    combos = [(n, sid, lbl, b) for n in scene_meta
              for (sid, lbl) in STRATEGIES for b in budgets]
    total = len(combos)
    done = skipped = errors = 0
    t0 = time.time()
    print(f"Sweep: {total} runs ({len(scene_meta)} scenes x {len(STRATEGIES)} strategies x {len(budgets)} budgets)")
    print(f"Resuming dataset with {len(ds)} existing rows -> {out_path}\n")

    for i, (name, sid, label, budget) in enumerate(combos):
        scene_id, scene, rc, lip = scene_meta[name]
        config = {
            "scene": name,
            "scene_category": scene.category,
            "strategy": label,
            "strategy_id": sid,
            "backend": "gpu",
            "resolution": res,
            "max_iterations": budget,
            "hit_threshold": hit_threshold,
            "reference": REFERENCE_TAG,
        }
        ch = config_hash(config)
        if ds.has(ch):
            skipped += 1
            continue

        mc = MarchConfig(max_iterations=budget, hit_threshold=hit_threshold)
        try:
            timing = measure_ms(runner, scene_id, sid, rc, mc, lip, warmup, repeats)
            cap = runner.capture(scene_id, sid, rc, mc, lipschitz=lip)
            sc = score_capture(cap, refs[name])
            iters = cap["geom"][..., 1] * budget
            row = {
                "config_hash": ch,
                "status": "ok",
                "config": config,
                "provenance": prov,
                "perf": {
                    **timing,
                    "iter": _dist(iters),
                    "evals": _dist(cap["evals"]),
                },
                "accuracy": {
                    "hit_rate": float(cap["hit"].mean()),
                    "false_hit": sc["hit"]["false_hit_rate"],
                    "false_miss": sc["hit"]["false_miss_rate"],
                    "iou": sc["hit"]["iou"],
                    "depth_rmse": sc["depth"]["rmse"],
                    "depth_p95": sc["depth"]["p95"],
                    "normal_deg": sc["normal"]["mean_deg"],
                    "depth_ssim": sc["ssim"]["depth_ssim"],
                    "normal_ssim": sc["ssim"]["normal_ssim"],
                    "color_ssim": sc["ssim"]["color_ssim"],
                },
            }
            ds.append(row)
            done += 1
        except Exception as e:
            ds.append({"config_hash": ch, "status": "error", "config": config,
                       "provenance": prov, "error": f"{type(e).__name__}: {e}"})
            errors += 1

        if (i + 1) % 10 == 0 or (i + 1) == total:
            el = time.time() - t0
            rate = (done + errors) / el if el > 0 else 0
            remaining = total - skipped - done - errors
            eta = remaining / rate if rate > 0 else 0
            print(f"  [{i+1:4d}/{total}] done={done} skip={skipped} err={errors} "
                  f"| {rate:.1f}/s ETA {eta:5.0f}s")

    print(f"\nFinished: {done} new, {skipped} skipped, {errors} errors -> {out_path}")


def main(argv=None) -> int:
    _force_utf8()
    p = argparse.ArgumentParser(description="GPU-primary raymarching sweep -> JSONL.")
    p.add_argument("--out", type=str, default="sweep.jsonl")
    p.add_argument("--scenes", type=str, default=",".join(DEFAULT_SCENES))
    p.add_argument("--budgets", type=str, default=",".join(map(str, DEFAULT_BUDGETS)))
    p.add_argument("--res", type=int, default=384)
    p.add_argument("--hit-threshold", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=8)
    args = p.parse_args(argv)

    scene_names = [s.strip() for s in args.scenes.split(",") if s.strip()]
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    run_sweep(args.out, scene_names, budgets, args.res, args.hit_threshold,
              args.warmup, args.repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
