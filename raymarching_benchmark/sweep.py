"""GPU-primary sweep engine.

Runs scene x strategy x LEVEL, where the level axis is either:
  * ``budget``   — sweep the iteration budget at a fixed epsilon (default), or
  * ``residual`` — sweep the target residual epsilon (= hit_threshold) under a
                   fixed iteration cap (matched-residual, SWEEP_PLAN §B): march
                   every method to the same closeness-to-surface and report what
                   reaching it COST. Cost is the dependent variable.

Per run it measures three cost axes (never collapsed to one):
  * runtime         (median GPU ms via GL timer query — hardware cost)
  * SDF evals/ray   (workload; under-charges adaptive methods — see caveat)
  * iter_divergence (neighbor iteration spread — efficiency / warp divergence)
plus the iteration distribution (diagnostic) and a did-not-converge fraction,
and scores accuracy against the cached dense-march oracle (IoU primary; depth
error + SSIM secondary).

Appends one JSONL row per run; resumable (skips combos already present) and
crash-tolerant (a failed combo writes a status="error" row and the sweep
continues).

    uv run python -m raymarching_benchmark.sweep --out sweep.jsonl --res 384
    uv run python -m raymarching_benchmark.sweep --mode residual --out sweep.jsonl
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
from .gpu.groundtruth import (
    _resolve_scene, reference_render_config,
    dense_march_config, dense_march_params, DENSE_MARCH_STRATEGY_ID, DENSE_MIN_STEP,
)
from .metrics.scoring import score_capture
from .data.dataset import JsonlDataset
from .data.provenance import provenance, config_hash


DEFAULT_SCENES = ["Sphere", "Grazing Plane", "Thin Torus", "Mandelbulb"]
DEFAULT_BUDGETS = [32, 64, 128, 256, 512]
# Matched-residual axis (SWEEP_PLAN §B): march every method to the same
# closeness-to-surface epsilon (= hit_threshold) under a fixed high iteration
# cap, and report what reaching it COST. Accuracy is ~held constant by epsilon;
# cost (evals / iters / GPU ms) becomes the dependent variable — "what does
# equal accuracy cost." Iteration count is kept only as a diagnostic.
DEFAULT_EPSILONS = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
RESIDUAL_CAP = 2048   # hard iteration ceiling; rays that hit it = "did not converge"
# NOTE on naming (SWEEP_PLAN §I): ids 2 and 5 use NAIVE over-relaxation —
# t += d*ω with *post-hoc* overshoot backup, which cannot catch a step that
# tunnels cleanly through a thin feature. They are labeled "Naive-..." so the
# names mean what the field expects. id 8 (Safe-Relaxed) is the faithful
# Keinert et al. 2014 safe over-relaxation with the *predictive* disjoint-sphere
# fallback. id 9 (dense_march) is the calibration oracle and is intentionally
# NOT a competitor here.
STRATEGIES: List[Tuple[int, str]] = [
    (0, "Standard"), (1, "Overstep-Bisect"), (2, "Naive-Relaxed"), (3, "Segment"),
    (4, "Enhanced"), (5, "Naive-Auto-Relaxed"), (6, "Skipping-Spheres"), (7, "RevAA"),
    (8, "Safe-Relaxed"),
]
REFERENCE_TAG = f"Dense-March@minStep={DENSE_MIN_STEP}"


def _force_utf8():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _dist(a: np.ndarray) -> Dict[str, float]:
    a = a.astype(np.float64).ravel()
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def _divergence_proxy(iters2d: np.ndarray) -> float:
    """Proxy for warp divergence (SWEEP_PLAN §C): mean absolute difference in
    iteration count between spatially adjacent rays. Neighboring pixels map to
    neighboring GPU threads, so a high value means adjacent threads take wildly
    different path lengths — the thing that makes an adaptive method that wins on
    eval-count lose on wall-clock. Hardware-agnostic, computed from the capture."""
    a = iters2d.astype(np.float64)
    dx = np.abs(a[:, 1:] - a[:, :-1])
    dy = np.abs(a[1:, :] - a[:-1, :])
    return float((dx.sum() + dy.sum()) / (dx.size + dy.size))


def build_levels(mode: str, *, budgets: List[int], epsilons: List[float],
                 cap: int, hit_threshold: float
                 ) -> List[Tuple[float, MarchConfig, Dict]]:
    """Return the sweep levels as (level_value, MarchConfig, config_extra).

    ``budget`` mode sweeps max_iterations at a fixed epsilon; ``residual`` mode
    sweeps epsilon (= hit_threshold) at a fixed iteration cap. ``config_extra``
    is merged into the row config (and thus the config_hash), so the two axes
    never collide in the dataset.
    """
    levels: List[Tuple[float, MarchConfig, Dict]] = []
    if mode == "budget":
        for b in budgets:
            mc = MarchConfig(max_iterations=int(b), hit_threshold=hit_threshold)
            levels.append((float(b), mc, {
                "sweep_axis": "budget",
                "max_iterations": int(b),
                "hit_threshold": hit_threshold,
                "epsilon": hit_threshold,
            }))
    elif mode == "residual":
        for eps in epsilons:
            mc = MarchConfig(max_iterations=int(cap), hit_threshold=float(eps))
            levels.append((float(eps), mc, {
                "sweep_axis": "residual",
                "max_iterations": int(cap),
                "hit_threshold": float(eps),
                "epsilon": float(eps),
            }))
    else:
        raise ValueError(f"unknown sweep mode {mode!r}")
    return levels


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


def run_sweep(out_path: str, scene_names: List[str],
              levels: List[Tuple[float, MarchConfig, Dict]],
              res: int, warmup: int, repeats: int, *, mode: str = "budget") -> None:
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
        # Score against the calibrated dense-march oracle (validated to ~1e-7
        # depth / IoU 1.0 on analytic scenes), not v1's plain high-budget trace.
        refs[name] = runner.capture(scene_id, DENSE_MARCH_STRATEGY_ID, rc,
                                    dense_march_config(), lipschitz=lip,
                                    params=dense_march_params())
        scene_meta[name] = (scene_id, scene, rc, lip)

    combos = [(n, sid, lbl, lvl) for n in scene_meta
              for (sid, lbl) in STRATEGIES for lvl in levels]
    total = len(combos)
    done = skipped = errors = 0
    t0 = time.time()
    print(f"Sweep[{mode}]: {total} runs ({len(scene_meta)} scenes x "
          f"{len(STRATEGIES)} strategies x {len(levels)} levels)")
    print(f"Resuming dataset with {len(ds)} existing rows -> {out_path}\n")

    for i, (name, sid, label, level) in enumerate(combos):
        level_value, mc, extra = level
        scene_id, scene, rc, lip = scene_meta[name]
        config = {
            "scene": name,
            "scene_category": scene.category,
            "strategy": label,
            "strategy_id": sid,
            "backend": "gpu",
            "resolution": res,
            "reference": REFERENCE_TAG,
            **extra,
        }
        ch = config_hash(config)
        if ds.has(ch):
            skipped += 1
            continue

        cap_iters = int(extra["max_iterations"])
        try:
            timing = measure_ms(runner, scene_id, sid, rc, mc, lip, warmup, repeats)
            cap = runner.capture(scene_id, sid, rc, mc, lipschitz=lip)
            sc = score_capture(cap, refs[name])
            iter_ratio = cap["geom"][..., 1]
            iters = iter_ratio * cap_iters
            # A ray that exhausts the iteration cap without a hit did NOT reach
            # epsilon — the matched-residual "did-not-converge" signal.
            maxed = iter_ratio >= 0.999
            dnc = float(np.logical_and(maxed, ~cap["hit"]).mean())
            row = {
                "config_hash": ch,
                "status": "ok",
                "config": config,
                "provenance": prov,
                "perf": {
                    # Three cost axes (never collapse to one — they disagree):
                    #  ms_*            : GPU wall-clock (GL timer) — hardware cost
                    #  evals           : SDF evaluations / ray — workload, but an
                    #                    eval is NOT constant-cost across methods,
                    #                    so this under-charges adaptive/relaxed
                    #                    methods; always read it next to ms.
                    #  iter_divergence : neighbor iteration spread — efficiency;
                    #                    explains eval-vs-ms rank disagreements.
                    **timing,
                    "iter": _dist(iters),
                    "evals": _dist(cap["evals"]),
                    "iter_divergence": _divergence_proxy(iters),
                    "did_not_converge": dnc,
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
    p.add_argument("--mode", type=str, default="budget", choices=["budget", "residual"],
                   help="budget: sweep max_iterations at fixed epsilon. "
                        "residual: sweep epsilon (matched-residual) at a fixed cap.")
    p.add_argument("--budgets", type=str, default=",".join(map(str, DEFAULT_BUDGETS)))
    p.add_argument("--epsilons", type=str, default=",".join(map(str, DEFAULT_EPSILONS)),
                   help="Target residuals for --mode residual.")
    p.add_argument("--cap", type=int, default=RESIDUAL_CAP,
                   help="Iteration ceiling for --mode residual (did-not-converge above it).")
    p.add_argument("--res", type=int, default=384)
    p.add_argument("--hit-threshold", type=float, default=1e-4,
                   help="Fixed epsilon for --mode budget.")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=8)
    args = p.parse_args(argv)

    scene_names = [s.strip() for s in args.scenes.split(",") if s.strip()]
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    epsilons = [float(e) for e in args.epsilons.split(",") if e.strip()]
    levels = build_levels(args.mode, budgets=budgets, epsilons=epsilons,
                          cap=args.cap, hit_threshold=args.hit_threshold)
    run_sweep(args.out, scene_names, levels, args.res, args.warmup, args.repeats,
              mode=args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
