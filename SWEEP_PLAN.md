# Sweep Engine Plan — "Churn for hours → analyzable dataset"

> Created 2026-06-04. Supersedes `PLAN.md` (which documents the now-complete
> CPU/GPU benchmark foundation — Phases 1–5 there are done).

## Goal

Let the GPU run unattended for hours over a large grid of
`scene × strategy × parameters × resolution`, scoring every run against a
ground-truth reference, and append results to a durable dataset we can later
analyze to answer: **which technique+settings win on which type of scene, at
what accuracy cost.**

## Decisions locked in

- **Ground truth:** high-budget reference march (Standard tracing at very high
  `max_iterations` + tiny `hit_threshold`), per scene+camera+resolution, cached.
- **Dataset format:** append-only **JSONL**, one line per run.
- **Engine:** GPU-primary. CPU march becomes optional spot-check, not per-combo.

## The reframe

Today we have a benchmark *reporter* (fixed config → human-readable tables).
We need an experiment *engine* + *dataset*. The 6 gaps below are what separate
the two. Phase 1 makes "best" meaningful; Phase 2 makes it scale; Phase 3 is the
payoff.

---

## Current state (what's already built)

- Dual CPU (`metrics/`) + GPU (`gpu/runner.py`, moderngl headless) backends with
  Python/GLSL strategy parity (11 strategies, dispatched by id).
- 14 scenes with `category()`, `suggested_camera()`, `known_lipschitz_bound()`.
- Per-run telemetry: iteration distribution, hit rate, warp-divergence proxy,
  GPU median timing, per-pixel iteration/depth/hit maps.
- Reporting: CSV matrices, markdown, charts.

### The 6 gaps

1. **No sweep engine** — `main.py:203` loops `scene × strategy` only; every
   `MarchConfig` knob is a single fixed value per invocation.
2. **Params can't reach the GPU** — `omega`/`kappa` hardcoded in
   `runner.py:97-100`; skipping-spheres `margin` hardcoded in GLSL. Sweeping a
   parameter currently can't affect the GPU path.
3. **No ground-truth oracle** — strategies self-report hits, so "best" can
   reward fast-but-wrong algorithms (cf. Skipping-Spheres 0.115 vs 0.458 hit
   rate on Grazing Plane; RevAA ≡ Standard on Bad Lipschitz).
4. **No durable dataset** — results are in-memory per invocation, then rendered
   to reports. No append-only table, no provenance.
5. **No resumability** — a multi-hour run can't skip completed combos or survive
   a crash (and `main.py` already dies on the cp1252 console bug mid-run).
6. **CPU is the wrong primary engine** — CPU per-pixel Python march dominates the
   hours; GPU is only a validation sidecar (`main.py:209`).

---

## Phase 0 — Robustness prerequisites (small, unblock everything)

- **0.1 Fix the cp1252 crash.** Make `visualization/tables.py` output
  encoding-safe (reconfigure stdout to utf-8 at entry, or ASCII-only table
  glyphs). A sweep can't run for hours if printing kills it.
- **0.2 GPU timing rigor (optional but cheap).** Consider `GL_TIMESTAMP` query
  objects instead of `perf_counter` + `ctx.finish()` to exclude dispatch/sync
  overhead. Keep median-over-repeats.

## Phase 1 — Make "best" meaningful (ground truth + scored dataset)

Deliver value even with today's small loop.

- **1.1 `gpu/groundtruth.py` — reference oracle.**
  - `reference_maps(scene_name, camera, resolution) -> {depth_map, hit_map}`.
  - Renders Standard (id 0) at `max_iterations ≈ 4096`, `hit_threshold ≈ 1e-6`,
    large `max_distance`, via the existing GPURunner.
  - Cache to disk as `.npy` keyed by a hash of (scene, camera, resolution,
    reference-config). Compute once per scene/resolution, reuse across all runs.
- **1.2 `metrics/scoring.py` — accuracy vs reference.**
  Given a run's `depth_map`/`hit_map` and the reference, compute:
  - `false_hit_rate` (run hit where reference miss),
  - `false_miss_rate` (run miss where reference hit),
  - `depth_rmse` / `depth_p95_err` over pixels both agree are hits,
  - `silhouette_err` (disagreement concentrated near edges — optional).
  These are the columns that let "fast" be weighed against "correct."
- **1.3 `data/dataset.py` — JSONL writer + run schema.**
  One line per run (see schema below). Append-only, flushed after each row.
- **1.4 `data/provenance.py`** — capture git SHA, dirty flag, GPU
  renderer/vendor string (from moderngl `ctx.info`), driver, hostname, UTC
  timestamp, python/moderngl versions, and a `config_hash` (stable hash of the
  full param set) used both as row id and for resume.

## Phase 2 — Scale it (sweep engine + GPU param threading + resumable runner)

- **2.1 Un-hardcode GLSL params → uniforms.** Promote `omega`, `kappa`,
  `margin` (skipping-spheres), `initial_relaxation`, `min_step_fraction`, etc.
  to uniforms in the relevant GLSL functions, with current values as defaults.
- **2.2 Extend `GPURunner.render` to accept a `params: dict`** and set those
  uniforms, so any swept parameter actually reaches the shader.
- **2.3 Sweep spec.** A YAML/py file describing the grid, e.g.:
  ```yaml
  scenes: all            # or [Sphere, Grazing Plane, ...]
  strategies: all        # or [Standard, Segment, ...]
  resolutions: [[1920,1080]]
  repeats: 7
  param_grid:            # cartesian product, per-strategy overrides allowed
    hit_threshold: [1e-3, 1e-4]
    max_iterations: [256, 512]
    Segment: { kappa: [1.5, 2.0, 3.0] }
    Relaxed: { omega: [1.2, 1.6, 1.9] }
    Skipping-Spheres: { margin: [0.02, 0.05, 0.1] }
  ```
- **2.4 `sweep.py` — the engine.**
  - Expand spec → list of run configs (cartesian product, with per-strategy
    param subsets so we don't test `kappa` on Standard).
  - **GPU-primary:** measure timing + iteration/depth/hit maps on GPU; score
    against cached reference; write JSONL row.
  - **Resumable:** on start, read existing JSONL, collect completed
    `config_hash`es, skip them.
  - **Crash-tolerant:** wrap each run in try/except; on failure write a row with
    `status: "error"` + message and continue.
  - **Progress/ETA:** count done/total, running rate, est. completion.
  - CPU march only when `--validate-cpu` sampling flag is set (e.g. 1 in N).

## Phase 3 — The payoff (analysis layer)

- **3.1 `analyze.py` / notebook.** Load JSONL → pandas/Polars.
- **3.2 Per scene-category winners.** Group by `scene.category`; rank strategies
  by a chosen objective (e.g. GPU time at accuracy ≤ ε).
- **3.3 Speed/accuracy Pareto front** per scene or category — the headline
  artifact: who is on the frontier, who is dominated.
- **3.4 Parameter sensitivity** — for each strategy, how its winning params
  shift across scene types.

---

## JSONL row schema (draft)

```json
{
  "config_hash": "ab12…",
  "status": "ok",
  "provenance": {
    "git_sha": "dbe6788", "git_dirty": true,
    "gpu": "NVIDIA …", "driver": "…", "host": "…",
    "timestamp_utc": "2026-06-04T…", "moderngl": "5.x"
  },
  "config": {
    "scene": "Grazing Plane", "scene_category": "stress_grazing",
    "strategy": "Segment", "backend": "gpu",
    "resolution": [1920, 1080], "repeats": 7,
    "params": { "max_iterations": 512, "hit_threshold": 1e-4, "kappa": 2.0 }
  },
  "perf": {
    "gpu_ms_median": 4.2, "gpu_ms_p95": 4.9,
    "iter_mean": 11.6, "iter_p95": 20, "iter_max": 48,
    "warp_divergence": 1.81, "sample_count": 35712000
  },
  "accuracy": {
    "false_hit_rate": 0.001, "false_miss_rate": 0.34,
    "depth_rmse": 0.012, "depth_p95_err": 0.04, "hit_rate": 0.115
  }
}
```

---

## Suggested file layout (new)

```
raymarching_benchmark/
├── sweep.py                 # Phase 2 — engine entrypoint (CLI: --spec sweep.yaml)
├── gpu/groundtruth.py       # Phase 1 — reference maps + disk cache
├── metrics/scoring.py       # Phase 1 — accuracy vs reference
├── data/
│   ├── dataset.py           # Phase 1 — JSONL writer + schema
│   └── provenance.py        # Phase 1 — git/gpu/host/config_hash
└── analyze.py               # Phase 3 — load JSONL → pareto / winners
sweeps/
└── example.yaml             # a sample sweep spec
```

## Build order (dependency-correct)

0.1 → 1.4 (provenance/hash) → 1.3 (JSONL) → 1.1 (ground truth) → 1.2 (scoring)
→ 2.1/2.2 (GPU params) → 2.3/2.4 (spec + engine) → 3.x (analysis).

Phase 1 alone, run over the existing small grid, already produces a *scored*
dataset — the first time "best" means "fast **and** correct."
