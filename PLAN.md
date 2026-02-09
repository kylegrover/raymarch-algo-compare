# Ray Marching Algorithm Benchmark — Plan

> Last updated: 2026-02-08

## Current Status

We have a working single-strategy, single-scene CLI runner:
```
uv run -m raymarching_benchmark --width 64 --height 48
→ Strategy: Standard | Scene: Sphere — runs and produces stats.json + heatmap PNG
```

### What's Done (✅)

| Module | Status |
|--------|--------|
| Core math (`Vec3`, `Ray`, `Camera`) | Complete, working |
| Type system (`MarchResult`, `RayMarchStats`) | Complete (one bug — see below) |
| 14 SDF test scenes (smooth, sharp, thin, fractal, CSG, stress) | Complete |
| SDF primitives + combinators + transforms | Complete |
| All 7 strategy implementations | Complete (Standard, Overstep-Bisect, Adaptive Hybrid, Relaxed, Auto-Relaxed, Enhanced, Segment) |
| Per-ray metrics collection (`MetricsCollector`) | Working |
| Iteration heatmap PNG output | Working |
| `MetricsAnalyzer` (comparison logic) | Implemented |
| `print_comparison_tables()` | Implemented |
| `BenchmarkConfig` dataclass | Implemented |
| Scene `suggested_camera()` overrides | Implemented on all 14 scenes |
| Scene `known_lipschitz_bound()` | Implemented on all 14 scenes |

### What's Broken (🐛)

1. **`types.py` — `compute()` never assigns `self.hit_map`.**
   `compute()` creates a local `hit_map` array but forgets `self.hit_map = hit_map`.
   → Hit-map and depth-map output is silently skipped.

2. **`main.py` — `--strategy` flag is a no-op.**
   Strategy is hardcoded: `strategy = StandardSphereTracing()`.
   No strategy registry/dispatch exists.

### What's Implemented but Dead Code (🔌)

- `MetricsAnalyzer` — never instantiated or called by CLI
- `print_comparison_tables()` — never called
- `BenchmarkConfig` — never used (CLI builds `RenderConfig`/`MarchConfig` directly)
- Scene `suggested_camera()` — never consumed by `main.py`
- Scene `known_lipschitz_bound()` → `SegmentTracing.lipschitz` — never wired
- 6 of 7 strategies (everything except Standard)
- `visualization/charts.py` — stub only

### What's Missing Entirely (❌)

- **Strategy dispatch** — no registry, no `--strategy all` support
- **Matrix mode** — run all strategies × all scenes in one invocation
- **Comparative analysis output** — tables, rankings, per-scene best/worst
- **Charts** (`charts.py` is an empty stub)
- **Tests** — `tests/` directory is empty
- **`--json` multi-scene bug** — overwrites file each iteration when multiple scenes given
- **Warp divergence** in output (metric is computed but not surfaced)

---

## Phase 1 — Bug Fixes & Wiring (get existing code working)

1. **Fix `hit_map` bug in `types.py`** — add `self.hit_map = hit_map` in `compute()`.
2. **Build strategy registry** — add `get_strategy_by_name(name)` in `strategies/__init__.py` and wire `--strategy` flag in `main.py`.
3. **Wire scene `suggested_camera()`** — use scene's camera defaults when the user doesn't override.
4. **Wire `known_lipschitz_bound()`** → pass to `SegmentTracing` automatically.
5. **Fix `--json` multi-scene overwrite** — accumulate results, write once at end.

## Phase 2 — Matrix Mode & Comparative Analysis

6. **Add `--strategy all` and `--scene all`** — support running the full matrix.
7. **Wire `MetricsAnalyzer`** — accumulate results across all runs in a session.
8. **Wire `print_comparison_tables()`** — display after matrix run completes.
9. **Surface warp divergence** — include in stats.json and comparison tables.
10. **Improve console output** — progress bar or at least per-run status during matrix runs.

## Phase 3 — Visualization & Reporting

11. **Implement `charts.py`** — bar charts (iterations by strategy per scene), scatter plots (accuracy vs speed), heatmap grids.
12. **Side-by-side heatmap comparison** — iteration heatmaps for all strategies on one scene, tiled.
13. **Generate summary report** — markdown or HTML with tables + embedded charts.
14. **Save comparative results** — single JSON and/or CSV with full matrix data.

## Phase 4 — Validation & Polish

15. **Add smoke tests** — `tests/test_smoke.py` covering Vec3, Camera, each strategy on a simple scene.
16. **Add strategy correctness tests** — verify all strategies agree on hit/miss for known rays.
17. **Validate `SegmentTracing` iteration counting** — it currently counts SDF evaluations, not march steps. Decide on convention and document.
18. **Scene name matching** — replace fragile substring matching with exact-name lookup (+ alias support).
19. **README** — usage examples, sample output, architecture diagram.

## Phase 5 — Stretch Goals

20. **Performance profiling** — identify bottleneck (likely pure-Python SDF eval), consider numpy vectorization.
21. **Taichi GPU backend** — optional GPU acceleration for large resolutions.
22. **Additional scenes** — user-contributed scenes, parametric scene generators.
23. **Interactive mode** — pick scene/strategy from a menu, see live heatmap.

---

## Architecture Reference

```
raymarching_benchmark/
├── main.py                  # CLI entry point and orchestrator
├── config.py                # RenderConfig, MarchConfig, BenchmarkConfig
├── core/
│   ├── vec3.py              # Vec3 + utilities
│   ├── ray.py               # Ray(origin, direction)
│   ├── camera.py            # Perspective camera, ray generation
│   └── types.py             # MarchResult (per-ray), RayMarchStats (aggregate)
├── scenes/
│   ├── base.py              # Abstract SDFScene
│   ├── primitives.py        # SDF functions + combinators
│   └── catalog.py           # 14 test scenes
├── strategies/
│   ├── base.py              # Abstract MarchStrategy
│   ├── standard_sphere.py   # Standard sphere tracing
│   ├── overstep_bisect.py   # Overstep + bisection
│   ├── adaptive_hybrid.py   # Adaptive hybrid (mode switching)
│   ├── relaxed_sphere.py    # Fixed-omega relaxed
│   ├── auto_relaxed.py      # Auto-relaxed (AR-ST, EMA-based omega)
│   ├── enhanced_sphere.py   # Enhanced (planar extrapolation)
│   └── segment_tracing.py   # Segment tracing (Lipschitz bounds)
├── metrics/
│   ├── collector.py         # MetricsCollector — run rays, time, compute stats
│   └── analyzer.py          # MetricsAnalyzer — cross-strategy comparison
└── visualization/
    ├── heatmaps.py          # Iteration/hit/depth heatmap PNGs
    ├── tables.py            # Console comparison tables
    └── charts.py            # Comparative charts (stub)
```
