# Ray Marching Algorithm Benchmark

A modular framework for benchmarking raymarching strategies on a variety of SDF scenes.

### Strategies 

- Standard
- Relaxed
- Heuristic-Auto-Relaxed
- Slope-Auto-Relaxed
- Enhanced
- Curvature-aware
- Overstep-Bisect
- Adaptive-Hybrid
- Segment (Lipschitz-aware)


- **14 Test Scenes** covering smooth geometry, sharp edges, thin/tunneling features, grazing angles, fractals, and intentionally invalid Lipschitz bounds.
- **Metrics**: iterations, hit rate, iteration distribution (p95/max), time-per-ray (CPU & optional GPU), and a warp-divergence proxy.
- **Outputs**: per-run images, CSV matrices, and an aggregated `REPORT.md` with comparative charts.


## Requirements & quick install (Windows-friendly) ⚙️
- Python: **3.9+** (see `pyproject.toml`).
- Optional for GPU measurements: a working OpenGL driver and the `moderngl`/`glcontext` stack.

Recommended (dev) setup:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .[dev]
```

If you prefer a minimal runtime install:

```powershell
pip install -r requirements.txt
```

Note: the project declares precise versions in `pyproject.toml` (use that for reproducible installs).

---

## Quickstart — run a benchmark ▶️
(Primary workflow uses the `uv` task-runner; a Python fallback is provided below.)

Run a single CPU benchmark (recommended — uv):

```bash
uv run -m raymarching_benchmark --scene Sphere --strategy Standard --width 160 --height 120
```

Run the full comparative matrix (uv — CPU + optional GPU validation):

```bash
uv run -m raymarching_benchmark --scene all --strategy all --width 80 --height 60 --json summary.json
```

GPU-focused run (uv shortcut to 1080p):

```bash
uv run -m raymarching_benchmark --scene Sphere --strategy Standard --gpu-1080p --gpu-warmup 3 --gpu-repeats 7
```

Advanced / fallback — direct Python entrypoint (useful in CI or when `uv` is not available):

```bash
python -m raymarching_benchmark --scene Sphere --strategy Standard --width 160 --height 120
```

```bash
python -m raymarching_benchmark --scene all --strategy all --width 80 --height 60 --json summary.json
```

---

## Where outputs go & how to inspect results 📁
- Default output folder: `results/`.
- Per-run outputs: `results/<Scene>__<Strategy>__<timestamp>/` (contains `iterations.png`, `inv_depth.png`, `hit_map.png`, `depth_map.npy`, `stats.json`).
- Aggregate outputs (comparisons & matrices): `results/REPORT.md`, `matrix_iteration_mean.csv`, `matrix_time_per_ray_us.csv`, `matrix_hit_rate.csv`, etc.

---

## GPU notes 🖥️
- GPU timing is **best-effort**: runner attempts a GPU validation for each scene/strategy and will gracefully fall back to CPU-only if OpenGL/context creation or shaders fail.
- Flags: `--gpu-width`, `--gpu-height`, `--gpu-1080p`, `--gpu-warmup`, `--gpu-repeats`.
- Requirements: up-to-date GPU drivers and an OpenGL-compatible environment. If you see OpenGL/context errors, try running the CPU path (no extra action needed).

---

## Development & tests 🔧
Run the fast smoke/test suite locally:

```bash
# recommended
uv run -m pytest -q
# or single test
uv run -m pytest tests/test_smoke.py -q
```

For very fast iteration during development:

```bash
uv run -m raymarching_benchmark --width 16 --height 12
```

---

## Project layout (key modules) 🔎
- `raymarching_benchmark/core/` — math primitives, `Vec3`, `Ray`, `Camera`.
- `raymarching_benchmark/scenes/` — scene catalog and SDF primitives.
- `raymarching_benchmark/strategies/` — strategy implementations and registry (`list_strategies()`).
- `raymarching_benchmark/metrics/` — collectors and analyzers.
- `raymarching_benchmark/visualization/` — charts, heatmaps, and report writers.

---

## Adding content (quick checklist)
- Add a Strategy: implement `MarchStrategy` in `strategies/` and register it in `strategies/__init__.py` (add a readable key to `STRATEGIES`).
- Add a Scene: add an `SDFScene` subclass to `scenes/catalog.py` and include it in `get_all_scenes()`.
- Add tests: include a focused unit/integration test under `tests/` that reproduces the behavior.

---

## Troubleshooting & tips
- If the GPU run fails: confirm drivers and try `--no-save-images` (helps when headless). The runner will not abort the whole run for GPU failures.
- If you get unexpected iteration counts: try lowering `--width/--height` and compare heatmaps in `results/`.
- Use `--json summary.json` to produce a compact machine-readable summary for CI.