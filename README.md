# Ray Marching Algorithm Benchmark

A modular framework for benchmarking and analyzing different ray marching strategies across a variety of SDF (Signed Distance Field) stress-test scenes.

## Features

- **7 Strategies**: Standard, Relaxed, Auto-Relaxed (AR-ST), Enhanced, Overstep-Bisect, Adaptive-Hybrid, and Segment Tracing.
- **14 Test Scenes**: Covering smooth surfaces, sharp edges, thin features, grazing angles, fractals (Mandelbulb, Menger Sponge), and invalid Lipschitz bounds.
- **Rich Metrics**: Iteration counts, hit rates, accuracy (SDF error), time per ray, and Warp Divergence (workload variance).
- **Visualization**: Heatmaps (iteration, hit, depth), comparative bar charts, speed-vs-accuracy scatter plots, and side-by-side tiled comparisons.
- **Comprehensive Reporting**: Automatically generates Markdown reports with embedded charts and tables, plus CSV data for external analysis.

## Quickstart

Ensure you have `uv` installed.

### 1. Install dependencies
```bash
uv sync
```

### 2. Run a single benchmark
```bash
uv run -m raymarching_benchmark --scene Sphere --strategy Standard --width 160 --height 120
```

### 3. Run a comparative matrix
Compare all strategies across all scenes:
```bash
uv run -m raymarching_benchmark --scene all --strategy all --width 80 --height 60 --json summary.json
```

### 4. View Results
Results are saved in `results/` by default, including:
- `REPORT.md`: A summary of the whole benchmark session.
- `compare__<Scene>.png`: Side-by-side heatmaps for each scene.
- `chart_iterations.png`: Performance bar chart.
- `matrix_iteration_mean.csv`: Raw data matrix.
- `stats.json`: Individual run stats.

## Project Structure

- `core/`: Fundamental math (Vec3), Ray, and Camera logic.
- `scenes/`: SDF primitives, combinators, and the test scene catalog.
- `strategies/`: Modular implementations of ray marching algorithms.
- `metrics/`: Collection and statistical analysis components.
- `visualization/`: Charting, heatmap generation, and report writers.

## Development

### Running Tests
```bash
uv run python -m pytest tests/test_smoke.py
```

### Adding a new Strategy
1. Create a new subclass of `MarchStrategy` in `strategies/`.
2. Register it in `strategies/__init__.py`.

### Adding a new Scene
1. Create a new subclass of `SDFScene` in `scenes/catalog.py`.
2. Add it to the `get_all_scenes()` list.

- Run the MVP (small, fast):
  - uv run -m raymarching_benchmark --width 64 --height 48

- Inspect outputs:
  - results/<Scene>__<Strategy>__<timestamp>/iterations.png
  - results/<Scene>__<Strategy>__<timestamp>/inv_depth.png
  - results/<Scene>__<Strategy>__<timestamp>/depth_map.npy

Goals

- Provide a compact, testable baseline to benchmark and compare ray-marching strategies.
- Iterate on strategies and metrics without needing GPU or complex setup.

Development notes

- Use `uv run -m raymarching_benchmark --width 16 --height 12` for very fast feedback.
- Tests: `pytest -q` (optional; MVP focuses on runnable baseline).

Contributing

Open a PR with focused changes and include a short reproducer (command and expected output).