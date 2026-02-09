Ray Marching Algorithm Analytical Testing — MVP

Quickstart (uv)

- Create or refresh venv and install deps:
  - uv venv create --prompt .venv
  - uv add numpy
  - (optional) uv add pillow

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