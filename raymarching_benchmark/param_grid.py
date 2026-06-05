"""Per-strategy tunable-parameter grid (SWEEP_PLAN §F).

Now that ω / segment growth κ / fattening margin are shader uniforms threaded
through ``GPURunner.{render,capture}(params=...)``, the sweep can brute-force
each method's parameter space and from the one dataset extract (a) each method's
best-tuned params per scene — its *fair shot*, and (b) its parameter
sensitivity. Strategies with no meaningful tunable run a single ``default``
config so they still appear in the grid.

Only parameters a strategy actually reads are listed — sweeping ω for Standard
would just waste runs.
"""
from __future__ import annotations
import itertools
from typing import Dict, Iterator, List

# strategy_id -> {uniform_name: [grid values]}. The first value is the default.
# Keep grids small; the cartesian product across all strategies is what fills an
# overnight run, so per-strategy lists stay short and meaningful.
STRATEGY_PARAM_GRID: Dict[int, Dict[str, List[float]]] = {
    2: {"omega": [1.2, 1.4, 1.6, 1.8]},      # Naive-Relaxed
    3: {"kappa": [1.5, 2.0, 3.0]},           # Segment (candidate-segment growth)
    5: {"omega": [1.2, 1.4, 1.6, 1.8]},      # Naive-Auto-Relaxed
    6: {"margin": [0.02, 0.05, 0.1, 0.2]},   # Skipping-Spheres (SDF fattening)
    8: {"omega": [1.2, 1.5, 1.8, 2.0]},      # Safe-Relaxed (predictive fallback
                                             # protects larger ω)
}


def param_combos(strategy_id: int) -> Iterator[Dict[str, float]]:
    """Yield each param-override dict for a strategy (cartesian product of its
    grid). Strategies with no tunables yield a single empty dict."""
    grid = STRATEGY_PARAM_GRID.get(strategy_id)
    if not grid:
        yield {}
        return
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def param_label(params: Dict[str, float]) -> str:
    """Stable short label for a param combo (used in the row config so each
    tuned variant is a distinct, resumable cell)."""
    if not params:
        return "default"
    return ",".join(f"{k}={v:g}" for k, v in sorted(params.items()))


def count_param_combos(strategy_id: int) -> int:
    grid = STRATEGY_PARAM_GRID.get(strategy_id)
    if not grid:
        return 1
    n = 1
    for v in grid.values():
        n *= len(v)
    return n
