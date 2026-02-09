"""Minimal charting stubs for later expansion. Currently provides a small
PNG heatmap writer wrapper (delegates to heatmaps.save_iteration_heatmap).
"""
from typing import Any


def render_charts(*, out_dir: str, analyzer: Any = None):
    # Placeholder: heavy visualization is out of scope for MVP.
    print(f"[charts] stub: charts are not implemented yet (out_dir={out_dir})")
