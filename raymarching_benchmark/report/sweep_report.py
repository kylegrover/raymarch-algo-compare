"""Turn a sweep JSONL into a standalone HTML analysis page.

For each scene it plots the chosen accuracy metric against all three fairness
axes (runtime ms / SDF evals / iteration steps) — one line per strategy across
budgets — and computes a runtime-matched winners table ("best accuracy
achievable under T ms"). This is where "measure by steps OR evals OR runtime"
pays off: the same data, three x-axes, plus the practical time-budget ranking.

    uv run python -m raymarching_benchmark.report.sweep_report \
        --in sweep.jsonl --out sweep_report.html --metric iou
"""
from __future__ import annotations
import io
import sys
import html
import base64
import argparse
from collections import defaultdict
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..data.dataset import JsonlDataset


ACCURACY_LABELS = {
    "iou": "Hit IoU (primary)",            # objective geometric — the ranking metric
    "color_ssim": "Color SSIM", "depth_ssim": "Depth SSIM",
    "normal_ssim": "Normal SSIM",          # SSIM = descriptive only
}
HIGHER_BETTER = True  # all the metrics above are higher-is-better
TAB = plt.get_cmap("tab10")


def _force_utf8():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _flatten(rows: List[Dict]) -> Dict[str, Dict[str, List[Dict]]]:
    """scene -> strategy -> [points sorted by budget]."""
    by_scene: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("status") != "ok":
            continue
        c, p, a = r["config"], r["perf"], r["accuracy"]
        pt = {
            "budget": c["max_iterations"],
            "ms": p["ms_median"],
            "iters": p["iter"]["mean"],
            "evals": p["evals"]["mean"],
            **{k: a[k] for k in a},
        }
        by_scene[c["scene"]][c["strategy"]].append(pt)
    for scene in by_scene:
        for strat in by_scene[scene]:
            by_scene[scene][strat].sort(key=lambda x: x["budget"])
    return by_scene


def _chart(strat_points: Dict[str, List[Dict]], xkey: str, xlabel: str,
           metric: str) -> str:
    fig, ax = plt.subplots(figsize=(4.7, 3.5), dpi=110)
    for idx, strat in enumerate(sorted(strat_points)):
        pts = strat_points[strat]
        xs = [p[xkey] for p in pts]
        ys = [p[metric] for p in pts]
        ax.plot(xs, ys, marker="o", ms=3.5, lw=1.4, color=TAB(idx % 10), label=strat)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ACCURACY_LABELS.get(metric, metric), fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.margins(y=0.05)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _legend(strats: List[str]) -> str:
    items = []
    for idx, s in enumerate(sorted(strats)):
        r, g, b, _ = TAB(idx % 10)
        col = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
        items.append(f'<span class="lg"><span class="sw" style="background:{col}"></span>{html.escape(s)}</span>')
    return '<div class="legend">' + "".join(items) + "</div>"


def _runtime_table(strat_points: Dict[str, List[Dict]], metric: str) -> str:
    all_pts = [(s, p) for s, pts in strat_points.items() for p in pts]
    ms_vals = [p["ms"] for _, p in all_pts]
    lo, hi = min(ms_vals), max(ms_vals)
    targets = np.geomspace(max(lo, 1e-3), hi, 5)

    rows = []
    for T in targets:
        cand = [(s, p) for s, p in all_pts if p["ms"] <= T * 1.0000001]
        if not cand:
            rows.append(f"<tr><td>{T:.2f}</td><td colspan=4>—</td></tr>")
            continue
        s, p = max(cand, key=lambda sp: sp[1][metric])
        rows.append(
            f"<tr><td>≤ {T:.2f}</td><td class='name'>{html.escape(s)}</td>"
            f"<td>{p[metric]:.3f}</td><td>{int(p['budget'])}</td>"
            f"<td>{p['ms']:.2f}</td><td>{p['evals']:.0f}</td></tr>")
    head = ("<tr><th>time budget (ms)</th><th class='name'>best strategy</th>"
            f"<th>{html.escape(ACCURACY_LABELS.get(metric, metric))}</th>"
            "<th>iters</th><th>ms</th><th>evals</th></tr>")
    return f"<table>{head}{''.join(rows)}</table>"


CSS = """
* { box-sizing: border-box; }
body { font-family:-apple-system,Segoe UI,Roboto,sans-serif; margin:0; background:#14161a; color:#e6e8ec; }
header { padding:20px 28px; border-bottom:1px solid #2a2e36; background:#181b20; }
h1 { margin:0 0 6px; font-size:20px; }
.meta { color:#9aa1ab; font-size:12px; }
section { padding:22px 28px; border-bottom:1px solid #20242b; }
h2 { font-size:17px; margin:0 0 2px; }
.cat { color:#8b93a0; font-size:12px; margin-bottom:12px; }
.charts { display:flex; gap:14px; flex-wrap:wrap; }
.charts img { background:#fff; border-radius:8px; border:1px solid #2a2f38; max-width:100%; }
.legend { margin:10px 0 14px; font-size:12px; color:#cfd4dc; }
.lg { display:inline-flex; align-items:center; margin-right:14px; }
.sw { width:11px; height:11px; border-radius:2px; display:inline-block; margin-right:5px; }
table { border-collapse:collapse; margin-top:8px; font-size:12.5px; }
th,td { padding:5px 12px; text-align:right; border-bottom:1px solid #262b33; white-space:nowrap; }
th { color:#9aa1ab; }
td.name, th.name { text-align:left; }
"""


def build_html(rows: List[Dict], metric: str) -> str:
    by_scene = _flatten(rows)
    meta_src = next((r for r in rows if r.get("status") == "ok"), None)
    gpu = (meta_src or {}).get("provenance", {}).get("gpu", {}).get("renderer", "unknown GPU")
    res = (meta_src or {}).get("config", {}).get("resolution", "?")

    parts = ['<!doctype html><html><head><meta charset="utf-8">',
             '<title>Sweep Analysis</title>', f"<style>{CSS}</style></head><body>"]
    parts.append('<header><h1>Ray Marching Sweep — Fairness Analysis</h1>')
    parts.append(f'<div class="meta">{html.escape(str(gpu))} · {res}² · '
                 f'metric: {html.escape(ACCURACY_LABELS.get(metric, metric))} · '
                 f'{len([r for r in rows if r.get("status")=="ok"])} runs · '
                 'accuracy vs three cost axes (runtime / SDF evals / steps)</div>')
    parts.append('<div class="meta" style="margin-top:6px;max-width:70ch">'
                 '⚠ The three cost axes can <b>disagree</b>, and the disagreement is '
                 'itself a result: an SDF eval is not constant-cost across methods, so '
                 'eval-count under-charges adaptive/relaxed tracers — read it next to '
                 'GPU wall-clock. Per-run <code>iter_divergence</code> (neighbor '
                 'iteration spread) is why an eval-count winner can lose on-GPU.'
                 '</div></header>')

    for scene in by_scene:
        sp = by_scene[scene]
        cat = next((r["config"]["scene_category"] for r in rows
                    if r.get("status") == "ok" and r["config"]["scene"] == scene), "")
        parts.append("<section>")
        parts.append(f'<h2>{html.escape(scene)}</h2><div class="cat">{html.escape(cat)}</div>')
        parts.append(_legend(list(sp.keys())))
        parts.append('<div class="charts">')
        parts.append(f'<img src="{_chart(sp, "ms", "GPU time (ms, log)", metric)}">')
        parts.append(f'<img src="{_chart(sp, "evals", "SDF evals / ray (log)", metric)}">')
        parts.append(f'<img src="{_chart(sp, "iters", "iteration steps (log)", metric)}">')
        parts.append("</div>")
        parts.append("<h3 style='font-size:13px;color:#9aa1ab;margin:14px 0 0'>Runtime-matched winners</h3>")
        parts.append(_runtime_table(sp, metric))
        parts.append("</section>")

    parts.append("</body></html>")
    return "".join(parts)


def main(argv=None) -> int:
    _force_utf8()
    p = argparse.ArgumentParser(description="Sweep JSONL -> standalone HTML analysis.")
    p.add_argument("--in", dest="inp", type=str, default="sweep.jsonl")
    p.add_argument("--out", type=str, default="sweep_report.html")
    p.add_argument("--metric", type=str, default="iou",
                   choices=list(ACCURACY_LABELS.keys()),
                   help="Ranking metric for runtime-matched winners. Default IoU "
                        "(primary objective geometric); SSIM is descriptive only.")
    args = p.parse_args(argv)

    rows = JsonlDataset.load(args.inp)
    if not rows:
        print(f"No rows in {args.inp}")
        return 1
    html_str = build_html(rows, args.metric)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html_str)
    import os
    print(f"Wrote {args.out} ({os.path.getsize(args.out)/1e3:.0f} KB) from {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
