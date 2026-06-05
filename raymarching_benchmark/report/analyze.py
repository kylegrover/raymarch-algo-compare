"""Distil a sweep JSONL into the project's headline questions, as Markdown.

The sweep answers "which technique + settings win on which kind of scene, at what
cost — and why." This module turns the raw rows into the tables that answer that:

  1. Accuracy winner per scene-view (best-tuned, at matched residual ε).
  2. Cost at equal accuracy (evals vs GPU ms at ε) — and where the two disagree.
  3. Structurally limited methods (max IoU < threshold at *any* cost).
  4. Does tuning help? best-param vs default-param IoU, and the winning values.
  5. Warp-divergence ranking (why an eval-count winner can lose on-GPU).

Run:  uv run python -m raymarching_benchmark.report.analyze --in sweep_grid.jsonl
"""
from __future__ import annotations
import sys
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np


def _force_utf8():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_records(path: str) -> List[Dict]:
    recs = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("status") != "ok":
            continue
        c, p, a = r["config"], r["perf"], r["accuracy"]
        recs.append({
            "scene": c["scene"], "view": c["viewpoint"], "view_cat": c["view_category"],
            "strategy": c["strategy"], "params": c.get("params", "default"),
            "axis": c["sweep_axis"], "budget": c["max_iterations"],
            "epsilon": c.get("epsilon"),
            "iou": a["iou"], "false_miss": a["false_miss"], "hit_rate": a["hit_rate"],
            "ms": p["ms_median"], "evals": p["evals"]["mean"],
            "divergence": p.get("iter_divergence", float("nan")),
            "dnc": p.get("did_not_converge", 0.0),
        })
    return recs


def _closest_eps(recs: List[Dict], target: float) -> List[Dict]:
    """Residual-axis records at the epsilon closest to target."""
    res = [r for r in recs if r["axis"] == "residual" and r["epsilon"] is not None]
    if not res:
        return []
    eps = sorted({r["epsilon"] for r in res}, key=lambda e: abs(e - target))[0]
    return [r for r in res if r["epsilon"] == eps], eps


def _best_param(group: List[Dict]) -> Dict:
    """Pick the fair-shot record: highest IoU, ties broken by lower GPU ms."""
    return max(group, key=lambda r: (round(r["iou"], 4), -r["ms"]))


SCENE_ORDER = ["Sphere", "Cube", "Thin Torus", "Grazing Plane", "Mandelbulb"]
STRAT_ORDER = ["Standard", "Overstep-Bisect", "Enhanced", "Segment", "Naive-Relaxed",
               "Naive-Auto-Relaxed", "Safe-Relaxed", "Skipping-Spheres", "RevAA"]


def section_accuracy_winner(recs: List[Dict], target_eps: float) -> List[str]:
    res, eps = _closest_eps(recs, target_eps)
    out = [f"## 1. Accuracy winner per scene-view (matched residual ε≈{eps:g}, best-tuned)\n",
           "Best-tuned IoU each method reaches at a fixed closeness-to-surface, per view. "
           "**Winner** = highest IoU (ties → cheaper GPU ms). `n≥0.99` lists every method "
           "that effectively matches the oracle.\n",
           "| scene | view (cat) | winner | IoU | reach ≥0.99 |",
           "|---|---|---|---|---|"]
    by_sv = defaultdict(lambda: defaultdict(list))
    for r in res:
        by_sv[(r["scene"], r["view"], r["view_cat"])][r["strategy"]].append(r)
    for (scene, view, cat) in sorted(by_sv, key=lambda k: (SCENE_ORDER.index(k[0]) if k[0] in SCENE_ORDER else 9, k[1])):
        best_per_strat = {s: _best_param(g) for s, g in by_sv[(scene, view, cat)].items()}
        win_s = max(best_per_strat, key=lambda s: (round(best_per_strat[s]["iou"], 4), -best_per_strat[s]["ms"]))
        win = best_per_strat[win_s]
        good = sorted([s for s, r in best_per_strat.items() if r["iou"] >= 0.99],
                      key=lambda s: STRAT_ORDER.index(s) if s in STRAT_ORDER else 9)
        out.append(f"| {scene} | {view} ({cat}) | **{win_s}** | {win['iou']:.3f} | "
                   f"{', '.join(good) if good else '—'} |")
    return out


def section_cost_at_accuracy(recs: List[Dict], target_eps: float) -> List[str]:
    res, eps = _closest_eps(recs, target_eps)
    out = [f"\n## 2. Cost at equal accuracy (ε≈{eps:g}) — evals vs GPU ms\n",
           "Per scene, the median **cost to reach ε** across views (best-tuned per method, "
           "only methods that actually reach IoU≥0.95 there). Ranks by evals and by ms can "
           "**disagree** — that gap is the eval-cost caveat made real.\n"]
    by_scene = defaultdict(lambda: defaultdict(list))
    for r in res:
        by_scene[r["scene"]][(r["strategy"], r["view"])].append(r)
    for scene in [s for s in SCENE_ORDER if s in by_scene]:
        # best param per (strategy,view), keep only accurate ones, median over views
        per_strat = defaultdict(list)
        for (strat, view), g in by_scene[scene].items():
            b = _best_param(g)
            if b["iou"] >= 0.95:
                per_strat[strat].append(b)
        rows = []
        for strat, lst in per_strat.items():
            rows.append((strat, float(np.median([r["evals"] for r in lst])),
                         float(np.median([r["ms"] for r in lst]))))
        if not rows:
            continue
        by_evals = sorted(rows, key=lambda x: x[1])
        by_ms = sorted(rows, key=lambda x: x[2])
        out.append(f"\n**{scene}** — cheapest by evals: "
                   + ", ".join(f"{s}({e:.0f})" for s, e, _ in by_evals[:4]))
        out.append(f"  · cheapest by ms: "
                   + ", ".join(f"{s}({m:.2f}ms)" for s, _, m in by_ms[:4]))
        # inversion: a method top-3 by evals but bottom-half by ms
        ev_rank = {s: i for i, (s, _, _) in enumerate(by_evals)}
        ms_rank = {s: i for i, (s, _, _) in enumerate(by_ms)}
        n = len(rows)
        inv = [s for s in ev_rank if ev_rank[s] < n / 2 and ms_rank[s] >= n / 2]
        if inv:
            out.append(f"  · ⚠ eval-cheap but ms-dear (divergence/overhead): {', '.join(inv)}")
    return out


def section_limited(recs: List[Dict], thresh: float = 0.95) -> List[str]:
    out = [f"\n## 3. Structurally limited methods (max IoU < {thresh} at *any* cost)\n",
           "Best IoU each method reaches over the **entire** grid (every budget, every ε, "
           "every param) per scene-view. Below threshold = no setting rescues it — a real "
           "failure mode, not a budget shortfall.\n",
           "| scene | view | method | best IoU (any cost) |",
           "|---|---|---|---|"]
    best = defaultdict(lambda: defaultdict(float))
    for r in recs:
        k = (r["scene"], r["view"])
        best[k][r["strategy"]] = max(best[k][r["strategy"]], r["iou"])
    any_row = False
    for (scene, view) in sorted(best, key=lambda k: (SCENE_ORDER.index(k[0]) if k[0] in SCENE_ORDER else 9, k[1])):
        for strat in sorted(best[(scene, view)], key=lambda s: best[(scene, view)][s]):
            v = best[(scene, view)][strat]
            if v < thresh:
                out.append(f"| {scene} | {view} | {strat} | {v:.3f} |")
                any_row = True
    if not any_row:
        out.append("| — | — | (none) | — |")
    return out


def section_tuning(recs: List[Dict], budget: int = 256) -> List[str]:
    out = [f"\n## 4. Does tuning help? (budget={budget}, best-param vs default)\n",
           "For each tunable method: mean IoU gain from its best param over `default`, "
           "averaged across scene-views, and the most-often-best value. A positive gain "
           "means the brute-force 'fair shot' mattered.\n",
           "| method | mean IoU gain (best−default) | best value (mode) |",
           "|---|---|---|"]
    bud = [r for r in recs if r["axis"] == "budget" and r["budget"] == budget]
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # strat -> (scene,view) -> paramlabel -> [iou]
    for r in bud:
        by[r["strategy"]][(r["scene"], r["view"])][r["params"]].append(r["iou"])
    for strat in STRAT_ORDER:
        if strat not in by:
            continue
        sv_map = by[strat]
        # only methods with real tunables (>1 param label somewhere)
        labels = set()
        for sv, pm in sv_map.items():
            labels |= set(pm.keys())
        if labels == {"default"} or len(labels) <= 1:
            continue
        gains = []
        best_vals = []
        for sv, pm in sv_map.items():
            if "default" in pm:
                base = np.mean(pm["default"])
            else:
                base = min(np.mean(v) for v in pm.values())
            best_label = max(pm, key=lambda L: np.mean(pm[L]))
            gains.append(np.mean(pm[best_label]) - base)
            best_vals.append(best_label)
        from collections import Counter
        mode_val = Counter(best_vals).most_common(1)[0][0]
        out.append(f"| {strat} | {np.mean(gains):+.3f} | {mode_val} |")
    return out


def section_divergence(recs: List[Dict]) -> List[str]:
    out = ["\n## 5. Warp-divergence ranking (efficiency axis)\n",
           "Mean neighbor iteration-spread per method (budget axis). Higher = adjacent GPU "
           "threads take more divergent paths — the mechanism by which an eval-count winner "
           "loses on wall-clock.\n",
           "| method | mean iter_divergence | mean GPU ms |",
           "|---|---|---|"]
    bud = [r for r in recs if r["axis"] == "budget"]
    by = defaultdict(list)
    for r in bud:
        by[r["strategy"]].append(r)
    rows = [(s, float(np.nanmean([r["divergence"] for r in g])),
             float(np.median([r["ms"] for r in g]))) for s, g in by.items()]
    for s, d, m in sorted(rows, key=lambda x: -x[1]):
        out.append(f"| {s} | {d:.2f} | {m:.2f} |")
    return out


def build_markdown(recs: List[Dict], target_eps: float = 1e-4) -> str:
    n_sv = len({(r["scene"], r["view"]) for r in recs})
    n_strat = len({r["strategy"] for r in recs})
    head = [f"# Sweep findings\n",
            f"_{len(recs)} scored runs · {n_sv} scene-views · {n_strat} strategies · "
            f"budget + matched-residual axes · scored vs the calibrated dense-march oracle._\n",
            "> Generated by `raymarching_benchmark.report.analyze`. IoU is the primary "
            "(objective geometric) metric; cost is reported on three axes (GPU ms, SDF "
            "evals/ray, iteration divergence) and never collapsed.\n"]
    parts = head
    parts += section_accuracy_winner(recs, target_eps)
    parts += section_cost_at_accuracy(recs, target_eps)
    parts += section_limited(recs)
    parts += section_tuning(recs)
    parts += section_divergence(recs)
    return "\n".join(parts) + "\n"


def main(argv=None) -> int:
    _force_utf8()
    p = argparse.ArgumentParser(description="Distil a sweep JSONL into Markdown findings.")
    p.add_argument("--in", dest="inp", type=str, default="sweep_grid.jsonl")
    p.add_argument("--out", type=str, default=None, help="Write Markdown here (else stdout).")
    p.add_argument("--eps", type=float, default=1e-4, help="Matched-residual target epsilon.")
    args = p.parse_args(argv)

    recs = load_records(args.inp)
    md = build_markdown(recs, target_eps=args.eps)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"Wrote {args.out} ({len(recs)} records)")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
