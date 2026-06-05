"""Phase 6.2 — feature → Pareto-optimal-method characterization.

Joins the per scene-view features (6.1, ``features.jsonl``) to the existing grid
(``sweep_grid.jsonl``) by (scene, viewpoint) — **no GPU re-run** — and asks the
project's durable question: *does a small set of cheap scene/ray features predict
which method wins?*

Winner is regime-aware (a method's "win" means different things by difficulty):
  • **cost regime** — if ≥1 method reaches IoU ≥ ``acc`` (best-tuned, any budget),
    the winner is the **cheapest** such method (fewest SDF evals; ms as tiebreak).
    "All these methods are accurate enough; which is cheapest here?"
  • **accuracy regime** — if none reach ``acc``, the winner is the **most
    accurate** method. "This scene-view is hard; which method survives at all?"

Output: a joined table (features + winner per scene-view) and a separation
summary (mean feature value per winning method) so a human can see *which*
feature distinguishes the regimes — or see that the winner is degenerate
(e.g. Standard everywhere), which empirically answers the diversity question.

Run:
    uv run python -m raymarching_benchmark.report.discovery \
        --features features.jsonl --grid sweep_grid.jsonl
"""
from __future__ import annotations
import sys
import json
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from .analyze import load_records

FEATURE_ORDER = ["hit_rate", "grazing_frac", "silhouette_cplx", "hardness_mean",
                 "hardness_cv", "lipschitz_p99", "thin_slab"]


def _force_utf8():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_features(path: str) -> Dict[Tuple[str, str], Dict[str, float]]:
    feats: Dict[Tuple[str, str], Dict[str, float]] = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        feats[(r["scene"], r["viewpoint"])] = r["features"]
    return feats


def winner_per_view(recs: List[Dict], acc: float, cost: str = "evals"
                    ) -> Dict[Tuple[str, str], Dict]:
    """Per (scene, view): the regime-aware winning method + diagnostics.

    ``cost`` selects the cost axis for the cost regime: ``"evals"`` (SDF
    evaluations/ray) or ``"ms"`` (GPU wall-clock). Phase 4 showed the two
    *disagree* for adaptive tracers (eval-cheap but ms-dear), so the winner can
    flip with this choice — running both is the honest cross-check."""
    other = "ms" if cost == "evals" else "evals"
    by_sv = defaultdict(lambda: defaultdict(list))
    for r in recs:
        by_sv[(r["scene"], r["view"])][r["strategy"]].append(r)

    out: Dict[Tuple[str, str], Dict] = {}
    for sv, by_strat in by_sv.items():
        best_iou = {s: max(r["iou"] for r in lst) for s, lst in by_strat.items()}
        # cheapest record (by the chosen cost axis) that reaches `acc`, per strategy
        reach: Dict[str, Tuple[float, float]] = {}
        for s, lst in by_strat.items():
            ok = [r for r in lst if r["iou"] >= acc]
            if ok:
                b = min(ok, key=lambda r: (r[cost], r[other]))
                reach[s] = (b[cost], b[other])
        if reach:
            regime = "cost"
            win = min(reach, key=lambda s: (reach[s][0], reach[s][1]))
            win_val = reach[win][0]                          # chosen cost axis
            runner = sorted(reach.values())                  # by (cost, other)
            margin = (runner[1][0] - runner[0][0]) if len(runner) > 1 else float("inf")
        else:
            regime = "accuracy"
            win = max(best_iou, key=lambda s: best_iou[s])
            win_val = best_iou[win]
            ordered = sorted(best_iou.values(), reverse=True)
            margin = (ordered[0] - ordered[1]) if len(ordered) > 1 else float("inf")
        out[sv] = {"regime": regime, "winner": win, "win_val": win_val,
                   "margin": margin, "n_reach": len(reach),
                   "n_strats": len(by_strat), "best_iou_max": max(best_iou.values())}
    return out


def build_markdown(feats: Dict[Tuple[str, str], Dict[str, float]],
                   wins: Dict[Tuple[str, str], Dict], acc: float) -> str:
    keys = sorted(set(feats) & set(wins))
    missing_grid = sorted(set(feats) - set(wins))

    out: List[str] = [
        "# Discovery: feature → winning method\n",
        f"_{len(keys)} scene-views joined (features 6.1 × grid). Winner is regime-aware: "
        f"**cost** = cheapest method reaching IoU≥{acc:g} (evals); **accuracy** = most "
        f"accurate where none do. No GPU re-run — features joined to the existing grid._\n",
    ]

    # 1. Joined table.
    out += ["## 1. Joined table (features + winner per scene-view)\n",
            "| scene · view | " + " | ".join(FEATURE_ORDER) + " | regime | winner | n≥acc |",
            "|---|" + "---|" * (len(FEATURE_ORDER) + 3)]
    for sv in keys:
        f = feats[sv]; w = wins[sv]
        fcells = " | ".join(f"{f.get(k, float('nan')):.3g}" for k in FEATURE_ORDER)
        out.append(f"| {sv[0]} · {sv[1]} | {fcells} | {w['regime']} | "
                   f"**{w['winner']}** | {w['n_reach']}/{w['n_strats']} |")

    # 2. Winner distribution.
    wc = Counter(wins[sv]["winner"] for sv in keys)
    rc = Counter(wins[sv]["regime"] for sv in keys)
    out += ["\n## 2. Winner distribution\n",
            "| winner | scene-views won | | regime | count |", "|---|---|---|---|---|"]
    wl = wc.most_common()
    rl = rc.most_common()
    for i in range(max(len(wl), len(rl))):
        a = f"| {wl[i][0]} | {wl[i][1]} " if i < len(wl) else "|  |  "
        b = f"| {rl[i][0]} | {rl[i][1]} |" if i < len(rl) else "|  |  |"
        out.append(a + "| " + b)
    if len(wc) == 1:
        out.append(f"\n> ⚠ **Degenerate:** one method ({wl[0][0]}) wins every joined "
                   "scene-view. Features cannot 'predict' a constant — this empirically "
                   "says the current scene set lacks the geometric diversity to expose a "
                   "feature→winner relationship (the §G concern), not that no relationship "
                   "exists. Add regimes that defeat the dominant method.")

    # 3. Feature separation: mean feature value per winning method.
    out += ["\n## 3. Feature separation (mean feature per winning method)\n",
            "Which feature distinguishes the winners? A feature that varies a lot "
            "*across rows* of this table is a candidate predictor.\n",
            "| winner (n) | " + " | ".join(FEATURE_ORDER) + " |",
            "|---|" + "---|" * len(FEATURE_ORDER)]
    by_winner: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for sv in keys:
        by_winner[wins[sv]["winner"]].append(feats[sv])
    for win, flist in sorted(by_winner.items(), key=lambda kv: -len(kv[1])):
        cells = []
        for k in FEATURE_ORDER:
            vals = [f.get(k, np.nan) for f in flist]
            vals = [v for v in vals if v == v]
            cells.append(f"{np.mean(vals):.3g}" if vals else "—")
        out.append(f"| {win} ({len(flist)}) | " + " | ".join(cells) + " |")

    # 4. Per-feature discriminative spread (range across winner-group means).
    out += ["\n## 4. Which features carry signal?\n",
            "Spread = (max − min) of each feature's per-winner-group mean, normalised by "
            "the feature's overall mean. Higher ⇒ the feature separates winners more.\n",
            "| feature | normalised spread |", "|---|---|"]
    spreads = []
    for k in FEATURE_ORDER:
        gmeans = []
        for win, flist in by_winner.items():
            vals = [f.get(k, np.nan) for f in flist]
            vals = [v for v in vals if v == v]
            if vals:
                gmeans.append(np.mean(vals))
        allvals = [feats[sv].get(k, np.nan) for sv in keys]
        allvals = [v for v in allvals if v == v]
        base = np.mean(allvals) if allvals else 0.0
        spread = (max(gmeans) - min(gmeans)) / base if (gmeans and base) else 0.0
        spreads.append((k, spread))
    for k, s in sorted(spreads, key=lambda x: -x[1]):
        out.append(f"| {k} | {s:.2f} |")

    if missing_grid:
        out += ["\n## Note: scene-views with features but no grid rows\n",
                "These were captured by 6.1 but are absent from the grid (the grid ran a "
                "narrower scene/viewpoint set). They become joinable when the grid widens:\n",
                ", ".join(f"{s}·{v}" for s, v in missing_grid)]

    return "\n".join(out) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    _force_utf8()
    p = argparse.ArgumentParser(description="Join features to the grid; characterize feature→winner (6.2).")
    p.add_argument("--features", type=str, default="features.jsonl")
    p.add_argument("--grid", type=str, default="sweep_grid.jsonl")
    p.add_argument("--acc", type=float, default=0.99, help="IoU bar for the cost regime.")
    p.add_argument("--cost", choices=["evals", "ms"], default="evals",
                   help="Cost axis for the cost regime (Phase 4: the two can disagree).")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args(argv)

    feats = load_features(args.features)
    recs = load_records(args.grid)
    wins = winner_per_view(recs, args.acc, args.cost)
    md = build_markdown(feats, wins, args.acc)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"Wrote {args.out}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
