"""Perceptual A/B harness — does the metric-predicted winner actually look best?

The discovery metric (6.2) claims a Pareto winner per scene-view, but "wins by
IoU/ms" is not the same as "looks best to a human." This page makes that testable:
for each (scene, viewpoint) it renders every method at the **same wall-clock
budget**, side by side, and badges the method the metric *predicts* should win —
so a human can rate which render is actually the most pleasing / least
perceptually discontinuous, and we see where the metric and the eye disagree.

Equal-time framing (the honest part): a fixed iteration budget is *not* equal
time — adaptive tracers cost more ms per iteration. So we anchor each tier to
**Standard's** GPU ms at an anchor budget, then give every other method the
iteration budget whose measured ms is closest to that same target. Every panel in
a tier therefore costs ~the same wall-clock; what differs is what each method
*did* with it.

Per (scene, viewpoint) × time-tier, three rows:
  color        — lit shading (the "is it pleasing" view)
  depth        — geometry (shared scale)
  discontinuity— |Laplacian(depth)| in the hit region: raymarching under-stepping
                 shows as stair-steps / banding; this makes that visible

Run:
    uv run python -m raymarching_benchmark.report.perceptual_review --res 384
    # writes perceptual_review.html (self-contained)
"""
from __future__ import annotations
import io
import os
import base64
import argparse
import statistics
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..config import MarchConfig
from ..scenes.catalog import get_all_scenes
from ..viewpoints import viewpoints_for, Viewpoint
from ..data.capture_io import depth_to_image
from ..gpu.runner import GPURunner
from ..gpu.groundtruth import (
    dense_march_config, dense_march_params, DENSE_MARCH_STRATEGY_ID, force_utf8_stdout,
)
from .analyze import load_records
from .discovery import load_features, winner_per_view

# Methods shown (the ones that win *somewhere* in 6.2 — keeps the grid legible).
METHODS: List[Tuple[int, str]] = [
    (0, "Standard"), (4, "Enhanced"), (2, "Naive-Relaxed"),
    (8, "Safe-Relaxed"), (3, "Segment"),
]
# Iteration budgets scanned to match a target ms (fine near the low end).
_BUDGETS = [12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
# Anchor budgets of Standard that define the per-scene time tiers.
_ANCHORS = [64, 256]
ANCHOR_LABEL = {64: "fast", 256: "slow"}

# (scene, viewpoint) chosen to expose the 6.2 regimes.
_SELECTION: List[Tuple[str, str]] = [
    ("Sphere", "ortho"),                  # easy baseline
    ("Grazing Plane", "extreme-grazing"), # grazing → Segment/over-relax regime
    ("Thin Torus", "grazing-edge"),       # thin + grazing → Safe-Relaxed regime
    ("Hollow Cube (CSG)", "grazing-face"),# CSG thin walls
    ("Mandelbulb", "ortho"),              # fractal → high-lipschitz regime
]


def data_uri(arr_u8: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr_u8).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _color_image(cap: Dict) -> np.ndarray:
    return np.clip(cap["color"][..., :3] * 255.0, 0, 255).astype(np.uint8)


def _discontinuity_image(depth: np.ndarray, hit: np.ndarray) -> np.ndarray:
    """|Laplacian(depth)| inside the hit mask, as a heat map. Smooth surfaces →
    near-black; stair-stepping / banding from under-stepping → bright. Silhouette
    edges are masked out (they are real depth jumps, not artifacts)."""
    d = np.where(hit, depth, 0.0)
    lap = np.zeros_like(d)
    lap[1:-1, 1:-1] = (d[:-2, 1:-1] + d[2:, 1:-1] + d[1:-1, :-2] + d[1:-1, 2:]
                       - 4.0 * d[1:-1, 1:-1])
    # interior = hit pixel whose 4-neighbours are all hits (drop silhouette)
    pad = np.pad(hit, 1, constant_values=False)
    interior = pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:] & hit
    mag = np.abs(lap)
    scale = np.percentile(mag[interior], 99) if interior.any() else 1.0
    v = np.clip(mag / max(scale, 1e-9), 0.0, 1.0) * interior
    # heat: black → orange → white
    img = np.stack([np.clip(v * 2.0, 0, 1), np.clip(v * 1.4 - 0.2, 0, 1),
                    np.clip(v * 1.4 - 0.6, 0, 1)], axis=-1)
    return (img * 255).astype(np.uint8)


def _iou(a, b) -> float:
    inter = np.logical_and(a, b).sum()
    return float(inter) / float(max(np.logical_or(a, b).sum(), 1))


def _measure_ms(runner, scene_id, sid, rc, budget, lip, params, repeats=6) -> float:
    mc = MarchConfig(max_iterations=int(budget), hit_threshold=1e-4)
    runner.render(scene_id, sid, rc, mc, lipschitz=lip, params=params)  # warmup
    ts = []
    for _ in range(repeats):
        _, t = runner.render(scene_id, sid, rc, mc, lipschitz=lip, params=params)
        ts.append(t * 1000.0)
    return float(statistics.median(ts))


def _budget_for_target(ms_by_budget: Dict[int, float], target: float) -> int:
    return min(ms_by_budget, key=lambda b: abs(ms_by_budget[b] - target))


def _capture_section(scene, vp: Viewpoint, runner: GPURunner, res: int,
                     pred_ms: str, pred_evals: str) -> Dict:
    sid_scene = next(i for i, s in enumerate(get_all_scenes()) if s.name == scene.name)
    rc = vp.render_config(res, res)
    lip = scene.known_lipschitz_bound()

    oracle = runner.capture(sid_scene, DENSE_MARCH_STRATEGY_ID, rc, dense_march_config(),
                            lipschitz=lip, params=dense_march_params())
    dhit = oracle["hit"]
    drange = ((float(oracle["depth"][dhit].min()), float(oracle["depth"][dhit].max()))
              if dhit.any() else (0.0, 1.0))

    # Measure ms for every method at every budget once (reused across tiers).
    ms_tbl: Dict[int, Dict[int, float]] = {}
    for (sid, _name) in METHODS:
        ms_tbl[sid] = {b: _measure_ms(runner, sid_scene, sid, rc, b, lip, None)
                       for b in _BUDGETS}

    tiers = []
    for anchor in _ANCHORS:
        target = ms_tbl[0][anchor]              # Standard's ms at the anchor budget
        panels = []
        for (sid, name) in METHODS:
            b = _budget_for_target(ms_tbl[sid], target)
            mc = MarchConfig(max_iterations=int(b), hit_threshold=1e-4)
            cap = runner.capture(sid_scene, sid, rc, mc, lipschitz=lip)
            iou = _iou(cap["hit"], dhit)
            badge = []
            if name == pred_ms:
                badge.append("ms★")
            if name == pred_evals:
                badge.append("evals★")
            panels.append({
                "name": name, "budget": b, "ms": ms_tbl[sid][b], "iou": iou,
                "badge": " ".join(badge),
                "color": data_uri(_color_image(cap)),
                "depth": data_uri(depth_to_image(cap["depth"], cap["hit"], drange)),
                "disc": data_uri(_discontinuity_image(cap["depth"], cap["hit"])),
            })
        tiers.append({"anchor": anchor, "label": ANCHOR_LABEL[anchor],
                      "target_ms": target, "panels": panels})

    return {"title": f"{scene.name} · {vp.name} ({vp.category})",
            "pred_ms": pred_ms, "pred_evals": pred_evals, "tiers": tiers}


_CSS = """
body{background:#11131a;color:#dde;font:14px/1.5 system-ui,sans-serif;margin:0;padding:24px}
h1{font-size:20px} h2{font-size:16px;margin:30px 0 4px;border-bottom:1px solid #333;padding-bottom:4px}
h3{font-size:13px;color:#9bd;margin:16px 0 4px;font-weight:600}
.intro,.pred{max-width:1000px;color:#aab;margin-bottom:8px}
.pred{color:#cbd;background:#191c26;border-left:3px solid #4a7;padding:6px 10px;border-radius:4px}
.rowlabel{color:#789;font-size:11px;text-transform:uppercase;letter-spacing:.5px;margin:6px 0 2px}
.row{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 2px}
figure{margin:0;width:170px}
figure img{width:170px;height:170px;image-rendering:pixelated;border:1px solid #2a2d38;background:#000;border-radius:4px}
figcaption{font-size:11px;margin-top:2px;color:#9aa}
figcaption b{color:#fff}
.win{outline:2px solid #5c6;border-radius:5px;padding:2px}
.badge{color:#5e6;font-weight:700}
"""

_INTRO = """<div class=intro>
<b>Equal wall-clock A/B.</b> Within each tier, every method got the <i>same GPU ms</i>
(matched to Standard's cost at the anchor budget — the per-method iteration count
that buys that time is in each caption). So differences are <i>what each method did
with equal time</i>, not a budget handicap. <b class=badge>★</b> marks the method the
discovery metric predicts should win (ms★ = by GPU-ms cost; evals★ = by SDF-eval
cost — they often differ). <b>Your job:</b> ignore the badges at first — rate which
<i>color</i> render is most pleasing and which <i>discontinuity</i> map is darkest
(= least banding) — then see whether the ★ agrees with your eye. Where it doesn't,
the metric is missing something perceptual.
</div>"""


def render_html(sections: List[Dict]) -> str:
    out = [f"<!doctype html><html><head><meta charset=utf-8><style>{_CSS}</style>",
           "<title>Perceptual review — equal-time A/B</title></head><body>",
           "<h1>Perceptual A/B — does the metric-predicted winner look best?</h1>",
           _INTRO]
    for s in sections:
        out.append(f"<h2>{s['title']}</h2>")
        out.append(f"<div class=pred>Metric prediction — by GPU ms: <b>{s['pred_ms']}</b>"
                   f" · by SDF evals: <b>{s['pred_evals']}</b></div>")
        for tier in s["tiers"]:
            out.append(f"<h3>{tier['label']} tier — equal budget ≈ {tier['target_ms']:.3f} ms/frame "
                       f"(Standard @ {tier['anchor']} iters)</h3>")
            for rowkey, lbl in (("color", "lit color — “most pleasing?”"),
                                ("depth", "depth — geometry"),
                                ("disc", "discontinuity — darker = less banding")):
                out.append(f"<div class=rowlabel>{lbl}</div><div class=row>")
                for p in tier["panels"]:
                    cls = " class=win" if p["badge"] else ""
                    cap = ""
                    if rowkey == "color":
                        b = f" <span class=badge>{p['badge']}</span>" if p["badge"] else ""
                        cap = (f"<figcaption><b>{p['name']}</b>{b}<br>"
                               f"{p['budget']} it · {p['ms']:.3f} ms · IoU {p['iou']:.3f}</figcaption>")
                    out.append(f"<figure{cls}><img src='{p[rowkey]}'>{cap}</figure>")
                out.append("</div>")
    return "".join(out)


def build(selection, res: int, features_path: str, grid_path: str) -> List[Dict]:
    runner = GPURunner()
    # Metric predictions per scene-view (best-effort; "—" if scene-view absent).
    preds_ms, preds_ev = {}, {}
    try:
        recs = load_records(grid_path)
        preds_ms = {sv: w["winner"] for sv, w in winner_per_view(recs, 0.99, "ms").items()}
        preds_ev = {sv: w["winner"] for sv, w in winner_per_view(recs, 0.99, "evals").items()}
    except FileNotFoundError:
        print(f"  [warn] {grid_path} not found — predictions left blank")

    sections = []
    for sname, vname in selection:
        scene = next((s for s in get_all_scenes() if s.name == sname), None)
        if scene is None:
            print(f"  [skip] unknown scene {sname!r}"); continue
        vp = next((v for v in viewpoints_for(scene) if v.name == vname), None)
        if vp is None:
            print(f"  [skip] {sname}/{vname}"); continue
        print(f"  rendering {sname} / {vp.name} ...")
        sections.append(_capture_section(
            scene, vp, runner, res,
            preds_ms.get((sname, vname), "—"), preds_ev.get((sname, vname), "—")))
    return sections


def main(argv: Optional[List[str]] = None) -> int:
    force_utf8_stdout()
    p = argparse.ArgumentParser(description="Equal-time perceptual A/B review page.")
    p.add_argument("--res", type=int, default=384)
    p.add_argument("--out", type=str, default="perceptual_review.html")
    p.add_argument("--features", type=str, default="features.jsonl")
    p.add_argument("--grid", type=str, default="sweep_grid.jsonl")
    args = p.parse_args(argv)

    print(f"Building perceptual A/B at {args.res}x{args.res} ...")
    sections = build(_SELECTION, args.res, args.features, args.grid)
    html = render_html(sections)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nWrote {args.out} ({len(html)//1024} KB, {len(sections)} scene-views). Open in a browser.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
