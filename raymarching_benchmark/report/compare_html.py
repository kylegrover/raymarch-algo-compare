"""Run a small strategy x scene comparison on the GPU and emit a single
self-contained HTML report (images embedded as base64, channel toggle in JS).

Each (scene, strategy) cell records GPU timing, iteration stats, and accuracy
scored against a high-budget ground-truth reference (raw depth + SSIM on
depth/normal/lit color + hit-mask agreement).

    uv run python -m raymarching_benchmark.report.compare_html --res 448 \
        --out comparison_report.html
"""
from __future__ import annotations
import io
import os
import sys
import html
import base64
import argparse
import datetime
import statistics
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from ..config import RenderConfig, MarchConfig
from ..gpu.runner import GPURunner
from ..gpu.groundtruth import _resolve_scene, reference_render_config, reference_march_config
from ..data.capture_io import depth_to_image, normal_to_image, color_to_image
from ..metrics.scoring import score_capture


# Representative 4-scene subset spanning the difficulty spectrum.
DEFAULT_SCENES = ["Sphere", "Grazing Plane", "Thin Torus", "Mandelbulb"]

# (gpu strategy id, display name) — matches main.glsl dispatch.
STRATEGIES: List[Tuple[int, str]] = [
    (0, "Standard"),
    (1, "Overstep-Bisect"),
    (2, "Naive-Relaxed"),          # naive over-relaxation (post-hoc backup)
    (3, "Segment"),
    (4, "Enhanced"),
    (5, "Naive-Auto-Relaxed"),     # naive over-relaxation (post-hoc backup)
    (6, "Skipping-Spheres"),
    (7, "RevAA"),
    (8, "Safe-Relaxed"),           # Keinert 2014 safe over-relaxation (predictive)
]


def _force_utf8():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def data_uri(arr_u8: np.ndarray) -> str:
    img = Image.fromarray(arr_u8)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def encode_images(cap: Dict, depth_range: Tuple[float, float]) -> Dict[str, str]:
    return {
        "color": data_uri(color_to_image(cap["color"])),
        "normal": data_uri(normal_to_image(cap["normal"], cap["hit"])),
        "depth": data_uri(depth_to_image(cap["depth"], cap["hit"], depth_range)),
    }


def measure_ms(runner: GPURunner, scene_id: int, sid: int,
               rc: RenderConfig, mc: MarchConfig, lip: float,
               warmup: int = 3, repeats: int = 8) -> float:
    for _ in range(warmup):
        runner.render(scene_id, sid, rc, mc, lipschitz=lip)
    ts = []
    for _ in range(repeats):
        _, t = runner.render(scene_id, sid, rc, mc, lipschitz=lip)
        ts.append(t)
    return float(statistics.median(ts)) * 1000.0


def run_comparison(scene_names: List[str], res: int,
                   method_iters: int, method_threshold: float) -> Dict:
    runner = GPURunner()
    gpu = runner.gpu_info()
    method_mc = MarchConfig(max_iterations=method_iters, hit_threshold=method_threshold)

    scenes_out = []
    for name in scene_names:
        scene_id, scene = _resolve_scene(name)
        if scene is None:
            print(f"  [skip] unknown scene {name!r}")
            continue
        rc = reference_render_config(scene, res, res)
        lip = scene.known_lipschitz_bound()

        # High-budget ground truth.
        ref = runner.capture(scene_id, 0, rc, reference_march_config(), lipschitz=lip)
        rh = ref["hit"]
        drange = (float(ref["depth"][rh].min()), float(ref["depth"][rh].max())) if rh.any() else (0.0, 1.0)
        ref_images = encode_images(ref, drange)

        print(f"\n>> {name}  (ref hit {rh.mean():.1%}, depth [{drange[0]:.2f},{drange[1]:.2f}])")
        rows = []
        for sid, label in STRATEGIES:
            ms = measure_ms(runner, scene_id, sid, rc, method_mc, lip)
            cap = runner.capture(scene_id, sid, rc, method_mc, lipschitz=lip)
            sc = score_capture(cap, ref)
            iters = cap["geom"][..., 1] * method_mc.max_iterations
            row = {
                "strategy": label,
                "gpu_ms": ms,
                "iter_mean": float(iters.mean()),
                "iter_p95": float(np.percentile(iters, 95)),
                "iter_max": float(iters.max()),
                "hit_rate": float(cap["hit"].mean()),
                "false_hit": sc["hit"]["false_hit_rate"],
                "false_miss": sc["hit"]["false_miss_rate"],
                "iou": sc["hit"]["iou"],
                "depth_rmse": sc["depth"]["rmse"],
                "depth_p95": sc["depth"]["p95"],
                "normal_deg": sc["normal"]["mean_deg"],
                "depth_ssim": sc["ssim"]["depth_ssim"],
                "normal_ssim": sc["ssim"]["normal_ssim"],
                "color_ssim": sc["ssim"]["color_ssim"],
                "images": encode_images(cap, drange),
            }
            rows.append(row)
            print(f"   {label:<24} {ms:6.2f}ms  iter~{row['iter_mean']:5.1f}  "
                  f"IoU {row['iou']:.3f}  dSSIM {row['depth_ssim']:.3f}  cSSIM {row['color_ssim']:.3f}")

        scenes_out.append({
            "name": name,
            "category": scene.category,
            "hit_rate": float(rh.mean()),
            "depth_range": list(drange),
            "ref_images": ref_images,
            "rows": rows,
        })

    return {
        "meta": {
            "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "resolution": res,
            "gpu": gpu,
            "method_iters": method_iters,
            "method_threshold": method_threshold,
            "reference": "Standard @ 4096 iters, threshold 1e-6",
        },
        "scenes": scenes_out,
    }


# --- HTML rendering -------------------------------------------------------

# (key, label, higher_is_better, fmt)
METRIC_COLS = [
    ("gpu_ms", "GPU ms", False, "{:.2f}"),
    ("iter_mean", "iter μ", False, "{:.1f}"),
    ("iter_max", "iter max", False, "{:.0f}"),
    ("hit_rate", "hit%", None, "{:.1%}"),
    ("false_miss", "false miss", False, "{:.2%}"),
    ("false_hit", "false hit", False, "{:.2%}"),
    ("iou", "IoU", True, "{:.3f}"),
    ("depth_rmse", "depth RMSE", False, "{:.4f}"),
    ("normal_deg", "normal °", False, "{:.2f}"),
    ("depth_ssim", "depth SSIM", True, "{:.3f}"),
    ("normal_ssim", "normal SSIM", True, "{:.3f}"),
    ("color_ssim", "color SSIM", True, "{:.3f}"),
]

CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0;
       background: #14161a; color: #e6e8ec; }
header { padding: 20px 28px; border-bottom: 1px solid #2a2e36; background:#181b20; position:sticky; top:0; z-index:5;}
h1 { margin: 0 0 6px; font-size: 20px; }
.meta { color:#9aa1ab; font-size: 12px; }
.controls { margin-top:10px; }
.controls button { background:#222831; color:#cfd4dc; border:1px solid #39414d;
       padding:6px 14px; border-radius:6px; cursor:pointer; font-size:13px; margin-right:6px;}
.controls button.active { background:#3b82f6; color:#fff; border-color:#3b82f6; }
section { padding: 22px 28px; border-bottom: 1px solid #20242b; }
h2 { font-size: 17px; margin: 0 0 2px; }
.cat { color:#8b93a0; font-size:12px; margin-bottom:14px; }
.gallery { display:grid; grid-template-columns: repeat(auto-fill, minmax(170px,1fr)); gap:12px; margin-bottom:18px;}
.card { background:#1b1f25; border:1px solid #2a2f38; border-radius:8px; overflow:hidden; }
.card.ref { border-color:#3b82f6; }
.card img { width:100%; display:block; background:#000; aspect-ratio:1/1; object-fit:contain; }
.card .cap { padding:6px 8px; font-size:12px; }
.card .cap .nm { font-weight:600; }
.card .cap .sub { color:#8b93a0; font-size:11px; }
table { border-collapse: collapse; width:100%; font-size:12.5px; }
th, td { padding:5px 8px; text-align:right; border-bottom:1px solid #262b33; white-space:nowrap; }
th { color:#9aa1ab; font-weight:600; position:sticky; }
td.name, th.name { text-align:left; }
tr:hover td { background:#1f242c; }
.best { color:#4ade80; font-weight:700; }
.worst { color:#f87171; }
"""

JS = """
function setChannel(ch){
  document.querySelectorAll('.controls button').forEach(b=>b.classList.toggle('active', b.dataset.ch===ch));
  document.querySelectorAll('.card img').forEach(img=>{ img.src = img.dataset[ch]; });
}
window.addEventListener('DOMContentLoaded',()=>setChannel('color'));
"""


def _card(name: str, sub: str, images: Dict[str, str], is_ref=False) -> str:
    cls = "card ref" if is_ref else "card"
    return (
        f'<div class="{cls}">'
        f'<img data-color="{images["color"]}" data-normal="{images["normal"]}" '
        f'data-depth="{images["depth"]}" src="{images["color"]}" alt="{html.escape(name)}">'
        f'<div class="cap"><div class="nm">{html.escape(name)}</div>'
        f'<div class="sub">{html.escape(sub)}</div></div></div>'
    )


def _table(rows: List[Dict]) -> str:
    # find best/worst per column
    best = {}
    worst = {}
    for key, _, hib, _ in METRIC_COLS:
        if hib is None:
            continue
        vals = [(i, r[key]) for i, r in enumerate(rows)
                if isinstance(r[key], float) and not np.isnan(r[key])]
        if not vals:
            continue
        bi = (max if hib else min)(vals, key=lambda x: x[1])[0]
        wi = (min if hib else max)(vals, key=lambda x: x[1])[0]
        best[key] = bi
        worst[key] = wi

    head = '<tr><th class="name">Strategy</th>' + ''.join(
        f'<th>{html.escape(lbl)}</th>' for _, lbl, _, _ in METRIC_COLS) + '</tr>'
    body = []
    for i, r in enumerate(rows):
        cells = [f'<td class="name">{html.escape(r["strategy"])}</td>']
        for key, _, _, fmt in METRIC_COLS:
            v = r[key]
            txt = fmt.format(v) if isinstance(v, float) and not np.isnan(v) else "—"
            cls = ""
            if best.get(key) == i:
                cls = ' class="best"'
            elif worst.get(key) == i:
                cls = ' class="worst"'
            cells.append(f'<td{cls}>{txt}</td>')
        body.append('<tr>' + ''.join(cells) + '</tr>')
    return f'<table>{head}{"".join(body)}</table>'


def build_html(report: Dict) -> str:
    m = report["meta"]
    gpu = m["gpu"].get("renderer") or "unknown GPU"
    parts = ['<!doctype html><html><head><meta charset="utf-8">',
             '<title>Ray Marching Strategy Comparison</title>',
             f'<style>{CSS}</style></head><body>']
    parts.append('<header>')
    parts.append('<h1>Ray Marching Strategy Comparison</h1>')
    parts.append(
        f'<div class="meta">{html.escape(gpu)} &nbsp;·&nbsp; {m["resolution"]}² '
        f'&nbsp;·&nbsp; methods: {m["method_iters"]} iters / thr {m["method_threshold"]:g} '
        f'&nbsp;·&nbsp; reference: {html.escape(m["reference"])} '
        f'&nbsp;·&nbsp; {m["generated"]}</div>')
    parts.append('<div class="controls">View: '
                 '<button data-ch="color" onclick="setChannel(\'color\')">Color</button>'
                 '<button data-ch="normal" onclick="setChannel(\'normal\')">Normal</button>'
                 '<button data-ch="depth" onclick="setChannel(\'depth\')">Depth</button></div>')
    parts.append('</header>')

    for sc in report["scenes"]:
        parts.append('<section>')
        parts.append(f'<h2>{html.escape(sc["name"])}</h2>')
        parts.append(f'<div class="cat">{html.escape(sc["category"])} &nbsp;·&nbsp; '
                     f'reference hit rate {sc["hit_rate"]:.1%}</div>')
        # gallery: reference first, then each strategy
        cards = [_card("REFERENCE", "Standard @ 4096", sc["ref_images"], is_ref=True)]
        for r in sc["rows"]:
            sub = f'{r["gpu_ms"]:.2f}ms · IoU {r["iou"]:.3f}'
            cards.append(_card(r["strategy"], sub, r["images"]))
        parts.append(f'<div class="gallery">{"".join(cards)}</div>')
        parts.append(_table(sc["rows"]))
        parts.append('</section>')

    parts.append(f'<script>{JS}</script></body></html>')
    return ''.join(parts)


def main(argv=None) -> int:
    _force_utf8()
    p = argparse.ArgumentParser(description="GPU strategy comparison -> standalone HTML.")
    p.add_argument("--scenes", type=str, default=",".join(DEFAULT_SCENES))
    p.add_argument("--res", type=int, default=448)
    p.add_argument("--method-iters", type=int, default=256)
    p.add_argument("--method-threshold", type=float, default=1e-4)
    p.add_argument("--out", type=str, default="comparison_report.html")
    args = p.parse_args(argv)

    scene_names = [s.strip() for s in args.scenes.split(",") if s.strip()]
    print(f"Comparing {len(scene_names)} scenes x {len(STRATEGIES)} strategies at {args.res}²")
    report = run_comparison(scene_names, args.res, args.method_iters, args.method_threshold)
    html_str = build_html(report)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html_str)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"\nWrote {args.out} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
