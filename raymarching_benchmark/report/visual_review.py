"""Visual review page for the interval oracle + faithful method ceilings.

Builds ONE self-contained HTML page of side-by-side depth maps, normal maps, and
— most importantly — *difference / disagreement* overlays, so a human can gut-
check what the IoU numbers can't show: do the surfaces look geometrically right,
and where exactly do the methods disagree (silhouette-only = fine; whole chunks =
a real problem).

Per (scene, viewpoint) it shows, on a shared depth scale:
  row 1 (depth)      Analytic · Interval oracle · Dense-march · STRAWMAN Segment · Faithful Segment
  row 2 (diagnostics) Δ interval−analytic · Δ dense−interval · disagree(strawman,oracle)
                      · disagree(faithful,oracle) · interval normals

The Mandelbulb section is deliberately honest: no analytic, no interval oracle —
the dense march is shown *unverified*, the one regime we did not certify.

Run:
    uv run python -m raymarching_benchmark.report.visual_review --res 384
    # writes interval_review.html in the current directory
"""
from __future__ import annotations
import io
import os
import base64
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..config import MarchConfig
from ..scenes.catalog import get_all_scenes
from ..viewpoints import viewpoints_for, Viewpoint
from ..data.capture_io import depth_to_image, normal_to_image
from ..gpu.runner import GPURunner
from ..gpu.groundtruth import (
    dense_march_config, dense_march_params, DENSE_MARCH_STRATEGY_ID, force_utf8_stdout,
)
from ..gpu import analytic
from ..gpu.interval import has_interval
from ..gpu.interval_oracle import interval_capture
from ..gpu.faithful_offline import faithful_capture
from ..gpu.oracle_calibration import residual, silhouette_band

STRAWMAN_SEGMENT_ID = 3

# (scene, viewpoint-name) pairs chosen to expose each claim. None viewpoint = all.
_SELECTION: List[Tuple[str, str]] = [
    ("Sphere", "ortho"),                 # baseline: everything must match
    ("Cube", "grazing-face"),            # strawman Segment failed here (~0.68)
    ("Thin Torus", "macro"),             # the dramatic one: strawman 0.003
    ("Thin Torus", "grazing-edge"),      # thin + grazing
    ("Grazing Plane", "grazing"),        # directional-Lipschitz win + far-clip
]
_MANDELBULB: List[Tuple[str, str]] = [("Mandelbulb", "ortho")]


def data_uri(arr_u8: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr_u8).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _diff_image(da, db, co_hit, scale: float) -> np.ndarray:
    """|da-db| over co-hit pixels as grayscale (white = `scale` units of error);
    non-co-hit pixels are dark navy so 'not compared' reads differently from
    'zero error'."""
    H, W = da.shape
    img = np.zeros((H, W, 3), dtype=np.float64)
    img[~co_hit] = np.array([0.12, 0.12, 0.28])     # navy = no comparison
    err = np.clip(np.abs(da - db) / max(scale, 1e-9), 0.0, 1.0)
    g = np.where(co_hit, err, 0.0)
    img[co_hit] = np.stack([g, g, g], axis=-1)[co_hit]
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def _disagree_image(hit_method, hit_truth) -> np.ndarray:
    """Hit-mask disagreement. blue = truth hits but method MISSES (tunnel);
    red = method hits where truth is empty (phantom); gray = both hit; black =
    both miss."""
    H, W = hit_method.shape
    img = np.zeros((H, W, 3), dtype=np.uint8)
    both = hit_method & hit_truth
    tunnel = hit_truth & ~hit_method
    phantom = hit_method & ~hit_truth
    img[both] = (90, 90, 90)
    img[tunnel] = (40, 90, 230)     # blue
    img[phantom] = (230, 60, 50)    # red
    return img


def _iou(hit_a, hit_b) -> float:
    inter = np.logical_and(hit_a, hit_b).sum()
    union = np.logical_or(hit_a, hit_b).sum()
    return float(inter) / float(max(union, 1))


def _capture_section(scene, vp: Viewpoint, runner: GPURunner, res: int) -> Dict:
    name = scene.name
    sid = next(i for i, s in enumerate(get_all_scenes()) if s.name == name)
    rc = vp.render_config(res, res)

    an = analytic.analytic_capture(name, rc) if analytic.has_analytic(name) else None
    iv = interval_capture(name, rc) if has_interval(name) else None
    fc = faithful_capture(name, rc) if has_interval(name) else None

    dense = runner.capture(sid, DENSE_MARCH_STRATEGY_ID, rc, dense_march_config(),
                           lipschitz=scene.known_lipschitz_bound(),
                           params=dense_march_params())
    straw = runner.capture(sid, STRAWMAN_SEGMENT_ID, rc,
                           MarchConfig(max_iterations=512, hit_threshold=1e-4, max_distance=100.0),
                           lipschitz=scene.known_lipschitz_bound())

    # Shared depth scale: prefer the most-trusted hit set available.
    truth = iv or an or {"depth": dense["depth"], "hit": dense["hit"]}
    if truth["hit"].any():
        d = truth["depth"][truth["hit"]]
        drange = (float(d.min()), float(d.max()))
    else:
        drange = (0.0, 1.0)

    panels: List[Dict] = []

    def depth_panel(cap, label, sub=""):
        if cap is None:
            return
        panels.append({"row": "depth", "label": label, "sub": sub,
                       "img": data_uri(depth_to_image(cap["depth"], cap["hit"], drange))})

    depth_panel(an, "Analytic", "closed-form truth")
    depth_panel(iv, "Interval oracle", "sound, no-tunnel")
    depth_panel({"depth": dense["depth"], "hit": dense["hit"]}, "Dense-march", "the grid's oracle")
    straw_iou = _iou(straw["hit"], (iv or an or dense)["hit"])
    depth_panel({"depth": straw["depth"], "hit": straw["hit"]},
                "STRAWMAN Segment", f"GLSL id 3 · IoU {straw_iou:.3f}")
    if fc is not None:
        fc_iou = _iou(fc["hit"], (iv or an)["hit"])
        depth_panel(fc, "Faithful Segment", f"sound · IoU {fc_iou:.3f}")

    # Diagnostics row.
    diag: List[Dict] = []
    diff_scale = max(drange[1] - drange[0], 1e-3) * 0.02   # 2% of depth span = white

    def diag_panel(label, sub, img):
        diag.append({"label": label, "sub": sub, "img": data_uri(img)})

    if iv is not None and an is not None:
        co = iv["hit"] & an["hit"]
        diag_panel("Δ interval − analytic", f"white = {diff_scale:.2g} units",
                   _diff_image(iv["depth"], an["depth"], co, diff_scale))
    if iv is not None:
        co = dense["hit"] & iv["hit"]
        diag_panel("Δ dense − interval", f"white = {diff_scale:.2g} units",
                   _diff_image(dense["depth"], iv["depth"], co, diff_scale))
    truth_hit = (iv or an or dense)["hit"]
    diag_panel("disagree: strawman vs oracle", "blue=tunnel · red=phantom",
               _disagree_image(straw["hit"], truth_hit))
    if fc is not None:
        diag_panel("disagree: faithful vs oracle", "blue=tunnel · red=phantom",
                   _disagree_image(fc["hit"], (iv or an)["hit"]))
    if iv is not None:
        diag_panel("Interval normals", "RGB = world normal",
                   normal_to_image(iv["normal"], iv["hit"]))
    else:
        diag_panel("Dense-march color", "shaded (no oracle here)",
                   np.clip(dense["color"] * 255, 0, 255).astype(np.uint8))

    return {"title": f"{name} · {vp.name} ({vp.category})",
            "depth": panels, "diag": diag,
            "drange": drange}


def _resolve(scene_name: str, vp_name: str):
    scene = next((s for s in get_all_scenes() if s.name == scene_name), None)
    if scene is None:
        return None, None
    vp = next((v for v in viewpoints_for(scene) if v.name == vp_name), None)
    if vp is None:
        vps = viewpoints_for(scene)
        vp = vps[0] if vps else None
    return scene, vp


def build(selection: List[Tuple[str, str]], res: int, include_mandelbulb: bool) -> List[Dict]:
    runner = GPURunner()
    sections = []
    sel = list(selection) + (_MANDELBULB if include_mandelbulb else [])
    for sname, vname in sel:
        scene, vp = _resolve(sname, vname)
        if scene is None or vp is None:
            print(f"  [skip] {sname}/{vname}")
            continue
        print(f"  rendering {sname} / {vp.name} ...")
        sections.append(_capture_section(scene, vp, runner, res))
    return sections


_CSS = """
body{background:#11131a;color:#dde;font:14px/1.5 system-ui,sans-serif;margin:0;padding:24px}
h1{font-size:20px} h2{font-size:16px;margin:28px 0 6px;border-bottom:1px solid #333;padding-bottom:4px}
.intro{max-width:900px;color:#aab;margin-bottom:8px}
.row{display:flex;gap:10px;flex-wrap:wrap;margin:8px 0 4px}
figure{margin:0;width:200px}
figure img{width:200px;height:200px;image-rendering:pixelated;border:1px solid #2a2d38;background:#000;border-radius:4px}
figcaption{font-size:12px;margin-top:3px}
figcaption b{color:#fff} figcaption span{color:#889}
.legend{color:#9aa;font-size:12px;margin:4px 0 16px}
.warn{color:#fb6}
"""

_INTRO = """<div class=intro>
Side-by-side for human gut-checks. <b>Depth row</b>: near = bright, all share one
scale per scene, so panels line up. <b>Diagnostics row</b>: the difference maps are
the point — <span style="color:#fff">white</span> in a Δ map means depth disagreement
(should be ~black except a 1px silhouette); in a <i>disagree</i> map,
<span style="color:#4ae">blue</span> = the method <b>missed a real surface</b>
(tunnelling) and <span style="color:#e54">red</span> = it hit empty space.
<br><b>What to look for:</b> the STRAWMAN Segment panel should look visibly broken
(blue chunks) where Faithful Segment and the oracle look clean; the two Δ maps
(interval−analytic, dense−interval) should be essentially black — that is the
visual form of "the oracle is trustworthy."
</div>"""


def render_html(sections: List[Dict]) -> str:
    out = [f"<!doctype html><html><head><meta charset=utf-8><style>{_CSS}</style>",
           "<title>Interval oracle — visual review</title></head><body>",
           "<h1>Interval oracle &amp; faithful method ceilings — visual review</h1>",
           _INTRO]
    for s in sections:
        out.append(f"<h2>{s['title']}</h2>")
        out.append(f"<div class=legend>depth scale [{s['drange'][0]:.2f}, {s['drange'][1]:.2f}] units</div>")
        for rowkey in ("depth", "diag"):
            out.append("<div class=row>")
            for p in s[rowkey]:
                out.append(f"<figure><img src='{p['img']}'>"
                           f"<figcaption><b>{p['label']}</b><br><span>{p['sub']}</span></figcaption></figure>")
            out.append("</div>")
    out.append("</body></html>")
    return "".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    force_utf8_stdout()
    p = argparse.ArgumentParser(description="Build the interval-oracle visual review HTML.")
    p.add_argument("--res", type=int, default=384)
    p.add_argument("--out", type=str, default="interval_review.html")
    p.add_argument("--no-mandelbulb", action="store_true")
    args = p.parse_args(argv)

    print(f"Building visual review at {args.res}x{args.res} ...")
    sections = build(_SELECTION, args.res, include_mandelbulb=not args.no_mandelbulb)
    html = render_html(sections)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nWrote {args.out} ({len(html)//1024} KB, {len(sections)} sections). Open it in a browser.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
