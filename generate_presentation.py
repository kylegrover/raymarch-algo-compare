"""Regenerate findings_presentation.html as a sober technical report.
Pulls the equal-time render frames from perceptual_review.html (pure file I/O,
no GPU) and writes a single self-contained, lightly-styled academic page."""
import re, html

SRC = "perceptual_review.html"
DST = "findings_presentation.html"
METHODS = ["Standard", "Enhanced", "Naive-Relaxed", "Safe-Relaxed", "Segment"]
CHANNELS = ["color", "depth", "discont"]
src = open(SRC, encoding="utf-8", errors="replace").read()

def scene_body(name_sub):
    m = re.search(r'<h2[^>]*>(' + re.escape(name_sub) + r'[^<]*)</h2>(.*?)(?=<h2|\Z)', src, re.S)
    if not m: raise SystemExit(f"scene not found: {name_sub}")
    return m.group(2)

def tier_cells(body, tier_i):
    figs = re.findall(r'<figure[^>]*>(.*?)</figure>', re.split(r'<h3[^>]*>', body)[1:][tier_i], re.S)
    cells = []
    for i, f in enumerate(figs):
        m = re.search(r'src=["\'](data:image[^"\']+)["\']', f)
        cap = re.search(r'<figcaption>(.*?)</figcaption>', f, re.S)
        cap = re.sub(r'\s+', ' ', html.unescape(re.sub('<[^>]+>', ' ', cap.group(1)))).strip() if cap else ""
        cells.append({"img": m.group(1) if m else None, "method": METHODS[i % 5],
                      "channel": CHANNELS[i // 5], "cap": cap})
    return cells

def cell(cells, channel, method):
    for c in cells:
        if c["channel"] == channel and c["method"] == method: return c
    raise SystemExit(f"missing {channel}/{method}")

def vals(cap):
    iou = re.search(r'IoU ([\d.]+)', cap)
    ms  = re.search(r'([\d.]+) ms', cap)
    return (iou.group(1) if iou else "?"), (ms.group(1) if ms else "?")

# (scene, tier, [methods], figure caption template) ------------------------
FIGS = [
    ("Grazing Plane · extreme-grazing", 0, ["Standard", "Segment"],
     "Grazing plane viewed at a near-tangent angle, both methods given the same "
     "wall-clock budget (≈{ms} ms/frame). Plain sphere tracing under-marches the "
     "skimming surface; segment tracing resolves it. Top row: shaded colour; bottom: depth."),
    ("Mandelbulb · ortho", 0, ["Standard", "Naive-Relaxed"],
     "Mandelbulb, a non-metric field (Lipschitz > 1) on which the distance estimate "
     "overshoots, at equal wall-clock budget (≈{ms} ms/frame). Standard must take "
     "conservative steps and lags the relaxed stepper."),
    ("Thin Torus · grazing-edge", 0, ["Standard", "Segment"],
     "Thin torus seen edge-on, equal budget (≈{ms} ms/frame). The counter-example: "
     "segment tracing tunnels through the thin ring (IoU 0.05) and never reaches the "
     "accuracy bar, while plain sphere tracing is correct."),
]

def render_figure(scene, tier_i, methods, captpl, n):
    cells = tier_cells(scene_body(scene), tier_i)
    ms_ref = vals(cell(cells, "color", "Standard")["cap"])[1]
    panels = ""
    for me in methods:
        iou, _ = vals(cell(cells, "color", me)["cap"])
        col = cell(cells, "color", me)["img"]; dep = cell(cells, "depth", me)["img"]
        panels += (f'<div class="panel"><div class="plab">{html.escape(me)} '
                   f'<span class="iou">IoU {iou}</span></div>'
                   f'<img src="{col}" alt="{me} colour"><img src="{dep}" class="d" alt="{me} depth"></div>')
    cap = captpl.format(ms=ms_ref)
    name = html.escape(scene)
    return (f'<figure class="rf"><div class="panels">{panels}</div>'
            f'<figcaption><span class="fn">Figure {n}.</span> {cap}</figcaption></figure>')

fig_html = "\n".join(render_figure(s, t, m, c, i + 1) for i, (s, t, m, c) in enumerate(FIGS))

CSS = """
  :root{--ink:#1b1b19;--mut:#5c5c57;--line:#d9d7cf;--rule:#2a2a26;--bg:#fbfaf7;--accent:#5a3e2b}
  *{box-sizing:border-box}
  html{-webkit-text-size-adjust:100%}
  body{margin:0;background:var(--bg);color:var(--ink);
    font:18px/1.62 Charter,"Iowan Old Style",Georgia,"Times New Roman",serif}
  .col{max-width:720px;margin:0 auto;padding:0 26px}
  h1{font-size:27px;line-height:1.25;font-weight:700;margin:0 0 6px;letter-spacing:-.01em}
  .byline{color:var(--mut);font-size:14px;letter-spacing:.02em;margin:0}
  h2{font-size:19px;font-weight:700;margin:42px 0 10px;letter-spacing:-.005em}
  p{margin:0 0 16px}
  a{color:var(--accent)}
  .mut{color:var(--mut)}
  .mono{font-family:"SF Mono",ui-monospace,Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}
  em{font-style:italic}

  header{padding:54px 0 26px;border-bottom:1px solid var(--line)}
  .kick{font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:var(--mut);margin:0 0 16px}
  .abstract{margin:26px 0 8px;font-size:16.5px;line-height:1.6}
  .abstract b{font-variant:small-caps;font-weight:700;letter-spacing:.03em;font-size:15px;margin-right:6px}

  section{padding:6px 0}
  /* tables — booktabs style */
  table{border-collapse:collapse;width:100%;margin:6px 0 8px;font-size:15.5px}
  caption{caption-side:bottom;text-align:left;color:var(--mut);font-size:14px;line-height:1.5;
    padding-top:9px}
  caption .tn{color:var(--ink);font-weight:700}
  thead th{border-bottom:1.5px solid var(--rule);text-align:left;padding:6px 12px 6px 0;font-weight:700}
  th.r,td.r{text-align:right;padding-right:0}
  tbody td,tbody th{padding:5px 12px 5px 0;border-bottom:1px solid var(--line);font-weight:400}
  tbody tr:last-child td,tbody tr:last-child th{border-bottom:1.5px solid var(--rule)}
  td.num{font-family:"SF Mono",ui-monospace,Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}
  .win{font-weight:700}
  table.tight{margin-top:14px}

  /* render figures */
  figure.rf{margin:22px 0 6px}
  .panels{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  @media(max-width:560px){.panels{grid-template-columns:1fr}}
  .panel{border:1px solid var(--line);background:#fff}
  .plab{font-size:13.5px;padding:6px 9px;border-bottom:1px solid var(--line);
    font-family:system-ui,sans-serif;display:flex;justify-content:space-between;gap:8px}
  .plab .iou{color:var(--mut);font-variant-numeric:tabular-nums}
  .panel img{width:100%;display:block;background:#000;aspect-ratio:1/1;object-fit:cover}
  .panel img.d{border-top:1px solid var(--line);filter:saturate(.15)}
  figure.rf figcaption{font-size:14px;color:var(--mut);line-height:1.55;margin-top:10px}
  figure.rf .fn{color:var(--ink);font-weight:700}

  ul{margin:0 0 16px;padding-left:22px}
  li{margin:4px 0}
  footer{margin:40px 0 60px;padding-top:18px;border-top:1px solid var(--line);
    color:var(--mut);font-size:14px;line-height:1.55}
  footer code{font-family:"SF Mono",ui-monospace,Menlo,Consolas,monospace;font-size:13px}
  hr.sep{border:0;border-top:1px solid var(--line);margin:0}
"""

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Which sphere-tracing algorithm is fastest, and on what geometry</title>
<style>%CSS%</style>
</head>
<body>
<header><div class="col">
  <p class="kick">Ray-marching benchmark &middot; technical note</p>
  <h1>Which sphere-tracing algorithm is fastest,<br>and on what geometry?</h1>
  <p class="byline">9 strategies &middot; 38 scene&middot;viewpoints &middot; 12,236 scored runs &middot; NVIDIA RTX&nbsp;3090 &middot; June&nbsp;2026</p>
  <p class="abstract"><b>Abstract</b> We benchmark nine sphere-tracing strategies across 38
  scene&middot;viewpoints, scoring each run geometrically (intersection-over-union of the hit
  mask) against a dense-march oracle and reporting cost on two independent axes: SDF
  evaluations per ray and GPU wall-clock time. Plain sphere tracing is the fastest method
  on only 13 of 38 viewpoints by wall-clock; adaptive variants take the remainder. The
  choice of winner is governed almost entirely by a single, strategy-independent property
  of the view &mdash; the fraction of rays that approach the surface at a grazing angle &mdash;
  and the two cost axes do not agree on a ranking. Where the geometry forces small steps
  (grazing incidence, thin or non-metric surfaces) adaptive methods are 2&ndash;8&times; faster;
  on smooth, well-conditioned fields the baseline is both fastest and most reliable.</p>
</div></header>

<div class="col">

<section>
<h2>1.&ensp;Setup</h2>
<p>Each strategy is run on every scene&middot;viewpoint over a sweep of iteration budgets
(32&ndash;2048) and matched target residuals, at 512&sup2; resolution. Accuracy is the IoU of the
rendered hit mask against a dense-march reference validated to ≈1e&minus;7 on analytic
scenes. Cost is measured three ways &mdash; GPU milliseconds (timer queries), SDF
evaluations per ray, and warp divergence &mdash; and never collapsed into a single score. A
method &ldquo;wins&rdquo; a viewpoint if it is the cheapest to reach IoU&nbsp;&ge;&nbsp;0.98. The nine
strategies are Standard, Overstep-Bisect, Naive-Relaxed, Segment, Enhanced, Naive-Auto-Relaxed,
Skipping-Spheres, Reverse-AA, and Safe-Relaxed (Keinert).</p>
</section>

<section>
<h2>2.&ensp;The cost axis determines the winner</h2>
<p>Counting wins across all 38 viewpoints (Table&nbsp;1), the baseline shares the lead on
wall-clock time but is decisively beaten on work done. The two axes disagree because
SDF-evaluation count does not charge a method for the per-step arithmetic and warp
divergence it incurs: Enhanced needs the fewest evaluations almost everywhere yet rarely
wins in milliseconds.</p>
<table>
<caption><span class="tn">Table&nbsp;1.</span> Number of viewpoints (of 38) on which each
method is the cheapest to reach IoU&nbsp;&ge;&nbsp;0.98, by cost axis.</caption>
<thead><tr><th>Method</th><th class="r">Wins by GPU&nbsp;ms</th><th class="r">Wins by SDF&nbsp;evals</th></tr></thead>
<tbody>
<tr><th>Standard</th><td class="num r">13</td><td class="num r">3</td></tr>
<tr><th>Safe-Relaxed</th><td class="num r">13</td><td class="num r">3</td></tr>
<tr><th>Naive-Relaxed</th><td class="num r">9</td><td class="num r">15</td></tr>
<tr><th>Enhanced</th><td class="num r">0</td><td class="num r">14</td></tr>
<tr><th>Segment</th><td class="num r">2</td><td class="num r">0</td></tr>
<tr><th>Naive-Auto-Relaxed</th><td class="num r">1</td><td class="num r">3</td></tr>
</tbody></table>
</section>

<section>
<h2>3.&ensp;The deciding property is grazing incidence</h2>
<p>Each viewpoint carries strategy-independent geometric features. Ranking them by how
strongly they separate the winning method (Table&nbsp;2), one dominates: the fraction of
rays that grazes the surface, followed by the variance of step hardness. The mechanism is
simple &mdash; adaptive tracers earn their cost only where sphere tracing is forced into many
small steps (grazing incidence; thin or non-metric surfaces). Smooth fields converge in
roughly ten steps and leave nothing to recover.</p>
<table>
<caption><span class="tn">Table&nbsp;2.</span> Per-feature separation of the wall-clock winner,
as normalised spread of the per-method mean (larger&nbsp;=&nbsp;more discriminative).</caption>
<thead><tr><th>Feature</th><th class="r">Separation</th></tr></thead>
<tbody>
<tr><th>grazing fraction</th><td class="num r">4.26</td></tr>
<tr><th>step-hardness CV</th><td class="num r">2.88</td></tr>
<tr><th>thin-slab fraction</th><td class="num r">1.63</td></tr>
<tr><th>step-hardness mean</th><td class="num r">1.25</td></tr>
<tr><th>hit rate</th><td class="num r">1.07</td></tr>
<tr><th>Lipschitz p99</th><td class="num r">0.98</td></tr>
<tr><th>silhouette complexity</th><td class="num r">0.52</td></tr>
</tbody></table>
</section>

<section>
<h2>4.&ensp;Magnitude of the adaptive advantage</h2>
<p>On views where the geometry forces small steps, the fastest adaptive method reaches the
accuracy bar in a fraction of the baseline's wall-clock time (Table&nbsp;3). The grazing-plane
case is the most defensible: it is a metric field on which the dense-march oracle is
provably sound, so the speed-up is certifiable rather than an artefact of oracle error.</p>
<table>
<caption><span class="tn">Table&nbsp;3.</span> Wall-clock time (GPU&nbsp;ms) to reach
IoU&nbsp;&ge;&nbsp;0.98 for the fastest method versus Standard, on step-limited views.</caption>
<thead><tr><th>Scene&middot;view</th><th>Fastest method</th><th class="r">Fastest&nbsp;ms</th><th class="r">Standard&nbsp;ms</th><th class="r">Factor</th></tr></thead>
<tbody>
<tr><th>Mandelbulb &middot; ortho</th><td>Naive-Relaxed</td><td class="num r">0.54</td><td class="num r">4.18</td><td class="num r win">7.8&times;</td></tr>
<tr><th>Sphere Cloud &middot; ortho</th><td>Safe-Relaxed</td><td class="num r">0.46</td><td class="num r">2.65</td><td class="num r win">5.8&times;</td></tr>
<tr><th>Thin Planes &middot; ortho</th><td>Safe-Relaxed</td><td class="num r">0.92</td><td class="num r">4.22</td><td class="num r win">4.6&times;</td></tr>
<tr><th>Grazing Plane &middot; extreme</th><td>Segment</td><td class="num r">0.36</td><td class="num r">0.86</td><td class="num r win">2.4&times;</td></tr>
</tbody></table>
</section>

<section>
<h2>5.&ensp;The transition occurs within a single scene</h2>
<p>The dependence on grazing incidence is not a confound of scene identity. Holding the
plane and the code fixed and changing only the camera angle moves the winner (Table&nbsp;4):
at a grazing angle segment tracing leads; viewed head-on, the same plane converges in a
handful of steps and nothing beats the baseline.</p>
<table class="tight">
<caption><span class="tn">Table&nbsp;4.</span> Same plane, two camera angles. Wall-clock ms to
the accuracy bar.</caption>
<thead><tr><th>View</th><th class="r">grazing frac.</th><th>Winner</th><th class="r">Winner&nbsp;ms</th><th class="r">Standard&nbsp;ms</th></tr></thead>
<tbody>
<tr><th>grazing</th><td class="num r">0.37</td><td class="win">Segment</td><td class="num r">0.36</td><td class="num r">0.86</td></tr>
<tr><th>steep (head-on)</th><td class="num r">0.00</td><td class="win">Standard</td><td class="num r">0.10</td><td class="num r">0.10</td></tr>
</tbody></table>
</section>

<section>
<h2>6.&ensp;Equal-time renders</h2>
<p>The following frames give each method the same wall-clock budget per frame, so the
differences are what an equal compute budget actually buys. IoU is the geometric agreement
with the oracle.</p>
%FIGS%
</section>

<section>
<h2>7.&ensp;Where the baseline remains optimal</h2>
<p>On smooth, low-grazing, well-conditioned distance fields the adaptive methods are
frequently <em>structurally capped</em> &mdash; unable to reach the accuracy bar at any budget
&mdash; while the baseline is both fastest and safest. This holds across all three views of the
Thin Torus and the Hollow-Cube (CSG) shell, and on the head-on Cube, Sphere, and Onion-Shell
views. Two caveats temper the adaptive results:</p>
<ul>
<li>Segment tracing is capped everywhere except the grazing plane; its advantage is real
and certifiable but narrow, and it fails on curvature and thin features (Figure&nbsp;3).</li>
<li>Naive-Relaxed leads many wall-clock rows but is the <em>naive</em> over-relaxation, which
can tunnel thin features; its wins clear the IoU&nbsp;&ge;&nbsp;0.98 bar but warrant visual
confirmation before being relied upon.</li>
</ul>
</section>

<footer>
<hr class="sep">
<p style="margin-top:16px">Reproduced from <code>sweep_grid512.jsonl</code> (12,236 rows, 2026-06-06)
via <code>report.analyze</code> and <code>report.discovery --cost {evals,ms}</code>; equal-time
frames from the perceptual A/B harness. Oracle and method definitions are described in the
project's <code>SWEEP_PLAN.md</code> and <code>FINDINGS.md</code>.</p>
</footer>

</div>
</body>
</html>
"""

out = HTML.replace("%CSS%", CSS).replace("%FIGS%", fig_html)
open(DST, "w", encoding="utf-8").write(out)
print(f"Wrote {DST}: {len(out)//1024} KB, {out.count('data:image')} embedded frames, "
      f"{out.count('src=&quot;None&quot;') + out.count('src=\"None\"')} unfilled.")
