"""Backfill the gallery <img src="None"> placeholders in findings_presentation.html
with real base64 frames from perceptual_review.html, in document order.
Pure file I/O (no GPU). Idempotent-ish: only touches src="None" slots."""
import re, html

SRC = "perceptual_review.html"
DST = "findings_presentation.html"
METHODS = ["Standard", "Enhanced", "Naive-Relaxed", "Safe-Relaxed", "Segment"]
CHANNELS = ["color", "depth", "discont"]
src = open(SRC, encoding="utf-8", errors="replace").read()

def scene_body(name_sub):
    m = re.search(r'<h2[^>]*>(' + re.escape(name_sub) + r'[^<]*)</h2>(.*?)(?=<h2|\Z)', src, re.S)
    if not m:
        raise SystemExit(f"scene not found: {name_sub}")
    return m.group(2)

def tier_cells(body, tier_i):
    tiers = re.split(r'<h3[^>]*>', body)[1:]
    figs = re.findall(r'<figure[^>]*>(.*?)</figure>', tiers[tier_i], re.S)
    cells = []
    for i, f in enumerate(figs):
        # source uses single OR double quotes around the data URI
        m = re.search(r'src=["\'](data:image[^"\']+)["\']', f)
        cells.append({"img": m.group(1) if m else None,
                      "method": METHODS[i % 5], "channel": CHANNELS[i // 5]})
    return cells

def pick(cells, channel, method):
    for c in cells:
        if c["channel"] == channel and c["method"] == method:
            if not c["img"]:
                raise SystemExit(f"no img for {channel}/{method}")
            return c["img"]
    raise SystemExit(f"cell not found {channel}/{method}")

# Must match the card/image order already written into the HTML:
# per comparison, per method: color img then depth img.
COMPARISONS = [
    ("Grazing Plane · extreme-grazing", 0, ["Standard", "Segment"]),
    ("Mandelbulb · ortho",              0, ["Standard", "Naive-Relaxed"]),
    ("Thin Torus · grazing-edge",       0, ["Standard", "Segment"]),
]
ordered = []
for scene, ti, methods in COMPARISONS:
    cells = tier_cells(scene_body(scene), ti)
    for me in methods:
        ordered.append(pick(cells, "color", me))
        ordered.append(pick(cells, "depth", me))

dst = open(DST, encoding="utf-8").read()
n_slots = dst.count('src="None"')
if n_slots != len(ordered):
    raise SystemExit(f"slot mismatch: {n_slots} src=None vs {len(ordered)} images")

def repl(_m, _it=iter(ordered)):
    return 'src="' + next(_it) + '"'
dst = re.sub(r'src="None"', repl, dst)
open(DST, "w", encoding="utf-8").write(dst)
print(f"Filled {n_slots} image slots. New size: {len(dst)//1024} KB, "
      f"{dst.count('data:image')} data URIs.")
