"""Python↔GLSL SDF parity guard.

Every grid-eligible scene's Python `sdf()` (used by features.py on CPU) must match
its GLSL `map_impl` branch (used by the GPU sweep) — a drift between the two files
silently corrupts the feature→winner join and is exactly what benched Smooth Blend,
Pillar Forest, and Bad-Lipschitz historically (see SWEEP_PLAN §6.5).

GPU evaluation can't run in this (non-interactive) session, so instead of probing
the shader we transcribe each GLSL branch *literally and independently* here, then
assert it equals the catalog `sdf()` over a cloud of random points. This catches the
real failure mode — the two source files describing different geometry — without a
GPU. (It can't catch GLSL-compiler quirks, but for these arithmetic SDFs that risk
is negligible; true GPU parity is exercised on the next user-run sweep.)
"""
import math
import random

from raymarching_benchmark.core.vec3 import Vec3
from raymarching_benchmark.scenes.catalog import get_scene_by_name


# ── Independent transcriptions of the GLSL map_impl branches (scenes.glsl). ──
# Written straight from the shader text, NOT from catalog.py, on purpose.

def _glsl_sphere(p):  # sceneId 0 — sanity anchor
    return p.length() - 1.0


def _glsl_box(p, b):
    q = Vec3(abs(p.x) - b, abs(p.y) - b, abs(p.z) - b)
    outside = Vec3(max(q.x, 0.0), max(q.y, 0.0), max(q.z, 0.0)).length()
    return outside + min(max(q.x, max(q.y, q.z)), 0.0)


def _glsl_smooth_union(d1, d2, k):
    h = max(0.0, min(1.0, 0.5 + 0.5 * (d2 - d1) / k))
    return (d2 * (1.0 - h) + d1 * h) - k * h * (1.0 - h)


def _glsl_gyroid(p):  # sceneId 16
    qx, qy, qz = p.x * 3.0, p.y * 3.0, p.z * 3.0
    g = (math.sin(qx) * math.cos(qy)
         + math.sin(qy) * math.cos(qz)
         + math.sin(qz) * math.cos(qx))
    sheet = g / 10.392304845413264
    ball = p.length() - 2.2
    return max(sheet, ball)


def _glsl_capped_torus(p):  # sceneId 17
    scx, scy = 0.9092974268256817, -0.4161468365471424
    px = abs(p.x)
    k = (px * scx + p.y * scy) if (scy * px > scx * p.y) else math.sqrt(px * px + p.y * p.y)
    return math.sqrt(px * px + p.y * p.y + p.z * p.z + 1.2 * 1.2 - 2.0 * 1.2 * k) - 0.2


def _glsl_box_lattice(p):  # sceneId 18
    cx = max(-2.0, min(2.0, math.floor(p.x + 0.5)))
    cy = max(-2.0, min(2.0, math.floor(p.y + 0.5)))
    cz = max(-2.0, min(2.0, math.floor(p.z + 0.5)))
    return _glsl_box(Vec3(p.x - cx, p.y - cy, p.z - cz), 0.3)


def _glsl_metaballs(p):  # sceneId 19
    d = p.length() - 0.8
    for (cx, cy, cz) in [(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                         (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)]:
        d = _glsl_smooth_union(d, Vec3(p.x - cx, p.y - cy, p.z - cz).length() - 0.6, 0.45)
    return d


def _glsl_smooth_blend(p):  # sceneId 7 — un-deferred by the smin fix
    d1 = Vec3(p.x + 0.5, p.y, p.z).length() - 0.8
    d2 = _glsl_box(Vec3(p.x - 0.5, p.y, p.z), 0.6)
    return _glsl_smooth_union(d1, d2, 0.5)


_CASES = {
    "Sphere": _glsl_sphere,
    "Smooth Blend": _glsl_smooth_blend,
    "Gyroid": _glsl_gyroid,
    "Capped Torus": _glsl_capped_torus,
    "Box Lattice": _glsl_box_lattice,
    "Metaballs": _glsl_metaballs,
}


def test_python_glsl_sdf_parity():
    rng = random.Random(1234)
    N = 6000
    for scene_name, glsl_fn in _CASES.items():
        scene = get_scene_by_name(scene_name)
        assert scene is not None, f"scene {scene_name!r} not found"
        worst = 0.0
        for _ in range(N):
            p = Vec3(rng.uniform(-3.5, 3.5),
                     rng.uniform(-3.5, 3.5),
                     rng.uniform(-3.5, 3.5))
            a = scene.sdf(p)
            b = glsl_fn(p)
            worst = max(worst, abs(a - b))
        assert worst < 1e-6, f"{scene_name}: Python↔GLSL SDF drift {worst:.3e} (>1e-6)"
