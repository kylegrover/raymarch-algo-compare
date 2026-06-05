"""Closed-form ray/surface intersection for scenes with analytic depth.

This is the *independent* anchor of the oracle-trust chain: it shares no code
and no algorithm with sphere tracing, so the residual between it and the
dense-march reference is a real, un-circular error bar (validation gate #1).

It reproduces the GPU camera-ray generation **exactly** so per-pixel depths line
up with captured arrays:

  * the fragment shader uses ``uv = (gl_FragCoord.xy - 0.5*resolution)/resolution.y``
    and ``rd = normalize(camDir + uv.x*camRight + uv.y*camUp)`` — note it ignores
    ``fov_degrees`` entirely; only the camera *basis* matters;
  * ``GPURunner.capture`` flips the framebuffer vertically (``arr[::-1]``) to a
    top-left image origin. We generate rays directly in that flipped layout, so
    ``analytic_capture(...)['depth']`` is pixel-aligned with ``capture()['depth']``.

Closed forms: Sphere (quadratic), infinite Plane (linear), Cube (slab method),
Thin Torus (quartic; our ``sd_torus`` has its major circle in the xz-plane /
axis = y, so the implicit form is ``(|P|^2+R^2-r^2)^2 = 4R^2(Px^2+Pz^2)``).
"""
from __future__ import annotations
import numpy as np
from typing import Callable, Dict, Optional

from ..config import RenderConfig


def _normalize(v: np.ndarray, axis=-1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)


def camera_basis(rc: RenderConfig):
    """Replicate core.camera.Camera's orthonormal basis (the vectors the GPU
    receives as camDir/camRight/camUp)."""
    pos = np.asarray(rc.camera_position, dtype=np.float64)
    target = np.asarray(rc.camera_target, dtype=np.float64)
    up0 = np.asarray(rc.camera_up, dtype=np.float64)
    forward = _normalize(target - pos)
    right = _normalize(np.cross(forward, up0))
    true_up = _normalize(np.cross(right, forward))
    return pos, forward, right, true_up


def generate_rays(rc: RenderConfig):
    """Return (ro, rd) where ro is (3,) and rd is (H, W, 3), normalized, in the
    same top-left layout that GPURunner.capture returns."""
    W, H = rc.width, rc.height
    pos, forward, right, up = camera_basis(rc)

    cols = np.arange(W, dtype=np.float64)
    rows = np.arange(H, dtype=np.float64)
    fx = cols + 0.5                       # gl_FragCoord.x at pixel centers
    # Stored row r -> GL pixel row (H-1-r); fy = (H-1-r)+0.5 = H-0.5-r
    fy = (H - 0.5) - rows

    uvx = (fx - 0.5 * W) / H              # (W,)
    uvy = (fy - 0.5 * H) / H              # (H,)
    UVX, UVY = np.meshgrid(uvx, uvy)      # (H, W) each

    rd = (forward[None, None, :]
          + UVX[..., None] * right[None, None, :]
          + UVY[..., None] * up[None, None, :])
    rd = _normalize(rd, axis=-1)
    return pos, rd


def _empty(H: int, W: int):
    return (np.zeros((H, W)), np.zeros((H, W), dtype=bool), np.zeros((H, W, 3)))


# ── primitives ──────────────────────────────────────────────────────────────

def intersect_sphere(ro, rd, radius: float = 1.0, center=(0.0, 0.0, 0.0)):
    H, W, _ = rd.shape
    oc = ro - np.asarray(center, dtype=np.float64)          # (3,)
    b = rd @ oc                                              # (H,W) since |rd|=1
    c = float(oc @ oc) - radius * radius
    disc = b * b - c
    hit = disc >= 0.0
    sq = np.sqrt(np.maximum(disc, 0.0))
    t0 = -b - sq
    t1 = -b + sq
    # nearest positive root
    t = np.where(t0 > 1e-6, t0, t1)
    hit = hit & (t > 1e-6)
    depth = np.where(hit, t, 0.0)
    pos = ro[None, None, :] + depth[..., None] * rd
    normal = np.where(hit[..., None], _normalize(pos - np.asarray(center)), 0.0)
    return depth, hit, normal


def intersect_plane(ro, rd, normal=(0.0, 1.0, 0.0), h: float = -0.5):
    H, W, _ = rd.shape
    n = np.asarray(normal, dtype=np.float64)
    denom = rd @ n                                           # (H,W)
    safe = np.abs(denom) > 1e-12
    t = np.where(safe, (h - (ro @ n)) / np.where(safe, denom, 1.0), -1.0)
    hit = safe & (t > 1e-6)
    depth = np.where(hit, t, 0.0)
    # face the camera
    nrm = n if (rd @ n).mean() < 0 else n
    normal_map = np.zeros((H, W, 3))
    normal_map[hit] = _normalize(n[None, :])[0]
    return depth, hit, normal_map


def intersect_box(ro, rd, half=(1.0, 1.0, 1.0)):
    H, W, _ = rd.shape
    b = np.asarray(half, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        invd = 1.0 / rd                                     # (H,W,3)
        t1 = (-b[None, None, :] - ro[None, None, :]) * invd
        t2 = (b[None, None, :] - ro[None, None, :]) * invd
    tmin = np.minimum(t1, t2)
    tmax = np.maximum(t1, t2)
    # ray parallel to a slab and outside it -> no hit on that axis
    parallel = ~np.isfinite(invd)
    outside = parallel & ((ro[None, None, :] < -b[None, None, :]) |
                          (ro[None, None, :] > b[None, None, :]))
    tmin = np.where(parallel, -np.inf, tmin)
    tmax = np.where(parallel, np.inf, tmax)

    axis_near = np.argmax(tmin, axis=-1)                    # slab that sets tnear
    tnear = np.max(tmin, axis=-1)
    tfar = np.min(tmax, axis=-1)
    hit = (tnear <= tfar) & (tfar > 1e-6) & ~np.any(outside, axis=-1)
    t = np.where(tnear > 1e-6, tnear, tfar)                 # exterior vs interior
    hit = hit & (t > 1e-6)
    depth = np.where(hit, t, 0.0)

    normal = np.zeros((H, W, 3))
    ii, jj = np.nonzero(hit)
    ax = axis_near[ii, jj]
    sgn = -np.sign(rd[ii, jj, ax])
    normal[ii, jj, ax] = sgn
    return depth, hit, normal


def intersect_torus(ro, rd, R: float = 1.5, r: float = 0.05):
    """Solve the torus quartic per ray. Major circle in the xz-plane (axis = y),
    matching ``sd_torus``. Only rays passing within the bounding sphere R+r are
    solved (the rest are guaranteed misses), so np.roots runs over few pixels."""
    H, W, _ = rd.shape
    depth = np.zeros((H, W))
    hit = np.zeros((H, W), dtype=bool)
    normal = np.zeros((H, W, 3))

    O = ro.astype(np.float64)
    # distance^2 from torus center (origin) to each ray line; prune the rest
    OdotD = rd @ O                                          # (H,W)
    dist2 = float(O @ O) - OdotD * OdotD
    bound = (R + r) * 1.01
    candidate = dist2 <= bound * bound
    iy, ix = np.nonzero(candidate)

    a1 = 2.0 * OdotD[iy, ix]                                # |P|^2 = t^2 + a1 t + a0
    a0 = float(O @ O)
    Dx = rd[iy, ix, 0]; Dz = rd[iy, ix, 2]
    Ox, Oz = O[0], O[2]
    # S = |P|^2 + (R^2 - r^2): s2=1, s1=a1, s0=a0 + R^2 - r^2
    s1 = a1
    s0 = a0 + R * R - r * r
    # Px^2 + Pz^2 = q2 t^2 + q1 t + q0
    q2 = Dx * Dx + Dz * Dz
    q1 = 2.0 * (Ox * Dx + Oz * Dz)
    q0 = Ox * Ox + Oz * Oz
    fourR2 = 4.0 * R * R
    # F = S^2 - 4R^2 (Px^2+Pz^2): quartic coeffs (descending)
    c4 = np.ones_like(s1)
    c3 = 2.0 * s1
    c2 = (s1 * s1 + 2.0 * s0) - fourR2 * q2
    c1 = 2.0 * s1 * s0 - fourR2 * q1
    c0 = (s0 * s0 - fourR2 * q0) + np.zeros_like(s1)

    for k in range(len(iy)):
        roots = np.roots([c4[k], c3[k], c2[k], c1[k], c0[k]])
        real = roots[np.abs(roots.imag) < 1e-7].real
        real = real[real > 1e-6]
        if real.size == 0:
            continue
        t = float(real.min())
        depth[iy[k], ix[k]] = t
        hit[iy[k], ix[k]] = True

    # SDF-gradient normals at the hits
    ih, iw = np.nonzero(hit)
    P = O[None, :] + depth[ih, iw][:, None] * rd[ih, iw]
    q = np.sqrt(P[:, 0] ** 2 + P[:, 2] ** 2)
    w0 = q - R
    w1 = P[:, 1]
    wlen = np.sqrt(w0 * w0 + w1 * w1)
    n2x = w0 / np.maximum(wlen, 1e-12)
    n2y = w1 / np.maximum(wlen, 1e-12)
    qn = np.maximum(q, 1e-12)
    grad = np.stack([n2x * P[:, 0] / qn, n2y, n2x * P[:, 2] / qn], axis=-1)
    normal[ih, iw] = _normalize(grad)
    return depth, hit, normal


# ── registry ────────────────────────────────────────────────────────────────

ANALYTIC_SCENES: Dict[str, Callable] = {
    "Sphere": lambda ro, rd: intersect_sphere(ro, rd, radius=1.0),
    "Grazing Plane": lambda ro, rd: intersect_plane(ro, rd, (0.0, 1.0, 0.0), -0.5),
    "Cube": lambda ro, rd: intersect_box(ro, rd, (1.0, 1.0, 1.0)),
    "Thin Torus": lambda ro, rd: intersect_torus(ro, rd, R=1.5, r=0.05),
}


def has_analytic(scene_name: str) -> bool:
    return scene_name in ANALYTIC_SCENES


def analytic_capture(scene_name: str, rc: RenderConfig) -> Optional[Dict[str, np.ndarray]]:
    """Closed-form depth/hit/normal for a scene, pixel-aligned with capture().

    Returns None if the scene has no closed form.
    """
    fn = ANALYTIC_SCENES.get(scene_name)
    if fn is None:
        return None
    ro, rd = generate_rays(rc)
    depth, hit, normal = fn(ro, rd)
    return {"depth": depth, "hit": hit, "normal": normal}
