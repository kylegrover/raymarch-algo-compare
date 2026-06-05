"""Curated canonical viewpoints per scene (SWEEP_PLAN §E).

A single hand-framed camera is a cherry-picking risk and grazing behavior is
extremely viewpoint-sensitive, so each scene gets a few explicitly-categorized
viewpoints and a high-quality reference is rendered *per viewpoint*. Categories:

  * orthogonal — straight-on / easy framing (baseline).
  * grazing    — low-angle skim (high step counts; worst case for sphere tracing).
  * macro      — close-up (thin-feature resolution).
  * interior   — inside / claustrophobic (high bounding-volume overlap).

Curated tables exist for the core scenes; everything else falls back to its
``suggested_camera`` (or an origin-facing default) as a single orthogonal view.
Cameras are framed by *distance* — the GPU pinhole ignores fov, the basis is all
that matters (effective vertical fov ≈ 53°).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple

from .config import RenderConfig


@dataclass(frozen=True)
class Viewpoint:
    name: str
    category: str                       # orthogonal | grazing | macro | interior
    position: Tuple[float, float, float]
    target: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0)

    def render_config(self, width: int, height: int) -> RenderConfig:
        return RenderConfig(width=width, height=height,
                            camera_position=self.position,
                            camera_target=self.target,
                            camera_up=self.up)


# Hand-curated per core scene. Kept deliberately small (2–3 each) so the grid
# stays tractable; categories chosen to stress different failure modes.
_CURATED = {
    "Sphere": [
        Viewpoint("ortho", "orthogonal", (0.0, 0.0, 3.2)),
        Viewpoint("macro", "macro", (0.0, 0.0, 1.7)),
    ],
    "Grazing Plane": [
        Viewpoint("steep", "orthogonal", (0.0, 4.0, 4.0), (0.0, -0.5, 0.0)),
        Viewpoint("grazing", "grazing", (0.0, 0.6, 8.0), (0.0, -0.4, 0.0)),
        Viewpoint("extreme-grazing", "grazing", (0.0, 0.28, 13.0), (0.0, -0.46, 0.0)),
    ],
    "Cube": [
        Viewpoint("face", "orthogonal", (0.0, 0.0, 3.6)),
        Viewpoint("corner", "orthogonal", (2.4, 2.0, 2.6)),
        Viewpoint("grazing-face", "grazing", (3.4, 0.0, 0.5)),
    ],
    "Thin Torus": [
        Viewpoint("ring-face", "orthogonal", (0.0, 0.0, 4.0)),
        Viewpoint("grazing-edge", "grazing", (0.0, 0.35, 4.0)),
        Viewpoint("macro", "macro", (0.0, 0.0, 2.2)),
    ],
    "Mandelbulb": [
        Viewpoint("ortho", "orthogonal", (0.0, 0.0, 3.0)),
        Viewpoint("macro", "macro", (0.0, 0.0, 1.9)),
        Viewpoint("angled", "orthogonal", (2.0, 1.4, 2.0)),
    ],
    # ── Phase 6 expansion: Python≡GLSL-verified scenes, multi-angle. ──────────
    "Cylinder": [   # r=1, half-height=1.5, axis=Y → sharp circular cap edges
        Viewpoint("side", "orthogonal", (0.0, 0.0, 4.2)),
        Viewpoint("cap-grazing", "grazing", (3.6, 1.55, 0.6)),   # skim the top edge
        Viewpoint("macro", "macro", (0.0, 0.0, 2.4)),
    ],
    "Near Miss": [  # two r=1 spheres at x=±1.01 → pinched gap
        Viewpoint("ortho", "orthogonal", (0.0, 0.0, 5.2)),
        Viewpoint("gap-grazing", "grazing", (0.0, 2.6, 4.4), (0.0, 0.0, 0.0)),
        Viewpoint("macro-gap", "macro", (0.0, 0.0, 3.0)),
    ],
    "Hollow Cube (CSG)": [  # box(1) − sphere(1.3): concave spherical bites
        Viewpoint("face", "orthogonal", (0.0, 0.0, 4.0)),
        Viewpoint("corner", "orthogonal", (2.4, 2.0, 2.6)),
        Viewpoint("grazing-face", "grazing", (3.4, 0.0, 0.5)),
    ],
    "Onion Shell": [  # nested thin shells of a sphere (~r=2.15 outer)
        Viewpoint("ortho", "orthogonal", (0.0, 0.0, 5.4)),
        Viewpoint("grazing", "grazing", (0.0, 0.55, 5.4), (0.0, 0.0, 0.0)),
        Viewpoint("macro", "macro", (0.0, 0.0, 3.2)),
    ],
    "Thin Planes Stack": [  # repeated 0.01-thick shells, spacing 0.5 → tunneling
        Viewpoint("ortho", "orthogonal", (0.0, 0.25, 5.0), (0.0, 0.25, 0.0)),
        Viewpoint("grazing", "grazing", (0.0, 0.12, 6.2), (0.0, 0.05, 0.0)),
        Viewpoint("edge", "orthogonal", (3.0, 0.25, 4.0), (0.0, 0.25, 0.0)),
    ],
}


def viewpoints_for(scene) -> List[Viewpoint]:
    """Curated viewpoints for a scene, or a single sensible default."""
    vps = _CURATED.get(scene.name)
    if vps:
        return list(vps)
    sc = scene.suggested_camera()
    if sc is not None:
        return [Viewpoint("default", "orthogonal",
                          tuple(sc.camera_position), tuple(sc.camera_target),
                          tuple(sc.camera_up))]
    return [Viewpoint("default", "orthogonal", (0.0, 0.0, 5.0))]
