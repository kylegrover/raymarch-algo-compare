"""Catalog of all test scenes with stress-test characteristics."""

import math
from typing import List, Optional, Dict
from .base import SDFScene
from .primitives import *
from ..core.vec3 import Vec3
from ..config import RenderConfig


# ──────────────────────────────────────────────
# Scene 1: Simple Sphere
# ──────────────────────────────────────────────

class SphereScene(SDFScene):
    @property
    def name(self): return "Sphere"

    @property
    def description(self): return "Unit sphere at origin. Baseline: all methods should handle easily."

    @property
    def category(self): return "smooth"

    def sdf(self, p: Vec3) -> float:
        return sd_sphere(p, 1.0)


# ──────────────────────────────────────────────
# Scene 2: Ground Plane (grazing angle stress test)
# ──────────────────────────────────────────────

class GrazingPlaneScene(SDFScene):
    @property
    def name(self): return "Grazing Plane"

    @property
    def description(self):
        return "Plane viewed at <5 degree angle. Worst-case for standard sphere tracing (tiny steps)."

    @property
    def category(self): return "stress_grazing"

    def sdf(self, p: Vec3) -> float:
        return sd_plane(p, Vec3(0.0, 1.0, 0.0), -0.5)

    def suggested_camera(self) -> Optional[RenderConfig]:
        return RenderConfig(
            camera_position=(0.0, 0.6, 8.0),
            camera_target=(0.0, -0.4, 0.0),
        )


# ──────────────────────────────────────────────
# Scene 3: Unit Cube (sharp edges)
# ──────────────────────────────────────────────

class CubeScene(SDFScene):
    @property
    def name(self): return "Cube"

    @property
    def description(self): return "Unit cube. Tests sharp edges and corners."

    @property
    def category(self): return "sharp_edges"

    def sdf(self, p: Vec3) -> float:
        return sd_box(p, Vec3(1.0, 1.0, 1.0))


# ──────────────────────────────────────────────
# Scene 4: Thin Torus
# ──────────────────────────────────────────────

class ThinTorusScene(SDFScene):
    @property
    def name(self): return "Thin Torus"

    @property
    def description(self):
        return "Torus with small minor radius. Tests thin feature handling."

    @property
    def category(self): return "thin_features"

    def sdf(self, p: Vec3) -> float:
        return sd_torus(p, 1.5, 0.05)


# ──────────────────────────────────────────────
# Scene 5: Cylinder
# ──────────────────────────────────────────────

class CylinderScene(SDFScene):
    @property
    def name(self): return "Cylinder"

    @property
    def description(self): return "Cylinder with sharp circular edges."

    @property
    def category(self): return "sharp_edges"

    def sdf(self, p: Vec3) -> float:
        return sd_cylinder(p, 1.0, 1.5)


# ──────────────────────────────────────────────
# Scene 6: Near-miss multi-object
# ──────────────────────────────────────────────

class NearMissScene(SDFScene):
    @property
    def name(self): return "Near Miss"

    @property
    def description(self):
        return "Two spheres nearly touching. Rays through gap test convergence behavior."

    @property
    def category(self): return "near_miss"

    def sdf(self, p: Vec3) -> float:
        d1 = sd_sphere(op_translate(p, Vec3(-1.01, 0.0, 0.0)), 1.0)
        d2 = sd_sphere(op_translate(p, Vec3(1.01, 0.0, 0.0)), 1.0)
        return op_union(d1, d2)


# ──────────────────────────────────────────────
# Scene 7: CSG Difference (hollow cube)
# ──────────────────────────────────────────────

class HollowCubeScene(SDFScene):
    @property
    def name(self): return "Hollow Cube (CSG)"

    @property
    def description(self):
        return "Cube with sphere subtracted. Tests CSG boundary handling."

    @property
    def category(self): return "csg"

    def sdf(self, p: Vec3) -> float:
        d_box = sd_box(p, Vec3(1.0, 1.0, 1.0))
        d_sphere = sd_sphere(p, 1.3)
        return op_subtract(d_box, d_sphere)


# ──────────────────────────────────────────────
# Scene 8: Smooth blended shapes
# ──────────────────────────────────────────────

class SmoothBlendScene(SDFScene):
    @property
    def name(self): return "Smooth Blend"

    @property
    def description(self):
        return "Smoothly blended sphere+box. Tests smooth union SDF quality."

    @property
    def category(self): return "smooth"

    def sdf(self, p: Vec3) -> float:
        d1 = sd_sphere(op_translate(p, Vec3(-0.5, 0.0, 0.0)), 0.8)
        d2 = sd_box(op_translate(p, Vec3(0.5, 0.0, 0.0)), Vec3(0.6, 0.6, 0.6))
        return op_smooth_union(d1, d2, 0.5)


# ──────────────────────────────────────────────
# Scene 9: Onion Shell (thin features)
# ──────────────────────────────────────────────

class OnionShellScene(SDFScene):
    @property
    def name(self): return "Onion Shell"

    @property
    def description(self):
        return "Nested onion shells of a sphere. Extreme thin feature stress test."

    @property
    def category(self): return "thin_features"

    def sdf(self, p: Vec3) -> float:
        d = sd_sphere(p, 2.0)
        d = op_onion(d, 0.1)
        d = op_onion(d, 0.05)
        return d


# ──────────────────────────────────────────────
# Scene 10: Menger Sponge (fractal)
# ──────────────────────────────────────────────

class MengerSpongeScene(SDFScene):
    def __init__(self, iterations: int = 3):
        self._iterations = iterations

    @property
    def name(self): return f"Menger Sponge (iter={self._iterations})"

    @property
    def description(self):
        return "Menger sponge fractal. Tests tunneling and high iteration demands."

    @property
    def category(self): return "fractal"

    def sdf(self, p: Vec3) -> float:
        d = sd_box(p, Vec3(1.0, 1.0, 1.0))
        s = 1.0

        for i in range(self._iterations):
            # Fold into positive octant with repetition
            a = Vec3(
                ((p.x * s + 1.0) % 2.0) - 1.0,
                ((p.y * s + 1.0) % 2.0) - 1.0,
                ((p.z * s + 1.0) % 2.0) - 1.0,
            )
            s *= 3.0
            r = Vec3(
                abs(1.0 - 3.0 * abs(a.x)),
                abs(1.0 - 3.0 * abs(a.y)),
                abs(1.0 - 3.0 * abs(a.z)),
            )

            # Cross-shaped hole
            da = max(r.x, r.y)
            db = max(r.y, r.z)
            dc = max(r.z, r.x)
            c = (min(da, min(db, dc)) - 1.0) / s

            d = max(d, c)

        return d


# ──────────────────────────────────────────────
# Scene 11: Mandelbulb (infinite curvature)
# ──────────────────────────────────────────────

class MandelbulbScene(SDFScene):
    def __init__(self, power: float = 8.0, max_iter: int = 8):
        self._power = power
        self._max_iter = max_iter

    @property
    def name(self): return "Mandelbulb"

    @property
    def description(self):
        return "Mandelbulb fractal. Infinite curvature breaks planar assumptions."

    @property
    def category(self): return "fractal"

    def known_lipschitz_bound(self) -> Optional[float]:
        return None  # SDF is approximate

    def sdf(self, p: Vec3) -> float:
        z = Vec3(p.x, p.y, p.z)
        dr = 1.0
        r = 0.0

        for _ in range(self._max_iter):
            r = z.length()
            if r > 4.0:
                break

            # Convert to polar coordinates
            theta = math.acos(max(-1.0, min(1.0, z.z / max(r, 1e-12))))
            phi = math.atan2(z.y, z.x)

            dr = r ** (self._power - 1.0) * self._power * dr + 1.0

            # Scale and rotate
            zr = r ** self._power
            theta *= self._power
            phi *= self._power

            z = Vec3(
                zr * math.sin(theta) * math.cos(phi),
                zr * math.sin(theta) * math.sin(phi),
                zr * math.cos(theta),
            ) + p

        return 0.5 * math.log(max(r, 1e-12)) * r / max(dr, 1e-12)

    def suggested_camera(self) -> Optional[RenderConfig]:
        return RenderConfig(
            camera_position=(0.0, 0.0, 3.0),
            camera_target=(0.0, 0.0, 0.0),
        )


# ──────────────────────────────────────────────
# Scene 12: Bad Lipschitz Sphere
# ──────────────────────────────────────────────

class BadLipschitzSphereScene(SDFScene):
    @property
    def name(self): return "Bad Lipschitz Sphere"

    @property
    def description(self):
        return "Sphere with SDF scaled by 2x (invalid). Tests overshoot recovery."

    @property
    def category(self): return "invalid_sdf"

    def known_lipschitz_bound(self) -> Optional[float]:
        return 2.0  # Intentionally wrong

    def sdf(self, p: Vec3) -> float:
        return (p.length() - 1.0) * 2.0


# ──────────────────────────────────────────────
# Scene 13: Pillars forest (many near-misses)
# ──────────────────────────────────────────────

class PillarsScene(SDFScene):
    @property
    def name(self): return "Pillar Forest"

    @property
    def description(self):
        return "Grid of thin cylinders. Many near-miss rays, tests throughput."

    @property
    def category(self): return "complex"

    def sdf(self, p: Vec3) -> float:
        # Repeat in XZ
        q = op_repeat(p, Vec3(2.0, 0.0, 2.0))
        d_pillar = sd_cylinder(q, 0.15, 3.0)
        d_floor = sd_plane(p, Vec3(0.0, 1.0, 0.0), -3.0)
        return op_union(d_pillar, d_floor)

    def suggested_camera(self) -> Optional[RenderConfig]:
        return RenderConfig(
            camera_position=(1.0, 1.0, 8.0),
            camera_target=(0.0, 0.0, 0.0),
        )


# ──────────────────────────────────────────────
# Scene 14: Stacked thin planes
# ──────────────────────────────────────────────

class ThinPlanesScene(SDFScene):
    @property
    def name(self): return "Thin Planes Stack"

    @property
    def description(self):
        return "Multiple thin parallel planes. Tests tunneling through thin geometry."

    @property
    def category(self): return "thin_features"

    def sdf(self, p: Vec3) -> float:
        # Repeated thin shell
        d = sd_plane(p, Vec3(0.0, 1.0, 0.0), 0.0)
        d = abs(d)  # Infinite plane shell at y=0
        # Now repeat in Y
        spacing = 0.5
        py_mod = ((p.y + spacing * 0.5) % spacing) - spacing * 0.5
        q = Vec3(p.x, py_mod, p.z)
        d_shell = abs(sd_plane(q, Vec3(0.0, 1.0, 0.0), 0.0)) - 0.01
        return d_shell

    def suggested_camera(self) -> Optional[RenderConfig]:
        return RenderConfig(
            camera_position=(0.0, 0.25, 5.0),
            camera_target=(0.0, 0.25, 0.0),
        )


# ──────────────────────────────────────────────
# Scene registry
# ──────────────────────────────────────────────

def get_all_scenes() -> List[SDFScene]:
    """Return all registered test scenes."""
    return [
        SphereScene(),
        GrazingPlaneScene(),
        CubeScene(),
        ThinTorusScene(),
        CylinderScene(),
        NearMissScene(),
        HollowCubeScene(),
        SmoothBlendScene(),
        OnionShellScene(),
        MengerSpongeScene(iterations=3),
        MandelbulbScene(),
        BadLipschitzSphereScene(),
        PillarsScene(),
        ThinPlanesScene(),
    ]


def get_scene_by_name(name: str) -> Optional[SDFScene]:
    """Find scene by name (case-insensitive exact match or start match)."""
    name_low = name.lower().replace(' ', '')
    scenes = get_all_scenes()
    
    # 1. Try exact (no spaces)
    for scene in scenes:
        if scene.name.lower().replace(' ', '') == name_low:
            return scene
            
    # 2. Try starts-with
    for scene in scenes:
        if scene.name.lower().replace(' ', '').startswith(name_low):
            return scene
            
    return None


def get_scenes_by_category(category: str) -> List[SDFScene]:
    """Get all scenes in a category."""
    return [s for s in get_all_scenes() if s.category == category]