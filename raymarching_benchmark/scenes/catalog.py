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
            # Fold into positive octant with repetition. NOTE: phase matches the
            # GLSL `mod(p*s, 2.0) - 1.0` exactly (no +1 offset) so the Python
            # feature SDF and the GPU-rendered SDF describe the *same* fractal —
            # a prerequisite for the discovery join (see SWEEP_PLAN §6.5).
            a = Vec3(
                ((p.x * s) % 2.0) - 1.0,
                ((p.y * s) % 2.0) - 1.0,
                ((p.z * s) % 2.0) - 1.0,
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
# Scene 15: Sphere Cloud (expensive metric SDF)
# ──────────────────────────────────────────────

class SphereCloudScene(SDFScene):
    """Union of 24 fixed spheres. Purpose: an SDF that is EXPENSIVE to evaluate
    (map() = min over 24 primitives, ~24× a single sphere) yet still **metric**
    (union of metric SDFs is 1-Lipschitz) so the dense-march oracle stays sound.
    This is the regime that gives adaptive tracers a fair shot — saving distance
    evals saves real wall-clock — *with* certifiable accuracy, unlike the fractals.
    Positions/radii are hardcoded identically in scenes.glsl (sceneId 14)."""

    # (center, radius) — generated once (seed 7), mirrored verbatim in GLSL.
    SPHERES = [
        ((0.4253, 1.3505, 0.9373), 0.4723), ((-0.9343, -0.6794, 1.2701), 0.4257),
        ((-1.6821, 1.0922, 1.0100), 0.3090), ((-0.1090, -0.6697, -0.7534), 0.4659),
        ((-0.8334, -0.1867, 0.0155), 0.4879), ((0.1819, 1.6847, 0.9951), 0.4789),
        ((0.4154, 1.6625, -0.9680), 0.4053), ((-1.1553, 0.3826, -1.5506), 0.3120),
        ((-1.5787, 0.0506, -0.1149), 0.3223), ((1.4184, 0.4394, 0.0480), 0.4841),
        ((-0.0106, -0.8584, -1.6599), 0.4015), ((-1.0458, 0.6529, -1.0179), 0.3197),
        ((-0.4436, -1.6873, 1.1222), 0.4745), ((-1.1748, -0.7902, 1.2931), 0.4211),
        ((0.0333, 1.1803, 0.4750), 0.4053), ((0.8220, -1.3889, 0.1399), 0.3628),
        ((0.0264, 1.2626, -0.4717), 0.3704), ((0.3338, -1.4985, -0.3821), 0.3327),
        ((-0.6017, -1.1893, 1.0755), 0.2884), ((-0.4099, 1.6277, 0.3060), 0.4728),
        ((0.3572, 0.4692, 0.5999), 0.3829), ((-1.1873, -0.2029, -0.8855), 0.4005),
        ((-0.3315, -1.3712, 1.5906), 0.3509), ((-0.9690, 0.5840, -0.6786), 0.4453),
    ]

    @property
    def name(self): return "Sphere Cloud"

    @property
    def description(self):
        return "Union of 24 fixed spheres. Expensive metric SDF: costly eval, sound oracle."

    @property
    def category(self): return "expensive_metric"

    def sdf(self, p: Vec3) -> float:
        d = 1e10
        for (cx, cy, cz), r in self.SPHERES:
            d = min(d, sd_sphere(op_translate(p, Vec3(cx, cy, cz)), r))
        return d

    def suggested_camera(self) -> Optional[RenderConfig]:
        return RenderConfig(camera_position=(0.0, 0.0, 7.0), camera_target=(0.0, 0.0, 0.0))


# ──────────────────────────────────────────────
# Scene 16: Bumpy Sphere (expensive metric + grazing crawl + thin bumps)
# ──────────────────────────────────────────────

class BumpySphereScene(SDFScene):
    """Base sphere (r=1.4) + 30 protruding bump-spheres (r=0.18) on a fibonacci
    lattice. Designed to combine the THREE conditions that produce an adaptive-
    tracer win (see SWEEP_PLAN §6.5): (1) expensive eval (31 primitives), (2)
    grazing crawl at the limb + between bumps (where Standard takes tiny steps),
    (3) thin protrusions (where naive over-relaxation tunnels but safe/predictive
    does not). Metric ⇒ the dense-march oracle is sound, so any win is certifiable
    — unlike the Mandelbulb. Bump centers mirrored verbatim in scenes.glsl (id 15)."""

    BASE_R = 1.4
    BUMP_R = 0.18
    BUMPS = [
        (0.3841, 1.4500, 0.0000), (-0.4821, 1.3500, 0.4417), (0.0725, 1.2500, -0.8260),
        (0.5860, 1.1500, 0.7643), (-1.0548, 1.0500, -0.1866), (0.9794, 0.9500, -0.6230),
        (-0.3209, 0.8500, 1.1935), (-0.5987, 0.7500, -1.1528), (1.2698, 0.6500, 0.4637),
        (-1.2900, 0.5500, 0.5325), (0.6065, 0.4500, -1.2960), (0.4365, 0.3500, 1.3917),
        (-1.2797, 0.2500, -0.7416), (1.4577, 0.1500, -0.3205), (-0.8622, 0.0500, 1.2264),
        (-0.1927, -0.0500, -1.4867), (1.1412, -0.1500, 0.9618), (-1.4778, -0.2500, 0.0611),
        (1.0339, -0.3500, -1.0289), (-0.0661, -0.4500, 1.4294), (-0.8941, -0.5500, -1.0715),
        (1.3398, -0.6500, 0.1803), (-1.0663, -0.7500, 0.7419), (0.2713, -0.8500, -1.2058),
        (0.5771, -0.9500, 1.0072), (-1.0205, -1.0500, -0.3256), (0.8743, -1.1500, -0.4039),
        (-0.3201, -1.2500, 0.7649), (-0.2213, -1.3500, -0.6152), (0.3400, -1.4500, 0.1787),
    ]

    @property
    def name(self): return "Bumpy Sphere"

    @property
    def description(self):
        return "Sphere + 30 bumps. Expensive metric SDF with grazing crawl + thin features."

    @property
    def category(self): return "expensive_metric"

    def sdf(self, p: Vec3) -> float:
        d = sd_sphere(p, self.BASE_R)
        for cx, cy, cz in self.BUMPS:
            d = min(d, sd_sphere(op_translate(p, Vec3(cx, cy, cz)), self.BUMP_R))
        return d

    def suggested_camera(self) -> Optional[RenderConfig]:
        return RenderConfig(camera_position=(0.0, 0.0, 5.0), camera_target=(0.0, 0.0, 0.0))


# ──────────────────────────────────────────────
# Scene 17: Gyroid solid (periodic curved surface, grazing-rich)
# ──────────────────────────────────────────────

class GyroidScene(SDFScene):
    """A gyroid labyrinth { g(p) < 0 } clipped to a ball. The raw gyroid implicit
    g = Σ sin·cos is NON-metric (|∇g| ≤ freq·2√3), so we divide by that bound to
    get a conservative 1-Lipschitz under-estimator — zero on the surface, never
    overshooting — which keeps the dense-march oracle sound. The draw is a smooth,
    periodic, highly-curved surface with broad grazing walls: a regime no other
    scene covers, and a clean test of whether grazing-rich curvature alone (cheap
    eval) triggers an adaptive-tracer win. Mirrored verbatim in scenes.glsl (id 16).
    """
    FREQ = 3.0
    LIP = FREQ * 2.0 * (3.0 ** 0.5)   # = 10.3923…, the gradient bound of g
    RADIUS = 2.2

    @property
    def name(self): return "Gyroid"

    @property
    def description(self):
        return "Gyroid labyrinth clipped to a ball. Smooth periodic curved surface; grazing-rich, metric."

    @property
    def category(self): return "periodic_surface"

    def sdf(self, p: Vec3) -> float:
        qx, qy, qz = self.FREQ * p.x, self.FREQ * p.y, self.FREQ * p.z
        g = (math.sin(qx) * math.cos(qy)
             + math.sin(qy) * math.cos(qz)
             + math.sin(qz) * math.cos(qx))
        sheet = g / self.LIP
        ball = sd_sphere(p, self.RADIUS)
        return max(sheet, ball)

    def suggested_camera(self) -> Optional[RenderConfig]:
        return RenderConfig(camera_position=(0.0, 0.0, 6.0), camera_target=(0.0, 0.0, 0.0))


# ──────────────────────────────────────────────
# Scene 18: Capped Torus (open thin ring)
# ──────────────────────────────────────────────

class CappedTorusScene(SDFScene):
    """iq's open ("C"-shaped) torus: thin like Thin Torus but with an *open*
    boundary edge, adding silhouette and a grazing-prone gap. Exact-ish metric
    SDF (1-Lipschitz). Mirrored verbatim in scenes.glsl (id 17)."""

    SC = (math.sin(2.0), math.cos(2.0))   # half-angle 2.0 rad → wide arc
    RA = 1.2
    RB = 0.2

    @property
    def name(self): return "Capped Torus"

    @property
    def description(self):
        return "Open C-shaped torus. Thin feature with a boundary edge → extra silhouette + grazing."

    @property
    def category(self): return "thin_features"

    def sdf(self, p: Vec3) -> float:
        return sd_capped_torus(p, self.SC, self.RA, self.RB)

    def suggested_camera(self) -> Optional[RenderConfig]:
        return RenderConfig(camera_position=(0.0, 0.0, 4.5), camera_target=(0.0, 0.0, 0.0))


# ──────────────────────────────────────────────
# Scene 19: Finite Box Lattice (certifiable near-miss throughput)
# ──────────────────────────────────────────────

class BoxLatticeScene(SDFScene):
    """A finite 5×5×5 grid of boxes via iq's limited domain repetition. Folding
    by a per-cell-constant integer translation is distance-preserving, so this is
    fully METRIC (oracle-sound) — the certifiable counterpart to the benched
    Pillar Forest. Many near-miss rays thread the 0.4-wide gaps between boxes,
    stressing empty-space traversal and silhouette. NOTE: rounding uses
    floor(x+0.5), NOT round(), because GLSL round() breaks ties in an
    implementation-defined direction (parity hazard). Mirrored in scenes.glsl
    (id 18)."""

    C = 1.0      # cell spacing
    L = 2.0      # half-extent in cells → indices −2…2
    BB = 0.3     # box half-size (0.6 wide, 0.4 gap)

    @property
    def name(self): return "Box Lattice"

    @property
    def description(self):
        return "Finite 5×5×5 box grid (metric). Many near-miss rays through the gaps; throughput + silhouette."

    @property
    def category(self): return "near_miss"

    def sdf(self, p: Vec3) -> float:
        def cell(x: float) -> float:
            r = math.floor(x / self.C + 0.5)        # deterministic round
            r = max(-self.L, min(self.L, r))
            return r
        q = Vec3(p.x - self.C * cell(p.x),
                 p.y - self.C * cell(p.y),
                 p.z - self.C * cell(p.z))
        return sd_box(q, Vec3(self.BB, self.BB, self.BB))

    def suggested_camera(self) -> Optional[RenderConfig]:
        return RenderConfig(camera_position=(0.0, 0.0, 7.0), camera_target=(0.0, 0.0, 0.0))


# ──────────────────────────────────────────────
# Scene 20: Metaballs (smooth-union blobs)
# ──────────────────────────────────────────────

class MetaballsScene(SDFScene):
    """Six spheres fused with the canonical polynomial smin (now identical in
    Python and GLSL — see op_smooth_union). smin under-estimates min(), so it
    never overshoots → oracle-sound. Smooth-blend regime companion to the
    un-deferred Smooth Blend. Mirrored verbatim in scenes.glsl (id 19)."""

    K = 0.45
    # (center, radius) — central blob + 5 lobes.
    BALLS = [
        ((0.0, 0.0, 0.0), 0.8),
        ((1.0, 0.0, 0.0), 0.6),
        ((-1.0, 0.0, 0.0), 0.6),
        ((0.0, 1.0, 0.0), 0.6),
        ((0.0, -1.0, 0.0), 0.6),
        ((0.0, 0.0, 1.0), 0.6),
    ]

    @property
    def name(self): return "Metaballs"

    @property
    def description(self):
        return "Six spheres fused with polynomial smin. Smooth-union regime; metric (under-estimating)."

    @property
    def category(self): return "smooth"

    def sdf(self, p: Vec3) -> float:
        (cx, cy, cz), r = self.BALLS[0]
        d = sd_sphere(op_translate(p, Vec3(cx, cy, cz)), r)
        for (cx, cy, cz), r in self.BALLS[1:]:
            d = op_smooth_union(d, sd_sphere(op_translate(p, Vec3(cx, cy, cz)), r), self.K)
        return d

    def suggested_camera(self) -> Optional[RenderConfig]:
        return RenderConfig(camera_position=(0.0, 0.0, 5.0), camera_target=(0.0, 0.0, 0.0))


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
        SphereCloudScene(),
        BumpySphereScene(),
        GyroidScene(),        # sceneId 16
        CappedTorusScene(),   # sceneId 17
        BoxLatticeScene(),    # sceneId 18
        MetaballsScene(),     # sceneId 19
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