# Ray Marching Algorithm Analytical Testing System

I'll build a comprehensive, modular system in Python with optional GPU acceleration via Taichi. This system benchmarks multiple ray marching strategies across stress-test scenes, collecting deep performance metrics and producing comparative analysis.

## Project Structure

```
raymarching_benchmark/
├── main.py                  # Entry point and orchestrator
├── core/
│   ├── __init__.py
│   ├── vec3.py              # Vector math primitives
│   ├── ray.py               # Ray representation
│   ├── camera.py            # Camera and ray generation
│   └── types.py             # Shared types and result structures
├── scenes/
│   ├── __init__.py
│   ├── base.py              # Abstract SDF scene interface
│   ├── primitives.py        # SDF primitive functions
│   └── catalog.py           # All test scene definitions
├── strategies/
│   ├── __init__.py
│   ├── base.py              # Abstract strategy interface
│   ├── standard_sphere.py   # Standard sphere tracing
│   ├── overstep_bisect.py   # Overstep + bisection
│   ├── adaptive_hybrid.py   # Adaptive hybrid
│   ├── relaxed_sphere.py    # Over-relaxed sphere tracing
│   ├── auto_relaxed.py      # Auto-relaxed (AR-ST)
│   ├── enhanced_sphere.py   # Enhanced sphere tracing (planar assumption)
│   └── segment_tracing.py   # Segment tracing (Lipschitz bounds)
├── metrics/
│   ├── __init__.py
│   ├── collector.py         # Per-ray and aggregate metric collection
│   └── analyzer.py          # Statistical analysis and comparison
├── visualization/
│   ├── __init__.py
│   ├── tables.py            # Console and file table output
│   ├── heatmaps.py          # Iteration/cost heatmap rendering
│   └── charts.py            # Comparative charts
└── config.py                # Global configuration
```

## Implementation

### `config.py`

```python
"""Global configuration for the ray marching benchmark system."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RenderConfig:
    """Rendering resolution and camera settings."""
    width: int = 320
    height: int = 240
    fov_degrees: float = 60.0
    camera_position: tuple = (0.0, 0.0, 5.0)
    camera_target: tuple = (0.0, 0.0, 0.0)
    camera_up: tuple = (0.0, 1.0, 0.0)


@dataclass
class MarchConfig:
    """Default marching parameters (strategies may override)."""
    max_iterations: int = 512
    hit_threshold: float = 1e-4
    max_distance: float = 100.0
    min_step_fraction: float = 0.01  # For overstep methods
    initial_relaxation: float = 1.6  # For relaxed sphere tracing
    bisection_steps: int = 10  # For bisection phase
    stuck_threshold: int = 5  # Consecutive small steps before switching
    stuck_step_ratio: float = 0.001  # Step / max_distance ratio for "stuck"


@dataclass
class BenchmarkConfig:
    """Overall benchmark configuration."""
    render: RenderConfig = field(default_factory=RenderConfig)
    march: MarchConfig = field(default_factory=MarchConfig)
    num_warmup_runs: int = 1
    num_timed_runs: int = 3
    output_dir: str = "results"
    save_heatmaps: bool = True
    save_renders: bool = True
    verbose: bool = True
```

### `core/__init__.py`

```python
from .vec3 import Vec3
from .ray import Ray
from .camera import Camera
from .types import MarchResult, RayMarchStats

__all__ = ['Vec3', 'Ray', 'Camera', 'MarchResult', 'RayMarchStats']
```

### `core/vec3.py`

```python
"""Lightweight vector3 class using numpy for batch operations and pure Python for single vectors."""

import numpy as np
from typing import Union


class Vec3:
    """3D vector with standard operations. Supports both scalar and numpy-backed batch operations."""

    __slots__ = ('x', 'y', 'z')

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __add__(self, other: 'Vec3') -> 'Vec3':
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Vec3') -> 'Vec3':
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> 'Vec3':
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> 'Vec3':
        return self.__mul__(scalar)

    def __neg__(self) -> 'Vec3':
        return Vec3(-self.x, -self.y, -self.z)

    def __truediv__(self, scalar: float) -> 'Vec3':
        inv = 1.0 / scalar
        return Vec3(self.x * inv, self.y * inv, self.z * inv)

    def dot(self, other: 'Vec3') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'Vec3') -> 'Vec3':
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def length(self) -> float:
        return (self.x * self.x + self.y * self.y + self.z * self.z) ** 0.5

    def length_squared(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z

    def normalized(self) -> 'Vec3':
        l = self.length()
        if l < 1e-12:
            return Vec3(0.0, 0.0, 0.0)
        return self / l

    def abs(self) -> 'Vec3':
        return Vec3(abs(self.x), abs(self.y), abs(self.z))

    def max_component(self) -> float:
        return max(self.x, self.y, self.z)

    def min_component(self) -> float:
        return min(self.x, self.y, self.z)

    def to_tuple(self) -> tuple:
        return (self.x, self.y, self.z)

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @staticmethod
    def from_array(arr) -> 'Vec3':
        return Vec3(float(arr[0]), float(arr[1]), float(arr[2]))

    def __repr__(self):
        return f"Vec3({self.x:.6f}, {self.y:.6f}, {self.z:.6f})"

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z


def vec3_max(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z))


def vec3_min(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z))


def vec3_clamp(v: Vec3, lo: float, hi: float) -> Vec3:
    return Vec3(
        max(lo, min(hi, v.x)),
        max(lo, min(hi, v.y)),
        max(lo, min(hi, v.z))
    )
```

### `core/ray.py`

```python
"""Ray representation for ray marching."""

from .vec3 import Vec3


class Ray:
    """A ray defined by origin and normalized direction."""

    __slots__ = ('origin', 'direction')

    def __init__(self, origin: Vec3, direction: Vec3):
        self.origin = origin
        self.direction = direction.normalized()

    def at(self, t: float) -> Vec3:
        """Point along ray at parameter t."""
        return self.origin + self.direction * t

    def __repr__(self):
        return f"Ray(origin={self.origin}, dir={self.direction})"
```

### `core/camera.py`

```python
"""Camera for generating rays through image pixels."""

import math
from .vec3 import Vec3
from .ray import Ray


class Camera:
    """Perspective camera that generates rays for each pixel."""

    def __init__(self, position: Vec3, target: Vec3, up: Vec3,
                 fov_degrees: float, width: int, height: int):
        self.position = position
        self.width = width
        self.height = height

        # Build orthonormal basis
        forward = (target - position).normalized()
        right = forward.cross(up).normalized()
        true_up = right.cross(forward).normalized()

        self.forward = forward
        self.right = right
        self.up = true_up

        # Compute image plane dimensions
        aspect = width / height
        fov_rad = math.radians(fov_degrees)
        half_height = math.tan(fov_rad / 2.0)
        half_width = aspect * half_height

        self.half_width = half_width
        self.half_height = half_height

    def get_ray(self, px: int, py: int) -> Ray:
        """Generate ray for pixel (px, py). Uses center-of-pixel sampling."""
        u = (2.0 * (px + 0.5) / self.width - 1.0) * self.half_width
        v = (1.0 - 2.0 * (py + 0.5) / self.height) * self.half_height

        direction = self.forward + self.right * u + self.up * v
        return Ray(self.position, direction)

    def get_all_rays(self):
        """Generator yielding (px, py, ray) for all pixels."""
        for py in range(self.height):
            for px in range(self.width):
                yield px, py, self.get_ray(px, py)
```

### `core/types.py`

```python
"""Shared data types for march results and statistics."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np


@dataclass
class MarchResult:
    """Result of marching a single ray."""
    hit: bool  # Did we find a surface intersection?
    t: float  # Parameter along ray at termination
    iterations: int  # Total iterations used
    final_sdf: float  # SDF value at termination point
    position_x: float = 0.0
    position_y: float = 0.0
    position_z: float = 0.0
    bisection_steps: int = 0  # For methods that use bisection
    stuck_count: int = 0  # Times detected as "stuck"
    strategy_switches: int = 0  # For hybrid methods


@dataclass
class RayMarchStats:
    """Aggregate statistics for a complete render pass."""
    strategy_name: str
    scene_name: str
    total_rays: int = 0
    hit_count: int = 0
    miss_count: int = 0

    # Iteration statistics
    iteration_counts: List[int] = field(default_factory=list)
    iteration_mean: float = 0.0
    iteration_median: float = 0.0
    iteration_std: float = 0.0
    iteration_min: int = 0
    iteration_max: int = 0
    iteration_p95: float = 0.0
    iteration_p99: float = 0.0

    # Accuracy statistics (for hits only)
    final_sdf_values: List[float] = field(default_factory=list)
    accuracy_mean: float = 0.0
    accuracy_max: float = 0.0
    accuracy_std: float = 0.0

    # Convergence
    hit_rate: float = 0.0

    # Timing
    total_time_seconds: float = 0.0
    time_per_ray_us: float = 0.0

    # Warp divergence proxy (std of iterations in local neighborhoods)
    warp_divergence_proxy: float = 0.0

    # Heatmap data (iteration count per pixel)
    iteration_heatmap: Optional[np.ndarray] = None
    hit_map: Optional[np.ndarray] = None

    def compute(self, results: List[MarchResult], width: int, height: int,
                elapsed_seconds: float):
        """Compute all aggregate statistics from raw results."""
        self.total_rays = len(results)
        self.total_time_seconds = elapsed_seconds

        iterations = []
        sdf_values_hits = []
        hit_map = np.zeros((height, width), dtype=bool)
        iter_map = np.zeros((height, width), dtype=np.int32)

        for i, r in enumerate(results):
            py, px = divmod(i, width)
            iterations.append(r.iterations)
            iter_map[py, px] = r.iterations

            if r.hit:
                self.hit_count += 1
                hit_map[py, px] = True
                sdf_values_hits.append(abs(r.final_sdf))
            else:
                self.miss_count += 1

        self.iteration_counts = iterations
        iter_arr = np.array(iterations, dtype=np.float64)

        self.iteration_mean = float(np.mean(iter_arr))
        self.iteration_median = float(np.median(iter_arr))
        self.iteration_std = float(np.std(iter_arr))
        self.iteration_min = int(np.min(iter_arr))
        self.iteration_max = int(np.max(iter_arr))
        self.iteration_p95 = float(np.percentile(iter_arr, 95))
        self.iteration_p99 = float(np.percentile(iter_arr, 99))

        if sdf_values_hits:
            sdf_arr = np.array(sdf_values_hits, dtype=np.float64)
            self.accuracy_mean = float(np.mean(sdf_arr))
            self.accuracy_max = float(np.max(sdf_arr))
            self.accuracy_std = float(np.std(sdf_arr))

        self.hit_rate = self.hit_count / max(self.total_rays, 1)
        self.time_per_ray_us = (elapsed_seconds * 1e6) / max(self.total_rays, 1)

        # Warp divergence proxy: std of iteration counts in 8x4 blocks (simulating 32-thread warps)
        warp_h, warp_w = 4, 8
        block_stds = []
        for by in range(0, height - warp_h + 1, warp_h):
            for bx in range(0, width - warp_w + 1, warp_w):
                block = iter_map[by:by+warp_h, bx:bx+warp_w].flatten().astype(np.float64)
                if len(block) == warp_h * warp_w:
                    block_stds.append(float(np.std(block)))
        self.warp_divergence_proxy = float(np.mean(block_stds)) if block_stds else 0.0

        self.iteration_heatmap = iter_map
        self.hit_map = hit_map
```

### `scenes/__init__.py`

```python
from .base import SDFScene
from .catalog import get_all_scenes, get_scene_by_name

__all__ = ['SDFScene', 'get_all_scenes', 'get_scene_by_name']
```

### `scenes/base.py`

```python
"""Abstract base class for SDF scenes."""

from abc import ABC, abstractmethod
from core.vec3 import Vec3
from config import RenderConfig
from typing import Optional


class SDFScene(ABC):
    """A scene defined by a signed distance function."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable scene name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what this scene tests."""
        ...

    @property
    def category(self) -> str:
        """Scene category for grouping in analysis."""
        return "general"

    @abstractmethod
    def sdf(self, p: Vec3) -> float:
        """Evaluate the signed distance function at point p."""
        ...

    def suggested_camera(self) -> Optional[RenderConfig]:
        """Override to suggest a camera config for this scene."""
        return None

    def known_lipschitz_bound(self) -> Optional[float]:
        """Return the Lipschitz constant if known. 1.0 for valid SDFs."""
        return 1.0
```

### `scenes/primitives.py`

```python
"""SDF primitive functions and combinators."""

import math
from core.vec3 import Vec3, vec3_max, vec3_min


# ──────────────────────────────────────────────
# Primitives
# ──────────────────────────────────────────────

def sd_sphere(p: Vec3, radius: float) -> float:
    return p.length() - radius

def sd_box(p: Vec3, half_extents: Vec3) -> float:
    q = p.abs() - half_extents
    outside = Vec3(max(q.x, 0.0), max(q.y, 0.0), max(q.z, 0.0)).length()
    inside = min(max(q.x, max(q.y, q.z)), 0.0)
    return outside + inside

def sd_plane(p: Vec3, normal: Vec3, offset: float) -> float:
    return p.dot(normal) - offset

def sd_cylinder(p: Vec3, radius: float, half_height: float) -> float:
    d_radial = (p.x * p.x + p.z * p.z) ** 0.5 - radius
    d_height = abs(p.y) - half_height
    outside = (max(d_radial, 0.0) ** 2 + max(d_height, 0.0) ** 2) ** 0.5
    inside = min(max(d_radial, d_height), 0.0)
    return outside + inside

def sd_torus(p: Vec3, major_radius: float, minor_radius: float) -> float:
    q_xz = (p.x * p.x + p.z * p.z) ** 0.5 - major_radius
    return (q_xz * q_xz + p.y * p.y) ** 0.5 - minor_radius

def sd_capsule(p: Vec3, a: Vec3, b: Vec3, radius: float) -> float:
    ab = b - a
    ap = p - a
    t = max(0.0, min(1.0, ap.dot(ab) / max(ab.dot(ab), 1e-12)))
    closest = a + ab * t
    return (p - closest).length() - radius

def sd_cone(p: Vec3, angle_rad: float, height: float) -> float:
    """Cone along Y axis, tip at origin, opening downward."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    q_len = (p.x * p.x + p.z * p.z) ** 0.5
    q = Vec3(q_len, p.y, 0.0)
    # 2D cone SDF
    tip_dist = Vec3(q.x, q.y, 0.0).length()
    d1 = q.y - (-height)
    d2 = q.x * c + q.y * s
    return max(-d1, d2)


# ──────────────────────────────────────────────
# Combinators (CSG operations)
# ──────────────────────────────────────────────

def op_union(d1: float, d2: float) -> float:
    return min(d1, d2)

def op_subtract(d1: float, d2: float) -> float:
    """Subtract d2 from d1."""
    return max(d1, -d2)

def op_intersect(d1: float, d2: float) -> float:
    return max(d1, d2)

def op_smooth_union(d1: float, d2: float, k: float) -> float:
    h = max(k - abs(d1 - d2), 0.0) / max(k, 1e-12)
    return min(d1, d2) - h * h * k * 0.25

def op_smooth_subtract(d1: float, d2: float, k: float) -> float:
    return -op_smooth_union(-d1, d2, k)

def op_smooth_intersect(d1: float, d2: float, k: float) -> float:
    return -op_smooth_union(-d1, -d2, k)


# ──────────────────────────────────────────────
# Transformations
# ──────────────────────────────────────────────

def op_translate(p: Vec3, offset: Vec3) -> Vec3:
    return p - offset

def op_repeat(p: Vec3, spacing: Vec3) -> Vec3:
    """Infinite repetition."""
    return Vec3(
        ((p.x + spacing.x * 0.5) % spacing.x) - spacing.x * 0.5 if spacing.x > 0 else p.x,
        ((p.y + spacing.y * 0.5) % spacing.y) - spacing.y * 0.5 if spacing.y > 0 else p.y,
        ((p.z + spacing.z * 0.5) % spacing.z) - spacing.z * 0.5 if spacing.z > 0 else p.z,
    )

def op_round(d: float, radius: float) -> float:
    return d - radius

def op_onion(d: float, thickness: float) -> float:
    return abs(d) - thickness
```

### `scenes/catalog.py`

```python
"""Catalog of all test scenes with stress-test characteristics."""

import math
from typing import List, Optional, Dict
from .base import SDFScene
from .primitives import *
from core.vec3 import Vec3
from config import RenderConfig


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
    """Find scene by name (case-insensitive partial match)."""
    for scene in get_all_scenes():
        if name.lower() in scene.name.lower():
            return scene
    return None


def get_scenes_by_category(category: str) -> List[SDFScene]:
    """Get all scenes in a category."""
    return [s for s in get_all_scenes() if s.category == category]
```

### `strategies/__init__.py`

```python
from .base import MarchStrategy
from .standard_sphere import StandardSphereTracing
from .overstep_bisect import OverstepBisectTracing
from .adaptive_hybrid import AdaptiveHybridTracing
from .relaxed_sphere import RelaxedSphereTracing
from .auto_relaxed import AutoRelaxedSphereTracing
from .enhanced_sphere import EnhancedSphereTracing
from .segment_tracing import SegmentTracing

__all__ = [
    'MarchStrategy',
    'StandardSphereTracing',
    'OverstepBisectTracing',
    'AdaptiveHybridTracing',
    'RelaxedSphereTracing',
    'AutoRelaxedSphereTracing',
    'EnhancedSphereTracing',
    'SegmentTracing',
]


def get_all_strategies():
    """Return default instances of all strategies."""
    return [
        StandardSphereTracing(),
        RelaxedSphereTracing(omega=1.6),
        AutoRelaxedSphereTracing(),
        EnhancedSphereTracing(),
        OverstepBisectTracing(),
        AdaptiveHybridTracing(),
        SegmentTracing(),
    ]
```

### `strategies/base.py`

```python
"""Abstract base class for ray marching strategies."""

from abc import ABC, abstractmethod
from core.ray import Ray
from core.types import MarchResult
from scenes.base import SDFScene
from config import MarchConfig
from typing import Callable


class MarchStrategy(ABC):
    """Abstract ray marching strategy."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for display/comparison."""
        ...

    @property
    def short_name(self) -> str:
        """Short name for tables."""
        return self.name

    @abstractmethod
    def march(self, ray: Ray, sdf_func: Callable, config: MarchConfig) -> MarchResult:
        """
        March a ray through the SDF scene.

        Args:
            ray: The ray to march.
            sdf_func: Callable(Vec3) -> float, the SDF.
            config: Marching configuration.

        Returns:
            MarchResult with all metrics populated.
        """
        ...
```

### `strategies/standard_sphere.py`

```python
"""Standard sphere tracing (naive)."""

from .base import MarchStrategy
from core.ray import Ray
from core.types import MarchResult
from config import MarchConfig
from typing import Callable


class StandardSphereTracing(MarchStrategy):
    """
    Classic sphere tracing: step by SDF distance each iteration.
    t += sdf(pos) until |sdf| < threshold or max_iterations reached.
    """

    @property
    def name(self) -> str:
        return "Standard Sphere Tracing"

    @property
    def short_name(self) -> str:
        return "Standard"

    def march(self, ray: Ray, sdf_func: Callable, config: MarchConfig) -> MarchResult:
        t = 0.0
        iterations = 0

        for i in range(config.max_iterations):
            iterations = i + 1
            pos = ray.at(t)
            d = sdf_func(pos)

            if abs(d) < config.hit_threshold:
                return MarchResult(
                    hit=True, t=t, iterations=iterations, final_sdf=d,
                    position_x=pos.x, position_y=pos.y, position_z=pos.z
                )

            t += d

            if t > config.max_distance:
                break

        pos = ray.at(t)
        d = sdf_func(pos)
        return MarchResult(
            hit=False, t=t, iterations=iterations, final_sdf=d,
            position_x=pos.x, position_y=pos.y, position_z=pos.z
        )
```

### `strategies/relaxed_sphere.py`

```python
"""Over-relaxed sphere tracing with fixed omega."""

from .base import MarchStrategy
from core.ray import Ray
from core.types import MarchResult
from config import MarchConfig
from typing import Callable


class RelaxedSphereTracing(MarchStrategy):
    """
    Over-relaxed sphere tracing (Keinert et al., 2014).
    Steps by omega * sdf(pos), where omega > 1.
    If overshoot detected (step > |sdf| at new position), fallback to safe step.
    """

    def __init__(self, omega: float = 1.6):
        self._omega = omega

    @property
    def name(self) -> str:
        return f"Relaxed Sphere Tracing (ω={self._omega})"

    @property
    def short_name(self) -> str:
        return f"Relaxed(ω={self._omega})"

    def march(self, ray: Ray, sdf_func: Callable, config: MarchConfig) -> MarchResult:
        t = 0.0
        iterations = 0
        prev_d = 0.0
        omega = self._omega

        for i in range(config.max_iterations):
            iterations = i + 1
            pos = ray.at(t)
            d = sdf_func(pos)

            if abs(d) < config.hit_threshold:
                return MarchResult(
                    hit=True, t=t, iterations=iterations, final_sdf=d,
                    position_x=pos.x, position_y=pos.y, position_z=pos.z
                )

            # Check if the previous relaxed step overshot
            # If |prev_step| + |current_d| < |prev_step * omega|, overshoot occurred
            step = d * omega

            # Safety check: if we took an over-relaxed step and the new SDF
            # is small enough that we might have overshot, fall back
            if i > 0 and (prev_d + d) < prev_d * omega:
                # Overshoot detected: fall back to conservative step
                step = d  # Use unrelaxed step
                # Could also reduce omega here

            t += step
            prev_d = d

            if t > config.max_distance:
                break

        pos = ray.at(t)
        d = sdf_func(pos)
        return MarchResult(
            hit=False, t=t, iterations=iterations, final_sdf=d,
            position_x=pos.x, position_y=pos.y, position_z=pos.z
        )
```

### `strategies/auto_relaxed.py`

```python
"""Auto-Relaxed Sphere Tracing (AR-ST) with adaptive omega."""

from .base import MarchStrategy
from core.ray import Ray
from core.types import MarchResult
from config import MarchConfig
from typing import Callable


class AutoRelaxedSphereTracing(MarchStrategy):
    """
    Auto-Relaxed Sphere Tracing.
    Uses exponential smoothing to estimate the local SDF slope along the ray
    and dynamically adjusts the relaxation parameter omega.

    Based on the idea that if SDF values are decreasing rapidly (approaching surface),
    omega should decrease toward 1.0 (conservative), and if SDF values are
    steady/increasing (open space), omega can increase (aggressive).
    """

    def __init__(self, omega_min: float = 1.0, omega_max: float = 2.0,
                 smoothing: float = 0.7, growth_rate: float = 1.05,
                 decay_rate: float = 0.7):
        self._omega_min = omega_min
        self._omega_max = omega_max
        self._smoothing = smoothing
        self._growth_rate = growth_rate
        self._decay_rate = decay_rate

    @property
    def name(self) -> str:
        return "Auto-Relaxed Sphere Tracing"

    @property
    def short_name(self) -> str:
        return "AR-ST"

    def march(self, ray: Ray, sdf_func: Callable, config: MarchConfig) -> MarchResult:
        t = 0.0
        iterations = 0
        omega = 1.2  # Start mildly aggressive
        prev_d = float('inf')
        ema_ratio = 1.0  # Exponential moving average of d[i]/d[i-1]

        for i in range(config.max_iterations):
            iterations = i + 1
            pos = ray.at(t)
            d = sdf_func(pos)

            if abs(d) < config.hit_threshold:
                return MarchResult(
                    hit=True, t=t, iterations=iterations, final_sdf=d,
                    position_x=pos.x, position_y=pos.y, position_z=pos.z
                )

            # Compute ratio of consecutive SDF values
            if prev_d > 1e-10 and i > 0:
                ratio = d / prev_d
                ema_ratio = self._smoothing * ema_ratio + (1.0 - self._smoothing) * ratio

                # If ratio < 1, we're approaching surface -> decrease omega
                # If ratio >= 1, we're in open space -> increase omega
                if ema_ratio < 0.8:
                    omega = max(self._omega_min, omega * self._decay_rate)
                elif ema_ratio > 1.0:
                    omega = min(self._omega_max, omega * self._growth_rate)
                # else: maintain current omega

            step = d * omega

            # Safety: detect overshoot (previous relaxed step was too large)
            if d < 0:
                # We've overshot - back up and go conservative
                t += d  # d is negative, so this backs up
                omega = self._omega_min
                prev_d = abs(d)
                continue

            t += step
            prev_d = d

            if t > config.max_distance:
                break

        pos = ray.at(t)
        d = sdf_func(pos)
        return MarchResult(
            hit=False, t=t, iterations=iterations, final_sdf=d,
            position_x=pos.x, position_y=pos.y, position_z=pos.z
        )
```

### `strategies/enhanced_sphere.py`

```python
"""Enhanced Sphere Tracing with planar surface assumption."""

from .base import MarchStrategy
from core.ray import Ray
from core.types import MarchResult
from config import MarchConfig
from typing import Callable


class EnhancedSphereTracing(MarchStrategy):
    """
    Enhanced Sphere Tracing (Keinert et al., 2014 / Hart improvements).

    Uses the history of previous SDF evaluations to estimate the surface
    as locally planar and predicts the intersection point geometrically.

    Key idea: if we have two consecutive SDF samples d_prev at t_prev and d_curr at t_curr,
    and assume the surface is a plane, the intersection is approximately at:
        t_intersection ≈ t_curr + d_curr * (t_curr - t_prev) / (d_prev - d_curr)

    Falls back to standard sphere tracing when the prediction is unreliable.
    """

    @property
    def name(self) -> str:
        return "Enhanced Sphere Tracing"

    @property
    def short_name(self) -> str:
        return "Enhanced"

    def march(self, ray: Ray, sdf_func: Callable, config: MarchConfig) -> MarchResult:
        t = 0.0
        iterations = 0
        prev_t = 0.0
        prev_d = float('inf')
        use_prediction = False

        for i in range(config.max_iterations):
            iterations = i + 1
            pos = ray.at(t)
            d = sdf_func(pos)

            if abs(d) < config.hit_threshold:
                return MarchResult(
                    hit=True, t=t, iterations=iterations, final_sdf=d,
                    position_x=pos.x, position_y=pos.y, position_z=pos.z
                )

            # Try planar prediction if we have history
            step = d  # Default: standard sphere trace step

            if i > 0 and prev_d > d and d > 0 and (prev_d - d) > 1e-10:
                dt = t - prev_t
                # Linear extrapolation: when will SDF reach 0?
                predicted_step = d * dt / (prev_d - d)

                # Sanity check: prediction should be positive and not too large
                if 0 < predicted_step < d * 3.0:
                    step = predicted_step
                    use_prediction = True
                else:
                    use_prediction = False
            else:
                use_prediction = False

            # Safety: if we overshot (d < 0), back up
            if d < 0:
                # Bisect between prev_t and t
                t = (prev_t + t) * 0.5
                prev_d = abs(d)
                continue

            prev_t = t
            prev_d = d
            t += step

            if t > config.max_distance:
                break

        pos = ray.at(t)
        d = sdf_func(pos)
        return MarchResult(
            hit=False, t=t, iterations=iterations, final_sdf=d,
            position_x=pos.x, position_y=pos.y, position_z=pos.z
        )
```

### `strategies/overstep_bisect.py`

```python
"""Overstep-Bisect ray marching strategy."""

from .base import MarchStrategy
from core.ray import Ray
from core.types import MarchResult
from config import MarchConfig
from typing import Callable


class OverstepBisectTracing(MarchStrategy):
    """
    Two-phase strategy:
    Phase 1 - Sphere trace with enforced minimum step size to avoid getting stuck.
              Track t_near (last SDF > 0) and t_far (first SDF < 0).
    Phase 2 - Once overshoot detected (sign change), bisect between t_near and t_far.
    """

    def __init__(self, min_step_factor: float = 0.01, bisection_steps: int = 16):
        self._min_step_factor = min_step_factor
        self._bisection_steps = bisection_steps

    @property
    def name(self) -> str:
        return "Overstep-Bisect"

    @property
    def short_name(self) -> str:
        return "Overstep-Bisect"

    def march(self, ray: Ray, sdf_func: Callable, config: MarchConfig) -> MarchResult:
        t = 0.0
        iterations = 0
        t_near = 0.0  # Last t where SDF > 0
        t_far = -1.0  # First t where SDF < 0 (invalid until set)
        bisection_steps_used = 0

        min_step = self._min_step_factor  # Minimum step size

        # Phase 1: Forward march with minimum step enforcement
        phase1_budget = config.max_iterations - self._bisection_steps
        for i in range(phase1_budget):
            iterations = i + 1
            pos = ray.at(t)
            d = sdf_func(pos)

            if abs(d) < config.hit_threshold:
                return MarchResult(
                    hit=True, t=t, iterations=iterations, final_sdf=d,
                    position_x=pos.x, position_y=pos.y, position_z=pos.z,
                    bisection_steps=0
                )

            if d > 0:
                t_near = t
                # Take the larger of SDF distance and minimum step
                step = max(d, min_step)
                t += step
            else:
                # Overshot! Record far bound and enter bisection
                t_far = t
                break

            if t > config.max_distance:
                pos = ray.at(t)
                d = sdf_func(pos)
                return MarchResult(
                    hit=False, t=t, iterations=iterations, final_sdf=d,
                    position_x=pos.x, position_y=pos.y, position_z=pos.z,
                    bisection_steps=0
                )

        # Phase 2: Bisection between t_near and t_far
        if t_far > 0:
            for j in range(self._bisection_steps):
                iterations += 1
                bisection_steps_used = j + 1
                t_mid = (t_near + t_far) * 0.5
                pos = ray.at(t_mid)
                d = sdf_func(pos)

                if abs(d) < config.hit_threshold:
                    return MarchResult(
                        hit=True, t=t_mid, iterations=iterations, final_sdf=d,
                        position_x=pos.x, position_y=pos.y, position_z=pos.z,
                        bisection_steps=bisection_steps_used
                    )

                if d > 0:
                    t_near = t_mid
                else:
                    t_far = t_mid

                if (t_far - t_near) < config.hit_threshold:
                    t_mid = (t_near + t_far) * 0.5
                    pos = ray.at(t_mid)
                    d = sdf_func(pos)
                    return MarchResult(
                        hit=True, t=t_mid, iterations=iterations, final_sdf=d,
                        position_x=pos.x, position_y=pos.y, position_z=pos.z,
                        bisection_steps=bisection_steps_used
                    )

            # Bisection converged to interval but not threshold
            t_final = (t_near + t_far) * 0.5
            pos = ray.at(t_final)
            d = sdf_func(pos)
            return MarchResult(
                hit=abs(d) < config.hit_threshold * 10,  # Slightly relaxed
                t=t_final, iterations=iterations, final_sdf=d,
                position_x=pos.x, position_y=pos.y, position_z=pos.z,
                bisection_steps=bisection_steps_used
            )

        # No overshoot detected - miss
        pos = ray.at(t)
        d = sdf_func(pos)
        return MarchResult(
            hit=False, t=t, iterations=iterations, final_sdf=d,
            position_x=pos.x, position_y=pos.y, position_z=pos.z,
            bisection_steps=0
        )
```

### `strategies/adaptive_hybrid.py`

```python
"""Adaptive hybrid: starts with sphere tracing, detects stuck, switches to overstep-bisect."""

from .base import MarchStrategy
from core.ray import Ray
from core.types import MarchResult
from config import MarchConfig
from typing import Callable


class AdaptiveHybridTracing(MarchStrategy):
    """
    Starts with standard sphere tracing for efficiency.
    If it detects it's "stuck" (many consecutive tiny steps), switches to
    the overstep-bisect strategy to force progress.
    """

    def __init__(self, stuck_threshold: int = 5, stuck_step_ratio: float = 0.001,
                 min_step_factor: float = 0.005, bisection_steps: int = 12):
        self._stuck_threshold = stuck_threshold
        self._stuck_step_ratio = stuck_step_ratio
        self._min_step_factor = min_step_factor
        self._bisection_steps = bisection_steps

    @property
    def name(self) -> str:
        return "Adaptive Hybrid"

    @property
    def short_name(self) -> str:
        return "Hybrid"

    def march(self, ray: Ray, sdf_func: Callable, config: MarchConfig) -> MarchResult:
        t = 0.0
        iterations = 0
        consecutive_small = 0
        mode = "sphere"  # "sphere" or "overstep"
        strategy_switches = 0
        stuck_count = 0

        # Overstep mode state
        t_near = 0.0
        t_far = -1.0
        min_step = self._min_step_factor

        for i in range(config.max_iterations):
            iterations = i + 1
            pos = ray.at(t)
            d = sdf_func(pos)

            if abs(d) < config.hit_threshold:
                return MarchResult(
                    hit=True, t=t, iterations=iterations, final_sdf=d,
                    position_x=pos.x, position_y=pos.y, position_z=pos.z,
                    stuck_count=stuck_count,
                    strategy_switches=strategy_switches
                )

            if mode == "sphere":
                # Standard sphere tracing
                step = d

                # Detect stuck condition
                if d > 0 and d < self._stuck_step_ratio * max(t, 1.0):
                    consecutive_small += 1
                else:
                    consecutive_small = 0

                if consecutive_small >= self._stuck_threshold:
                    mode = "overstep"
                    strategy_switches += 1
                    stuck_count += 1
                    t_near = t
                    t_far = -1.0
                    consecutive_small = 0
                    continue  # Re-evaluate in new mode

                if d < 0:
                    # Unexpected overshoot in sphere mode - switch to bisection
                    t_far = t
                    t_near = max(0.0, t + d)  # Back up by |d|
                    mode = "bisect"
                    strategy_switches += 1
                    continue

                t += step

            elif mode == "overstep":
                # Overstep with minimum step enforcement
                if d > 0:
                    t_near = t
                    step = max(d, min_step)
                    t += step
                else:
                    # Overshot - enter bisection
                    t_far = t
                    mode = "bisect"
                    continue

            elif mode == "bisect":
                # Binary search between t_near and t_far
                if t_far < 0:
                    mode = "sphere"
                    continue

                t_mid = (t_near + t_far) * 0.5
                pos = ray.at(t_mid)
                d = sdf_func(pos)

                if abs(d) < config.hit_threshold:
                    return MarchResult(
                        hit=True, t=t_mid, iterations=iterations, final_sdf=d,
                        position_x=pos.x, position_y=pos.y, position_z=pos.z,
                        stuck_count=stuck_count,
                        strategy_switches=strategy_switches
                    )

                if d > 0:
                    t_near = t_mid
                else:
                    t_far = t_mid

                if (t_far - t_near) < config.hit_threshold:
                    t = (t_near + t_far) * 0.5
                    pos = ray.at(t)
                    d = sdf_func(pos)
                    return MarchResult(
                        hit=True, t=t, iterations=iterations, final_sdf=d,
                        position_x=pos.x, position_y=pos.y, position_z=pos.z,
                        stuck_count=stuck_count,
                        strategy_switches=strategy_switches
                    )

                t = t_mid
                continue

            if t > config.max_distance:
                break

        pos = ray.at(t)
        d = sdf_func(pos)
        return MarchResult(
            hit=False, t=t, iterations=iterations, final_sdf=d,
            position_x=pos.x, position_y=pos.y, position_z=pos.z,
            stuck_count=stuck_count,
            strategy_switches=strategy_switches
        )
```

### `strategies/segment_tracing.py`

```python
"""Segment Tracing inspired by Galin et al. 2020."""

import math
from .base import MarchStrategy
from core.ray import Ray
from core.types import MarchResult
from config import MarchConfig
from typing import Callable
from core.vec3 import Vec3


class SegmentTracing(MarchStrategy):
    """
    Segment Tracing: computes Lipschitz bounds over ray segments rather than points.

    For a Lipschitz-1 SDF, the maximum possible step from t is:
        step = (d(t) + d(t + candidate)) / 2  (approximately, using segment analysis)

    Simplified implementation: uses two SDF evaluations per step to get a tighter
    bound on where the surface can be within the segment [t, t + d(t)].

    This doubles the SDF evaluations per iteration but can dramatically reduce
    the total number of steps needed.
    """

    def __init__(self, lipschitz: float = 1.0):
        self._lipschitz = lipschitz

    @property
    def name(self) -> str:
        return "Segment Tracing"

    @property
    def short_name(self) -> str:
        return "Segment"

    def march(self, ray: Ray, sdf_func: Callable, config: MarchConfig) -> MarchResult:
        t = 0.0
        iterations = 0
        L = self._lipschitz

        for i in range(config.max_iterations):
            iterations = i + 1
            pos = ray.at(t)
            d = sdf_func(pos)

            if abs(d) < config.hit_threshold:
                return MarchResult(
                    hit=True, t=t, iterations=iterations, final_sdf=d,
                    position_x=pos.x, position_y=pos.y, position_z=pos.z
                )

            if d < 0:
                # Overshot - back up conservatively
                t -= abs(d) * 0.5
                continue

            # Standard step candidate
            candidate = d / L

            # Evaluate at the end of the candidate segment
            pos_end = ray.at(t + candidate)
            d_end = sdf_func(pos_end)
            iterations += 1  # Count the extra evaluation

            if abs(d_end) < config.hit_threshold:
                t += candidate
                return MarchResult(
                    hit=True, t=t, iterations=iterations, final_sdf=d_end,
                    position_x=pos_end.x, position_y=pos_end.y, position_z=pos_end.z
                )

            if d_end < 0:
                # Surface is within this segment - bisect
                t_lo = t
                t_hi = t + candidate
                for _ in range(8):
                    iterations += 1
                    t_mid = (t_lo + t_hi) * 0.5
                    d_mid = sdf_func(ray.at(t_mid))
                    if abs(d_mid) < config.hit_threshold:
                        pos_mid = ray.at(t_mid)
                        return MarchResult(
                            hit=True, t=t_mid, iterations=iterations, final_sdf=d_mid,
                            position_x=pos_mid.x, position_y=pos_mid.y, position_z=pos_mid.z
                        )
                    if d_mid > 0:
                        t_lo = t_mid
                    else:
                        t_hi = t_mid
                t = (t_lo + t_hi) * 0.5
                pos_final = ray.at(t)
                d_final = sdf_func(ray.at(t))
                return MarchResult(
                    hit=True, t=t, iterations=iterations, final_sdf=d_final,
                    position_x=pos_final.x, position_y=pos_final.y, position_z=pos_final.z
                )

            # Both endpoints safe - use the larger safe step
            # The segment [t, t+candidate] is surface-free
            # We can potentially step further using d_end
            extended_step = candidate + d_end / L
            t += extended_step

            if t > config.max_distance:
                break

        pos = ray.at(t)
        d = sdf_func(pos)
        return MarchResult(
            hit=False, t=t, iterations=iterations, final_sdf=d,
            position_x=pos.x, position_y=pos.y, position_z=pos.z
        )
```

### `metrics/__init__.py`

```python
from .collector import MetricsCollector
from .analyzer import MetricsAnalyzer

__all__ = ['MetricsCollector', 'MetricsAnalyzer']
```

### `metrics/collector.py`

```python
"""Metrics collection for ray marching benchmarks."""

import time
import numpy as np
from typing import List, Callable
from core.camera import Camera
from core.types import MarchResult, RayMarchStats
from strategies.base import MarchStrategy
from scenes.base import SDFScene
from config import MarchConfig


class MetricsCollector:
    """Collects per-ray metrics and computes aggregate statistics."""

    def __init__(self, config: MarchConfig):
        self.config = config

    def benchmark_strategy(self, strategy: MarchStrategy, scene: SDFScene,
                           camera: Camera, verbose: bool = True) -> RayMarchStats:
        """
        Run a complete benchmark of one strategy on one scene.

        Returns aggregate statistics.
        """
        width = camera.width
        height = camera.height
        total_rays = width * height

        sdf_func = scene.sdf

        if verbose:
            print(f"  Benchmarking: {strategy.short_name} on {scene.name} "
                  f"({width}x{height} = {total_rays} rays)...")

        results: List[MarchResult] = []

        start_time = time.perf_counter()

        for py in range(height):
            for px in range(width):
                ray = camera.get_ray(px, py)
                result = strategy.march(ray, sdf_func, self.config)
                results.append(result)

            if verbose and (py + 1) % max(1, height // 10) == 0:
                pct = (py + 1) / height * 100
                elapsed = time.perf_counter() - start_time
                eta = elapsed / (py + 1) * (height - py - 1)
                print(f"    {pct:5.1f}% complete, ETA: {eta:.1f}s")

        elapsed = time.perf_counter() - start_time

        stats = RayMarchStats(
            strategy_name=strategy.short_name,
            scene_name=scene.name
        )
        stats.compute(results, width, height, elapsed)

        if verbose:
            print(f"    Done in {elapsed:.2f}s. "
                  f"Hit rate: {stats.hit_rate:.1%}, "
                  f"Mean iters: {stats.iteration_mean:.1f}, "
                  f"Max iters: {stats.iteration_max}")

        return stats
```

### `metrics/analyzer.py`

```python
"""Statistical analysis and cross-comparison of benchmark results."""

import numpy as np
from typing import List, Dict, Tuple, Optional
from core.types import RayMarchStats
from dataclasses import dataclass


@dataclass
class ComparisonResult:
    """Result of comparing strategies across scenes."""
    scene_name: str
    rankings: List[Tuple[str, float]]  # (strategy_name, score) sorted best-first
    best_strategy: str
    best_mean_iterations: float
    worst_strategy: str
    worst_mean_iterations: float
    speedup_ratio: float  # worst/best


class MetricsAnalyzer:
    """Analyzes and compares benchmark results across strategies and scenes."""

    def __init__(self):
        self.all_stats: List[RayMarchStats] = []

    def add_result(self, stats: RayMarchStats):
        self.all_stats.append(stats)

    def add_results(self, stats_list: List[RayMarchStats]):
        self.all_stats.extend(stats_list)

    def get_strategies(self) -> List[str]:
        return sorted(set(s.strategy_name for s in self.all_stats))

    def get_scenes(self) -> List[str]:
        seen = []
        for s in self.all_stats:
            if s.scene_name not in seen:
                seen.append(s.scene_name)
        return seen

    def get_stat(self, strategy: str, scene: str) -> Optional[RayMarchStats]:
        for s in self.all_stats:
            if s.strategy_name == strategy and s.scene_name == scene:
                return s
        return None

    def compare_by_iterations(self) -> List[ComparisonResult]:
        """Compare strategies by mean iteration count per scene."""
        results = []
        for scene in self.get_scenes():
            rankings = []
            for strategy in self.get_strategies():
                stat = self.get_stat(strategy, scene)
                if stat:
                    rankings.append((strategy, stat.iteration_mean))

            rankings.sort(key=lambda x: x[1])

            if rankings:
                best = rankings[0]
                worst = rankings[-1]
                speedup = worst[1] / max(best[1], 0.01)

                results.append(ComparisonResult(
                    scene_name=scene,
                    rankings=rankings,
                    best_strategy=best[0],
                    best_mean_iterations=best[1],
                    worst_strategy=worst[0],
                    worst_mean_iterations=worst[1],
                    speedup_ratio=speedup
                ))

        return results

    def strategy_summary(self) -> Dict[str, Dict[str, float]]:
        """
        For each strategy, compute summary metrics across all scenes:
        - avg_mean_iterations
        - avg_hit_rate
        - avg_accuracy
        - num_wins (scenes where it had lowest mean iterations)
        - avg_warp_divergence
        """
        comparisons = self.compare_by_iterations()
        strategies = self.get_strategies()

        summary = {}
        for strat in strategies:
            stats_for_strat = [s for s in self.all_stats if s.strategy_name == strat]
            if not stats_for_strat:
                continue

            wins = sum(1 for c in comparisons if c.best_strategy == strat)
            avg_iters = np.mean([s.iteration_mean for s in stats_for_strat])
            avg_hit_rate = np.mean([s.hit_rate for s in stats_for_strat])
            avg_accuracy = np.mean([s.accuracy_mean for s in stats_for_strat if s.accuracy_mean > 0])
            avg_warp_div = np.mean([s.warp_divergence_proxy for s in stats_for_strat])
            avg_p99 = np.mean([s.iteration_p99 for s in stats_for_strat])
            avg_time = np.mean([s.time_per_ray_us for s in stats_for_strat])

            summary[strat] = {
                'avg_mean_iterations': float(avg_iters),
                'avg_hit_rate': float(avg_hit_rate),
                'avg_accuracy': float(avg_accuracy) if not np.isnan(avg_accuracy) else 0.0,
                'num_wins': wins,
                'total_scenes': len(stats_for_strat),
                'avg_warp_divergence': float(avg_warp_div),
                'avg_p99_iterations': float(avg_p99),
                'avg_time_per_ray_us': float(avg_time),
            }

        return summary

    def per_scene_matrix(self, metric: str = 'iteration_mean') -> Tuple[List[str], List[str], np.ndarray]:
        """
        Build a matrix of [scenes x strategies] for a given metric.
        Returns (scene_names, strategy_names, matrix).
        """
        scenes = self.get_scenes()
        strategies = self.get_strategies()

        matrix = np.full((len(scenes), len(strategies)), np.nan)

        for si, scene in enumerate(scenes):
            for sti, strategy in enumerate(strategies):
                stat = self.get_stat(strategy, scene)
                if stat:
                    matrix[si, sti] = getattr(stat, metric, np.nan)

        return scenes, strategies, matrix
```

### `visualization/__init__.py`

```python
from .tables import print_comparison_tables
from .heatmaps import render_heatmaps
from .charts import render_charts

__all__ = ['print_comparison_tables', 'render_heatmaps', 'render_charts']
```

### `visualization/tables.py`

```python
"""Console and file table output for benchmark results."""

import os
from typing import List, Dict, Optional
from metrics.analyzer import MetricsAnalyzer, ComparisonResult
from core.types import RayMarchStats


def print_comparison_tables(analyzer: MetricsAnalyzer, output_dir: Optional[str] = None):
    """Print comprehensive comparison tables to console and optionally save to file."""

    lines = []

    def emit(text=""):
        print(text)
        lines.append(text)

    emit("=" * 100)
    emit("RAY MARCHING STRATEGY BENCHMARK RESULTS")
    emit("=" * 100)
    emit()

    # ── Per-Scene Comparison Table ──
    scenes = analyzer.get_scenes()
    strategies = analyzer.get_strategies()

    # Header
    col_width = 14
    header = f"{'Scene':<30}"
    for strat in strategies:
        header += f" {strat:>{col_width}}"
    emit(header)
    emit("-" * len(header))

    # Mean iterations per scene
    emit("\n── Mean Iterations per Ray ──")
    emit(header)
    emit("-" * len(header))

    for scene in scenes:
        row = f"{scene:<30}"
        min_val = float('inf')
        for strat in strategies:
            stat = analyzer.get_stat(strat, scene)
            if stat:
                min_val = min(min_val, stat.iteration_mean)

        for strat in strategies:
            stat = analyzer.get_stat(strat, scene)
            if stat:
                val = stat.iteration_mean
                marker = " *" if abs(val - min_val) < 0.1 else "  "
                row += f" {val:>{col_width - 2}.1f}{marker}"
            else:
                row += f" {'N/A':>{col_width}}"
        emit(row)

    emit()
    emit("  * = best for this scene")

    # Hit rate per scene
    emit("\n── Hit Rate (%) ──")
    emit(header)
    emit("-" * len(header))

    for scene in scenes:
        row = f"{scene:<30}"
        for strat in strategies:
            stat = analyzer.get_stat(strat, scene)
            if stat:
                row += f" {stat.hit_rate * 100:>{col_width}.1f}"
            else:
                row += f" {'N/A':>{col_width}}"
        emit(row)

    # P95 iterations
    emit("\n── P95 Iterations ──")
    emit(header)
    emit... (incomplete)