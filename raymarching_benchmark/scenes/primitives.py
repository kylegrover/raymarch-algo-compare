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
