"""Soundness gates for the interval oracle (gpu/interval.py, gpu/interval_oracle.py).

Gate 1 (pointwise): the interval extension at a degenerate point equals the
canonical scalar SDF in scenes/catalog.py.
Gate 2 (inclusion): for a box, [lo, hi] encloses the SDF at every interior point
— the property the oracle's no-tunnel guarantee rests on.
Gate 3 (first_hit): the isolator finds the true first intersection, including a
thin-torus ray that point-samplers tunnel through.
"""
import numpy as np
import pytest

from raymarching_benchmark.core.vec3 import Vec3
from raymarching_benchmark.scenes.catalog import get_scene_by_name
from raymarching_benchmark.gpu.interval import (
    Interval, IVec3, INTERVAL_SCENES, i_sphere, i_box, i_torus, i_plane,
)
from raymarching_benchmark.gpu.interval_oracle import first_hit, _scalar_sdf
from raymarching_benchmark.gpu.analytic import ANALYTIC_SCENES

SCENES = ["Sphere", "Grazing Plane", "Cube", "Thin Torus"]


@pytest.mark.parametrize("scene_name", SCENES)
def test_pointwise_matches_catalog(scene_name):
    """Gate 1: degenerate interval == catalog scalar SDF."""
    scene = get_scene_by_name(scene_name)
    rng = np.random.default_rng(0)
    P = rng.uniform(-3.0, 3.0, size=(2000, 3))
    iv = _scalar_sdf(scene_name, P)
    ref = np.array([scene.sdf(Vec3(*p)) for p in P])
    assert np.allclose(iv, ref, atol=1e-9), \
        f"{scene_name}: max diff {np.max(np.abs(iv - ref)):.2e}"


@pytest.mark.parametrize("scene_name", SCENES)
def test_inclusion_soundness(scene_name):
    """Gate 2: [lo, hi] encloses the SDF over the whole box (no underestimate)."""
    scene = get_scene_by_name(scene_name)
    fn = INTERVAL_SCENES[scene_name]
    rng = np.random.default_rng(1)
    n_boxes = 400
    centers = rng.uniform(-2.5, 2.5, size=(n_boxes, 3))
    halves = rng.uniform(0.01, 0.8, size=(n_boxes, 3))

    box = IVec3(
        Interval(centers[:, 0] - halves[:, 0], centers[:, 0] + halves[:, 0]),
        Interval(centers[:, 1] - halves[:, 1], centers[:, 1] + halves[:, 1]),
        Interval(centers[:, 2] - halves[:, 2], centers[:, 2] + halves[:, 2]),
    )
    d = fn(box)

    # Sample interior points per box and confirm the enclosure holds.
    n_samp = 64
    u = rng.uniform(-1.0, 1.0, size=(n_boxes, n_samp, 3))
    pts = centers[:, None, :] + u * halves[:, None, :]
    vals = np.array([[scene.sdf(Vec3(*p)) for p in box_pts] for box_pts in pts])

    tol = 1e-9
    assert np.all(vals >= d.lo[:, None] - tol), "lower bound violated (would tunnel)"
    assert np.all(vals <= d.hi[:, None] + tol), "upper bound violated"


def test_primitive_inclusion_units():
    """Gate 2 (focused): each primitive's enclosure brackets its center value."""
    box = IVec3(Interval(-0.5, 0.5), Interval(-0.5, 0.5), Interval(-0.5, 0.5))
    c = Vec3(0.0, 0.0, 0.0)
    for d, val in [
        (i_sphere(box, 1.0), get_scene_by_name("Sphere").sdf(c)),       # -1.0
        (i_box(box, (1.0, 1.0, 1.0)), get_scene_by_name("Cube").sdf(c)),  # -1.0
        (i_plane(box, (0.0, 1.0, 0.0), -0.5),
         get_scene_by_name("Grazing Plane").sdf(c)),                      # +0.5
        (i_torus(box, 1.5, 0.05), get_scene_by_name("Thin Torus").sdf(c)),
    ]:
        assert float(d.lo) <= float(d.hi)
        assert float(d.lo) - 1e-9 <= val <= float(d.hi) + 1e-9


def test_first_hit_sphere():
    """Gate 3: straight-on sphere ray hits at t = dist - radius."""
    ro = np.array([0.0, 0.0, 5.0])
    rd = np.array([[0.0, 0.0, -1.0]])
    t = first_hit(ro, rd, INTERVAL_SCENES["Sphere"], t_max=10.0, tol=1e-6)
    assert abs(t[0] - 4.0) < 1e-4, t


def test_first_hit_torus_no_tunnel():
    """Gate 3: a ray through the thin-torus tube is found (point-samplers miss
    it). Tube center is the major circle radius R=1.5 in the xz-plane; aim a
    downward ray straight through the tube top at x=1.5."""
    ro = np.array([1.5, 3.0, 0.0])
    rd = np.array([[0.0, -1.0, 0.0]])
    t = first_hit(ro, rd, INTERVAL_SCENES["Thin Torus"], t_max=10.0, tol=1e-6)
    # tube minor radius 0.05 centered at y=0 ⇒ first hit at y = +0.05, t ≈ 2.95
    assert np.isfinite(t[0]), "tunneled through the thin tube"
    assert abs(t[0] - 2.95) < 1e-3, t


@pytest.mark.parametrize("scene_name", ["Sphere", "Cube", "Thin Torus"])
def test_first_hit_matches_analytic_batch(scene_name):
    """Gate 3: a batch of rays agrees with the closed-form intersection."""
    from raymarching_benchmark.gpu.analytic import generate_rays
    from raymarching_benchmark.config import RenderConfig
    rc = RenderConfig(width=48, height=48)
    ro, rd = generate_rays(rc)
    rd_flat = rd.reshape(-1, 3)
    t_iv = first_hit(ro, rd_flat, INTERVAL_SCENES[scene_name], t_max=100.0, tol=1e-6)
    depth_a, hit_a, _ = ANALYTIC_SCENES[scene_name](ro, rd)
    hit_a = hit_a.reshape(-1); depth_a = depth_a.reshape(-1)
    hit_iv = np.isfinite(t_iv)
    # hit masks agree on the vast majority of pixels (silhouette tol)
    agree = np.mean(hit_iv == hit_a)
    assert agree > 0.99, f"{scene_name}: hit-mask agreement {agree:.4f}"
    both = hit_iv & hit_a
    if both.any():
        derr = np.max(np.abs(t_iv[both] - depth_a[both]))
        assert derr < 1e-3, f"{scene_name}: depth err {derr:.2e}"
