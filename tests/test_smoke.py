import pytest
import numpy as np
from raymarching_benchmark.core.vec3 import Vec3
from raymarching_benchmark.core.camera import Camera
from raymarching_benchmark.main import run_once
from raymarching_benchmark.strategies import list_strategies
from raymarching_benchmark.config import RenderConfig, MarchConfig

def test_vec3_math():
    v1 = Vec3(1.0, 2.0, 3.0)
    v2 = Vec3(0.0, 1.0, 0.0)
    
    assert (v1 + v2).to_tuple() == (1.0, 3.0, 3.0)
    assert (v1 * 2.0).to_tuple() == (2.0, 4.0, 6.0)
    assert v1.dot(v2) == 2.0
    assert v2.length() == 1.0

def test_camera_rays():
    cam = Camera(
        position=Vec3(0,0,5), 
        target=Vec3(0,0,0), 
        up=Vec3(0,1,0),
        fov_degrees=60.0,
        width=10, 
        height=10
    )
    ray = cam.get_ray(5, 5)
    assert ray.origin.z == 5.0
    assert ray.direction.z < 0 # Pointing towards origin

@pytest.mark.parametrize("strategy", list_strategies())
def test_strategy_sphere_hit(strategy):
    """Verify that every strategy can successfully hit a simple sphere."""
    # Use a small resolution for speed
    rc = RenderConfig(width=16, height=12)
    mc = MarchConfig(max_iterations=100)
    
    stats = run_once(render=rc, march=mc, scene_name="Sphere", strategy_name=strategy)
    
    # Standard sphere at origin from z=5 should have hits
    assert stats.hit_count > 0, f"Strategy {strategy} failed to hit the sphere"
    assert stats.iteration_mean > 0
    assert stats.iteration_max <= 100

def test_consistency():
    """Verify that strategies roughly agree on hit rate for the same scene."""
    rc = RenderConfig(width=20, height=20)
    
    s1 = run_once(render=rc, scene_name="Sphere", strategy_name="Standard")
    s2 = run_once(render=rc, scene_name="Sphere", strategy_name="Overstep-Bisect")
    
    # Hit counts should be identical for a clean sphere
    assert s1.hit_count == s2.hit_count, "Strategies disagree on hit count for simple sphere"
