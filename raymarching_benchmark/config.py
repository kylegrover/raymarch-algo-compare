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
    kappa: float = 2.0                 # Segment-tracing growth factor
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
