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

### `config.py` - [IMPLEMENTED](raymarching_benchmark/config.py)

### `core/__init__.py` - [IMPLEMENTED](raymarching_benchmark/core/__init__.py)

### `core/vec3.py` - [IMPLEMENTED](raymarching_benchmark/core/vec3.py)

### `core/ray.py` - [IMPLEMENTED](raymarching_benchmark/core/ray.py)

### `core/camera.py` - [IMPLEMENTED](raymarching_benchmark/core/camera.py)

### `core/types.py` - [IMPLEMENTED](raymarching_benchmark/core/types.py)

### `scenes/__init__.py` - [IMPLEMENTED](raymarching_benchmark/scenes/__init__.py)

### `scenes/base.py` - [IMPLEMENTED](raymarching_benchmark/scenes/base.py)

### `scenes/primitives.py` - [IMPLEMENTED](raymarching_benchmark/scenes/primitives.py)

### `scenes/catalog.py` - [IMPLEMENTED](raymarching_benchmark/scenes/catalog.py)

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