from typing import Dict, Type, Optional

from .base import MarchStrategy
from .standard_sphere import StandardSphereTracing
from .relaxed_sphere import RelaxedSphereTracing
from .auto_relaxed import AutoRelaxedSphereTracing
from .enhanced_sphere import EnhancedSphereTracing
from .overstep_bisect import OverstepBisectTracing
from .adaptive_hybrid import AdaptiveHybridTracing
from .segment_tracing import SegmentTracing

STRATEGIES: Dict[str, Type[MarchStrategy]] = {
    'Standard': StandardSphereTracing,
    'Relaxed': RelaxedSphereTracing,
    'Auto-Relaxed': AutoRelaxedSphereTracing,
    'Enhanced': EnhancedSphereTracing,
    'Overstep-Bisect': OverstepBisectTracing,
    'Adaptive-Hybrid': AdaptiveHybridTracing,
    'Segment': SegmentTracing,
}


def get_strategy_by_name(name: str) -> Optional[MarchStrategy]:
    """Look up a strategy by name (case-insensitive substring match)."""
    name_low = name.lower()

    # Try exact-ish match first
    for strat_name, strat_class in STRATEGIES.items():
        if strat_name.lower() == name_low:
            return strat_class()

    # Try substring match
    for strat_name, strat_class in STRATEGIES.items():
        if name_low in strat_name.lower():
            return strat_class()

    return None


def list_strategies() -> list[str]:
    """Return all available strategy names."""
    return list(STRATEGIES.keys())


__all__ = [
    'MarchStrategy',
    'StandardSphereTracing',
    'RelaxedSphereTracing',
    'AutoRelaxedSphereTracing',
    'EnhancedSphereTracing',
    'OverstepBisectTracing',
    'AdaptiveHybridTracing',
    'SegmentTracing',
    'get_strategy_by_name',
    'list_strategies',
    'STRATEGIES'
]
