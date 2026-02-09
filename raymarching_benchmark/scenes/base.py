"""Abstract base class for SDF scenes."""

from abc import ABC, abstractmethod
from ..core.vec3 import Vec3
from ..config import RenderConfig
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
