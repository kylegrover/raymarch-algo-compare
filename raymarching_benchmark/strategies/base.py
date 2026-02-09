"""Abstract base class for ray marching strategies."""

from abc import ABC, abstractmethod
from ..core.ray import Ray
from ..core.types import MarchResult
from ..scenes.base import SDFScene
from ..config import MarchConfig
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
