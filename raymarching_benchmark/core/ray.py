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
