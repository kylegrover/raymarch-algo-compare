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
