"""Slope-Based Auto-Relaxation Sphere Tracing (Bán & Valasek, 2023)."""

from .base import MarchStrategy
from ..core.ray import Ray
from ..core.types import MarchResult
from ..config import MarchConfig
from typing import Callable


class SlopeBasedAutoRelaxedTracing(MarchStrategy):
    """
    Auto-Relaxed Sphere Tracing using Slope Exponential Averaging.
    
    Based on "Automatic Step Size Relaxation in Sphere Tracing" 
    by Bán & Valasek (Eurographics 2023).

    Instead of heuristically adjusting an omega multiplier, this method 
    tracks the slope 'm' of the SDF along the ray. It calculates the 
    next step size based on the intersection of the estimated surface line
    and the ray.

    Algorithm 4 from the paper.
    """

    def __init__(self, beta: float = 0.3):
        """
        Args:
            beta: Exponential averaging coefficient (0 < beta < 1). 
                  Paper recommends 0.3 for robust performance.
        """
        self.beta = beta

    @property
    def name(self) -> str:
        return f"Slope-Based Auto-Relaxed (β={self.beta})"

    @property
    def short_name(self) -> str:
        return f"Slope-AR(β={self.beta})"

    def march(self, ray: Ray, sdf_func: Callable, config: MarchConfig) -> MarchResult:
        t = 0.0
        iterations = 0
        
        # Initial Setup (Line 4 of Alg 4)
        pos = ray.at(t)
        r = sdf_func(pos)
        z = r  # Initial step size is just the SDF value
        m = -1.0 # Initial slope (implies ray is perpendicular to plane)

        for i in range(config.max_iterations):
            iterations = i + 1

            # Hit check
            if abs(r) < config.hit_threshold:
                 return MarchResult(
                    hit=True, t=t, iterations=iterations, final_sdf=r,
                    position_x=pos.x, position_y=pos.y, position_z=pos.z
                )
            
            # Max distance check
            if t > config.max_distance:
                break

            # Try the calculated step (Line 6)
            T_candidate = t + z
            
            # We must evaluate SDF at candidate to check validity
            pos_candidate = ray.at(T_candidate)
            R_candidate = sdf_func(pos_candidate)
            
            # Check if the step is correct/valid (Line 7)
            # This checks if the unbounding spheres overlap (Lipschitz condition)
            if z <= r + abs(R_candidate):
                # Valid Step: Update slope and advance
                
                # Calculate instantaneous slope M (Line 8)
                # Avoid division by zero if z is extremely small
                denom = (T_candidate - t)
                M = (R_candidate - r) / denom if denom > 1e-12 else -1.0
                
                # Update smoothed slope m (Line 9)
                m = (1.0 - self.beta) * m + self.beta * M
                
                # Apply the step (Line 10)
                t = T_candidate
                r = R_candidate
                pos = pos_candidate # Update current position object
                
            else:
                # Invalid Step (Over-relaxed into disjoint sphere)
                # Revert to basic Sphere Tracing logic (Line 12)
                # We do NOT advance t, we just reset slope to -1
                m = -1.0
                # r remains the same, t remains the same
            
            # Calculate next step size z (Line 14)
            # z = 2r / (1 - m)
            # Safety: ensure (1-m) doesn't cause division by zero or negative step
            # If m approaches 1 (parallel to ray), step goes to infinity.
            denom = 1.0 - m
            if denom < 1e-6:
                denom = 1e-6
                
            z = (2.0 * r) / denom
            
            # Safety clamp: avoid negative steps if slope calculation goes wild
            if z < 0:
                z = r 

        # Final return if max iterations reached
        return MarchResult(
            hit=False, t=t, iterations=iterations, final_sdf=r,
            position_x=pos.x, position_y=pos.y, position_z=pos.z
        )