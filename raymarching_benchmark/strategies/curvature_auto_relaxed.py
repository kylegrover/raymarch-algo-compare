"""Curvature-Aware Tracing: Uses 3 history points to predict surface."""

from .base import MarchStrategy
from ..core.ray import Ray
from ..core.types import MarchResult
from ..config import MarchConfig
from typing import Callable

class CurvatureAwareTracing(MarchStrategy):
    """
    Maintains a history of the last 3 SDF evaluations.
    Fits a quadratic polynomial (parabola) to (t, sdf) to estimate
    the root (where sdf=0).
    
    If the parabola has no real root or predicts a step < standard step,
    it falls back to Standard Sphere Tracing.
    """

    @property
    def name(self) -> str:
        return "Curvature-Aware Tracing"

    def march(self, ray: Ray, sdf_func: Callable, config: MarchConfig) -> MarchResult:
        t = 0.0
        iterations = 0
        
        # History buffers
        t_hist = [0.0, 0.0, 0.0]
        d_hist = [0.0, 0.0, 0.0]
        hist_count = 0

        for i in range(config.max_iterations):
            iterations = i + 1
            pos = ray.at(t)
            d = sdf_func(pos)

            if abs(d) < config.hit_threshold:
                return MarchResult(hit=True, t=t, iterations=iterations, final_sdf=d,
                                   position_x=pos.x, position_y=pos.y, position_z=pos.z)

            # Update history
            t_hist.pop(0)
            d_hist.pop(0)
            t_hist.append(t)
            d_hist.append(d)
            hist_count = min(hist_count + 1, 3)

            step = d # Default safe step

            # Attempt Quadratic Prediction
            if hist_count == 3:
                t1, t2, t3 = t_hist
                d1, d2, d3 = d_hist
                
                # Check for numerical stability (avoid dividing by tiny numbers)
                if abs(t3-t2) > 1e-5 and abs(t2-t1) > 1e-5:
                    # Lagrangian Interpolation or Inverse Quadratic Interpolation
                    # Inverse Quadratic: fit t = f(d), find t at d=0
                    # This is generally more stable for finding roots than solving ax^2+bx+c=0
                    
                    try:
                        # Inverse Quadratic Interpolation formula
                        # t = sum( t_j * product( (0 - d_k)/(d_j - d_k) ) )
                        
                        term1 = t1 * ((0 - d2) / (d1 - d2)) * ((0 - d3) / (d1 - d3))
                        term2 = t2 * ((0 - d1) / (d2 - d1)) * ((0 - d3) / (d2 - d3))
                        term3 = t3 * ((0 - d1) / (d3 - d1)) * ((0 - d2) / (d3 - d2))
                        
                        t_pred = term1 + term2 + term3
                        pred_step = t_pred - t
                        
                        # Trust logic:
                        # 1. Must advance forward
                        # 2. Don't be too aggressive (limit to 3x standard step)
                        # 3. Only use if d is decreasing (we are approaching surface)
                        if d < d_hist[1] and 0 < pred_step < 3.0 * d:
                            step = pred_step
                    except ZeroDivisionError:
                        pass

            t += step
            
            if t > config.max_distance:
                break

        pos = ray.at(t)
        d = sdf_func(pos)
        return MarchResult(hit=False, t=t, iterations=iterations, final_sdf=d,
                           position_x=pos.x, position_y=pos.y, position_z=pos.z)