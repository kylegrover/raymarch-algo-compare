// Ray marching strategies for GLSL

uniform int maxIterations;
uniform float hitThreshold;
uniform float maxDistance;

// Strategy-specific uniforms
uniform float omega;      // for Relaxed & Heuristic AR
uniform float beta;       // for Slope-Based AR (usually 0.3)
uniform float lipschitz;  // global Lipschitz (optional)

// Segment-tracing tuning
uniform float kappa;      // growth factor for next candidate segment
uniform float minStep;    // minimum allowed march step (safety)

// Small epsilon to avoid division-by-zero
const float EPS = 1e-6;

struct MarchResult {
    bool hit;
    float t;
    int iterations;
    float finalSdf;
};

// ============================================================================
// 1. Standard Sphere Tracing
// ============================================================================
MarchResult standard(vec3 ro, vec3 rd) {
    float t = 0.0;
    int it = 0;
    float d;
    for(int i=0; i<maxIterations; i++) {
        it = i + 1;
        d = map(ro + rd*t);
        if(abs(d) < hitThreshold) return MarchResult(true, t, it, d);
        t += d;
        if(t > maxDistance) break;
    }
    return MarchResult(false, t, it, d);
}

// ============================================================================
// 2. Overstep-Bisect
// ============================================================================
MarchResult overstep_bisect(vec3 ro, vec3 rd) {
    float t = 0.0;
    int it = 0;
    float d;
    for(int i=0; i<maxIterations; i++) {
        it = i + 1;
        d = map(ro + rd*t);
        
        if (abs(d) < hitThreshold) return MarchResult(true, t, it, d);
        
        // Enforce minimum step to prevent "stuck" rays
        float step_size = max(d, hitThreshold * 2.0);
        float next_t = t + step_size;
        
        // Peek ahead
        float next_d = map(ro + rd*next_t);
        
        if (next_d < 0.0) {
            // Overshot detected: Bisect
            float a = t;
            float b = next_t;
            for(int j=0; j<8; j++) {
                float mid = (a + b) * 0.5;
                float dm = map(ro + rd*mid);
                if (dm < 0.0) b = mid;
                else a = mid;
            }
            return MarchResult(true, a, it + 8, map(ro + rd*a));
        }
        
        t = next_t;
        if(t > maxDistance) break;
    }
    return MarchResult(false, t, it, d);
}

// ============================================================================
// 3. Relaxed Sphere Tracing (Fixed Omega)
// ============================================================================
MarchResult relaxed(vec3 ro, vec3 rd) {
    float t = 0.0;
    int it = 0;
    float d;
    float prev_t = 0.0;
    
    for(int i=0; i<maxIterations; i++) {
        it = i + 1;
        d = map(ro + rd*t);
        
        if (abs(d) < hitThreshold) return MarchResult(true, t, it, d);
        
        if (d < 0.0) { 
            // Overshot: back up to midpoint of previous step
            t = (prev_t + t) * 0.5;
            continue;
        }
        
        prev_t = t;
        t += d * omega;
        
        if(t > maxDistance) break;
    }
    return MarchResult(false, t, it, d);
}

// ============================================================================
// 4. Enhanced Sphere Tracing (Planar Prediction)
// ============================================================================
MarchResult enhanced(vec3 ro, vec3 rd) {
    float t = 0.0;
    int it = 0;
    float d, prev_d = 1e10;
    float prev_t = 0.0;
    
    for(int i=0; i<maxIterations; i++) {
        it = i + 1;
        d = map(ro + rd*t);
        
        if (abs(d) < hitThreshold) return MarchResult(true, t, it, d);
        
        if (d < 0.0) { 
             t = (prev_t + t) * 0.5;
             continue;
        }

        float step_size = d;
        // If we have history and are approaching the surface (prev_d > d)
        if (i > 0 && prev_d > d && (prev_d - d) > EPS) {
             float dt = t - prev_t;
             float predicted = d * dt / (prev_d - d);
             // Geometric sanity check (limit prediction)
             if (predicted > 0.0 && predicted < d * 3.0) {
                 step_size = predicted;
             }
        }
        
        prev_t = t;
        prev_d = d;
        t += step_size;
        
        if(t > maxDistance) break;
    }
    return MarchResult(false, t, it, d);
}

// ============================================================================
// 5. Heuristic Auto-Relaxed (Old "ar_st")
// ============================================================================
// Dynamically adjusts omega based on ratio of consecutive SDF values.
MarchResult heuristic_auto_relaxed(vec3 ro, vec3 rd) {
    float t = 0.0;
    int it = 0;
    float d, prev_d = 1e10;
    float prev_t = 0.0;
    float curr_omega = omega; // Start with default aggressive omega
    
    for(int i=0; i<maxIterations; i++) {
        it = i + 1;
        d = map(ro + rd*t);
        
        if (abs(d) < hitThreshold) return MarchResult(true, t, it, d);
        
        if (d < 0.0) {
             // Overshot: Back up and reset omega
             t = (prev_t + t) * 0.5;
             curr_omega = 1.0; 
             continue;
        }

        prev_t = t;
        t += d * curr_omega;
        
        // Heuristic: if d is decreasing, we are safe to keep relaxed.
        // If d is increasing/stable, reset.
        if (d < prev_d) curr_omega = mix(curr_omega, omega, 0.1);
        else curr_omega = 1.0;
        
        prev_d = d;
        
        if(t > maxDistance) break;
    }
    return MarchResult(false, t, it, d);
}

// ============================================================================
// 6. Slope-Based Auto-Relaxed (PDF Method - NEW)
// ============================================================================
// Uses exponential averaging of slope 'm' to calculate next step 'z'.
// Matches "SlopeBasedAutoRelaxedTracing" in Python.
MarchResult slope_auto_relaxed(vec3 ro, vec3 rd) {
    float t = 0.0;
    int it = 0;
    
    // Initial state (Line 4 of Alg 4)
    float r = map(ro); // r is SDF value
    float z = r;       // Initial step size
    float m = -1.0;    // Initial slope
    
    for(int i=0; i<maxIterations; i++) {
        it = i + 1;
        
        if (abs(r) < hitThreshold) return MarchResult(true, t, it, r);
        if (t > maxDistance) break;
        
        // Candidate step (Line 6)
        float T_candidate = t + z;
        float R_candidate = map(ro + rd * T_candidate);
        
        // Check validity: Lipschtiz condition (Line 7)
        if (z <= r + abs(R_candidate)) {
            // Valid step
            
            // Calculate instantaneous slope M (Line 8)
            float denom = T_candidate - t;
            float M = (denom > EPS) ? (R_candidate - r) / denom : -1.0;
            
            // Update smoothed slope m (Line 9)
            m = (1.0 - beta) * m + beta * M;
            
            // Advance (Line 10)
            t = T_candidate;
            r = R_candidate;
        } else {
            // Invalid step (Line 12) - do not advance t
            m = -1.0; 
            // r stays the same, t stays the same
            // Next z will be calculated based on this reset m
        }
        
        // Calculate next step size z (Line 14)
        // z = 2r / (1 - m)
        // Safety: (1-m) can approach 0 if surface is parallel to ray
        float div = max(1.0 - m, EPS);
        z = (2.0 * r) / div;
        
        // Safety clamp against negative z
        if(z < 0.0) z = r;
    }
    
    return MarchResult(false, t, it, r);
}

// ============================================================================
// 7. Curvature-Aware Tracing (NEW)
// ============================================================================
// Uses 3 history points for Inverse Quadratic Interpolation
MarchResult curvature_aware(vec3 ro, vec3 rd) {
    float t = 0.0;
    int it = 0;
    
    // History buffer: [0]=oldest, [2]=newest
    vec3 t_hist = vec3(0.0);
    vec3 d_hist = vec3(0.0);
    int hist_count = 0;
    
    float d;
    
    for(int i=0; i<maxIterations; i++) {
        it = i + 1;
        d = map(ro + rd*t);
        
        if (abs(d) < hitThreshold) return MarchResult(true, t, it, d);
        
        // Shift history
        t_hist.x = t_hist.y; t_hist.y = t_hist.z; t_hist.z = t;
        d_hist.x = d_hist.y; d_hist.y = d_hist.z; d_hist.z = d;
        hist_count = min(hist_count + 1, 3);
        
        float step_size = d; // Default standard step
        
        // Attempt Quadratic Prediction if we have 3 points
        if (hist_count == 3) {
            float t1 = t_hist.x; float t2 = t_hist.y; float t3 = t_hist.z;
            float d1 = d_hist.x; float d2 = d_hist.y; float d3 = d_hist.z;
            
            // Stability check: denominators in inverse quadratic interp
            // (d1-d2), (d1-d3), (d2-d3) must not be zero
            if (abs(d1-d2) > EPS && abs(d1-d3) > EPS && abs(d2-d3) > EPS) {
                // Inverse Quadratic Interpolation: t = f(d) where d=0
                float term1 = t1 * ((0.0 - d2) / (d1 - d2)) * ((0.0 - d3) / (d1 - d3));
                float term2 = t2 * ((0.0 - d1) / (d2 - d1)) * ((0.0 - d3) / (d2 - d3));
                float term3 = t3 * ((0.0 - d1) / (d3 - d1)) * ((0.0 - d2) / (d3 - d2));
                
                float t_pred = term1 + term2 + term3;
                float pred_step = t_pred - t;
                
                // Trust logic:
                // 1. Must be forward (pred_step > 0)
                // 2. Not too aggressive (pred_step < 3 * d)
                // 3. Only use if we are strictly approaching surface (d < d2)
                if (d < d2 && pred_step > 0.0 && pred_step < 3.0 * d) {
                    step_size = pred_step;
                }
            }
        }
        
        t += step_size;
        if(t > maxDistance) break;
    }
    return MarchResult(false, t, it, d);
}

// ============================================================================
// 8. Segment Tracing
// ============================================================================

// Helper: Numerical K-segment estimator
float estimate_K_segment(vec3 a, vec3 b, float globalLipschitz) {
    float fa = map(a);
    float fb = map(b);
    float dist_ab = max(length(b - a), EPS);
    float L_ab = abs(fa - fb) / dist_ab;

    vec3 m = 0.5 * (a + b);
    float fm = map(m);
    float L_am = abs(fa - fm) / max(length(m - a), EPS);
    float L_mb = abs(fm - fb) / max(length(b - m), EPS);

    float L = max(L_ab, max(L_am, L_mb));
    L = max(L * 1.1, max(globalLipschitz, 1e-4));
    return L;
}

MarchResult segment(vec3 ro, vec3 rd) {
    float t = 0.0;
    int it = 0;
    float d = 0.0;

    float candidate = max(1.0, minStep);
    float globalL = max(lipschitz, 0.0);

    for (int i = 0; i < maxIterations; ++i) {
        it = i + 1;
        vec3 p = ro + rd * t;
        d = map(p);

        if (abs(d) < hitThreshold) return MarchResult(true, t, it, d);

        // Path A: Global Lipschitz known (Efficient)
        if (globalL > 1e-5) {
            candidate = max(minStep, abs(d) / max(globalL, EPS));
            
            vec3 pos_end = ro + rd * (t + candidate);
            float d_end = map(pos_end);
            it += 1; // Extra eval

            if (abs(d_end) < hitThreshold) {
                t += candidate;
                return MarchResult(true, t, it, d_end);
            }

            // Surface crossing detected -> Bisect
            if (d_end < 0.0) {
                float a = t;
                float b = t + candidate;
                for (int j = 0; j < 8; ++j) {
                    float mid = 0.5 * (a + b);
                    float dm = map(ro + rd * mid);
                    it += 1;
                    if (abs(dm) < hitThreshold) return MarchResult(true, mid, it, dm);
                    if (dm > 0.0) a = mid; else b = mid;
                }
                t = 0.5 * (a + b);
                return MarchResult(true, t, it, map(ro + rd * t));
            }

            // Safe extension
            float extended = candidate + d_end / max(globalL, EPS);
            float step = max(minStep, extended);
            t += step;
            candidate = kappa * step;
            if (t > maxDistance) break;
            continue;
        }

        // Path B: Numeric K-segment fallback (Expensive but robust)
        vec3 seg_end = ro + rd * (t + candidate);
        float Kseg = estimate_K_segment(p, seg_end, 1e-4);
        float step = abs(d) / max(Kseg, EPS);
        step = min(step, candidate);
        step = max(step, minStep);
        t += step;
        candidate = kappa * step;

        if (t > maxDistance) break;
    }

    return MarchResult(false, t, it, d);
}