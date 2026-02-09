// Ray marching strategies for GLSL

uniform int maxIterations;
uniform float hitThreshold;
uniform float maxDistance;
uniform float omega; // for Relaxed
uniform float lipschitz; // global Lipschitz (optional)
// Segment-tracing tuning
uniform float kappa;      // growth factor for next candidate segment
uniform float minStep;    // minimum allowed march step (safety)

// Small epsilon to avoid division-by-zero in estimators
const float SEG_EPS = 1e-6;

struct MarchResult {
    bool hit;
    float t;
    int iterations;
    float finalSdf;
};

// 1. Standard Sphere Tracing
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

// 2. Overstep-Bisect
MarchResult overstep_bisect(vec3 ro, vec3 rd) {
    float t = 0.0;
    int it = 0;
    float d;
    for(int i=0; i<maxIterations; i++) {
        it = i + 1;
        d = map(ro + rd*t);
        
        if (abs(d) < hitThreshold) return MarchResult(true, t, it, d);
        
        float step_size = max(d, hitThreshold * 2.0);
        float next_t = t + step_size;
        float next_d = map(ro + rd*next_t);
        
        if (next_d < 0.0) {
            // Bisection
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

// 3. Relaxed Sphere Tracing (with overshoot fallback)
MarchResult relaxed(vec3 ro, vec3 rd) {
    float t = 0.0;
    int it = 0;
    float d;
    float prev_t = 0.0;
    
    for(int i=0; i<maxIterations; i++) {
        it = i + 1;
        d = map(ro + rd*t);
        
        if (abs(d) < hitThreshold) return MarchResult(true, t, it, d);
        
        if (d < 0.0) { // Overshot
             t = (prev_t + t) * 0.5;
             continue;
        }
        
        prev_t = t;
        t += d * omega;
        
        if(t > maxDistance) break;
    }
    return MarchResult(false, t, it, d);
}

// 4. Enhanced Sphere Tracing (Planar)
MarchResult enhanced(vec3 ro, vec3 rd) {
    float t = 0.0;
    int it = 0;
    float d, prev_d = 1e10;
    float prev_t = 0.0;
    
    for(int i=0; i<maxIterations; i++) {
        it = i + 1;
        d = map(ro + rd*t);
        
        if (abs(d) < hitThreshold) return MarchResult(true, t, it, d);
        
        if (d < 0.0) { // Overshot
             t = (prev_t + t) * 0.5;
             continue;
        }

        float step_size = d;
        if (i > 0 && prev_d > d && (prev_d - d) > 1e-6) {
             float dt = t - prev_t;
             float predicted = d * dt / (prev_d - d);
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

// 5. Adaptive Relaxed (AR-ST) - simpler version
MarchResult ar_st(vec3 ro, vec3 rd) {
    float t = 0.0;
    int it = 0;
    float d, prev_d = 1e10;
    float prev_t = 0.0;
    float curr_omega = omega;
    
    for(int i=0; i<maxIterations; i++) {
        it = i + 1;
        d = map(ro + rd*t);
        
        if (abs(d) < hitThreshold) return MarchResult(true, t, it, d);
        
        if (d < 0.0) {
             t = (prev_t + t) * 0.5;
             curr_omega = 1.0; // Reduce relaxation on overshoot
             continue;
        }

        prev_t = t;
        t += d * curr_omega;
        if (d < prev_d) curr_omega = mix(curr_omega, omega, 0.1);
        else curr_omega = 1.0;
        
        prev_d = d;
        
        if(t > maxDistance) break;
    }
    return MarchResult(false, t, it, d);
}

// 6. Segment Tracing (numeric K-segment estimator)
// Implements the local-segment Lipschitz estimator approach (numeric fallback
// to the analytic KSegment used in the paper). Safer for arbitrary `map()`.

float estimate_K_segment(vec3 a, vec3 b, float globalLipschitz) {
    // Numerical slope estimates on [a,b] using endpoints + midpoint.
    float fa = map(a);
    float fb = map(b);
    float dist_ab = max(length(b - a), SEG_EPS);
    float L_ab = abs(fa - fb) / dist_ab;

    // Midpoint sample improves robustness for curved fields
    vec3 m = 0.5 * (a + b);
    float fm = map(m);
    float L_am = abs(fa - fm) / max(length(m - a), SEG_EPS);
    float L_mb = abs(fm - fb) / max(length(b - m), SEG_EPS);

    float L = max(L_ab, max(L_am, L_mb));

    // Apply small safety factor and enforce a sensible lower bound
    L = max(L * 1.1, max(globalLipschitz, 1e-4));
    return L;
}

MarchResult segment(vec3 ro, vec3 rd) {
    // Primary: mirror CPU `SegmentTracing.march()` behavior so GPU matches CPU.
    // Fallback: if no reliable global Lipschitz is provided, use a numeric K-segment.

    float t = 0.0;
    int it = 0;            // iteration / SDF-eval counter (we increment for extra map() calls)
    float d = 0.0;

    // candidate controls how far we probe ahead; start conservatively
    float candidate = max(1.0, minStep);
    float globalL = max(lipschitz, 0.0);

    for (int i = 0; i < maxIterations; ++i) {
        it = i + 1;
        vec3 p = ro + rd * t;
        d = map(p);

        if (abs(d) < hitThreshold) {
            return MarchResult(true, t, it, d);
        }

        // If a (user/scene) global Lipschitz is available, follow the CPU logic
        if (globalL > 1e-5) {
            // candidate based on global Lipschitz (same as Python implementation)
            candidate = max(minStep, abs(d) / max(globalL, SEG_EPS));

            // evaluate at the end of the candidate segment (extra SDF eval)
            vec3 pos_end = ro + rd * (t + candidate);
            float d_end = map(pos_end);
            it += 1; // account for the extra evaluation

            // Hit at the endpoint
            if (abs(d_end) < hitThreshold) {
                t += candidate;
                return MarchResult(true, t, it, d_end);
            }

            // Surface inside segment -> bisect and return if found
            if (d_end < 0.0) {
                float a = t;
                float b = t + candidate;
                for (int j = 0; j < 8; ++j) {
                    float mid = 0.5 * (a + b);
                    float dm = map(ro + rd * mid);
                    it += 1;
                    if (abs(dm) < hitThreshold) {
                        return MarchResult(true, mid, it, dm);
                    }
                    if (dm > 0.0) a = mid; else b = mid;
                }
                // conservative placement if bisect didn't converge exactly
                t = 0.5 * (a + b);
                float df = map(ro + rd * t);
                it += 1;
                return MarchResult(true, t, it, df);
            }

            // Both endpoints safe → can extend the step using endpoint slack (matches CPU)
            float extended = candidate + d_end / max(globalL, SEG_EPS);
            float step = max(minStep, extended);
            t += step;

            // next candidate grows from this step (paper: kappa > 1)
            candidate = kappa * step;

            if (t > maxDistance) break;
            continue;
        }

        // --- Fallback: numeric K-segment estimator (when no global Lipschitz known) ---
        vec3 seg_end = ro + rd * (t + candidate);
        float Kseg = estimate_K_segment(p, seg_end, 1e-4);
        float step = abs(d) / max(Kseg, SEG_EPS);
        step = min(step, candidate);
        step = max(step, minStep);
        t += step;
        candidate = kappa * step;

        if (t > maxDistance) break;
    }

    return MarchResult(false, t, it, d);
}
