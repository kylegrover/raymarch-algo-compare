// Ray marching strategies for GLSL

uniform int maxIterations;
uniform float hitThreshold;
uniform float maxDistance;
uniform float omega; // for Relaxed
uniform float lipschitz; // for Segment/Enhanced

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

// 6. Segment Tracing (Simplified)
MarchResult segment(vec3 ro, vec3 rd) {
    float t = 0.0;
    int it = 0;
    float d;
    float L = max(lipschitz, 0.01);
    
    for(int i=0; i<maxIterations; i++) {
        it = i + 1;
        d = map(ro + rd*t);
        if(abs(d) < hitThreshold) return MarchResult(true, t, it, d);
        
        float step_size = d / L;
        t += step_size;
        if(t > maxDistance) break;
    }
    return MarchResult(false, t, it, d);
}
