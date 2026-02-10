
out vec4 fragColor;

uniform vec2 resolution;
uniform vec3 camPos;
uniform vec3 camDir;
uniform vec3 camUp;
uniform vec3 camRight;

uniform int strategyId;

// Include scene and strategy code
// (In Python we will concatenate these)

void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * resolution.xy) / resolution.y;
    
    // Simple pinhole camera
    vec3 rd = normalize(camDir + uv.x * camRight + uv.y * camUp);
    vec3 ro = camPos;

    MarchResult res;
    if (strategyId == 0) res = standard(ro, rd);
    else if (strategyId == 1) res = overstep_bisect(ro, rd);
    else if (strategyId == 2) res = relaxed(ro, rd);
    else if (strategyId == 3) res = segment(ro, rd);
    else if (strategyId == 4) res = enhanced(ro, rd);
    else if (strategyId == 5) res = heuristic_auto_relaxed(ro, rd);
    else res = standard(ro, rd);

    // Encode result into texture
    // R: Hit (0/1)
    // G: Iterations (normalized)
    // B: Distance (log scaled or normalized)
    // A: Final SDF (signed)
    
    fragColor = vec4(
        res.hit ? 1.0 : 0.0,
        float(res.iterations) / float(maxIterations),
        res.t / maxDistance,
        res.finalSdf
    );
}
