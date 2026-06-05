
// Multiple render targets.
//  location 0 (outGeom): benchmark encoding — UNCHANGED layout so the timing
//                        path keeps working: R=hit, G=iter/maxIter, B=t/maxDist, A=finalSdf
//  location 1 (outNormalDepth): world-space normal in .xyz (raw, [-1,1]), raw depth t in .w
//  location 2 (outColor): shaded RGB in .rgb (gamma-encoded), hit flag in .a
// Capture targets are only written when captureMode == 1 (reference / snapshot capture).
layout(location = 0) out vec4 outGeom;
layout(location = 1) out vec4 outNormalDepth;
layout(location = 2) out vec4 outColor;
layout(location = 3) out vec4 outAux;   // .r = SDF-eval count during the march

uniform vec2 resolution;
uniform vec3 camPos;
uniform vec3 camDir;
uniform vec3 camUp;
uniform vec3 camRight;

uniform int strategyId;
uniform int captureMode;   // 0 = timing only (default), 1 = full capture (normal + shading)

// Include scene and strategy code
// (In Python we will concatenate these)

// Tetrahedron-technique surface normal via central differences of map().
vec3 calcNormal(vec3 p) {
    const float e = 0.0005;
    const vec2 k = vec2(1.0, -1.0);
    return normalize(
        k.xyy * map(p + k.xyy * e) +
        k.yyx * map(p + k.yyx * e) +
        k.yxy * map(p + k.yxy * e) +
        k.xxx * map(p + k.xxx * e)
    );
}

// Deterministic shading model shared by reference + every method snapshot so
// that differences in the lit image come only from geometry/normals, never
// from a randomized light. Single key light + hemisphere ambient, gamma 2.2.
vec3 shade(vec3 pos, vec3 n, vec3 rd) {
    vec3 L = normalize(vec3(0.6, 0.7, 0.5));
    float diff = max(dot(n, L), 0.0);
    float hemi = 0.5 + 0.5 * n.y;           // sky/ground ambient
    vec3 base = vec3(0.82, 0.80, 0.78);
    vec3 col = base * (0.15 * hemi + 0.85 * diff);
    return pow(clamp(col, 0.0, 1.0), vec3(0.4545));   // gamma encode
}

// Background for missed rays (consistent between reference and methods).
vec3 background(vec3 rd) {
    float t = 0.5 * (rd.y + 1.0);
    return mix(vec3(0.06, 0.07, 0.09), vec3(0.12, 0.14, 0.18), t);
}

void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * resolution.xy) / resolution.y;

    // Simple pinhole camera
    vec3 rd = normalize(camDir + uv.x * camRight + uv.y * camUp);
    vec3 ro = camPos;

    g_evals = 0;   // count SDF evaluations performed by the march itself
    MarchResult res;
    if (strategyId == 0) res = standard(ro, rd);
    else if (strategyId == 1) res = overstep_bisect(ro, rd);
    else if (strategyId == 2) res = relaxed(ro, rd);
    else if (strategyId == 3) res = segment(ro, rd);
    else if (strategyId == 4) res = enhanced(ro, rd);
    else if (strategyId == 5) res = heuristic_auto_relaxed(ro, rd);
    else if (strategyId == 6) res = skipping_spheres(ro, rd);
    else if (strategyId == 7) res = rev_affine(ro, rd);
    else if (strategyId == 8) res = safe_relaxed(ro, rd);
    else if (strategyId == 9) res = dense_march(ro, rd);
    else res = standard(ro, rd);

    int marchEvals = g_evals;   // snapshot before calcNormal adds its own evals

    // location 0 — benchmark encoding (unchanged)
    outGeom = vec4(
        res.hit ? 1.0 : 0.0,
        float(res.iterations) / float(maxIterations),
        res.t / maxDistance,
        res.finalSdf
    );

    // location 3 — raw SDF-eval count for the march (fairness denominator)
    outAux = vec4(float(marchEvals), 0.0, 0.0, 0.0);

    // Capture targets only computed on demand (uniform branch — cheap when off).
    if (captureMode == 1) {
        if (res.hit) {
            vec3 pos = ro + rd * res.t;
            vec3 n = calcNormal(pos);
            outNormalDepth = vec4(n, res.t);
            outColor = vec4(shade(pos, n, rd), 1.0);
        } else {
            outNormalDepth = vec4(0.0, 0.0, 0.0, 0.0);
            outColor = vec4(background(rd), 0.0);
        }
    } else {
        outNormalDepth = vec4(0.0);
        outColor = vec4(0.0);
    }
}
