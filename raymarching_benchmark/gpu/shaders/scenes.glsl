// Scene catalog for GLSL
// Must match Python scene IDs exactly

uniform int sceneId;

// Per-invocation SDF-evaluation counter. Every strategy (and calcNormal) calls
// map(), so this counts true work. main() resets it before the march and
// snapshots it afterwards, giving an "equal compute" denominator that is fairer
// than step count (one step can cost several map() calls).
int g_evals = 0;
float map_impl(vec3 p);
float map(vec3 p) { g_evals += 1; return map_impl(p); }

float map_impl(vec3 p) {
    if (sceneId == 0) { // Sphere
        return sdSphere(p, 1.0);
    }
    if (sceneId == 1) { // Grazing Plane
        return sdPlane(p, vec3(0.0, 1.0, 0.0), -0.5);
    }
    if (sceneId == 2) { // Cube
        return sdBox(p, vec3(1.0));
    }
    if (sceneId == 3) { // Thin Torus
        return sdTorus(p, 1.5, 0.05);
    }
    if (sceneId == 4) { // Cylinder
        return sdCylinder(p, 1.0, 1.5);
    }
    if (sceneId == 5) { // Near Miss
        float d1 = sdSphere(p - vec3(-1.01, 0.0, 0.0), 1.0);
        float d2 = sdSphere(p - vec3(1.01, 0.0, 0.0), 1.0);
        return min(d1, d2);
    }
    if (sceneId == 6) { // Hollow Cube
        float d_box = sdBox(p, vec3(1.0));
        float d_sphere = sdSphere(p, 1.3);
        return max(d_box, -d_sphere);
    }
    if (sceneId == 7) { // Smooth Blend
        float d1 = sdSphere(p - vec3(-0.5, 0.0, 0.0), 0.8);
        float d2 = sdBox(p - vec3(0.5, 0.0, 0.0), vec3(0.6));
        return opSmoothUnion(d1, d2, 0.5);
    }
    if (sceneId == 8) { // Onion Shell
        float d = sdSphere(p, 2.0);
        d = abs(d) - 0.1;
        d = abs(d) - 0.05;
        return d;
    }
    if (sceneId == 9) { // Menger Sponge
        return sdMenger(p, 3);
    }
    if (sceneId == 10) { // Mandelbulb
        return sdMandelbulb(p, 8);
    }
    if (sceneId == 11) { // Bad Lipschitz
        float d = sdSphere(p, 1.0);
        return d * 0.1; // Intentional underestimate
    }
    if (sceneId == 12) { // Pillar Forest
        vec3 q = opRepeat(p, vec3(2.0, 0.0, 2.0));
        return sdCylinder(q, 0.2, 5.0);
    }
    if (sceneId == 13) { // Thin Planes Stack
        float spacing = 0.5;
        float py_mod = mod(p.y + spacing * 0.5, spacing) - spacing * 0.5;
        return abs(py_mod) - 0.01;
    }
    if (sceneId == 14) { // Sphere Cloud (expensive metric SDF — 24 spheres)
        float d = 1e10;
        d=min(d,sdSphere(p-vec3(0.4253,1.3505,0.9373),0.4723));
        d=min(d,sdSphere(p-vec3(-0.9343,-0.6794,1.2701),0.4257));
        d=min(d,sdSphere(p-vec3(-1.6821,1.0922,1.0100),0.3090));
        d=min(d,sdSphere(p-vec3(-0.1090,-0.6697,-0.7534),0.4659));
        d=min(d,sdSphere(p-vec3(-0.8334,-0.1867,0.0155),0.4879));
        d=min(d,sdSphere(p-vec3(0.1819,1.6847,0.9951),0.4789));
        d=min(d,sdSphere(p-vec3(0.4154,1.6625,-0.9680),0.4053));
        d=min(d,sdSphere(p-vec3(-1.1553,0.3826,-1.5506),0.3120));
        d=min(d,sdSphere(p-vec3(-1.5787,0.0506,-0.1149),0.3223));
        d=min(d,sdSphere(p-vec3(1.4184,0.4394,0.0480),0.4841));
        d=min(d,sdSphere(p-vec3(-0.0106,-0.8584,-1.6599),0.4015));
        d=min(d,sdSphere(p-vec3(-1.0458,0.6529,-1.0179),0.3197));
        d=min(d,sdSphere(p-vec3(-0.4436,-1.6873,1.1222),0.4745));
        d=min(d,sdSphere(p-vec3(-1.1748,-0.7902,1.2931),0.4211));
        d=min(d,sdSphere(p-vec3(0.0333,1.1803,0.4750),0.4053));
        d=min(d,sdSphere(p-vec3(0.8220,-1.3889,0.1399),0.3628));
        d=min(d,sdSphere(p-vec3(0.0264,1.2626,-0.4717),0.3704));
        d=min(d,sdSphere(p-vec3(0.3338,-1.4985,-0.3821),0.3327));
        d=min(d,sdSphere(p-vec3(-0.6017,-1.1893,1.0755),0.2884));
        d=min(d,sdSphere(p-vec3(-0.4099,1.6277,0.3060),0.4728));
        d=min(d,sdSphere(p-vec3(0.3572,0.4692,0.5999),0.3829));
        d=min(d,sdSphere(p-vec3(-1.1873,-0.2029,-0.8855),0.4005));
        d=min(d,sdSphere(p-vec3(-0.3315,-1.3712,1.5906),0.3509));
        d=min(d,sdSphere(p-vec3(-0.9690,0.5840,-0.6786),0.4453));
        return d;
    }
    if (sceneId == 15) { // Bumpy Sphere (expensive metric: base + 30 bumps)
        float d = sdSphere(p, 1.4);
        d=min(d,sdSphere(p-vec3(0.3841,1.4500,0.0000),0.18));
        d=min(d,sdSphere(p-vec3(-0.4821,1.3500,0.4417),0.18));
        d=min(d,sdSphere(p-vec3(0.0725,1.2500,-0.8260),0.18));
        d=min(d,sdSphere(p-vec3(0.5860,1.1500,0.7643),0.18));
        d=min(d,sdSphere(p-vec3(-1.0548,1.0500,-0.1866),0.18));
        d=min(d,sdSphere(p-vec3(0.9794,0.9500,-0.6230),0.18));
        d=min(d,sdSphere(p-vec3(-0.3209,0.8500,1.1935),0.18));
        d=min(d,sdSphere(p-vec3(-0.5987,0.7500,-1.1528),0.18));
        d=min(d,sdSphere(p-vec3(1.2698,0.6500,0.4637),0.18));
        d=min(d,sdSphere(p-vec3(-1.2900,0.5500,0.5325),0.18));
        d=min(d,sdSphere(p-vec3(0.6065,0.4500,-1.2960),0.18));
        d=min(d,sdSphere(p-vec3(0.4365,0.3500,1.3917),0.18));
        d=min(d,sdSphere(p-vec3(-1.2797,0.2500,-0.7416),0.18));
        d=min(d,sdSphere(p-vec3(1.4577,0.1500,-0.3205),0.18));
        d=min(d,sdSphere(p-vec3(-0.8622,0.0500,1.2264),0.18));
        d=min(d,sdSphere(p-vec3(-0.1927,-0.0500,-1.4867),0.18));
        d=min(d,sdSphere(p-vec3(1.1412,-0.1500,0.9618),0.18));
        d=min(d,sdSphere(p-vec3(-1.4778,-0.2500,0.0611),0.18));
        d=min(d,sdSphere(p-vec3(1.0339,-0.3500,-1.0289),0.18));
        d=min(d,sdSphere(p-vec3(-0.0661,-0.4500,1.4294),0.18));
        d=min(d,sdSphere(p-vec3(-0.8941,-0.5500,-1.0715),0.18));
        d=min(d,sdSphere(p-vec3(1.3398,-0.6500,0.1803),0.18));
        d=min(d,sdSphere(p-vec3(-1.0663,-0.7500,0.7419),0.18));
        d=min(d,sdSphere(p-vec3(0.2713,-0.8500,-1.2058),0.18));
        d=min(d,sdSphere(p-vec3(0.5771,-0.9500,1.0072),0.18));
        d=min(d,sdSphere(p-vec3(-1.0205,-1.0500,-0.3256),0.18));
        d=min(d,sdSphere(p-vec3(0.8743,-1.1500,-0.4039),0.18));
        d=min(d,sdSphere(p-vec3(-0.3201,-1.2500,0.7649),0.18));
        d=min(d,sdSphere(p-vec3(-0.2213,-1.3500,-0.6152),0.18));
        d=min(d,sdSphere(p-vec3(0.3400,-1.4500,0.1787),0.18));
        return d;
    }
    if (sceneId == 16) { // Gyroid solid (1-Lipschitz scaled; clipped to ball)
        vec3 q = p * 3.0;                       // FREQ
        float g = sin(q.x) * cos(q.y)
                + sin(q.y) * cos(q.z)
                + sin(q.z) * cos(q.x);
        float sheet = g / 10.392304845413264;   // / (FREQ * 2*sqrt(3))
        float ball = sdSphere(p, 2.2);
        return max(sheet, ball);
    }
    if (sceneId == 17) { // Capped Torus (open thin ring; sc = (sin 2, cos 2))
        return sdCappedTorus(p, vec2(0.9092974268256817, -0.4161468365471424), 1.2, 0.2);
    }
    if (sceneId == 18) { // Finite box lattice (metric near-miss). round = floor(x+0.5)
        vec3 cell = clamp(floor(p + 0.5), -2.0, 2.0);   // C=1.0, L=2
        vec3 q = p - cell;
        return sdBox(q, vec3(0.3));
    }
    if (sceneId == 19) { // Metaballs (canonical polynomial smin, k=0.45)
        float d = sdSphere(p, 0.8);
        d = opSmoothUnion(d, sdSphere(p - vec3( 1.0, 0.0, 0.0), 0.6), 0.45);
        d = opSmoothUnion(d, sdSphere(p - vec3(-1.0, 0.0, 0.0), 0.6), 0.45);
        d = opSmoothUnion(d, sdSphere(p - vec3( 0.0, 1.0, 0.0), 0.6), 0.45);
        d = opSmoothUnion(d, sdSphere(p - vec3( 0.0,-1.0, 0.0), 0.6), 0.45);
        d = opSmoothUnion(d, sdSphere(p - vec3( 0.0, 0.0, 1.0), 0.6), 0.45);
        return d;
    }

    return 1e10;
}
