// Scene catalog for GLSL
// Must match Python scene IDs exactly

uniform int sceneId;

float map(vec3 p) {
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

    return 1e10;
}
