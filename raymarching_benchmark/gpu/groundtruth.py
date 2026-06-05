"""Ground-truth reference generation.

A reference is a *high-budget* march (Standard tracing at a very large iteration
count and a tiny hit threshold) treated as the truth for a scene+camera+
resolution. Every method snapshot is later scored against it (raw depth error +
SSIM on depth / normal / lit color).

Run directly to produce references + viewable PNGs:

    uv run python -m raymarching_benchmark.gpu.groundtruth \
        --scenes "Sphere,Grazing Plane,Mandelbulb" --res 512
"""
from __future__ import annotations
import os
import sys
import json
import argparse
from typing import Dict, List, Optional, Tuple

from ..config import RenderConfig, MarchConfig
from ..scenes.catalog import get_all_scenes
from .runner import GPURunner
from ..data.capture_io import save_capture, depth_range_of


def force_utf8_stdout() -> None:
    """Best-effort: make stdout UTF-8 so long runs don't die on cp1252."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # py3.7+
    except Exception:
        pass


# The interim ground-truth oracle understeps standard sphere tracing: each step
# is scaled by this factor (< 1) so the march reaches grazing/thin surfaces
# instead of stalling short, and tolerates mild Lipschitz violations (up to
# ~1/ORACLE_STEP_SCALE) without overshooting through the surface. Validated
# against analytic depth where closed form exists (see oracle_calibration.py).
ORACLE_STEP_SCALE = 0.6


def reference_march_config(**overrides) -> MarchConfig:
    """High-budget marching config used as ground truth.

    The iteration budget is large enough to absorb the understep factor
    (understepping by 0.6 needs ~1.7x as many steps to reach the surface).
    """
    cfg = MarchConfig(
        max_iterations=12000,
        hit_threshold=1e-6,
        max_distance=100.0,
    )
    cfg.min_step_fraction = 0.0
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def oracle_params() -> dict:
    """Shader uniform overrides that define the ground-truth oracle."""
    return {"stepScale": ORACLE_STEP_SCALE}


# ── Dense-march reference (additional, higher-trust oracle) ──────────────────
# An understepped trace with a HARD minimum-step floor and sign-change
# bisection (GLSL strategy id 9, `dense_march`). Unlike the understep oracle
# above it cannot stall short at grazing angles (the floor guarantees forward
# progress) and it brackets+bisects the first surface for sub-step-accurate
# depth. minStep bounds the thinnest resolvable feature. We capture BOTH this
# and the understep oracle per frame so the dataset can show the old oracle's
# bias explicitly. Validated against closed-form depth in oracle_calibration.py;
# DENSE_MIN_STEP is the value chosen from that convergence sweep.
DENSE_MARCH_STRATEGY_ID = 9
DENSE_STEP_SCALE = 0.5
# Chosen from the oracle_calibration.py minStep sweep: hit accuracy reaches its
# knee by 0.01 and depth/IoU are flat below it, while cost is ~constant across
# the whole range (adaptive strides dominate). 0.002 sits well past the knee,
# giving ~5x headroom below the thinnest non-analytic features (thin-planes
# shell ~0.01, onion 0.05) at no measurable cost. Validated: IoU 1.0000 +
# ~1e-7 depth on sphere/cube/torus; grazing-plane core IoU 0.9905 (the 1.9%
# shortfall is the shared maxDistance horizon clip, not a resolution error).
DENSE_MIN_STEP = 0.002


def dense_march_params(min_step: float = DENSE_MIN_STEP) -> dict:
    """Shader uniform overrides that define the dense-march reference."""
    return {"stepScale": DENSE_STEP_SCALE, "minStep": float(min_step)}


def dense_march_config(**overrides) -> MarchConfig:
    """High-iteration config for the dense march. The minStep floor means the
    march crawls near surfaces, so it needs a large iteration ceiling to absorb
    pathological grazing crawls before reporting non-convergence."""
    cfg = MarchConfig(
        max_iterations=40000,
        hit_threshold=1e-6,
        max_distance=100.0,
    )
    cfg.min_step_fraction = 0.0
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def capture_dense_reference(scene_name: str, *, width: int = 512, height: int = 512,
                            runner: Optional[GPURunner] = None,
                            min_step: float = DENSE_MIN_STEP
                            ) -> Tuple[Dict, RenderConfig, MarchConfig]:
    scene_id, scene = _resolve_scene(scene_name)
    if scene is None:
        raise ValueError(f"Unknown scene: {scene_name!r}")
    rc = reference_render_config(scene, width, height)
    mc = dense_march_config()
    runner = runner or GPURunner()
    cap = runner.capture(scene_id, DENSE_MARCH_STRATEGY_ID, rc, mc,
                         lipschitz=scene.known_lipschitz_bound(),
                         params=dense_march_params(min_step))
    return cap, rc, mc


def _resolve_scene(scene_name: str):
    scenes = get_all_scenes()
    scene = next((s for s in scenes if s.name == scene_name), None)
    if scene is None:
        return None, None
    return scenes.index(scene), scene


def reference_render_config(scene, width: int, height: int) -> RenderConfig:
    """Per-scene camera (its suggested_camera if any) at the given resolution."""
    sc = scene.suggested_camera()
    if sc is None:
        return RenderConfig(width=width, height=height)
    return RenderConfig(
        width=width, height=height,
        fov_degrees=sc.fov_degrees,
        camera_position=sc.camera_position,
        camera_target=sc.camera_target,
        camera_up=sc.camera_up,
    )


def safe_scene_name(name: str) -> str:
    out = name.replace(" ", "_")
    for ch in "()/\\":
        out = out.replace(ch, "")
    return out


def capture_reference(scene_name: str, *, width: int = 512, height: int = 512,
                      runner: Optional[GPURunner] = None,
                      march_cfg: Optional[MarchConfig] = None
                      ) -> Tuple[Dict, RenderConfig, MarchConfig]:
    scene_id, scene = _resolve_scene(scene_name)
    if scene is None:
        raise ValueError(f"Unknown scene: {scene_name!r}")
    rc = reference_render_config(scene, width, height)
    mc = march_cfg or reference_march_config()
    runner = runner or GPURunner()
    cap = runner.capture(scene_id, 0, rc, mc,
                         lipschitz=scene.known_lipschitz_bound(),
                         params=oracle_params())
    return cap, rc, mc


def generate_references(scene_names: List[str], out_dir: str = "references",
                        width: int = 512, height: int = 512) -> Dict[str, str]:
    """Capture + persist a reference (npz + PNGs + meta.json) per scene."""
    runner = GPURunner()
    gpu = runner.gpu_info()
    results: Dict[str, str] = {}

    for name in scene_names:
        # Primary (understep) oracle — shares the depth normalization for PNGs.
        cap, rc, mc = capture_reference(name, width=width, height=height, runner=runner)
        scene_dir = os.path.join(out_dir, safe_scene_name(name))
        drange = depth_range_of(cap)
        npz = save_capture(cap, scene_dir, "reference", depth_range=drange)
        hit_rate = float(cap["hit"].mean())

        # Additional dense-march oracle (higher trust). Same camera/resolution;
        # stored alongside so the two references can be compared per frame.
        dcap, _, dmc = capture_dense_reference(name, width=width, height=height, runner=runner)
        save_capture(dcap, scene_dir, "reference_dense", depth_range=drange)
        dense_hit_rate = float(dcap["hit"].mean())

        meta = {
            "scene": name,
            "resolution": [width, height],
            "gpu": gpu,
            "camera": {
                "position": list(rc.camera_position),
                "target": list(rc.camera_target),
                "up": list(rc.camera_up),
                "fov_degrees": rc.fov_degrees,
            },
            "reference_march": {
                "kind": "understep_standard",
                "step_scale": ORACLE_STEP_SCALE,
                "max_iterations": mc.max_iterations,
                "hit_threshold": mc.hit_threshold,
                "max_distance": mc.max_distance,
            },
            "reference_dense": {
                "kind": "dense_march",
                "step_scale": DENSE_STEP_SCALE,
                "min_step": DENSE_MIN_STEP,
                "max_iterations": dmc.max_iterations,
                "hit_threshold": dmc.hit_threshold,
                "max_distance": dmc.max_distance,
            },
            "hit_rate": hit_rate,
            "dense_hit_rate": dense_hit_rate,
            "depth_range": list(drange),
        }
        with open(os.path.join(scene_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        results[name] = npz
        print(f"  [ok] {name:<24} hit={hit_rate:6.2%}  dense_hit={dense_hit_rate:6.2%}"
              f"  depth=[{drange[0]:.3f},{drange[1]:.3f}]  -> {scene_dir}")

    return results


def main(argv: Optional[List[str]] = None) -> int:
    force_utf8_stdout()
    p = argparse.ArgumentParser(description="Generate ground-truth reference captures.")
    p.add_argument("--scenes", type=str, default="Sphere,Grazing Plane",
                   help="Comma-separated scene names, or 'all'.")
    p.add_argument("--res", type=int, default=512, help="Square resolution (width=height).")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--out-dir", type=str, default="references")
    args = p.parse_args(argv)

    if args.scenes.strip().lower() == "all":
        scene_names = [s.name for s in get_all_scenes()]
    else:
        scene_names = [s.strip() for s in args.scenes.split(",") if s.strip()]

    width = args.width or args.res
    height = args.height or args.res

    print(f"Generating references for {len(scene_names)} scene(s) at {width}x{height} -> {args.out_dir}/")
    generate_references(scene_names, out_dir=args.out_dir, width=width, height=height)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
