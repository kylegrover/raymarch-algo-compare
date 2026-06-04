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


def reference_march_config(**overrides) -> MarchConfig:
    """High-budget marching config used as ground truth."""
    cfg = MarchConfig(
        max_iterations=4096,
        hit_threshold=1e-6,
        max_distance=100.0,
    )
    cfg.min_step_fraction = 0.0
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


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
    cap = runner.capture(scene_id, 0, rc, mc, lipschitz=scene.known_lipschitz_bound())
    return cap, rc, mc


def generate_references(scene_names: List[str], out_dir: str = "references",
                        width: int = 512, height: int = 512) -> Dict[str, str]:
    """Capture + persist a reference (npz + PNGs + meta.json) per scene."""
    runner = GPURunner()
    gpu = runner.gpu_info()
    results: Dict[str, str] = {}

    for name in scene_names:
        cap, rc, mc = capture_reference(name, width=width, height=height, runner=runner)
        scene_dir = os.path.join(out_dir, safe_scene_name(name))
        drange = depth_range_of(cap)
        npz = save_capture(cap, scene_dir, "reference", depth_range=drange)

        hit_rate = float(cap["hit"].mean())
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
                "max_iterations": mc.max_iterations,
                "hit_threshold": mc.hit_threshold,
                "max_distance": mc.max_distance,
            },
            "hit_rate": hit_rate,
            "depth_range": list(drange),
        }
        with open(os.path.join(scene_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        results[name] = npz
        print(f"  [ok] {name:<24} hit_rate={hit_rate:6.2%}  depth=[{drange[0]:.3f},{drange[1]:.3f}]  -> {scene_dir}")

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
