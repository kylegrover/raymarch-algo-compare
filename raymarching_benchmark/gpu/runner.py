import os
import numpy as np
import moderngl
from typing import Optional, List
from ..core.types import MarchResult
from ..config import RenderConfig, MarchConfig
from ..scenes.catalog import get_all_scenes

class GPURunner:
    def __init__(self):
        # Create a headless context
        self.ctx = moderngl.create_standalone_context()
        self.prog = None
        self.fbo = None
        self.tex = None
        
    def _load_shader(self, shader_dir: str):
        with open(os.path.join(shader_dir, "primitives.glsl"), 'r') as f:
            primitives = f.read()
        with open(os.path.join(shader_dir, "scenes.glsl"), 'r') as f:
            scenes = f.read()
        with open(os.path.join(shader_dir, "strategies.glsl"), 'r') as f:
            strategies = f.read()
        with open(os.path.join(shader_dir, "main.glsl"), 'r') as f:
            main_src = f.read()
            
        # Concatenate in order
        full_src = f"#version 430\n{primitives}\n{scenes}\n{strategies}\n{main_src}"
        
        try:
            self.prog = self.ctx.program(
                vertex_shader='''
                #version 430
                in vec2 in_vert;
                void main() {
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                }
                ''',
                fragment_shader=full_src
            )
        except Exception as e:
            print("GLSL Compilation Error:")
            print(e)
            raise

    def render(self, scene_id: int, strategy_id: int, 
               render_cfg: RenderConfig, march_cfg: MarchConfig,
               lipschitz: float = 1.0) -> np.ndarray:
        
        if self.prog is None:
            shader_dir = os.path.join(os.path.dirname(__file__), "shaders")
            self._load_shader(shader_dir)

        width, height = render_cfg.width, render_cfg.height
        
        # Setup FBO and Texture
        if self.tex is None or (self.tex.width, self.tex.height) != (width, height):
            if self.tex: self.tex.release()
            if self.fbo: self.fbo.release()
            self.tex = self.ctx.texture((width, height), 4, dtype='f4')
            self.fbo = self.ctx.framebuffer(color_attachments=[self.tex])

        self.fbo.use()
        
        # Camera
        # We need to replicate the camera logic from core/camera.py
        from ..core.camera import Camera
        from ..core.vec3 import Vec3
        cam = Camera(
            position=Vec3(*render_cfg.camera_position),
            target=Vec3(*render_cfg.camera_target),
            up=Vec3(*render_cfg.camera_up),
            fov_degrees=render_cfg.fov_degrees,
            width=width,
            height=height
        )
        
        # Uniforms
        def set_uniform(name, value):
            if name in self.prog:
                self.prog[name].value = value

        set_uniform('resolution', (width, height))
        set_uniform('camPos', cam.position.to_tuple())
        set_uniform('camDir', cam.forward.to_tuple())
        set_uniform('camUp', cam.up.to_tuple())
        set_uniform('camRight', cam.right.to_tuple())
        
        if lipschitz is None:
            lipschitz = 1.0
            
        set_uniform('sceneId', scene_id)
        set_uniform('strategyId', strategy_id)
        set_uniform('maxIterations', march_cfg.max_iterations)
        set_uniform('hitThreshold', march_cfg.hit_threshold)
        set_uniform('maxDistance', march_cfg.max_distance)
        set_uniform('omega', 1.2)
        set_uniform('lipschitz', lipschitz)

        # Full screen quad
        vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0,
        ], dtype='f4')
        vbo = self.ctx.buffer(vertices)
        vao = self.ctx.simple_vertex_array(self.prog, vbo, 'in_vert')
        
        vao.render(moderngl.TRIANGLE_STRIP)
        
        # Read back
        data = self.fbo.read(components=4, dtype='f4')
        pixels = np.frombuffer(data, dtype='f4').reshape(height, width, 4)
        
        return pixels

def run_gpu_benchmark(scene_name: str, strategy_name: str, render_cfg: RenderConfig, march_cfg: MarchConfig):
    from ..strategies import list_strategies
    from ..scenes.catalog import get_all_scenes
    
    scenes = get_all_scenes()
    scene = next((s for s in scenes if s.name == scene_name), None)
    if not scene: return None
    
    scene_id = scenes.index(scene)
    
    # Map strategy name to ID
    strategy_map = {
        "Standard": 0,
        "Overstep-Bisect": 1,
        "Relaxed(ω=1.2)": 2,
        "Segment": 3,
        "Enhanced": 4,
        "AR-ST": 5,
        "Hybrid": 1, # Using Overstep-Bisect as Hybrid proxy for now
    }
    strat_id = strategy_map.get(strategy_name, 0)
    
    runner = GPURunner()
    pixels = runner.render(scene_id, strat_id, render_cfg, march_cfg, lipschitz=scene.known_lipschitz_bound())
    
    return pixels
