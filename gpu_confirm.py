import os
import numpy as np
import pandas as pd
from raymarching_benchmark.config import RenderConfig, MarchConfig
from raymarching_benchmark.gpu.runner import run_gpu_benchmark
from raymarching_benchmark.scenes.catalog import get_all_scenes

def main():
    print("="*60)
    print("GPU Ray Marching Benchmark Confirmation")
    print("="*60)

    rc = RenderConfig(width=160, height=120)
    mc = MarchConfig()
    
    # Load CPU results for comparison
    cpu_results_path = "results/matrix_iteration_mean.csv"
    if not os.path.exists(cpu_results_path):
        print(f"Error: CPU results not found at {cpu_results_path}. Run CPU benchmark first.")
        return
    
    cpu_df = pd.read_csv(cpu_results_path, index_col=0)
    
    scenes = get_all_scenes()
    strategies = ["Standard", "Overstep-Bisect", "Relaxed(ω=1.2)", "Segment", "Enhanced", "AR-ST"]
    
    comparison_data = []

    for scene in scenes:
        scene_name = scene.name
        if scene_name not in cpu_df.index:
            continue
            
        print(f"\n>> Validating Scene: {scene_name}")
        
        for strat_name in strategies:
            if strat_name not in cpu_df.columns:
                continue
                
            cpu_val = cpu_df.loc[scene_name, strat_name]
            
            # Run GPU
            try:
                pixels = run_gpu_benchmark(scene_name, strat_name, rc, mc)
                if pixels is None: continue
                
                # pixels[..., 1] is iterations / max_iterations
                gpu_iters = pixels[..., 1] * mc.max_iterations
                gpu_mean = np.mean(gpu_iters)
                
                diff = gpu_mean - cpu_val
                pct_diff = (diff / cpu_val * 100) if cpu_val != 0 else 0
                
                match_status = "OK" if abs(pct_diff) < 5.0 else "DRIFT"
                if abs(diff) < 0.1: match_status = "EXACT"
                
                print(f"  {strat_name:15} | CPU: {cpu_val:6.2f} | GPU: {gpu_mean:6.2f} | Diff: {diff:+6.2f} ({pct_diff:+.1f}%) | [{match_status}]")
                
                comparison_data.append({
                    "Scene": scene_name,
                    "Strategy": strat_name,
                    "CPU_Mean": cpu_val,
                    "GPU_Mean": gpu_mean,
                    "Diff": diff,
                    "Status": match_status
                })
            except Exception as e:
                print(f"  {strat_name:15} | FAILED: {e}")

    # Summary
    comp_df = pd.DataFrame(comparison_data)
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    if not comp_df.empty:
        status_counts = comp_df['Status'].value_counts()
        print(status_counts)
        
        avg_abs_diff = comp_df['Diff'].abs().mean()
        print(f"\nAverage Absolute Iteration Drift: {avg_abs_diff:.4f}")
        
        if avg_abs_diff < 1.0:
            print("\nRESULT: SUCCESS! GPU results statistically match CPU results.")
        else:
            print("\nRESULT: WARNING! Significant drift detected. Check GLSL implementation details.")
    else:
        print("No data collected.")

if __name__ == "__main__":
    main()
