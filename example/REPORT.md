# Ray Marching Algorithm Benchmark Report

## Overview
Tested **9** strategies across **14** scenes.

> Note: GPU timings (where shown) are measured by executing the GLSL shader via ModernGL and
> dividing the synchronous GPU render time by the number of pixels. Readback is excluded; driver/submit
> overhead may still be included. Use the `gpu_validation__*.csv` in the `results/` folder for raw data.
> GPU measurements were taken at: **1920x1080**

## Visual Comparisons
### Performance overview
![Iteration Count](chart_iterations.png)

### Workload Divergence (Warp Divergence)
![Divergence](chart_divergence.png)

### Speed vs Accuracy
![Speed vs Accuracy](chart_speed_accuracy.png)

## Aggregated Statistics

| Strategy | Iterations (avg) | Hit Rate | Wins | Warp Div | CPU Time (us/ray) | GPU Time (us/ray) (median) | GPU ms/frame (median) | GPU samples | GPU WD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AR-ST | 14.86 | 20.4% | 1 | 2.55 | 32.30 | 0.00 | 2.59 | 10.0 | 0.84 |
| Curvature-Aware Tracing | 16.38 | 21.1% | 0 | 3.14 | 41.97 | 0.00 | 2.41 | 10.0 | 0.84 |
| Enhanced | 16.00 | 20.7% | 0 | 3.21 | 33.56 | 0.00 | 3.16 | 10.0 | 0.90 |
| Hybrid | 15.77 | 21.2% | 0 | 2.80 | 33.49 | 0.00 | 2.24 | 10.0 | 0.84 |
| Overstep-Bisect | 14.70 | 21.2% | 0 | 2.62 | 30.17 | 0.00 | 2.88 | 10.0 | 0.91 |
| Relaxed(ω=1.2) | 15.02 | 20.7% | 0 | 2.91 | 31.55 | 0.00 | 2.37 | 10.0 | 0.84 |
| Segment | 10.50 | 22.4% | 13 | 2.05 | 36.73 | 0.00 | 2.40 | 10.0 | 0.96 |
| Slope-AR(β=0.3) | 12.95 | 21.2% | 0 | 2.35 | 28.27 | 0.00 | 2.39 | 10.0 | 0.84 |
| Standard | 17.38 | 21.1% | 0 | 3.19 | 33.94 | 0.00 | 2.91 | 10.0 | 0.84 |

## Per-Scene Analysis
### Sphere
![Sphere Comparison](compare__Sphere.png)

| Strategy | Iterations | Hit Rate | P95 | Warp Div | CPU Time (us/ray) | GPU Time (us/ray) (median) | GPU ms/frame (median) | GPU samples | GPU WD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AR-ST | 7.73 | 5.6% | 12.0 | 0.58 | 11.19 | 0.0002 | 0.38 | 10 | 0.26 |
| Curvature-Aware Tracing | 9.02 | 5.6% | 13.0 | 0.78 | 14.96 | 0.0005 | 1.04 | 10 | 0.26 |
| Enhanced | 8.38 | 5.6% | 12.0 | 0.80 | 11.59 | 0.0004 | 0.76 | 10 | 0.26 |
| Hybrid | 9.26 | 5.6% | 15.0 | 0.63 | 12.78 | 0.0003 | 0.55 | 10 | 0.26 |
| Overstep-Bisect | 9.27 | 5.6% | 14.0 | 0.58 | 11.45 | 0.0005 | 1.08 | 10 | 0.35 |
| Relaxed(ω=1.2) | 8.08 | 5.6% | 13.0 | 0.65 | 11.12 | 0.0002 | 0.37 | 10 | 0.26 |
| Segment | 5.86 | 5.6% | 8.0 | 0.40 | 11.67 | 0.0006 | 1.16 | 10 | 0.03 |
| Slope-AR(β=0.3) | 7.75 | 5.6% | 11.0 | 0.45 | 10.31 | 0.0006 | 1.28 | 10 | 0.26 |
| Standard | 9.32 | 5.6% | 14.0 | 0.81 | 11.11 | 0.0002 | 0.39 | 10 | 0.26 |

### Grazing Plane
![Grazing Plane Comparison](compare__Grazing_Plane.png)

| Strategy | Iterations | Hit Rate | P95 | Warp Div | CPU Time (us/ray) | GPU Time (us/ray) (median) | GPU ms/frame (median) | GPU samples | GPU WD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AR-ST | 38.31 | 59.3% | 138.0 | 10.09 | 43.67 | 0.0010 | 2.11 | 10 | 1.08 |
| Curvature-Aware Tracing | 42.18 | 59.1% | 166.0 | 11.12 | 66.09 | 0.0008 | 1.64 | 10 | 1.08 |
| Enhanced | 41.92 | 59.1% | 166.0 | 11.14 | 46.86 | 0.0031 | 6.41 | 10 | 1.10 |
| Hybrid | 38.99 | 59.4% | 116.0 | 9.52 | 42.84 | 0.0011 | 2.25 | 10 | 1.08 |
| Overstep-Bisect | 35.24 | 59.7% | 103.0 | 8.96 | 36.42 | 0.0013 | 2.66 | 10 | 1.08 |
| Relaxed(ω=1.2) | 38.34 | 59.3% | 138.0 | 9.97 | 40.67 | 0.0007 | 1.39 | 10 | 1.08 |
| Segment | 24.51 | 59.8% | 84.0 | 6.71 | 47.79 | 0.0006 | 1.21 | 10 | 0.50 |
| Slope-AR(β=0.3) | 27.69 | 59.8% | 91.0 | 6.97 | 33.19 | 0.0006 | 1.24 | 10 | 1.08 |
| Standard | 45.76 | 59.1% | 166.0 | 10.96 | 48.42 | 0.0010 | 2.09 | 10 | 1.08 |

### Cube
![Cube Comparison](compare__Cube.png)

| Strategy | Iterations | Hit Rate | P95 | Warp Div | CPU Time (us/ray) | GPU Time (us/ray) (median) | GPU ms/frame (median) | GPU samples | GPU WD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AR-ST | 6.52 | 2.3% | 9.0 | 0.24 | 18.41 | 0.0005 | 1.11 | 10 | 0.14 |
| Curvature-Aware Tracing | 7.95 | 3.3% | 12.0 | 0.57 | 24.81 | 0.0005 | 1.02 | 10 | 0.14 |
| Enhanced | 7.40 | 3.3% | 12.0 | 0.61 | 19.68 | 0.0005 | 1.10 | 10 | 0.15 |
| Hybrid | 7.95 | 3.3% | 12.0 | 0.57 | 20.29 | 0.0005 | 1.10 | 10 | 0.14 |
| Overstep-Bisect | 8.09 | 3.3% | 12.0 | 0.54 | 19.93 | 0.0003 | 0.59 | 10 | 0.17 |
| Relaxed(ω=1.2) | 6.82 | 2.3% | 10.0 | 0.31 | 19.21 | 0.0006 | 1.19 | 10 | 0.14 |
| Segment | 5.26 | 3.3% | 7.0 | 0.30 | 20.67 | 0.0005 | 1.13 | 10 | 0.02 |
| Slope-AR(β=0.3) | 6.87 | 3.3% | 9.0 | 0.31 | 16.06 | 0.0003 | 0.53 | 10 | 0.14 |
| Standard | 7.95 | 3.3% | 12.0 | 0.57 | 18.06 | 0.0003 | 0.65 | 10 | 0.14 |

### Thin Torus
![Thin Torus Comparison](compare__Thin_Torus.png)

| Strategy | Iterations | Hit Rate | P95 | Warp Div | CPU Time (us/ray) | GPU Time (us/ray) (median) | GPU ms/frame (median) | GPU samples | GPU WD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AR-ST | 6.61 | 0.2% | 9.0 | 0.21 | 11.08 | 0.0010 | 1.98 | 10 | 0.14 |
| Curvature-Aware Tracing | 8.04 | 0.4% | 12.0 | 0.30 | 15.29 | 0.0003 | 0.62 | 10 | 0.14 |
| Enhanced | 7.37 | 0.4% | 11.0 | 0.39 | 11.68 | 0.0006 | 1.25 | 10 | 0.15 |
| Hybrid | 8.07 | 0.4% | 12.0 | 0.29 | 12.83 | 0.0003 | 0.70 | 10 | 0.14 |
| Overstep-Bisect | 8.07 | 0.4% | 12.0 | 0.28 | 11.95 | 0.0010 | 2.12 | 10 | 0.15 |
| Relaxed(ω=1.2) | 6.81 | 0.2% | 10.0 | 0.21 | 10.51 | 0.0003 | 0.61 | 10 | 0.14 |
| Segment | 5.37 | 0.4% | 7.0 | 0.15 | 12.31 | 0.0006 | 1.19 | 10 | 0.01 |
| Slope-AR(β=0.3) | 7.00 | 0.4% | 9.0 | 0.20 | 12.45 | 0.0006 | 1.20 | 10 | 0.14 |
| Standard | 8.08 | 0.4% | 12.0 | 0.30 | 11.24 | 0.0005 | 1.08 | 10 | 0.14 |

### Cylinder
![Cylinder Comparison](compare__Cylinder.png)

| Strategy | Iterations | Hit Rate | P95 | Warp Div | CPU Time (us/ray) | GPU Time (us/ray) (median) | GPU ms/frame (median) | GPU samples | GPU WD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AR-ST | 6.61 | 3.6% | 10.0 | 0.31 | 12.22 | 0.0007 | 1.38 | 10 | 0.62 |
| Curvature-Aware Tracing | 7.88 | 4.5% | 11.0 | 0.64 | 16.38 | 0.0004 | 0.77 | 10 | 0.62 |
| Enhanced | 7.32 | 4.5% | 11.0 | 0.68 | 12.79 | 0.0007 | 1.36 | 10 | 0.62 |
| Hybrid | 7.99 | 4.5% | 12.0 | 0.47 | 13.83 | 0.0004 | 0.83 | 10 | 0.62 |
| Overstep-Bisect | 8.07 | 4.5% | 12.0 | 0.45 | 13.66 | 0.0011 | 2.29 | 10 | 0.83 |
| Relaxed(ω=1.2) | 6.87 | 3.6% | 10.0 | 0.34 | 12.38 | 0.0008 | 1.63 | 10 | 0.62 |
| Segment | 5.27 | 4.5% | 7.0 | 0.35 | 15.38 | 0.0005 | 1.13 | 10 | 0.04 |
| Slope-AR(β=0.3) | 6.96 | 4.5% | 9.0 | 0.36 | 11.67 | 0.0007 | 1.38 | 10 | 0.62 |
| Standard | 8.03 | 4.5% | 12.0 | 0.66 | 12.50 | 0.0004 | 0.85 | 10 | 0.62 |

### Near Miss
![Near Miss Comparison](compare__Near_Miss.png)

| Strategy | Iterations | Hit Rate | P95 | Warp Div | CPU Time (us/ray) | GPU Time (us/ray) (median) | GPU ms/frame (median) | GPU samples | GPU WD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AR-ST | 6.74 | 3.3% | 10.0 | 0.30 | 20.39 | 0.0006 | 1.27 | 10 | 0.49 |
| Curvature-Aware Tracing | 8.10 | 4.3% | 11.0 | 0.73 | 24.50 | 0.0006 | 1.27 | 10 | 0.49 |
| Enhanced | 7.54 | 4.3% | 11.0 | 0.77 | 20.06 | 0.0013 | 2.74 | 10 | 0.49 |
| Hybrid | 8.28 | 4.3% | 13.0 | 0.61 | 21.99 | 0.0005 | 1.05 | 10 | 0.49 |
| Overstep-Bisect | 8.28 | 4.3% | 12.0 | 0.54 | 21.72 | 0.0008 | 1.63 | 10 | 0.67 |
| Relaxed(ω=1.2) | 6.94 | 3.3% | 10.0 | 0.29 | 19.39 | 0.0009 | 1.86 | 10 | 0.49 |
| Segment | 5.46 | 4.3% | 7.0 | 0.38 | 22.75 | 0.0005 | 1.14 | 10 | 0.05 |
| Slope-AR(β=0.3) | 7.12 | 4.3% | 10.0 | 0.43 | 18.44 | 0.0004 | 0.84 | 10 | 0.49 |
| Standard | 8.32 | 4.3% | 12.0 | 0.76 | 21.00 | 0.0005 | 1.03 | 10 | 0.49 |

### Hollow Cube (CSG)
![Hollow Cube (CSG) Comparison](compare__Hollow_Cube_(CSG).png)

| Strategy | Iterations | Hit Rate | P95 | Warp Div | CPU Time (us/ray) | GPU Time (us/ray) (median) | GPU ms/frame (median) | GPU samples | GPU WD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AR-ST | 6.57 | 1.1% | 10.0 | 0.25 | 19.12 | 0.0002 | 0.49 | 10 | 0.24 |
| Curvature-Aware Tracing | 8.18 | 2.1% | 13.0 | 0.62 | 27.29 | 0.0005 | 0.96 | 10 | 0.24 |
| Enhanced | 7.65 | 1.7% | 13.0 | 0.64 | 21.04 | 0.0008 | 1.64 | 10 | 0.25 |
| Hybrid | 8.20 | 2.1% | 13.0 | 0.62 | 23.14 | 0.0005 | 1.11 | 10 | 0.24 |
| Overstep-Bisect | 8.28 | 2.1% | 13.0 | 0.55 | 23.37 | 0.0003 | 0.67 | 10 | 0.32 |
| Relaxed(ω=1.2) | 6.88 | 1.1% | 10.0 | 0.32 | 20.11 | 0.0005 | 1.10 | 10 | 0.24 |
| Segment | 5.39 | 2.1% | 8.0 | 0.32 | 23.12 | 0.0006 | 1.26 | 10 | 0.01 |
| Slope-AR(β=0.3) | 7.07 | 2.1% | 10.0 | 0.37 | 19.59 | 0.0006 | 1.16 | 10 | 0.24 |
| Standard | 8.20 | 2.1% | 13.0 | 0.62 | 22.03 | 0.0006 | 1.31 | 10 | 0.24 |

### Smooth Blend
![Smooth Blend Comparison](compare__Smooth_Blend.png)

| Strategy | Iterations | Hit Rate | P95 | Warp Div | CPU Time (us/ray) | GPU Time (us/ray) (median) | GPU ms/frame (median) | GPU samples | GPU WD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AR-ST | 6.27 | 1.4% | 9.0 | 0.22 | 27.34 | 0.0007 | 1.53 | 10 | 0.22 |
| Curvature-Aware Tracing | 7.63 | 2.2% | 11.0 | 0.51 | 32.59 | 0.0007 | 1.38 | 10 | 0.22 |
| Enhanced | 7.06 | 2.2% | 10.2 | 0.55 | 27.82 | 0.0017 | 3.47 | 10 | 0.22 |
| Hybrid | 7.66 | 2.2% | 11.0 | 0.40 | 31.14 | 0.0008 | 1.70 | 10 | 0.22 |
| Overstep-Bisect | 7.71 | 2.2% | 11.0 | 0.37 | 31.74 | 0.0007 | 1.50 | 10 | 0.27 |
| Relaxed(ω=1.2) | 6.50 | 1.4% | 9.0 | 0.24 | 26.19 | 0.0007 | 1.42 | 10 | 0.22 |
| Segment | 5.12 | 2.2% | 7.0 | 0.27 | 31.85 | 0.0003 | 0.55 | 10 | 0.02 |
| Slope-AR(β=0.3) | 6.76 | 2.2% | 9.0 | 0.27 | 25.26 | 0.0004 | 0.84 | 10 | 0.22 |
| Standard | 7.69 | 2.2% | 11.0 | 0.52 | 35.40 | 0.0004 | 0.83 | 10 | 0.22 |

### Onion Shell
![Onion Shell Comparison](compare__Onion_Shell.png)

| Strategy | Iterations | Hit Rate | P95 | Warp Div | CPU Time (us/ray) | GPU Time (us/ray) (median) | GPU ms/frame (median) | GPU samples | GPU WD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AR-ST | 8.65 | 10.3% | 17.0 | 1.14 | 14.01 | 0.0010 | 1.99 | 10 | 0.88 |
| Curvature-Aware Tracing | 9.12 | 10.3% | 14.0 | 1.21 | 17.21 | 0.0010 | 2.10 | 10 | 0.88 |
| Enhanced | 8.53 | 10.3% | 14.0 | 1.21 | 12.18 | 0.0022 | 4.58 | 10 | 0.89 |
| Hybrid | 9.65 | 10.3% | 17.0 | 1.05 | 14.14 | 0.0008 | 1.72 | 10 | 0.88 |
| Overstep-Bisect | 9.67 | 10.3% | 16.0 | 0.99 | 13.41 | 0.0021 | 4.34 | 10 | 1.14 |
| Relaxed(ω=1.2) | 8.96 | 10.3% | 18.0 | 1.36 | 12.74 | 0.0005 | 1.06 | 10 | 0.88 |
| Segment | 6.07 | 10.3% | 10.0 | 0.63 | 13.44 | 0.0007 | 1.43 | 10 | 0.13 |
| Slope-AR(β=0.3) | 8.06 | 10.3% | 13.0 | 0.71 | 11.66 | 0.0007 | 1.37 | 10 | 0.88 |
| Standard | 9.71 | 10.3% | 17.0 | 1.24 | 12.36 | 0.0018 | 3.64 | 10 | 0.88 |

### Menger Sponge (iter=3)
![Menger Sponge (iter=3) Comparison](compare__Menger_Sponge_(iter=3).png)

| Strategy | Iterations | Hit Rate | P95 | Warp Div | CPU Time (us/ray) | GPU Time (us/ray) (median) | GPU ms/frame (median) | GPU samples | GPU WD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AR-ST | 6.83 | 1.7% | 10.0 | 0.72 | 38.90 | 0.0008 | 1.61 | 10 | 0.90 |
| Curvature-Aware Tracing | 8.41 | 2.5% | 13.0 | 1.03 | 50.22 | 0.0012 | 2.40 | 10 | 0.90 |
| Enhanced | 7.91 | 2.5% | 13.0 | 1.08 | 43.67 | 0.0014 | 2.83 | 10 | 0.90 |
| Hybrid | 8.36 | 2.6% | 13.0 | 0.92 | 46.08 | 0.0013 | 2.61 | 10 | 0.90 |
| Overstep-Bisect | 8.40 | 2.6% | 13.0 | 0.80 | 46.35 | 0.0015 | 3.18 | 10 | 0.87 |
| Relaxed(ω=1.2) | 7.18 | 1.7% | 10.0 | 0.83 | 39.79 | 0.0014 | 2.83 | 10 | 0.90 |
| Segment | 5.50 | 2.6% | 8.0 | 0.54 | 47.87 | 0.0010 | 1.98 | 10 | 0.04 |
| Slope-AR(β=0.3) | 7.23 | 2.6% | 10.0 | 0.65 | 35.63 | 0.0007 | 1.52 | 10 | 0.90 |
| Standard | 8.43 | 2.6% | 13.0 | 1.07 | 43.83 | 0.0018 | 3.73 | 10 | 0.90 |

### Mandelbulb
![Mandelbulb Comparison](compare__Mandelbulb.png)

| Strategy | Iterations | Hit Rate | P95 | Warp Div | CPU Time (us/ray) | GPU Time (us/ray) (median) | GPU ms/frame (median) | GPU samples | GPU WD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AR-ST | 15.75 | 18.2% | 41.0 | 3.32 | 62.02 | 0.0047 | 9.84 | 10 | 1.34 |
| Curvature-Aware Tracing | 16.50 | 18.2% | 39.0 | 4.30 | 73.72 | 0.0041 | 8.47 | 10 | 1.34 |
| Enhanced | 15.65 | 18.2% | 38.0 | 4.34 | 58.16 | 0.0027 | 5.54 | 10 | 1.21 |
| Hybrid | 16.64 | 18.0% | 37.0 | 2.92 | 60.76 | 0.0034 | 7.06 | 10 | 1.34 |
| Overstep-Bisect | 15.09 | 17.8% | 31.0 | 2.41 | 48.57 | 0.0062 | 12.85 | 10 | 1.17 |
| Relaxed(ω=1.2) | 16.68 | 18.2% | 45.0 | 4.39 | 68.24 | 0.0022 | 4.55 | 10 | 1.34 |
| Segment | 10.27 | 18.2% | 24.0 | 2.19 | 68.49 | 0.0012 | 2.59 | 10 | 0.13 |
| Slope-AR(β=0.3) | 14.13 | 18.2% | 36.0 | 3.13 | 55.83 | 0.0047 | 9.73 | 10 | 1.34 |
| Standard | 18.24 | 18.2% | 47.0 | 4.37 | 66.46 | 0.0057 | 11.92 | 10 | 1.34 |

### Bad Lipschitz Sphere
![Bad Lipschitz Sphere Comparison](compare__Bad_Lipschitz_Sphere.png)

| Strategy | Iterations | Hit Rate | P95 | Warp Div | CPU Time (us/ray) | GPU Time (us/ray) (median) | GPU ms/frame (median) | GPU samples | GPU WD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AR-ST | 4.73 | 0.0% | 5.0 | 0.04 | 8.61 | 0.0009 | 1.78 | 10 | 1.11 |
| Curvature-Aware Tracing | 5.81 | 0.0% | 7.0 | 0.08 | 12.12 | 0.0008 | 1.71 | 10 | 1.11 |
| Enhanced | 5.50 | 0.0% | 7.0 | 0.11 | 10.46 | 0.0011 | 2.25 | 10 | 1.11 |
| Hybrid | 5.81 | 0.0% | 7.0 | 0.08 | 8.95 | 0.0009 | 1.89 | 10 | 1.11 |
| Overstep-Bisect | 5.81 | 0.0% | 7.0 | 0.08 | 8.82 | 0.0020 | 4.25 | 10 | 1.11 |
| Relaxed(ω=1.2) | 5.00 | 0.0% | 5.0 | 0.00 | 9.36 | 0.0018 | 3.65 | 10 | 1.11 |
| Segment | 7.24 | 16.6% | 12.0 | 0.78 | 13.55 | 0.0009 | 1.85 | 10 | 0.11 |
| Slope-AR(β=0.3) | 5.30 | 0.0% | 6.0 | 0.04 | 7.97 | 0.0013 | 2.64 | 10 | 1.11 |
| Standard | 5.81 | 0.0% | 7.0 | 0.08 | 8.63 | 0.0018 | 3.68 | 10 | 1.11 |

### Pillar Forest
![Pillar Forest Comparison](compare__Pillar_Forest.png)

| Strategy | Iterations | Hit Rate | P95 | Warp Div | CPU Time (us/ray) | GPU Time (us/ray) (median) | GPU ms/frame (median) | GPU samples | GPU WD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AR-ST | 25.45 | 80.7% | 65.0 | 10.82 | 58.96 | 0.0022 | 4.63 | 10 | 3.23 |
| Curvature-Aware Tracing | 25.82 | 85.6% | 67.0 | 13.89 | 72.50 | 0.0025 | 5.12 | 10 | 3.23 |
| Enhanced | 27.38 | 80.6% | 69.0 | 14.49 | 62.03 | 0.0019 | 3.99 | 10 | 5.21 |
| Hybrid | 28.65 | 85.6% | 65.0 | 13.14 | 64.50 | 0.0019 | 3.99 | 10 | 3.23 |
| Overstep-Bisect | 28.87 | 85.6% | 62.0 | 11.80 | 61.63 | 0.0010 | 2.16 | 10 | 4.56 |
| Relaxed(ω=1.2) | 26.43 | 85.6% | 66.0 | 14.31 | 56.38 | 0.0028 | 5.71 | 10 | 3.23 |
| Segment | 15.47 | 85.6% | 36.0 | 7.51 | 61.68 | 0.0028 | 5.88 | 10 | 1.55 |
| Slope-AR(β=0.3) | 25.26 | 85.6% | 57.0 | 11.01 | 58.90 | 0.0027 | 5.50 | 10 | 3.23 |
| Standard | 29.31 | 85.6% | 71.0 | 14.93 | 58.16 | 0.0016 | 3.34 | 10 | 3.23 |

### Thin Planes Stack
![Thin Planes Stack Comparison](compare__Thin_Planes_Stack.png)

| Strategy | Iterations | Hit Rate | P95 | Warp Div | CPU Time (us/ray) | GPU Time (us/ray) (median) | GPU ms/frame (median) | GPU samples | GPU WD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AR-ST | 61.27 | 97.7% | 251.0 | 7.47 | 106.32 | 0.0030 | 6.21 | 10 | 1.08 |
| Curvature-Aware Tracing | 64.65 | 96.9% | 302.0 | 8.14 | 139.96 | 0.0025 | 5.24 | 10 | 1.08 |
| Enhanced | 64.36 | 96.9% | 302.0 | 8.16 | 111.80 | 0.0031 | 6.36 | 10 | 0.00 |
| Hybrid | 55.29 | 98.1% | 190.0 | 8.02 | 95.56 | 0.0023 | 4.75 | 10 | 1.08 |
| Overstep-Bisect | 44.92 | 98.1% | 163.0 | 8.35 | 73.40 | 0.0005 | 0.98 | 10 | 0.00 |
| Relaxed(ω=1.2) | 58.84 | 97.7% | 251.0 | 7.47 | 95.54 | 0.0028 | 5.74 | 10 | 1.08 |
| Segment | 40.22 | 98.1% | 152.0 | 8.21 | 123.60 | 0.0053 | 11.02 | 10 | 10.72 |
| Slope-AR(β=0.3) | 44.03 | 98.1% | 159.0 | 8.07 | 78.81 | 0.0020 | 4.22 | 10 | 1.08 |
| Standard | 68.46 | 96.9% | 302.0 | 7.84 | 105.95 | 0.0030 | 6.13 | 10 | 1.08 |

