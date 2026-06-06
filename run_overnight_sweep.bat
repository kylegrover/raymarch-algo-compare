@echo off
REM ============================================================================
REM  Overnight GPU grid run  (SWEEP_PLAN.md  6.5 -- the DEEPENED 13-scene plan)
REM
REM  Launched by Windows Task Scheduler at midnight, in the logged-in INTERACTIVE
REM  session so it can acquire a GPU (WGL) context. NOTE: this same command hangs
REM  forever if launched by the Claude agent's shell (non-interactive session has
REM  no GPU) -- which is exactly why this runs as a Scheduled Task instead.
REM
REM  Output -> sweep_grid512.jsonl  (FRESH grid, NOT a resume of sweep.jsonl;
REM  res 512 busts the config_hash so old rows are not reused). Several hours.
REM  All console output is tee'd to overnight_run.log for a morning post-mortem.
REM ============================================================================
setlocal
cd /d "C:\Users\kyle\projects\glsl\raymarch-algo-compare"
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
set "LOG=overnight_run.log"
set "S=Sphere,Grazing Plane,Cube,Thin Torus,Mandelbulb,Cylinder,Near Miss,Hollow Cube (CSG),Onion Shell,Thin Planes Stack,Menger Sponge (iter=3),Sphere Cloud,Bumpy Sphere"

echo === overnight sweep started %DATE% %TIME% === > "%LOG%"

where uv >nul 2>&1 || (echo ERROR: uv not on PATH -- aborting >> "%LOG%" & exit /b 1)

echo [1/5] budget sweep (res 512, budgets 32..2048, full-score)... >> "%LOG%"
uv run python -m raymarching_benchmark.sweep --mode budget   --grid --full-score --res 512 --budgets 32,64,128,256,512,1024,2048 --out sweep_grid512.jsonl --scenes "%S%" >> "%LOG%" 2>&1

echo [2/5] residual sweep (matched-residual epsilon)... >> "%LOG%"
uv run python -m raymarching_benchmark.sweep --mode residual --grid --full-score --res 512 --out sweep_grid512.jsonl --scenes "%S%" >> "%LOG%" 2>&1

echo [3/5] regenerate features (res 256, strategy-independent)... >> "%LOG%"
uv run python -m raymarching_benchmark.report.features  --res 256 --out features.jsonl >> "%LOG%" 2>&1

echo [4/5] discovery join by evals... >> "%LOG%"
uv run python -m raymarching_benchmark.report.discovery --grid sweep_grid512.jsonl --cost evals --out discovery_by_evals.md >> "%LOG%" 2>&1

echo [5/5] discovery join by ms... >> "%LOG%"
uv run python -m raymarching_benchmark.report.discovery --grid sweep_grid512.jsonl --cost ms    --out discovery_by_ms.md >> "%LOG%" 2>&1

echo === finished %DATE% %TIME% === >> "%LOG%"
endlocal
