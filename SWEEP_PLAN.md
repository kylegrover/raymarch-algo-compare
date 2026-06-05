# Sweep Engine Plan — v2 (post external review)

> v1 built the capture → score → sweep → report pipeline (done, see "Status").
> v2 integrates two independent expert reviews. Their critiques converged on the
> same load-bearing issues — the **oracle**, the **fairness axis**, and the
> **silhouette/accuracy measure** — and those drive most of the changes below.

## Goal (unchanged)

Run a large grid of `scene × viewpoint × strategy × parameters × budget` on the
GPU for hours, score every run against a trustworthy ground truth, and append to
a durable dataset that answers **which technique + settings win on which kind of
scene, at what cost** — and ultimately *why*, in terms of cheap scene features.

---

## Status — what v1 already delivers

- GPU multi-target capture (depth / normal / lit color / hit) + per-invocation
  **SDF-evaluation counting**.
- Ground-truth + scoring (depth error, hit IoU/false-hit-miss, normal angle,
  SSIM on depth/normal/color).
- Resumable, provenance-stamped **JSONL sweep** over scene × strategy × budget.
- Two report views (image gallery; accuracy-vs-cost curves + runtime-matched
  winners).
- Fairness fix: every strategy honors the shared iteration budget.

v1's verdicts are **not yet trustworthy** — the reviews explain why, below.

---

## Reviewer-driven course corrections

### A. The oracle (the #1 threat)

Both reviewers independently flagged that a high-budget *same-renderer* reference
is circular **in a direction correlated with our hard cases**: plain sphere
tracing undershoots at grazing/thin features and can confidently converge to the
*wrong* surface on non-metric (L>1) SDFs — exactly the scenes whose verdicts we
care about. Plan:

1. **Interim oracle = understepped sphere trace.** Standard trace with the step
   multiplied by a fudge factor (~0.6) and a very high max-step count. Smaller
   steps actually *reach* grazing/thin surfaces instead of stalling short, and
   the 0.6 margin tolerates Lipschitz violations up to ~1/0.6 ≈ 1.67× without
   overshooting. Cheap, strictly better than v1's reference.
2. **Analytic calibration (nearly free).** For scenes with closed-form ray-surface
   depth (sphere, plane, cube, thin torus, CSG of primitives), render analytic
   depth and report the **oracle-minus-analytic residual**. This converts the
   oracle's bias from an unverifiable assumption into a measured error bar — the
   thing both reviewers said they need before trusting fractal verdicts.
3. **Interval-arithmetic root-finder (later, gold standard).** Evaluate the SDF
   over a bounding interval/box along the ray; `d_min>0` ⇒ provably empty, step
   on; `d_min≤0≤d_max` ⇒ bisect. Immune to Lipschitz violations and thin-feature
   skipping; guarantees the first intersection. Slow but offline. Build after the
   interim oracle; **compare the three references** on the scenes where all are
   available to bound interim-oracle error on the rest.

> Note: supersampling does *not* fix oracle bias (same algorithm, same biased
> limit, higher resolution) — use it only for anti-aliasing, not trust.

### B. Matched-residual as the primary experimental driver

"Equal iterations" is unfair (a step buys 1.0–2.9 evaluations); "equal
evaluations" still entangles *how efficiently* a method converges with *how far*
it got. Both reviewers: **march every method to the same closeness-to-surface ε,
sweep ε, and report what reaching it cost.** Accuracy is (approximately) held
constant by ε; cost becomes the dependent variable — "what does equal accuracy
cost," the question a practitioner actually has.

- Replace the iteration-budget sweep with a **target-residual sweep** (ε), with a
  hard max-budget cap for methods that can't reach ε on pathological scenes
  (record "did not converge").
- Keep iteration count as **diagnostic**, not the comparison axis.

### C. Tripartite cost reporting (with proper GPU timing)

Report all three, never collapse to one:

1. **Workload** — SDF evaluations / ray (hardware-agnostic). Caveat: an
   evaluation is *not* constant-cost for segment/interval methods (interval/
   Lipschitz-bound evals cost more per call), so eval-count under-charges them —
   which is why we always report wall-clock alongside.
2. **Hardware cost** — wall-clock via **GPU timer queries** (`GL_TIME_ELAPSED`
   around the dispatch), replacing CPU wall-clock + `finish()` (which folds in
   dispatch/readback/scheduler jitter that swamps small budgets). moderngl/OpenGL
   exposes this; **no engine switch needed.**
3. **Efficiency** — warp divergence (we proxy it via neighborhood iteration
   variance) and, where the toolchain exposes them, occupancy / register
   pressure. Divergence is a *first-class result* here: adaptive methods that win
   on eval-count can lose on-GPU because neighboring pixels take wildly different
   paths.

### D. Accuracy metrics — IoU-primary + FLIP, SSIM demoted

Whole-image SSIM is dominated by the large smooth regions everyone gets right and
dilutes the silhouette/thin-feature errors that matter. Decision:

- **Hit-mask IoU = primary objective geometric metric.** A skipped thin feature
  tanks IoU with no arbitrary weighting — un-confounded.
- **FLIP (NVIDIA, peer-reviewed perceptual) = primary perceptual metric.** It
  inherently penalizes high-contrast edge failures (ragged silhouettes) more than
  flat-surface shading variance — rigorous, not a hand-rolled weighting.
- **SSIM (depth/normal/color) demoted to secondary descriptive.**
- Objective silhouette-band geometric errors (errors restricted to the reference's
  depth-discontinuity band) kept as an *optional* later add — objective because
  the mask comes from ground truth, but deferred to avoid metric sprawl.

### E. Curated multi-viewpoint (no animation)

A single hand-framed camera is a cherry-picking risk, and grazing behavior is
extremely viewpoint-sensitive. Use **hand-curated canonical viewpoints per
scene**, explicitly categorized:

- Orthogonal / flat (easy), Grazing (high step counts),
  Macro / close-up (thin-feature resolution), Interior / claustrophobic (high
  bounding-volume overlap).

A high-quality reference is rendered **per viewpoint**. Animation/temporal
coherence is deferred (separate confound, only relevant if benchmarking TAA).

### F. Full parameter grid → derive best-shot *and* sensitivity

Compute is not the bottleneck (≈0.2 s/run ⇒ a ~10⁵–10⁶ grid is an overnight/
weekend run on the 3090). Decision: **brute-force the parameter grid** and from
the one dataset extract (a) each method's best-tuned params per scene — its "fair
shot," (b) its parameter sensitivity/robustness, and (c) feature correlations for
the discovery goal. Prerequisite: **thread tuning params as shader uniforms**
(un-hardcode ω, segment growth, fattening margin in the runner/GLSL).

### G. Instrument for *discovery* now

The durable contribution is characterizing **why** a method wins, in terms of
cheap scene/ray features, so the result survives the next new method. Compute and
**store per-run features now** (while iterating, to avoid re-running the grid):
local Lipschitz estimate, grazing-angle fraction, thin-feature density,
hit-complexity / silhouette length, iteration-count variance. Analysis target:
"does a small feature set predict the Pareto-optimal method?" → selection
criteria, then (second paper) an auto-tuning / scene-adaptive hybrid.

### H. Scale hygiene

At grid scale the bottlenecks are disk I/O and CPU-side scoring, not GPU:

- **Split cheap cost-runs (timer + eval count only) from scored runs (capture +
  IoU/FLIP).** Don't pay image scoring on every cell.
- **Persist images only for the Pareto frontier**, not every run.
- Batch JSONL writes.

### I. Correctness: fix mislabeled strategies (Reviewer 1's catch — confirmed)

Our `relaxed` and `heuristic_auto_relaxed` use **naive** over-relaxation:
`t += d·ω` with *post-hoc* overshoot detection (`if d<0 back up`). That check
cannot fire when a step tunnels cleanly through a thin shell, which is why they
"skip thin geometry." This is **not** Keinert et al. 2014 safe over-relaxation,
whose defining feature is a *predictive* disjoint-sphere fallback that refuses the
over-step before taking it. Actions:

- Implement **safe over-relaxation (Keinert)** with the predictive overlap test.
- Relabel the current ones honestly (e.g. "Naive-Over-Relaxed") so strategy names
  mean what the field expects.
- Re-examine the "over-relaxation skips thin geometry" finding under the safe
  variant.

---

## Re-examine v1's "findings" after the fixes

Reviewer 1's disambiguations — treat current v1 signals as *suspect* until
re-run against the improved oracle + matched-residual + safe variants:

- **"Interval method ≡ baseline":** indistinguishable on *which* axis? If on
  accuracy *and* eval-count → it never takes aggressive steps (a bug). If it
  matches accuracy at *lower* cost → a quiet win. Separate before calling it null.
- **"Best strategy changes with budget":** real Pareto crossover, or the
  conservative method converging to the same biased reference the oracle used?
  Disambiguate on analytic scenes.
- **"Over-relaxation skips thin geometry":** per §I, currently a property of our
  *naive* implementation, not of safe AR-ST.

---

## Revised phased roadmap

**Phase 3 — Trustworthy oracle + metrics — DONE**
3.1 ✅ Understepped (×0.6) reference kept; superseded as the *primary* oracle by
    a **dense march** (id 9): understep ×0.5 + hard `minStep` floor + sign-change
    bisection. Both are captured per frame (`reference` + `reference_dense`).
3.2 ✅ Closed-form ray intersections (sphere/plane/cube/torus) in `gpu/analytic.py`;
    `gpu/oracle_calibration.py` reports the residual vs analytic. **Result:** dense
    march = IoU 1.0000 / ~1e-7 depth on sphere/cube/torus; grazing-plane core IoU
    0.9905 (the 1.9% gap is the shared `maxDistance` horizon clip, invariant to
    minStep). The old understep oracle silently missed **11.6%** of the grazing
    plane — now quantified. `minStep=0.002` chosen (cost flat across the sweep).
3.3 ✅ IoU promoted to PRIMARY ranking metric (sweep_report + scoring hierarchy);
    SSIM demoted to descriptive. FLIP deferred.
3.4 ✅ Safe over-relaxation (Keinert, id 8) added; naive ids 2/5 relabeled
    "Naive-*". Re-exam confirms: Thin-Torus Naive-Relaxed IoU **0.667** vs
    Safe-Relaxed **1.000** — the "skips thin geometry" finding is a naive-only
    artifact. (Surfaced for Phase 4: Segment collapses on Thin Torus IoU 0.158 —
    *not* "≡ baseline"; needs its own re-exam, gate #3.)

**Phase 4 — Fair cost + matched residual — DONE**
4.1 ✅ GPU timer queries (`ctx.query(time=True)`, GL_TIME_ELAPSED) replace CPU
    wall-clock in `runner.render` — dispatch-only, ~0.28 ms @ budget 64 with
    median≈min. Eval count kept; added an `iter_divergence` proxy (neighbor
    iteration spread) as the efficiency/warp-divergence axis.
4.2 ✅ Matched-residual driver: `sweep.py --mode residual` sweeps ε (=hit_threshold)
    under a fixed cap, recording cost (evals/ms/iters) + `did_not_converge`.
    **Payoff:** reveals methods that no ε can rescue — Naive-Relaxed plateaus at
    Thin-Torus IoU 0.668; **Segment *degrades* as ε→0** (0.56→0.14, dnc=0 → it's
    confidently wrong, not out of budget). Refutes "interval ≡ baseline" (gate #3).
4.3 ✅ Eval-cost caveat made empirical + surfaced in the report header: on Grazing
    Plane @ ε=1e-4 Naive-Relaxed uses fewer evals than Standard (19 vs 23) but
    1.7× the GPU ms — eval-count under-charges adaptive tracers; `iter_divergence`
    explains the eval-vs-ms rank flips. Three axes reported, never collapsed.

**Phase 5 — Parameterization, viewpoints, scale — DONE**
5.1 ✅ ω / κ / fattening `margin` are uniforms threaded via `params=`; runner sets
    defaults so none default to 0. Per-strategy `param_grid.py` registry (name→grid,
    first = default) drives the brute-force grid.
5.2 ✅ `viewpoints.py`: curated categorized viewpoints (orthogonal/grazing/macro)
    per core scene with a per-view dense-march reference; default fallback for the
    rest. Sweep iterates scene×viewpoint; default scene set widened to include Cube.
5.3 ✅ Full grid via `sweep.py --grid` (scene×viewpoint×strategy×level×param-combo);
    cheap-vs-scored split (`--full-score`; IoU+depth always, SSIM opt-in); batched
    JSONL writes (`--flush-every`, `JsonlDataset.extend`). Per-run images: N/A — the
    sweep is JSONL-only, so "Pareto-only persistence" is already satisfied.

**First full grid run — DONE (N=3864, see FINDINGS.md)**
3864 scored runs (14 scene-views × 9 strategies × params × budget+residual), 0
errors, scored vs the dense-march oracle; `report/analyze.py` distils it. Headlines:
plain Standard is the hard-to-beat baseline (never structurally limited); Safe-
Relaxed is the only *advanced* method that's both robust (ω-insensitive) and
competitive; Naive over-relaxation is fragile (tunnels thin/sharp); **Segment is
broken off the plane** (Thin-Torus IoU 0.003); eval-count ≠ ms (Enhanced); the
**Mandelbulb is where the oracle's own trust runs out** (no method > ~0.98).

**Phase 6 — Discovery** (priorities sharpened by the grid)
6.3 Interval-arithmetic gold-standard reference — **DONE (metric scenes)**. Sound
    first-hit oracle by interval root isolation (`gpu/interval.py`,
    `gpu/interval_oracle.py`): `lo>0` *proves* a segment empty ⇒ no tunneling.
    Validated @384 (`gpu/interval_validation.py`): interval-vs-analytic IoU 1.0 on
    sphere/cube/torus (~5e-6 depth), and **dense-vs-interval IoU 1.0000 on all
    four** → the grid's dense-march oracle is sound on metric scenes to ~5e-6.
    Mandelbulb/Menger **deferred** (need interval *escape-time*, not IA on the
    distance estimator); fractal verdicts stay provisional.
6.4 Faithful method ceilings — **DONE (offline)**, refuting the strawman verdicts:
    • Segment: a *sound* Galin tracer (directional-Lipschitz K via interval
      autodiff, `gpu/interval_autodiff.py` + `gpu/faithful_offline.py`) reaches
      Thin-Torus **core IoU 1.0000** (strawman 0.003) and grazing-plane hits in
      ~3 steps — "broken" was an implementation artifact, not the technique.
    • RevAA: real revised affine arithmetic (`gpu/affine.py`) is **sound** (IoU 1.0,
      not "≡ Standard") and tighter than IA where correlation survives (Sphere
      1.17× fewer evals); no win on min/max-dominated SDFs. Honest, not tautological.
    These are offline accuracy/cost ceilings, **not** GPU-timed competitors;
    FINDINGS.md is left as-is until a faithful re-run/rewrite is decided.
6.1 Per scene-view feature extraction — **DONE** (`report/features.py`,
    `features.jsonl`, 23 scene-views). Strategy-*independent* features so they join
    to every grid row sharing (scene, viewpoint) — **no GPU re-run**: hit_rate,
    grazing_frac, silhouette_cplx, hardness_mean/cv (dense-march evals), plus CPU-
    sampled lipschitz_p99 and thin_slab. Sanity: lipschitz_p99 = 1.000 on every
    metric SDF, 2.000 on Bad-Lipschitz (the deliberate 2×), 7–13 on Mandelbulb;
    thin_slab orders Sphere 0.31 > Thin-Torus 0.04 > Mandelbulb 0.013.
6.2 Feature → Pareto-optimal-method — **DONE** (`report/discovery.py`). Joins 6.1
    to the grid; regime-aware winner (cost = cheapest reaching IoU≥0.99; accuracy =
    most-accurate where none do). **Two load-bearing results:**
    • *A small feature set predicts the winner.* By GPU ms: grazing_frac→**Segment**,
      high lipschitz_p99 (non-metric/fractal)→**Naive over-relaxation**, else→
      **Standard**. Top discriminators: grazing_frac, hardness_cv, lipschitz_p99.
    • *The answer depends on the cost axis.* Ranked by **SDF evals** Standard wins
      **0** views (Naive-Relaxed 6, Enhanced 5); ranked by **GPU ms** Standard wins
      **7** — eval-count under-charges the adaptive tracers' warp divergence (the
      Phase-4 caveat, here it *reorders the whole conclusion*). Regenerate:
      `report.discovery --cost {evals,ms}` → `discovery_by_{evals,ms}.md`.
    Caveat: only the 5 grid scenes × viewpoints are joinable; 9 more scene-views
    have features but no grid rows (Cylinder/CSG/Onion/Menger/Pillars/…). Widening
    the grid to them is the cheap, high-leverage next step.

6.5 **Overnight grid expansion — PLANNED.** Widen from 5 → **10 scenes** to give
    6.2 real geometric/winner diversity. Only scenes whose **Python SDF ≡ GLSL SDF**
    are admitted — features come from the Python SDF but the grid scores the GLSL
    render, so a mismatch silently invalidates the join. Verified-consistent set
    (10): Sphere, Grazing Plane, Cube, Thin Torus, Mandelbulb, **Cylinder, Near
    Miss, Hollow Cube (CSG), Onion Shell, Thin Planes Stack**. The 5 new scenes
    each got 3 curated viewpoints (`viewpoints.py`); all 15 verified to frame
    geometry (hit-rate > 0.2) and to add regimes — Hollow-Cube thin CSG walls
    (thin≈0.04), Thin-Planes grazing+thin (grazing 0.42, thin 0.013).

    Run (two invocations append to one resumable file; settings match the existing
    N=3864 so its rows are reused, not re-run):
    ```
    S="Sphere,Grazing Plane,Cube,Thin Torus,Mandelbulb,Cylinder,Near Miss,Hollow Cube (CSG),Onion Shell,Thin Planes Stack"
    uv run python -m raymarching_benchmark.sweep --mode budget   --grid --full-score --res 384 --out sweep_grid.jsonl --scenes "$S"
    uv run python -m raymarching_benchmark.sweep --mode residual --grid --full-score --res 384 --out sweep_grid.jsonl --scenes "$S"
    ```
    Scale: ~29 scene-views × 276 rows = **~8k rows** (~4k new), ≈1–2 h on the 3090
    at full-score. Then regenerate the join:
    ```
    uv run python -m raymarching_benchmark.report.features  --res 256 --out features.jsonl
    uv run python -m raymarching_benchmark.report.discovery --cost evals --out discovery_by_evals.md
    uv run python -m raymarching_benchmark.report.discovery --cost ms    --out discovery_by_ms.md
    ```

    **Deferred — needs Python↔GLSL SDF reconciliation before grid admission** (each
    is currently a *different scene* in the two backends):
    • Smooth Blend — different smooth-min (Python `h²k/4` vs GLSL `mix`-form).
    • Menger — fold phase offset (Python `mod(p·s+1,2)` vs GLSL `mod(p·s,2)`).
    • Bad Lipschitz — Python `×2` (overshoot test) vs GLSL `×0.1` (underestimate);
      a *semantic* choice (which failure mode to test) — decide direction, sync both.
    • Pillar Forest — radius/height differ + Python adds a floor plane, GLSL doesn't.
    Reconcile (sync Python→GLSL, the rendered authority) then add to the run.

---

## Validation gates (must answer before trusting any number)

1. ✅ **ANSWERED.** Dense-march oracle residual vs closed form: ~1e-7 depth /
   IoU 1.0000 on sphere/cube/torus; grazing-plane core IoU 0.9905 (rest = shared
   far-clip). Old understep oracle missed 11.6% of the grazing plane. The sweep
   now scores against the dense march. (`gpu/oracle_calibration.py`.)
2. ✅ **ANSWERED.** ids 2/5 = naive (post-hoc backup), relabeled "Naive-*"; id 8
   = Keinert safe. Thin-Torus IoU: naive 0.667 vs safe 1.000.
3. ✅ **ANSWERED.** Not a null: matched-residual shows Segment *degrades* as ε→0
   on Thin Torus (IoU 0.56→0.14, dnc=0 → confidently wrong, a bug in its
   extension/bracket on thin features), and it's far from baseline on Sphere too.
   Definitely not "≡ baseline at lower cost."
4. ◑ **Largely addressed by the grid.** Under the trusted oracle + matched
   residual, the well-behaved methods converge to the *same* IoU — the apparent
   "best method changes with budget" crossovers live almost entirely among the
   *broken* methods (Segment, Naive-relaxed) where low budget masks their failure.
   No evidence of an oracle-bias artifact on the analytic scenes. Remaining doubt
   is fractal-only (gate-1 territory → interval reference, Phase 6.3).

---

## Scale sanity (for the overnight grid)

≈0.2 s/run pilot rate ⇒ e.g. 14 scenes × 5 views × 8 strategies × 10 residual
steps × 100 param combos ≈ 5.6×10⁵ runs ≈ ~31 h — a feasible weekend run on one
3090. Therefore the engineering effort goes into I/O batching, Pareto-only image
persistence, and the cheap-vs-scored run split, **not** into shrinking the grid.
