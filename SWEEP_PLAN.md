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

6.5 **Overnight grid expansion — ✅ DONE (2026-06-06).** Ran via the Windows
    Task Scheduler task `RaymarchOvernightSweep` → `run_overnight_sweep.bat` (the
    agent shell can't acquire a GPU/WGL context — must launch from the user's
    interactive session). 02:00→03:09, **12,236 rows, 0 errors**, fresh
    `sweep_grid512.jsonl`. Canonical tables regenerated into `FINDINGS.md`
    (`report.analyze --in sweep_grid512.jsonl`); regime map in
    `discovery_by_evals.md` / `discovery_by_ms.md` (gitignored, regen via
    `report.discovery`). **Result — the headline reversal:**

    - **Standard does NOT always win.** By GPU **ms**: Safe-Relaxed 13, Standard
      13, Naive-Relaxed 9, Segment 2, Naive-Auto 1 (of 38 views). By **evals**:
      Naive-Relaxed 15, Enhanced 14, Standard only **3**. The two cost axes
      genuinely disagree — Enhanced wins the *workload* axis broadly but its
      per-iteration ALU cost means it rarely wins *wall-clock*.
    - **Wins track forced-small-steps, not eval cost** (the hypothesis, now
      quantified). Top winner-separating feature is **`grazing_frac`** (norm.
      spread 4.26 on ms), then **`hardness_cv`** (2.88). High → adaptive wins;
      low → Standard. Anchored ms multipliers vs Standard: Grazing-Plane grazing
      **Segment 2.4×** (certifiable); Mandelbulb·ortho **Naive 7.8×**; Thin-Planes
      grazing **Safe 4.2×**; Sphere-Cloud·ortho **Safe 5.8×**. Same Grazing Plane,
      *steep* (non-grazing) view → **Standard wins** — the flip is in one scene.
    - **Where Standard still rightly wins:** smooth, low-grazing metric scenes
      (Thin Torus all views, Hollow Cube CSG all views, Cube face/grazing, Sphere
      ortho) — there the adaptive methods are mostly *capped* (can't reach the bar).
    - **Caveats:** Segment is `cap` everywhere except the grazing plane (broken off
      that regime — consistent with the faithful-Segment fix); Naive-Relaxed leads
      many ms rows but is the *naive* over-relaxation (fast-but-can-miss) — its
      wins need a visual gut-check (perceptual A/B harness) before full trust.

    --- *(original plan, executed as written below)* ---

    Widen from 5 → **13 scenes**
    for real geometric/winner diversity. Only scenes whose **Python SDF ≡ GLSL SDF**
    are admitted — features come from the Python SDF but the grid scores the GLSL
    render, so a mismatch silently invalidates the join (each new scene verified to
    ~1e-7 by checking |Python SDF| at GLSL surface points). Verified-consistent set
    (13): Sphere, Grazing Plane, Cube, Thin Torus, Mandelbulb, Cylinder, Near Miss,
    Hollow Cube (CSG), Onion Shell, Thin Planes Stack, **Menger Sponge** (fold-phase
    reconciled), **Sphere Cloud** + **Bumpy Sphere** (new expensive-metric scenes,
    24/31 primitives, costly eval + sound oracle). All curated 3 viewpoints each.

    **Why this is the deeper run (resolved §6.5 review): what actually gives the
    adaptive methods a fair shot is NOT eval cost or more params — it is whether
    sphere-tracing is forced into *many tiny steps*.** Measured: smooth metric
    scenes (incl. the new expensive ones) converge in ~10–13 SDF evals/ray, so
    Standard wins — expensive eval just scales everyone's ms equally. Adaptive
    methods win where steps are forced small: **grazing** (Grazing Plane: Segment
    0.23 ms vs Standard 0.60 ms, certifiable) and **intricate/non-metric** surfaces
    (Mandelbulb: Enhanced 2.1×, Naive 9.8× at the lower bar). So the run leans into
    grazing views + the fractal/intricate scenes, and two changes vs the first grid:
      • **res 384 → 512** — the first grid sat in the GPU-overhead-dominated regime
        (sub-ms frames); 512 moves the expensive scenes into the multi-ms regime
        where eval savings translate to ms (Enhanced's Mandelbulb win is res-robust
        but only clears the noise floor above res ~384).
      • **budgets extended to 1024, 2048** — Mandelbulb/Menger don't converge by 512,
        which wrongly shunted them to the accuracy regime in the first pass.
    NOTE: res/budget changes mean the first N=3864 rows are NOT reused (config_hash
    includes resolution) — this is a full fresh grid.

    Run (two invocations append to one resumable file):
    ```
    S="Sphere,Grazing Plane,Cube,Thin Torus,Mandelbulb,Cylinder,Near Miss,Hollow Cube (CSG),Onion Shell,Thin Planes Stack,Menger Sponge (iter=3),Sphere Cloud,Bumpy Sphere"
    uv run python -m raymarching_benchmark.sweep --mode budget   --grid --full-score --res 512 --budgets 32,64,128,256,512,1024,2048 --out sweep_grid512.jsonl --scenes "$S"
    uv run python -m raymarching_benchmark.sweep --mode residual --grid --full-score --res 512 --out sweep_grid512.jsonl --scenes "$S"
    ```
    Scale: ~39 scene-views × ~330 rows ≈ **~13k rows**, several hours on the 3090 at
    res 512 + full-score (a genuine overnight run). Then regenerate the join (the
    per-view bar + cost-to-quality section surface where adaptive methods win):
    ```
    uv run python -m raymarching_benchmark.report.features  --res 256 --out features.jsonl
    uv run python -m raymarching_benchmark.report.discovery --grid sweep_grid512.jsonl --cost evals --out discovery_by_evals.md
    uv run python -m raymarching_benchmark.report.discovery --grid sweep_grid512.jsonl --cost ms    --out discovery_by_ms.md
    ```

    **Still deferred — Python↔GLSL mismatch, reconcile before admitting:**
    • Bad Lipschitz — Python `×2` (overshoot) vs GLSL `×0.1` (underestimate); a
      *semantic* choice (which failure mode to test) — decide direction, sync both.
    • Pillar Forest — radius/height differ + Python adds a floor plane, GLSL doesn't.
      (Superseded in practice by **Box Lattice**, the certifiable-metric near-miss
      scene below — Pillar's non-metric floor confound is no longer needed.)

    **+5 scenes pulled from the shadertoy scraps (2026-06-06), all metric / oracle-
    sound, all Python↔GLSL parity-checked by `tests/test_scene_parity.py`:**
    • **Smooth Blend — NO LONGER DEFERRED.** Root cause was the smooth-min form;
      `op_smooth_union` now uses the polynomial `mix`-form *identical* to GLSL
      `opSmoothUnion`. Parity ~0 by construction.
    • **Gyroid** (id 16) — gyroid labyrinth ∩ ball. The raw implicit is non-metric
      (|∇g| ≤ freq·2√3); divided by that bound → conservative 1-Lipschitz under-
      estimator (sound). New regime: smooth periodic curved surface, grazing-rich.
    • **Capped Torus** (id 17) — iq open "C" torus: thin feature + open boundary edge.
    • **Box Lattice** (id 18) — finite 5×5×5 box grid via limited domain repetition;
      fully metric (folded-translate is distance-preserving). round = `floor(x+0.5)`,
      NOT `round()` (GLSL ties are implementation-defined — parity hazard).
    • **Metaballs** (id 19) — 6 spheres fused with the now-canonical polynomial smin.

    Added to `run_overnight_sweep.bat` (18 scenes total); same output file ⇒ the
    sweep resumes and computes only the new scene-views. GPU parity is exercised on
    that next user-run (this agent session can't acquire a GPU context).

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

---

## Phase 7 — future work / open ideas (post-6.5)

Ordered roughly by value-for-effort. Each is independent.

1. **Visual gut-check of the ms-winners (highest priority).** The §6.5 ms table
   crowns Naive-Relaxed/Segment on many views, but Naive over-relaxation can
   *tunnel* thin/sharp features while still scoring IoU ≥ 0.98. Run the perceptual
   A/B harness (`report.perceptual_review` → `perceptual_review.html`) on the
   grazing/intricate winners at equal wall-clock and have a human confirm they're
   actually clean, not just numerically close. This is the credibility gate before
   publishing any "adaptive beats Standard by N×" claim.

2. **Reconcile the 2 remaining deferred scenes → widen the grid.** ✅ Smooth Blend
   resolved (smin aligned, 2026-06-06) and 4 new scraps-scenes added. Remaining:
   Bad Lipschitz (Python `×2` overshoot vs GLSL `×0.1` underestimate — a *semantic*
   choice: pick which failure mode to test), Pillar Forest (radius/height +
   Python-only floor plane; arguably superseded by the new metric Box Lattice).
   They already have 6.1 features but no grid rows (see the "no grid rows" note in
   the discovery reports). Fix Python≡GLSL (verify |SDF| ~1e-7 at surface), add curated
   viewpoints, re-run §6.5. Bad-Lipschitz especially should stress the methods that
   assume a true distance.

3. **Bumpy-floor-at-grazing scene (strongest missing certifiable demo).** The
   cleanest adaptive win so far (Segment on Grazing Plane) is a flat plane. A
   *bumpy* floor viewed at a grazing angle = grazing + intricate + metric (sound
   oracle) all at once — should be the most dramatic, *certifiable* adaptive win in
   the set, and harder to dismiss than the fractal (where oracle trust runs out).

4. **Build the feature→winner predictor.** §3/§4 of discovery show `grazing_frac`
   + `hardness_cv` separate the winners. Promote that from a correlation table to
   an actual classifier (even a 2-feature decision boundary) that, given a scene's
   strategy-independent features, predicts the ms-winner — the real deliverable of
   the "which algorithm, which scene, why" question. Validate on held-out views.

5. **Investigate Enhanced's per-iteration ms overhead.** Enhanced wins the *evals*
   axis 14× but rarely *ms* — its extra ALU + warp divergence (FINDINGS §5) eats
   the eval savings. If that overhead can be cut (simpler step rule / less
   divergence), Enhanced's broad eval-win could convert to ms-wins. Profile it.

6. **Decide Segment's scope.** It's `cap` everywhere except the grazing plane.
   Either extend its candidate-segment bracket to handle curvature/thin features
   (the long-standing bug), or formally scope it as a grazing-plane-only method and
   stop charging it on scenes it can't do.

7. **Interval (sound) gold oracle for the fractal.** Mandelbulb wins are real but
   sit at the dense-march oracle's trust limit (~0.98, unsound on L>1). The
   interval-arithmetic reference (Phase 6.3 work, deferred for Mandelbulb) would
   let us trust the fractal numbers to the same standard as the metric scenes.
