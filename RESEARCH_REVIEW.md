# Empirically Benchmarking Ray-Marching Strategies
### An approach + early-findings note for external review

*Prepared for a graphics researcher to critique the **methodology and structure**
of the study — not the implementation. Where you see numbers, treat them as
preliminary signals from a small pilot grid, not results.*

---

## 1. The question

For rendering signed-distance-field (SDF) scenes by ray marching, there is a
family of sphere-tracing variants — standard sphere tracing, over-relaxation,
auto-relaxation, segment tracing, enhanced/planar prediction, overstep-and-
bisect, and a few of our own — each claimed to win in some regime. We want an
**automated harness that can run for hours over a large grid of
`scene × strategy × settings` and return a dataset that answers: which technique
and which settings win on which kind of scene, and at what cost.**

The intended end state is "let the GPU churn overnight, come back to a dataset
you can mine," not a one-off comparison.

## 2. Why this is harder than "render each and time it"

Two methodological problems dominate, and most of our design effort goes into
them rather than into the marching code itself.

**(a) "Best" requires correctness, not just speed.** A strategy can be fast
precisely because it is *wrong* — it skips a thin feature, tunnels through a
hole, or leaves ragged silhouettes. Ranking by frame time alone systematically
rewards the least accurate methods. So we need a notion of ground truth and a
way to score visual/geometric error, and only then trade speed against it.

**(b) "Fair" comparison is non-obvious.** Giving each method "the same budget"
is ambiguous because a *step* is not a unit of work:

- standard tracing does one distance evaluation per step;
- overstep-and-bisect peeks ahead (≈2 evaluations/step) plus a bisection burst;
- segment tracing evaluates a Lipschitz bound over an interval;
- our interval method samples endpoints and bisects.

We measured this directly: at the *same iteration count*, evaluations-per-step
range from 1.0 (standard) to ~2.0 (overstep) to ~2.9 (interval). So "equal
steps," "equal distance-evaluations," and "equal wall-clock" are three
different fairness lenses, and they disagree.

## 3. The structural approach

We deliberately separated the study into independent layers so each can be
critiqued and replaced on its own.

**Ground truth as a high-budget reference render.** For each scene+camera we
render a reference with the most conservative method (plain sphere tracing) at a
very large iteration budget and a very tight convergence threshold, and treat
that as truth. This avoids needing closed-form depth for every scene and keeps
the reference in the same coordinate/camera frame as the methods under test.
*(We flag the obvious risk in §6.)*

**Multi-channel capture + structural scoring.** Every render emits, per pixel:
depth, surface normal, a deterministically-lit color image, and a hit/miss mask.
Each method is scored against the reference on four complementary families:

- raw depth error (per-pixel distance error where both agree there's a surface);
- hit-mask agreement (false-hit rate, false-miss rate, intersection-over-union);
- normal angular error;
- structural similarity (SSIM) on the depth image, the normal map, and the lit
  image — i.e. perceptual/structural agreement, not just per-pixel averages.

The lit image is shaded with a fixed light, so differences in it come only from
geometry and normals, not from rendering choices.

**Fairness instrumentation.** We count the true number of distance-function
evaluations each march performs (not just steps), measure GPU runtime, and
record step count. Then we **sweep the iteration budget** so that each method
traces out an *accuracy-vs-cost curve* rather than producing a single point.
The analysis layer can then ask the question on any axis: "best accuracy at
equal steps," "…at equal evaluations," or "…achievable within T milliseconds."

**Durable, resumable dataset.** Each run is one record carrying its full config,
the three cost axes, the accuracy scores, and provenance (code revision,
GPU/driver, host, timestamp). Runs are keyed by a config hash so an interrupted
multi-hour sweep resumes without repeating work, and a single failed
configuration is recorded as an error rather than killing the batch.

**Two consumers of the dataset:** a visual gallery (compare each method's
rendered depth/normal/color against the reference, by eye) and an analysis page
(accuracy-vs-cost curves per scene, plus a "who wins under a given time budget"
table).

## 4. What exists today

A working end-to-end pipeline: 8 strategies and 14 scenes implemented on the
GPU; ground-truth capture; the four scoring families; evaluation-counting; a
budget-sweeping runner that writes the dataset; and the two report views. A
pilot grid of 4 representative scenes (smooth, grazing-angle, thin-feature,
fractal) × 8 strategies × 5 budgets — 160 runs — completes in ~30 s on an
RTX 3090.

## 5. Early signals (preliminary)

These are from the pilot grid and are meant to show the methodology has
discriminating power, not to claim conclusions.

- **Fairness changes the verdict.** One of our two-pass strategies was, by an
  accident of a hard-coded step cap, getting roughly a fifth of the budget the
  others got. On the two hardest scenes it looked badly broken (hit-overlap with
  ground truth ~0.2–0.4). Giving it the *same* budget as everyone else moved it
  to competitive (~0.94–0.97). The lesson reinforces the whole premise: budget
  parity has to be enforced and verified before any ranking is meaningful.

- **Step count hides cost.** Two methods can report the same iteration count yet
  differ ~3× in actual work and ~5× in runtime, because of extra evaluations per
  step. Without the evaluation counter this would be invisible and misleading.

- **Winners are scene- *and* budget-dependent.** On easy scenes almost
  everything reaches near-perfect agreement cheaply; the interesting cases are
  the fractal and grazing scenes, where the best strategy *changes as the time
  budget grows* — a fast-but-approximate method wins at tight budgets and a
  more conservative one overtakes it given more time. This is exactly the
  Pareto structure we want the dataset to expose.

- **Known failure modes show up quantitatively.** Over-relaxation measurably
  skips thin geometry (low hit-overlap on the thin torus); one interval-based
  method is currently statistically indistinguishable from the baseline, which
  we suspect means it never actually takes the aggressive steps its theory
  permits.

## 6. Where we'd value your guidance

These are the assumptions we're least sure about — the reason for this review.

1. **Is a high-budget same-renderer march an acceptable oracle?** It is
   self-consistent, but it shares the renderer's biases (same SDF, same camera,
   same convergence pathologies). On deliberately non-metric ("bad Lipschitz")
   scenes the reference itself may be subtly wrong. Should we anchor on analytic
   depth where it exists, use an independent high-quality renderer, or a denser
   supersampled reference? How would you validate the oracle?

2. **Are these the right accuracy measures, and how should they be weighted?**
   We chose SSIM on depth/normal/lit-image plus hit-mask and depth error. These
   methods tend to fail specifically at **silhouettes and thin features** — should
   we be measuring edge/silhouette error explicitly rather than whole-image
   structural similarity, which may dilute exactly the errors we care about?

3. **Which fairness axis should be primary?** We lean toward
   distance-evaluations as the hardware-agnostic denominator, with wall-clock as
   the practical one. Is that the right call, or is there a more principled
   "equal work" notion (e.g. matched residual / matched convergence threshold
   rather than matched iterations)?

4. **Convergence-criteria parity.** All methods share a hit threshold but differ
   in stopping behavior. Is comparing them at a matched iteration budget fair, or
   should accuracy be evaluated at a matched *residual* (everyone marched until
   the same closeness-to-surface), decoupling "how far did it get" from "how many
   steps did that take"?

5. **Scene & viewpoint representativeness.** Each scene currently uses a single
   hand-framed camera. To avoid cherry-picked framing, should we sample many
   viewpoints per scene (and would view-to-view coherence, i.e. animation, change
   the conclusions)? Are there canonical stress scenes we're missing?

6. **Timing methodology.** We use full-screen GPU wall-clock with a hard sync and
   report a median over repeats, plus a warp-divergence proxy (variance of
   iteration counts in pixel neighborhoods). Is that adequate for honest GPU
   comparison, or do we need timer queries / occupancy analysis to avoid
   driver-level confounds?

7. **Parameter space.** Tuning parameters (relaxation factor, segment growth,
   fattening margin) are currently fixed; the next step is to sweep them as
   additional axes. Any priors on sensible ranges, or on which parameter×scene
   interactions are worth the combinatorial cost?

8. **Framing of the goal.** Is the most valuable output "a ranking table per
   scene class," or should we be aiming the dataset at *discovering* something —
   a scene-adaptive hybrid or an auto-tuner that picks/blends strategies from
   cheap scene features? That choice affects what we should be measuring now.

---

*Appendix — strategies under test:* standard sphere tracing, fixed
over-relaxation, auto-relaxation (slope/heuristic estimated relaxation),
enhanced (planar-prediction), segment tracing (interval Lipschitz bound),
overstep-and-bisect, a coarse-to-fine "skipping spheres" method, and an
interval-arithmetic approximation. *Scene classes:* smooth, sharp-edged, thin,
CSG, fractal (Menger/Mandelbulb), and deliberately non-metric/grazing stress
cases.
