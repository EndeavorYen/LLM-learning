# State space × inference factorial (Lab 13, exploratory, issue #22)

Script: `scripts/stage2_lab13_space_steps_factorial.py` · data: `benchmarks/analogy_v5.json` (same as the v5 test)
· Colab L4, torch 2.11 · seeds 42, 1, 2 · logs `seed*.log`, summary `aggregate.txt`.
All checks passed in every seed:
- F64 HOP and PREMISE-MLP reproduce Lab 12 exactly on CUDA.
- X1-ONLY (s1 only) is exactly 50%.
- The zero-noise / zero-perturbation levels equal the clean results.
- The hop trainer for the new spaces gives bit-identical weights to Lab 12's trainer on F64.

## Question
On v5, HOP (85.0% BR) trailed PREMISE-MLP (88.9%). The gap mixes three factors:
- **(a) state space:** the supervised 64-d funnel vs raw 896-d Qwen features.
- **(b) inference:** multi-step rollout vs a single decision.
- **(c) input form:** Qwen reading the two sentences jointly vs per-sentence vectors.

The factorial crosses (a) and (b) with per-sentence vectors everywhere.

State spaces, all from the same frozen Qwen features:
- **F64:** the supervised funnel.
- **R896:** raw features, centred and normalised.
- **P64:** unsupervised PCA to 64 dimensions.

## Results (mean of 3 seeds, 48 unseen items × stages 2–4; DEC = picker-free branch decision)
| state space | HOP (multi-step) | ONE-STEP | HOP − ONE-STEP |
|---|---|---|---|
| F64 (supervised funnel) | 91.2% | 91.7% | −0.5 |
| P64 (PCA, unsupervised) | 95.8% | 93.1% | +2.8 |
| R896 (raw, centred) | 95.8% (wide 13.7M-param hop: 95.8%) | **100.0%** (432/432) | −4.2 |
| PREMISE-MLP (joint text, reference) | | 97.9% | |

With a common sentence picker (BR) the same pattern holds: F64 85.7 / 85.7, P64 89.1 / 87.7, R896 88.9 / 93.1, PREMISE-MLP 91.4.

## Findings
1. **The supervised funnel is the weakest state space for transfer.** Training the same hop on raw centred
   Qwen features instead of the funnel's 64-d states raises it from 91.2% to 95.8% DEC; a one-step classifier
   rises from 91.7% to 100%. Unsupervised PCA to 64-d does as well as raw 896-d for the hop. The funnel
   separates the seen-domain anchors best (largest off-diagonal anchor cosine 0.40 vs 0.88–0.91), but that
   separation does not carry over to new domains: the bottleneck fits the seen domains at the cost of transfer.
   The earlier evidence for the funnel (v3: funnel 89% vs raw Qwen 56–58%) compared a trained funnel with an
   *untrained* nearest-anchor rule on raw features; with equal training on both sides the funnel loses.
2. **Factor (c) is not what favoured PREMISE-MLP.** Per-sentence raw vectors plus a one-step classifier
   (100%) beat Qwen reading the joint text (97.9%). The v5 gap was mainly the funnel, factor (a).
3. **Multi-step vs one-step makes little difference here**: −0.5, +2.8, −4.2 points, with no consistent sign. This is
   expected, since the branch is fixed by s0 in this task. Multi-step inference in 896-d does not beat the
   one-step classifier (Q1: no, on this task).
4. **No evidence that 64-d is more stable (Q2).**
   - **T1, input noise.** Raw 896-d is at least as accurate as F64 at every noise level (at α = 2: R896 HOP 78.7%,
     wide 81.2%, ONE-STEP 86.3% vs F64 HOP 78.2%). F64 loses fewer points (−13.0 vs −17.2), but it starts
     lower. PCA-64 is the least robust (−24.2): reducing dimension per se does not buy robustness.
   - **T2, perturbing the state after step 2** by up to one anchor spacing changes nothing in any space (≤ 1 point),
     isotropic or within the anchor span. The causal hop still sees s0 and s1 in its context and re-reads them,
     so this test cannot measure state drift in this architecture.
   - **T3, rolling out to step 12** (beyond the 5 trained positions, x0 leaves the window at step 5): every space
     drifts. At step 12 the share of items still on the step-2 law is F64 0.47, P64 0.34, R896 0.34, R896-wide
     0.53. None of the hops learned to hold a state beyond the trained length, so this is inconclusive.

## Implications for the next test
- Include raw centred Qwen features (R896) as a state space and as the one-step baseline. They are a strong
  opponent, and the funnel must beat them to justify itself.
- To test stability, the hop's state must be its **only memory** (a recurrent update: state + new input → state),
  not a transformer that can re-read the premise. T2 shows that otherwise perturbations are simply repaired from
  the context.
- The task must need the intermediate steps and must not be saturated: the one-step R896 classifier already
  reaches 100% here.

## Limits
- Exploratory, no pre-registered thresholds.
- The v5 unseen domains were already used once (by the confirmatory test).
- 48 items per seed.
- One encoder (Qwen2.5-0.5B).
- T1 noise scale for the premise row of PREMISE-MLP uses the sentence σ.
