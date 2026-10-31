# Hop learnability diagnosis (Lab 11, exploratory, issue #21)

Script: `scripts/stage2_lab11_hop_learnability.py` · data: `benchmarks/analogy_v4.json` · seeds 42, 1, 2 · Colab L4.
Logs: `seed42.log`, `seed1.log`, `seed2.log`.

## Question
In the v4 association test the hop (law_brain) chose the right branch on only 57% of unseen items, and only
56–69% on its own training chains. Was that a limit of the architecture, or of how it was trained?

## Variants (all trained on the 12 seen domains only)
| | training | positional emb. | hop loss |
|---|---|---|---|
| V0 | Lab 09: funnel + hop jointly, 400 steps | no | cosine regression |
| V1 | stage A: funnel alone (400 steps), frozen; stage B: hop alone, 2000 steps | no | cosine regression |
| V2 | as V1 | yes | cosine regression |
| V3 | as V1 | no | cross-entropy over the 27 anchors (τ = 0.1) |
| V4 | as V1 | yes | cross-entropy over the 27 anchors (τ = 0.1) |

`free` = the predicted vector is fed back as is; `snap` = it is replaced by its nearest anchor first.

## Results (mean of 3 seeds; BR = branch choice accuracy, mean of stages 2–4, chance 50%)
| variant | mode | seen BR (72 training chains) | unseen BR (48 items) | per seed | A-k | FLIP-k |
|---|---|---|---|---|---|---|
| V0 | free | 63.7% | 56.5% | 51.4 / 59.7 / 58.3 | 21.8% | 22.2% |
| V1 | free | 99.8% | 69.0% | 69.4 / 67.4 / 70.1 | 53.0% | 52.8% |
| V2 | free | 100% | 69.4% | 72.2 / 68.8 / 67.4 | 65.3% | 50.0% |
| V3 | free | 100% | 73.1% | 73.6 / 72.2 / 73.6 | 64.6% | 58.3% |
| **V4** | **free** | **100%** | **77.5%** | 75.0 / 77.8 / 79.9 | 79.6% | 66.2% |
| V4 | snap | 100% | 76.4% | 72.2 / 75.0 / 81.9 | 76.2% | 65.7% |
| PREMISE-MLP (reference) | | 100% | 77.3% | 76.4 / 77.8 / 77.8 | | |
| true law → anchor picker (stage-A funnel, ceiling) | | 100% | 83.3% | | | |

Checks: V0 reproduces Lab 09 exactly for every seed (HOP, ceiling and PREMISE-MLP numbers identical);
LAST is exactly 50% at every stage for both funnels; the funnel weights are unchanged by hop training (asserted).

## Findings
1. **The v4 failure was a training problem, not an architecture limit.** Trained jointly (V0) the hop never
   learns its own training chains (seen BR 64%). With the funnel trained first and frozen, the same law_brain
   learns them completely (seen BR 99.8–100%) in every variant and every seed.
2. **Why joint training fails:** with the hop loss (weight 2.5) in the same objective, the funnel's anchors are
   pulled together — the largest off-diagonal anchor cosine is 0.51–0.82 for the joint funnel vs 0.28–0.32 for
   the funnel trained alone, and the anchor push loss stays high (1.14 at step 400, seed 42). The hop's targets
   are then not separable.
3. **Transfer to unseen domains:** the two-stage hop chooses the right branch on 69–78% of unseen items
   (V0: 57%). The best variant, V4 (positional embeddings + cross-entropy over anchors), reaches 77.5%,
   the same as PREMISE-MLP (77.3%), which classifies the whole premise from 896-d Qwen features. The
   anchor-picker ceiling for this funnel is 83.3%, so the remaining gap is partly in recognising the
   candidate sentences, not in the branch decision.
4. FLIP (picks that change when only stage 0 is swapped for the partner law's) rises from 22% (V0) to 66% (V4):
   the hop's choice now depends on the premise.

## Caveats
- Exploratory: the variant was chosen on the v4 unseen items, so 77.5% is an optimistic estimate.
  The confirmatory test is v5 (8 brand-new domains, blind-judged 144/144), pre-registered on issue #21
  before it runs.
- 48 unseen items per seed; differences of a few points between variants are within noise.
