# v5 confirmatory association test (Lab 12, issue #21) — verdict

Script: `scripts/stage2_lab12_association_v5.py` (sha256 `fd36878b…2b6a`) · data: `benchmarks/analogy_v5.json`
(sha256 `d2a9e877…f196`) · Colab L4, torch 2.11, transformers 5.21 · seeds 42, 1, 2.
Pre-registration and amendment: issue #21 (both posted before any GPU run). Logs: `seed42.log`, `seed1.log`,
`seed2.log`; `seed42rerun.log` is the determinism check (bit-identical fingerprint and output lines).
`aggregate.txt` = `python tools/aggregate_v5.py seed42.log seed1.log seed2.log`.

## Setup
- Training: 20 seen domains (all v4 domains). Test: 8 domains never used before (足球比赛, 中学校园, 影视剧组,
  法院审判, 图书馆, 考古发掘, 建筑工地, 游乐园), 48 items (domain × law).
- An independent blind judge answered all 144 test questions correctly from stage 0 + stage 1 (`blind_judge/`).
- Task: given stage 0 (the system's structure: is there a regulator / bypass / backup?) and stage 1 (identical
  for the two laws of a pair), choose which of two branch sentences comes at stages 2, 3 and 4. Chance = 50%.
- HOP = Lab 11 V4: funnel trained alone and frozen, then the law_brain hop trained alone (positional embeddings,
  cross-entropy over the 27 state anchors), free rollout: ŝ2 → ŝ3 → ŝ4 from [s0, s1] with no text.

## Validity checks (all pass)
| check | result |
|---|---|
| 1. leak check; LAST exactly 50% (BR and DEC), every seed | pass (780 training texts checked; LAST 24/48 at every stage, every seed) |
| 2. HOP seen-domain BR ≥ 95%, every seed | pass (360/360 in every seed) |
| 3. RANDOM HOP BR ≤ 60% | pass (49.5%) |
| determinism (amendment) | seed 42 run twice: identical |

## Pre-registered hypotheses
| | criterion | result | verdict |
|---|---|---|---|
| **H1** hop chooses the right branch on new domains | BR ≥ 65% and ≥ 60% every seed | **84.95%** (367/432); seeds 82.6 / 88.2 / 84.0 | **supported** |
| **H2** hop vs whole-premise classifier | D = HOP − PREMISE-MLP; ≥ +5 better, \|D\| < 5 comparable, ≤ −5 worse | **D = −3.94** (367 vs 384 of 432); seeds −7.6 / +1.4 / −5.6 | **comparable** (by the registered rule; leaning towards PREMISE-MLP, see below) |
| **H3** two-stage training is what makes the hop work | HOP − V0 ≥ 10 points | **+26.16** (367 vs 254 of 432) | **supported** |

## All methods (mean of 3 seeds, 48 unseen items)
| method | BR | DEC | FLIP | note |
|---|---|---|---|---|
| **HOP** (V4, two-stage) | **84.95%** | 91.20% | 71.8% | 64-d s states only, no text after stage 1 |
| PREMISE-MLP | 88.89% | 97.92% | | 896-d Qwen features of the whole premise + sentence MLP |
| S0-RULE | 85.42% | 93.75% | | nearest stage-0 anchor of s0 |
| V0 (Lab 09 joint training) | 58.80% | 64.81% | 25.9% | seen-domain BR only 73.8% |
| Qwen2.5-0.5B LM few-shot | 58.33% | | | |
| Qwen2.5-0.5B LM zero-shot | 54.17% | | | |
| RANDOM HOP | 49.54% | 51.85% | 31.5% | |
| LAST (s1 only) | 50.00% | 50.00% | | exactly chance by construction |

Ceilings (true law given): anchor picker 92.1%, sentence-MLP picker 93.1%.
Same-picker comparison: HOP decision → MLP picker 85.65% vs premise decision → MLP picker 91.44%.

## What this shows
1. **The association hop works on genuinely new domains.** After seeing only s0 and s1 of a domain it has never
   met, the hop rolls forward three steps in its 64-d law space and chooses the right branch 85% of the time
   (chance 50%), while the language model itself (Qwen 0.5B ranking the sentences) gets 54–58%.
   The choice depends on the premise: swapping only stage 0 for the partner law's flips 72% of the picks.
2. **The v4 negative result was a training problem.** The same architecture trained jointly (V0) reaches only
   59% and does not even fit its training chains; trained in two stages it fits them completely and transfers.
   Joint training pulls the funnel's anchors together so the hop's targets are not separable (Lab 11).
3. **It is not better than reading the premise directly.** By the registered rule the hop is "comparable" to a
   classifier of the whole premise (D = −3.9 points), but the direction is consistent: PREMISE-MLP is ahead in
   two of three seeds by more than 5 points, on the candidate-independent decision (DEC 97.9% vs 91.2%), and with
   the same picker (91.4% vs 85.7%). The hop also does not beat S0-RULE (85.4%), which reads the law of s0 directly.
4. **Where it fails:** the hop's decision errors concentrate on law 3 (negative feedback → recovers): 15 of 18
   wrong decisions in seed 42, 5 of 10 and 6 of 10 in seeds 1 and 2, mostly answered as its partner law 2
   (positive feedback → runaway). In several of these items s0's nearest anchor is ambiguous or even correct
   (影视剧组 L3: s0 nearest (L3,S0), hop still goes to L2), so the hop has learned a default branch for
   off-distribution s0 inputs: it was trained only on seen-domain s vectors, which sit tightly on their anchors.

## Limits
- In this task the branch is fully determined by stage 0, so the best a hop can do is carry s0's law forward;
  the test shows it can, not that multi-step association adds information a premise classifier lacks. That needs
  a task where the next state depends on more than one earlier state (composition / novel paths).
- 48 test items per seed; differences of ~4 points (≈ 6 items per seed) are within noise.
- One frozen 0.5B encoder (Qwen2.5-0.5B, mean-pooled), Chinese text, six hand-written laws.

## Natural next steps (not run)
- Train the hop on out-of-fold s vectors (funnel trained on the other seen domains) or with input noise, so it
  sees off-anchor inputs like those of unseen domains — aimed at the law-3 default-branch error.
- A composition test, where the correct branch needs two earlier states, to test what the hop adds beyond s0.
