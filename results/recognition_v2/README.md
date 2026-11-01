# B1 recognition v2: anchor examples in new domains + domain-count curve (Lab 16, issue #25): result

- **Script**: `scripts/stage2_lab16_anchor_recognition.py` (commit d588560, sha256 `2df4d7c5…`; every executed copy was
  checked to equal it).
- **Data**:
  - `benchmarks/state_tracking_v1/stories.json` (`ed6d16bd…`)
  - `hard_cases.json` (`e145539b…`)
  - `extra_domains.json` (`c26febd2…`): 12 new training domains, blind check 84/84 (see `extra_blind/`)
- **Run**: Colab L4, torch 2.11, transformers 5.21, seeds 42, 1, 2, 10 example draws.
- **Logs and summaries**:

  | | logs | summary |
  |---|---|---|
  | Part A | `partA_l16_A1.log` (Q05, Q15, BGE), `partA_l16_A2.log` (Q7 + 7B reference) | `partA_aggregate.txt` |
  | Part B | `partB_l16_B1.log` (Q15, BGE), `partB_l16_B2.log` (Q7) | `partB_aggregate.txt` (`--a-best C-MLP`) |

  The summaries were produced by `tools/aggregate_lab16.py`.
- **Run notes**:
  - The live output of the first launch stalled for both sessions; those runs were discarded and restarted from
    scratch.
  - In the restart, each job also wrote its output to a file on the Colab machine.
  - The live output of `partA_l16_A1.log` stalled again, so that log is the copy from the Colab machine (machine
    lines complete, ending in `L16_CONFIG`).

Only training domains were used. The 8 test domains of `state_tracking_v1` were not touched.

## Part A: one example per event type in a new domain

### Verdict by the registered rule
| check | result |
|---|---|
| best anchor method (s = 1, k = 12, event sentences, 5-way) | **Q7-prompt + C-MLP 92.71%** (12015/12960) |
| ≥ 95% | **no** |
| gain over ZERO-MLP of the same encoder | **+3.47 points** (ZERO-MLP 89.24%, 11565/12960) |
| gain ≥ +5 → "anchors help, not enough" / < +5 → "anchors do not help much" | **< +5** |
| registered next step | **fine-tune the encoder (LoRA)** (Lab 15's fallback) |

- The best method is stable: folds range 90.6–93.7% and seeds 92.6–92.8%.
- The runner-up is Q15-prompt + C-MLP at 89.48%. The 7B-ICL reference is excluded from the selection.

### What the run shows

**1. The examples help as a calibration of the domain, not as prototypes.**

| s = 1, k = 12 | ZERO-MLP | C-MLP (centre the domain on its examples) | MLP-NN (nearest example) | RAW-NN |
|---|---|---|---|---|
| Q7-prompt | 89.24 | **92.71** | 87.21 | 76.42 |
| Q15-prompt | 86.32 | **89.48** | 82.72 | 75.89 |
| BGE | 83.08 | **87.30** | 80.39 | 68.79 |

- **Centring helps every encoder.** Subtracting the domain's mean (estimated from its 5 examples) before the trained
  classifier adds 2.9 to 5.5 points.
- **One example is too few to compare against.** Classifying by the nearest example is worse than using no example
  at all: 1.4 to 6.2 points worse in the MLP's hidden layer (MLP-NN), and 10 to 29 points worse on the raw features
  (RAW-NN).
- **Three examples change that.** With s = 3 the nearest-example methods overtake ZERO-MLP (Q7-prompt MLP-NN
  92.13%), but they still trail C-MLP.
- More examples add little to C-MLP: Q7-prompt goes 92.71 → 93.58 → 93.66% for s = 1, 2, 3.

**2. Examples stand in for training domains.** C-MLP trained on 4 domains is about as good as ZERO-MLP trained on 12:

| encoder | C-MLP, 4 domains | ZERO-MLP, 12 domains |
|---|---|---|
| Q7-prompt | 89.46% | 89.24% |
| Q15-prompt | 87.72% | 86.32% |
| BGE | 83.64% | 83.08% |

**3. The remaining errors are mostly the direction of change.**
- Q7-prompt + C-MLP makes 945 errors. 381 of them (40%) confuse IN and OUT.
- Without examples, the main errors are different: IN or FIX read as NOOP (569). Centring removes most of those.

**4. Initial-state sentences are close to solved** (8-way; the sentence group is known from its position).
- BGE ZERO-MLP gets 94.5%; BGE C-MLP gets **98.4%**.
- Qwen encoders reach only 57–85%.

**5. The 7B model benefits from examples but stays below the trained small recognizer.**
- Qwen2.5-7B-Instruct, right under both option orders:
  - without examples 67.5%;
  - with the domain's 5 examples in the prompt **89.2%**.
- Q7-prompt + C-MLP, also given the examples, reaches 92.7%.

**6. Examples do not fix the hard cases.**
- Negation is still mostly wrong: NEG → NOOP is at most 33% for any combination (6.7% for the best one).
- Another system's change is still mostly read as the literal event: OTHER → NOOP 2.5% for the best combination.
- Time markers are fine: TEMP 97.8% for the best combination.
- None of the examples is a negated or other-system sentence, so there is nothing to calibrate against.

## Part B: domain-count curve (28 training domains, 7 folds, k = 4 … 24)

### Verdict by the registered rule
g1 = acc(12) − acc(6), g2 = acc(24) − acc(12), each one doubling of the number of training domains.

| row | k=4 | k=6 | k=8 | k=12 | k=16 | k=20 | k=24 | g1 | g2 | shape |
|---|---|---|---|---|---|---|---|---|---|---|
| Q15-prompt ZERO-MLP 13-way | 71.52 | 77.26 | 78.96 | 82.53 | 84.43 | 86.46 | 87.20 | 5.27 | 4.67 | 持續上升 |
| Q7-prompt ZERO-MLP 13-way | 72.80 | 77.44 | 79.64 | 84.05 | 85.83 | 88.01 | 88.18 | 6.61 | 4.14 | 趨緩 |
| BGE ZERO-MLP 13-way | 69.29 | 76.61 | 80.57 | 83.72 | 85.42 | 86.96 | 87.98 | 7.11 | 4.26 | 趨緩 |
| Q15-prompt C-MLP (events, s=1) | 85.59 | 86.53 | 87.05 | 88.35 | 89.48 | 89.87 | 90.42 | 1.83 | 2.07 | 持續上升 |
| Q7-prompt C-MLP (events, s=1) | 88.59 | 89.51 | 90.13 | 90.98 | 91.58 | 92.28 | 92.45 | 1.46 | 1.47 | 持續上升 |
| BGE C-MLP (events, s=1) | 82.01 | 84.30 | 84.98 | 86.61 | 87.69 | 88.54 | 88.76 | 2.31 | 2.15 | 持續上升 |

### What the run shows
- **Without examples, accuracy still grows about 4–5 points per doubling at 12 → 24 domains**, a little less than at
  6 → 12. That counts as "趨緩" for two encoders and "持續上升" for one.
- **With examples, the curve starts higher but grows slowly**: 1.5–2 points per doubling.
  - The registered rule calls this "持續上升", because the gain per doubling does not shrink. The gain itself is small.
  - Without examples the curve is steeper. At k = 24 the advantage of the examples shrinks to 0.4–1.6 points
    (events, C-MLP vs ZERO-MLP; Q7-prompt 92.45% vs 91.59%).
- Descriptive extrapolation, a straight line in log₂(k) over k ≥ 6 (not registered):
  - ZERO-MLP 13-way reaches 95% at about 50–65 training domains.
  - C-MLP reaches 95% only at about 75–140, because its curve is flatter.
- **The extra domains are easier than the registered ones.** For example, held-out at k = 12, Q15-prompt
  ZERO-MLP 13-way is 79.6% on the 16 registered domains and 86.5% on the 12 extra ones. The new domains were written
  in one batch from one specification, so their phrasing is probably more uniform. The Part B curve is therefore
  somewhat optimistic for registered-style domains.

## Limits
- Everything here is cross-validation inside the training domains. The 8 test domains are untouched.
- "Best method" was chosen on the same data it is reported on, as registered. Its margin over the runner-up method
  on the same encoder (Q7-prompt MLP-NN 87.21%) is 5.5 points.
- The 7B-ICL reference covers 16 domains × 2 draws (864 queries per condition).

## Next step
- **Registered branch**: fine-tune the encoder (LoRA).
- **Where the errors are**:
  - the direction of a change (IN vs OUT);
  - negation and other-system sentences.
- The design will be discussed on #25 before anything is built.
