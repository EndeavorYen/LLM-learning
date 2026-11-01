# B1 recognition diagnostic v1 (Lab 15, issue #25): result

Script `scripts/stage2_lab15_recognition.py` (commit 914905e, sha256 `58a69226…`; the runner checked that both
executed copies equal it) · data `benchmarks/state_tracking_v1/stories.json` (`ed6d16bd…`), `hard_cases.json`
(`e145539b…`), `benchmarks/analogy_v5.json` (`d2a9e877…`) · Colab L4, torch 2.11, transformers 5.21 · seeds 42, 1, 2 ·
two sessions in parallel: `l15_A.log` (Q05, Q15, BGE) and `l15_B.log` (Q7 + zero-shot; its first VM stalled while
downloading the model and the run was restarted from scratch on a new VM) · `aggregate.txt` from
`tools/aggregate_lab15.py`.

Only the 16 training domains were used (4-fold leave-4-domains-out cross-validation). The 8 test domains of
`state_tracking_v1` and the 8 unseen domains of `analogy_v5` were not touched.

## Verdict by the registered rule
| check | result |
|---|---|
| best eye + recognizer, state tracking, 12 training domains | **BGE + MLP 83.65%** (1606/1920) |
| ≥ 95% → Phase 2 | **no** |
| no combination ≥ 95% → follow the k-curve | rises: 72.86% (k=4) → 80.21% (k=8) → 83.65% (k=12) |
| registered next step | **add training domains** |

- The top three are tied: BGE + MLP 83.65%, Q15-prompt + MLP 83.54%, Q7-prompt + MLP 82.97%. The difference
  between the first two is 2 sentences out of 1920, so this run does not pick one of them.
- With 83.65% per sentence, all 11 sentences of a 10-event story are right with probability 0.8365¹¹ ≈ 14%.

## What the run shows

### 1. A bigger eye does not help; the read-out does
| st, k=12 | NEAR | LIN | MLP | FUNNEL | XDOM |
|---|---|---|---|---|---|
| Q05-mean (Lab 14's features) | 45.8 | 59.6 | 67.3 | 66.0 | 59.4 |
| Q15-prompt | 71.7 | 77.5 | **83.5** | 80.2 | 77.6 |
| Q7-prompt | 72.0 | 81.1 | **83.0** | 78.6 | 77.3 |
| BGE (bge-m3) | 78.1 | 79.3 | **83.7** | 82.0 | 68.4 |

- Qwen 7B features are no better than 1.5B or bge-m3 (568M).
- How the sentence is read out matters more than size: the prompt's last token beats mean pooling by 6–8 points
  for Qwen 1.5B and 7B.
- A small MLP is the best recognizer for every eye. The funnel is 1–5 points lower. Cross-domain contrastive
  training (XDOM) did not help.

### 2. Different eyes are good at different sentence types
| k=12, MLP | event sentences (5 classes) | initial-state sentences (8 classes) |
|---|---|---|
| Q7-prompt | **88.0%** | 62.8% |
| Q15-prompt | 85.7% | 74.7% |
| BGE | 82.8% | **87.0%** (NEAR 89.1%) |

Each eye is strongest on a different sentence type, so a combination may do better. That has not been tested,
and the log has no per-sentence predictions to check how much their errors overlap.

### 3. More domains help, with diminishing returns
| k-curve (st) | k=4 | k=8 | k=12 |
|---|---|---|---|
| BGE + MLP | 72.86 | 80.21 | 83.65 |
| Q15-prompt + MLP | 73.49 | 80.42 | 83.54 |
| Q7-prompt + MLP | 75.00 | 79.90 | 82.97 |

- The gain from 8 to 12 domains (+3.1 to +3.4) is smaller than from 4 to 8 (+4.9 to +7.4).
- Exploratory, not registered: a straight line in log₂(k) through the three points reaches 95% at about 37–40
  training domains (BGE, Q15) or 64 (Q7), and 99% at 55 or more. The gain from 8 to 12 is already below that
  line, so these are optimistic.
- The same fit gives about 85–87% for a recognizer trained on all 16 domains.

### 4. Qwen 7B zero-shot is far below a trained small recognizer
| Qwen2.5-7B-Instruct, no training, right under both option orders | |
|---|---|
| event sentences | 67.8% (347/512); one order alone 77% |
| initial state (number of units and safeguard together) | 16.4% (21/128); units 23.4%, safeguard 70.3% |
| all 13 classes | 57.5% |

- In Lab 14's blind checks, an independent strong reader answered 54/54 whole test-domain stories twice, so
  these sentences can be read correctly.
- So the low zero-shot score is a limit of the 7B model, not ambiguous data, at least on the sampled items.

### 5. Auxiliary: law sentences (`analogy_v5`, 27 classes, 16 training domains)
- Best: 79.07%, a tie between Q7-prompt + LIN and BGE + MLP.
- The same pattern holds: prompt pooling beats mean pooling, and accuracy rises with more domains (BGE + MLP 60.2 → 71.2 → 79.1).

## Phase 1b: hard real-language cases (diagnostic; does not change the rule above)
Held-out domains, k=12, BGE + MLP (the registered best). The full table for every combination is in `aggregate.txt`.

| | trained on normal sentences | trained also on hard sentences of the training domains |
|---|---|---|
| NEG "X did not happen" → NOOP | **6.8%** (13/192); literal event 15.6% | **97.4%** (187/192); literal event 1.0% |
| OTHER "another system changed" → NOOP | **21.9%** (21/96); literal event 61.5% | 78.1% (75/96); literal event 18.8% |
| TEMP time markers / flashback | 79.2% | 80.2% |
| HEDGE: mean top probability, hedged vs matched plain | 0.64 vs 0.88 (drops) | 0.93 vs 0.87 (no longer drops) |
| normal held-out sentences | 83.65% | **78.85%** (1514/1920) |

- **Trained on normal sentences only, the recognizer fails on negation and on other systems.** By the registered
  reading, B1 is not done even after recognition of normal sentences reaches 95%.
- Most NEG errors are neither NOOP nor the literal event. The log does not record which class they went to.
- **Negation learned in some domains carries over to new domains** (97%). Other-system sentences carry over only
  partly (78%).
- **This has a cost.** The top three combinations lose 4.8–6.6 points on normal sentences (BGE + MLP
  83.65 → 78.85, Q15-prompt + MLP 83.54 → 76.9, Q7-prompt + MLP 82.97 → 78.2). The uncertainty signal on hedged
  sentences also disappears.
- Time markers cause almost no extra errors: TEMP accuracy is close to the normal-sentence accuracy.
- Qwen 7B zero-shot on the same hard cases, right under both option orders:
  - NEG 32.8%, OTHER 9.4%, TEMP 75.0%.
  - On hedged sentences (no "uncertain" option) it picks IN, BREAK or NOOP, never OUT or FIX.

## Limits
- The run logs no per-class confusion, so it cannot say which event types are confused or where NEG errors go.
- These are cross-validation estimates inside the 16 training domains (4 folds of 4 domains). Fold accuracy
  ranges from 76.7% to 87.7% for BGE + MLP, so domain-to-domain variance is large.
- The extrapolation in §3 is a three-point curve fit, not a registered measurement.

## Next step
The registered branch is to add training domains. Candidate designs are discussed on #25 before anything is built.
