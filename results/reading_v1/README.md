# B1 reading breadth (Lab 18a, issue #25): result

- **Script**: `scripts/stage2_lab18a_reading_breadth.py` (commit 92b052e, sha256 `cb96b8d7…`). Both executed copies
  were checked to equal it after removing the 9-line tee prefix.
- **Data**:
  - `benchmarks/reading_v1/train_groups.json` (`ffede6ec…`)
  - `benchmarks/reading_v1/test.json` (`49616f38…`; sealed, blind 1256/1256)
  - the Lab 17 / 17b files
- **Run**: Colab L4, seeds 42, 1, 2, 10 support draws, 2026-11-01. Peak memory 16.7 GiB.
- **Logs**: `l18a_A.log` (Q15), `l18a_B.log` (Q3E).
- **Aggregate**: `aggregate.txt` from `tools/aggregate_lab18a.py`; the registered-run checks passed.

## Verdict by the registered rule: **還沒讀懂 (not yet)**
The best cell is Q15 PAIRS with readout C.

| check | result | needed for 讀得廣 |
|---|---|---|
| group all-correct (base + 9–10 taught variants) | **56.49%** | ≥ 80 |
| normal (base) sentences of the new, naturally written domains | **92.85%** | ≥ 95 |
| HYPO (never taught) | **97.50%** | ≥ 70 |
| side effect (normal vs HARD17) | none (+1.18; template-style +1.21) | – |

## What the run shows (readout C)
| cell | group all | normal | taught items | HYPO | unrelated NOOP |
|---|---|---|---|---|---|
| Q15 HARD17 (Lab 17 recipe) | 6.7 | 91.7 | 75.2 | 15.7 | 90.0 |
| **Q15 PAIRS** | **56.5** | 92.9 | 92.9 | **97.5** | 93.8 |
| Q15 PAIRS+CR | 28.0 | 93.0 | 84.5 | 94.9 | 82.9 |
| Q3E HARD17 | 3.3 | 91.5 | 72.8 | 3.5 | 86.5 |
| Q3E PAIRS | 53.1 | 92.0 | 92.2 | 94.3 | 92.8 |
| Q3E PAIRS+CR | 26.4 | 89.5 | 82.4 | 79.4 | 83.3 |

1. **The contrast groups work, and they generalise to the untaught variant.**
   - Taught-item accuracy rises from 75% to 93%.
   - Hypotheticals, never seen in training (no 如果 / 要是 / 万一 … anywhere), rise from 16% to **97.5%** (Q15) and from
     4% to 94% (Q3E).
   - Negated, intended, question and future variants were taught; hypotheticals were learnt from them.
   - Caveat: this may be a broader "not actual → NOOP" rule rather than reading the conditional itself. Unrelated NOOP
     sentences and base sentences did not suffer, which argues against a blanket NOOP bias.
2. **Whole groups are still hard.** A group has 10–11 sentences, and at ~93% per sentence about half the groups are
   all right (56%).
   - The weakest variants in the best cell are CHATTER (85%), TRANSFER (82%), POSNEG (88%) and OTHER (89%).
   - Most failing items are in the same four variants: CHATTER 13, POSNEG 12, OTHER 11, TRANSFER 8.
3. **Natural writing is harder than template writing.**
   - Base sentences of the new domains are 92–93% in every cell, against 96–97% for the template-style sentences of
     the Lab 17b domains.
   - By domain (best cell), base accuracy ranges from 84% (输液架) to 100% (文物展柜).
   - This is the "narrow scenario" effect the author asked about: about 4 points are lost just from the change in
     writing style.
4. **Confidence regularisation hurt badly:** group all-correct −28 (Q15) / −27 (Q3E). Down-weighting the rows a
   keyword model can guess removed most of the NEG / INTENT signal (NEG 99 → 73).
5. **The Lab 17b probe, re-tested, now passes its own 看懂 line.**
   - Q15 PAIRS: DIR + REAL pairs 91.9% (Lab 17b rule: ≥ 90 and CUE gap ≤ 5; CUE 93.2 vs normal 96.9).
   - REAL-hypothetical 96.9% (was 32% in Lab 17b), SYS 87.5% (was 40%).
   - Run-to-run noise is visible: HARD17 here gives DIR + REAL 62.9%, against 71.7% for the same recipe in Lab 17b.

## Limits
- Test sentences were written by LLM writers (4 styles, no access to training data) and blind-checked by LLM
  annotators, not humans.
- LoRA cells are not bit-reproducible.
- The judgement is on readout C. ZERO gives the same picture (Q15 PAIRS group 55.2%, HYPO 97.2%).
