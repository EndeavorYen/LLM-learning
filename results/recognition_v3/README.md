# B1 recognition v3: fine-tuned eye, factorised head, hard sentences (Lab 17, issue #25): result

- **Script**: `scripts/stage2_lab17_finetune_recognition.py` (commit ae51031, sha256 `7b2e1d79…`; the three executed
  copies were checked to equal it).
- **Data**:
  - `stories.json` (`ed6d16bd…`)
  - `extra_domains.json` v1.1 (`f44dc28a…`)
  - `hard_cases.json` (`e145539b…`)
  - `hard_cases_extra.json` (`e31ad583…`; blind check 144/144, see `hard_extra_blind/`)
- **Run**: Colab L4, seeds 42, 1, 2, 10 example draws.
- **Logs**:
  - `l17_A.log` (Q15). Its exec stream stalled; this is the copy the job wrote on the VM, with complete machine lines.
  - `l17_B.log` (Q3E).
  - `l17_C.log` (Q7, frozen reference).
- **Aggregate**: `aggregate.txt` from `tools/aggregate_lab17.py`.
- The 8 test domains were not used. "Old" means the 16 registered-style domains (the registered primary metric);
  "new" means the 12 extra domains.

## Verdict by the registered rules
| check | result |
|---|---|
| main cell (LoRA + FACT + hard, one example), best eye | **Q15 92.19%** (11948/12960); Q3E 91.76% |
| ≥ 95% | **no** |
| negation gate: NEG → NOOP ≥ 90% **and** normal events drop ≤ 1 point vs hard off | Q15: NEG 96.20% but drop **2.44** → **fails** (Q3E: 99.48%, drop 1.00) |
| 93–95% (outcome C) | no |
| < 93% **and** LoRA gain < 2 points (outcome D) | no: LoRA gain **+6.80** (Q3E +12.27) |

**The result is not covered by the registered outcome table.** It is below 93%, but fine-tuning clearly helped. The
next step therefore has to be decided with the author; it is not applied automatically.

## What the run shows

### 1. Fine-tuning the eye is the big lever; the factorised head does not help
Main effects on old-domain events, one example, each averaged over the other two factors:

| eye | LoRA | FACT vs FLAT | hard sentences on |
|---|---|---|---|
| Q15 | **+6.20** | −0.23 | −2.79 |
| Q3E | **+9.38** | −1.38 | −2.37 |

Old-domain events, one example, FLAT head:

| | frozen | LoRA |
|---|---|---|
| Q15, hard sentences off | 88.80% | **94.81%** (new domains 95.74%) |
| Q15, hard sentences on | 86.07% | 92.08% |
| Q3E, hard sentences off | 85.90% | 92.86% |
| Q7 frozen (reference), hard sentences off | 92.48% | – |

- The best cell of the whole run, 94.81%, is just under 95%.
- A fine-tuned 1.5B eye beats the frozen 7B eye.
- **Fine-tuning roughly halves the IN ↔ OUT confusions.** Q15, hard sentences off: 1196 → 607.
- **With a fine-tuned eye, the example barely matters.** Q15 LoRA without an example (ZERO) gets 94.57%; with one
  example (C) it gets 94.81%. Without fine-tuning, the example added 2–5 points. The fine-tuned eye seems to have
  learned the calibration itself, which bears on phase 3 (no example).

### 2. Hard sentences fix negation, at a cost to normal sentences (the Lab 15 trade-off persists)
| main cell, old domains | NEG → NOOP | OTHER → NOOP | TEMP | normal events |
|---|---|---|---|---|
| Q15 LoRA, hard off | 6.61% | 4.69% | 97.29% | 94.63% |
| Q15 LoRA, hard on | **96.20%** | **85.00%** | 97.40% | 92.19% |
| Q3E LoRA, hard on | **99.48%** | 80.21% | 97.50% | 91.76% |

The cost appears as normal events read as NOOP. In the best main cell, 299 IN sentences and 111 FIX sentences went to
NOOP.

### 3. Initial-state sentences
With LoRA they are 98.6–99.2% right (old domains, hard sentences off), and 95–99% with hard sentences on.

## Limits
- LoRA cells are not bit-reproducible: the GPU attention backward pass is non-deterministic. The frozen cells are
  reproducible. Each LoRA run logs its feature fingerprint.
- The outcome table did not foresee this case: below 93%, yet fine-tuning helped.
- The best cell without hard sentences (94.81%) is not the registered main cell, and is reported descriptively only.
