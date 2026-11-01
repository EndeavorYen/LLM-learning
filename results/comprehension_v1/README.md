# B1 comprehension probe (Lab 17b, issue #25): result

- **Script**: `scripts/stage2_lab17b_comprehension_probe.py` (commit 1675cf4, sha256 `45aa8b0f…`). The three executed
  copies (9-line tee prefix removed) were checked to equal it; the logs print `script_sha=unavailable` because the
  script ran via `colab exec -f`.
- **Data**: `benchmarks/comprehension_probe_v1/probe.json` (sha256 `3beede52…`; validator PASS; blind check 306/312,
  the 9 revised items re-checked 30/30, see `blind/`), plus the Lab 17 training data (28 domains, hard sentences).
- **Run**: Colab L4, seeds 42, 1, 2, 10 example draws, 2026-11-01.
- **Logs**: `l17b_A.log` (Q15), `l17b_B.log` (Q3E), `l17b_C.log` (Q7 frozen eye + 7B direct answering and ICL).
- **Aggregate**: `aggregate.txt` from `tools/aggregate_lab17b.py` (registered run checks passed).

## Verdict by the registered rule: **部分看懂 (partly reads)**
Best small-eye cell (method C, highest DIR + REAL pair accuracy): **Q15, LoRA on, hard sentences on**.

| check | result | threshold |
|---|---|---|
| DIR + REAL pairs (pooled, 10 per domain) | **71.72%** | 看懂 ≥ 90; 猜字 < 70 |
| CUE vs normal sentences | 91.33% vs 95.31%, **3.98 lower** | 看懂 ≤ 5; 猜字 > 10 |

Pairs are above the 猜字 line but far below 看懂; removing surface cues costs little.

## What the probe shows (method C)
| cell | normal | CUE | STR | DIR pairs | REAL pairs | SYS pairs | MENTION pairs |
|---|---|---|---|---|---|---|---|
| Q15 frozen, hard off | 90.5 | 89.0 | 54.6 | 60.4 | 3.5 | 0.0 | 85.3 |
| Q15 LoRA, hard off | 98.3 | 97.3 | 70.1 | 81.9 | 4.4 | 0.0 | 92.8 |
| **Q15 LoRA, hard on** | 95.3 | 91.3 | 70.3 | 80.1 | 66.1 | 40.0 | 100 |
| Q3E LoRA, hard on | 95.0 | 90.6 | 60.6 | 73.2 | 55.6 | 56.1 | 100 |
| Q7 frozen, hard on | 89.1 | 87.8 | 55.1 | 65.6 | 59.4 | 18.6 | 91.4 |
| 7B direct (ZS0) | 58.3 | 66.7 | 52.1 | 29.2 | 13.9 | 8.3 | 58.3 |
| 7B with 5 examples (ICL) | 92.3 | 90.8 | 56.3 | 77.1 | 22.2 | 4.2 | 91.7 |

REAL pairs by form, best cell: actual vs negated 83.9%, vs intended 82.8%, vs **hypothetical 31.7%**.

1. **Cue removal is not the problem.** Sentences without "一 + measure" or the safeguard's name are only ~4 points
   below normal in every LoRA cell; the recogniser is not just spotting those two template cues.
2. **Reality status is learned only where taught.** Without hard sentences, REAL pairs are 1–6% for every eye
   (negated and intended sentences are read as the literal event). With hard sentences (which contain negation and
   other-system cases), negated / intended reach ~83%, but **hypotheticals ("如果……") stay at 0–32%**: never taught,
   never learned.
3. **"Which system" is weak.** The same event at a different system (隔壁 / 另一台) is read as this system's event:
   SYS pairs 0–3% without hard sentences, 19–56% with them.
4. **Direction is 80% at best.** Same-words direction pairs (A 给 B vs B 给 A) fail systematically on some items
   (e.g. `产蛋箱/DIR-2b`, `打印机纸盒/DIR-2b` read as OUT in all 30 scorings).
5. **Fine-tuning helps direction and transfers (+20 DIR points), not reality status.**
6. **The 7B model answering directly is worse than the small fine-tuned eye on every probe type**; with 5 examples
   it reads normal sentences well (92%) but REAL pairs are 22% and SYS 4%.

## Limits
- 6 probe domains written by LLM writers in one session; blind-checked by an independent annotator (an LLM), not humans.
- LoRA cells are not bit-reproducible (GPU attention backward pass).
- The judgement thresholds are coarse; the per-type numbers above matter more than the label.
