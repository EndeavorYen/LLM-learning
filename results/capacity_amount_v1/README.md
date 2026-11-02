# B1 rules that change: capacity + amounts (Lab 20, step 2b-1, issue #25): result

- **Script**: `scripts/stage2_lab20_capacity_amount.py` (commit f0cacfa, sha256 `3c1b7789…`).
  - All 3 executed job files are identical to it once the 3-line env prefix is removed (checked).
  - The logs' `script_sha` is the hash of each job file, so aggregate with `--script-verified`.
- **Data**:
  - `benchmarks/amount_v1/train.json` (`b2713147…`; validator PASS, blind sample 120/120).
  - `benchmarks/amount_v1/test.json` (`4bda68a5…`).
    - Blind check 878/880 sentences; 2 plan-like sentences were reworded, then re-checked 2/2.
    - 100/100 of the existing IN/OUT sentences confirmed as amount 1.
  - `benchmarks/story_v2/stories.json` (`5cefc226…`, from `tools/make_stories_v2.py`).
    - Rebuild check passes; blind check 39/40.
    - The one miss is a NOOP sentence that says the safeguard works normally, after the story had broken it. This is
      the same state-asserting NOOP caveat as Lab 19.
- **Pre-run amendments** (independent review, before any run):
  - Traps must change their final state if every amount is read as 1. A1 also needs level < capacity; A3 needs level ≥ 2.
  - The run checks that the stories were built from the registered amount test file.
  - Colab uploads use unique file names.
  - Logs add an amount-blind reference: correct events, every amount read as 1.
  - The aggregator checks coverage and that ORACLE+EXEC is 100%.
- **Run**: Colab L4, 2026-11-02, runner v3 with one seed per VM, no reruns.
- **Logs**: `l20_eye_s{42,1,2}_p.log`.
- **Aggregate**: `aggregate.txt`, from `python tools/aggregate_lab20.py <3 logs> --script-verified`.

## Verdicts by the registered rules
| question | result | rule | verdict |
|---|---|---|---|
| reading + rule with changing capacity / amounts | EYE+EXEC normal LONG **88.41%**, amount read right **99.57%** (2103/2112) | LONG ≥ 80 and amount ≥ 90 | **規則可變** |
| learned reasoner on the held-out capacity 4 | ORACLE+HOP **76.56%**, EYE+HOP 67.97 vs EYE+EXEC 90.19 | ORACLE+HOP ≥ 95 and gap ≤ 3 | **沒學到** |

Per seed, EYE+EXEC LONG: 90.7 / 84.5 / 90.0 (seeds 42 / 1 / 2). Amount: 702 / 701 / 700 of 704.

## Final-state accuracy (3 seeds)
| system | SHORT | MID | LONG | traps | all |
|---|---|---|---|---|---|
| correct events + rule | 100 | 100 | 100 | 100 | 100 |
| correct events, amounts read as 1 (reference) | 73.4 | 60.1 | 59.3 | 0 (by construction) | 56.4 |
| **eye + rule** | 94.3 | 90.6 | **88.4** | 89.6 | 90.8 |
| correct events + learned GRU | 96.2 | 94.1 | 92.1 | 90.3 | 93.5 |
| eye + learned GRU | 89.7 | 84.3 | 80.4 | 81.6 | 84.1 |

Normal stories by capacity (all lengths):
| system | cap 2 | cap 3 | cap 4 (held out from the GRU) | cap 5 |
|---|---|---|---|---|
| eye + rule | 93.2 | 90.5 | 90.2 | 89.7 |
| correct events + GRU | 99.9 | 99.7 | **76.6** | 99.8 |
| eye + GRU | 92.7 | 89.3 | **68.0** | 87.8 |

## What the run shows
1. **The eye reads amounts almost perfectly.**
   - 99.6% on new amount-2/3 sentences, including other ways of writing the number (98.3%) and asides that contain
     an unrelated number (100%).
   - It keeps the old one-unit sentences at 1 (99.0%).
   - Amount errors barely matter: only 93 + 32 normal stories have an amount misread at all.
2. **Errors still come from reading the event type, not the amount.**
   - Every story with no misread ends right (7,195 / 7,195 normal).
   - Of the normal stories that end wrong, 832 of 1,068 had only event-type misreads.
   - The sentence kinds that do the damage are the same as in Lab 19: POSNEG, DIRSWAP, CHATTER, PRONOUN, stripped cues.
3. **The rule-based pipeline does not care about capacity.** EYE+EXEC is about 90% at every capacity 2–5.
4. **The learned GRU does not interpolate to the capacity it never saw.**
   - It is about 99.8% on capacities 2, 3 and 5 but 76.6% on capacity 4, even with correct events.
   - On abstract one-hot stories it reached 82% at capacity 4 in all 3 seeds, against about 100% elsewhere.
   - A capacity encoding that a learned tracker can extend (as an input to compare against, not a learned constant)
     is an open design question.
5. **Comparison with Lab 19 (84.4% LONG):**
   - Lab 20's 88.4% is on a different story set, with more amount-2/3 sentences that the eye reads almost perfectly.
   - So this is not evidence that event-type reading improved.
   - Note also that the Lab 20 eye trained on amount_v1 too. Amount contrast groups are about 57% of the per-domain
     contrast-group picks, so it is not exactly the Lab 19 eye with extra data.

## Next
- 2b-2: two containers.
- Optionally, a capacity representation the learned reasoner can generalise.
- Reinforcing POSNEG / DIRSWAP / CHATTER reading remains the direct lever on long-story accuracy.
