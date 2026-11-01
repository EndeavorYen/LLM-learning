# B1 reading + reasoning joined (Lab 19, step 2a, issue #25): result

- **Script**: `scripts/stage2_lab19_read_and_reason.py` (commit 454c3d3, sha256 `d4668589…`).
  - All 6 executed job files equal it after the 4-line env prefix is removed (checked); the logs' `script_sha` is the
    hash of each job file, so it differs per job.
- **Data**: `benchmarks/story_v1/stories.json` (`69c5866b…`, 3,080 stories over 22 never-trained domains, blind 40/40);
  built by `tools/make_stories_v1.py`. The 8 state_tracking_v1 test domains are not touched.
- **Run**: Colab, 2026-11-02, runner v3 (one job per VM, detached, 2-min polls).
  - Eye seeds 42 / 1 / 2 and the 7B direct job on L4; Qwen3-8B thinking in two shards on A100.
  - Eye seed 42 lost its VM once after the job had finished; the rerun gave the same LoRA fingerprint
    (`0x1.478ee817d2000p+18`) and is the log used.
  - The runner hit the 2-hour harness limit while the thinking shards ran; they were collected by a takeover runner
    from the same VMs (no rerun).
- **Logs**: `l19_eye_s{42,1,2}_p.log`, `l19_direct_s0_p.log`, `l19_think_s0_p{0,1}of2.log`.
- **Aggregate**: `aggregate.txt` (`python tools/aggregate_lab19.py <the 6 logs>`).

## Verdict by the registered rule: **部分可用 (partly usable)**
| check | result | needed for 讀＋推可用 |
|---|---|---|
| EYE+EXEC, normal stories length 8–12 (3 seeds) | **84.42%** | ≥ 90 |
| same 220 stories: EYE+EXEC vs Qwen3-8B thinking | **85.61 vs 20.45** | not below thinking ✔ |

< 75% would have been 累積誤差太大; 84.4% is above that.

## Final-state accuracy (9 classes)
| system | SHORT 1–4 | MID 5–7 | LONG 8–12 | traps | all |
|---|---|---|---|---|---|
| correct events + rule | 100 | 100 | 100 | 100 | 100 |
| **eye + rule** (3 seeds) | 93.41 | 86.31 | **84.42** | 86.21 | 87.65 |
| eye + learned GRU | 92.88 | 86.06 | 83.91 | 86.82 | 87.35 |
| Qwen2.5-7B direct (both option orders) | 9.89 | 9.85 | 9.18 | 11.82 | 9.90 |
| Qwen3-8B thinking (LONG + traps only) | – | – | 20.45 | 23.41 | 22.42 |

Per seed, LONG: 87.5 / 84.2 / 81.6 (seeds 42 / 1 / 2).

## What the run shows
1. **Errors come from reading, not reasoning.** With correct events the rule is 100%. The eye reads 95.7% of the story
   sentences right; every story with zero misreads ends right (5,908 / 5,908 normal).
2. **Errors grow slowly with length**: 97.3% at 1 sentence → ~84% at 8–12. About half of the misreads do no harm
   (52.3% of normal stories with ≥ 1 misread still end right, e.g. a wrong OUT at 0, or a later event overwrites it).
3. **Which sentence types do the damage** (misreads, 3 seeds): DIRSWAP 434, POSNEG 383, CHATTER 377, PRONOUN 242,
   probe STR (stripped cues) 212. These are the targets for reinforcement.
4. **Learned GRU ≈ exact rule** (83.9 vs 84.4 LONG): the reasoner trained on length 1–4 abstract events carries over to
   length 12 and to the eye's soft inputs.
5. **Traps**: T1 (early break, late overflow) is the weakest at 76.4%; T2 / T4 are > 90%.
6. **The LLM baselines did not work in this format.**
   - Qwen3-8B thinking: in 490 / 660 stories (74.2%) the reasoning did not close within the registered 2,048 tokens,
     which counts as wrong. Post-hoc, on the stories it did answer (170, mean ~1,527 tokens), it was right 88.2%
     (normal LONG, n = 51; eye on the same stories 77.1%) and 86.6% (traps, n = 119; eye 90.2%). These answered
     stories are self-selected (the model finished on them), so this is not a registered comparison.
   - Qwen2.5-7B direct: ~10% at every length, including 1-sentence stories, so the floor is the registered scoring
     (letter scoring, right under BOTH option orders), not story length. It is not evidence about reading.
   - Compute: thinking used 1.26 M generated tokens and ~4.2 A100-hours for 660 stories; the eye + rule needs one
     forward pass per sentence (~6 L4-minutes per seed including LoRA training).

## Caveats
- Some NOOP sentences assert a state ("还是两个", etc.) that may contradict the tracked state; the label is NOOP, so the
  rule ignores them, while an LLM reading the text could be misled. This affects the LLM baselines, not the eye.
- The 7B and thinking numbers reflect these registered formats; a longer thinking budget or a different answer format
  would likely raise them. They are auxiliary observations.

## Next
Step 2b (rules that change: capacity, two-unit events, two containers; external Boxes benchmark) — design to be
discussed and registered first. Reinforcing DIRSWAP / POSNEG / CHATTER reading is the direct lever on LONG accuracy.
