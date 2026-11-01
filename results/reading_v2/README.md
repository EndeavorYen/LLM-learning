# B1 reading reinforcement (Lab 18b, issue #25): result

- **Script**: `scripts/stage2_lab18b_reading_reinforcement.py` (commit 18c9e49, sha256 `6eb08262…`).
  - All 10 executed job files equal it after the 3-line env prefix is removed (checked).
  - The logs' `script_sha` is the hash of each job file, so it differs per job; aggregate with `--script-verified`.
- **Data**: `benchmarks/reading_v2/train_groups_v2.json` (`1d7f1c6f…`), `test_v2.json` (`904a97b8…`, sealed, blind
  1256/1256), plus the Lab 18a / 17b files.
- **Run**: Colab L4, 2026-11-01/02, runner v3 (one seed per VM, detached job, 2-min polls).
  - Two earlier attempts lost their VMs mid-run (backend pruned them); their partial logs were not looked at and are
    not used.
  - One v3 job (Q3E seed 2) failed to launch on its first VM and ran on a fresh VM.
- **Logs**: `l18b_<eye>_s<seed>.log`, 10 files.
- **Aggregate**: `aggregate.txt` (`python tools/aggregate_lab18b.py --script-verified l18b_*.log`).

## Verdict by the registered rule: **還沒讀懂 (not yet)**, with normal sentences now above 95%
Best cell (new sealed test v2, readout C): **Q15, system-aware prompt on, noise practice off (A1B0)**.

| check | result | needed for 讀得廣 |
|---|---|---|
| group all-correct | **70.12%** | ≥ 80 |
| normal (base) sentences | **95.40%** | ≥ 95 ✔ |
| HYPO (never taught) | **98.54%** | ≥ 70 ✔ |

## Factor effects (registered rule: ≥ +3 points and the same sign in ≥ 4 of 5 seeds)
| eye | A: read with the question in mind | B: noise practice |
|---|---|---|
| Q15 | **+7.28**, 5/5 seeds → **effective** (normal +2.1) | +0.51, 3/5 → not effective |
| Q3E | **+5.98**, 5/5 seeds → **effective** (normal +3.8) | +0.40, 3/5 → not effective |

## What the run shows (readout C)
1. **Naming the system in the prompt is the lever.**
   - On the new test, group all-correct goes from 62% to 70% (Q15) and from 62% to 69% (Q3E).
   - Normal sentences cross 95% (95.4 / 95.8).
   - The biggest single gain is TRANSFER (79% → 92%), as expected from telling the eye which system "this" is.
   - On the Lab 18a test, Q15 goes from 58% to **75%** group all-correct, against 56% in Lab 18a.
2. **Noise practice adds nothing measurable.** CHATTER stays at 88–95% with or without it. The invariance loss neither
   helps nor hurts much (normal −0.6).
3. **The untaught hypothetical still generalises** (95–99% in every cell).
4. **Lab 17b probe, re-tested**: Q15 A1B0 reaches DIR + REAL pairs 97.3%, DIR 97.1% (direction swaps were the weak
   spot in 17b), and CUE 95.1% vs normal 96.5%.
5. **What still breaks whole groups** (best cell): CHATTER (89.8%), DIRSWAP (90.9%), PRONOUN (95.4%) and POSNEG
   (94.2%).
   - One domain (单车架, the student-diary bicycle rack) accounts for many of the failing items.
   - A group has 10–11 sentences, so ~96% per sentence still leaves ~30% of groups with at least one miss.

## Limits
- Data by LLM writers and LLM annotators, not humans.
- LoRA cells are not bit-reproducible.
- The judgement uses one sealed test of 8 domains (96 groups); the auxiliary tests agree in direction.
