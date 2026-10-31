# State-tracking pilot v1 (Lab 14, issue #23): result

Script `scripts/stage2_lab14_state_tracking.py` (commit 9806f2a) · data `benchmarks/state_tracking_v1/stories.json`
(sha256 `ed6d16bd…`) · Colab L4, torch 2.11 · seeds 42, 1, 2 (one session per seed, two in parallel; seed 42's
first attempt lost its output stream and was rerun from scratch) · logs `seed*.log` · `aggregate.txt` from
`tools/aggregate_lab14.py`. Blind answerability: `blind_round1/`, `blind_round2/` (54/54 each).

## Verdict by the registered rules
| check | result |
|---|---|
| V1 learnability (val_seen n = 1–4 ≥ 95%) | HOP-R896-d896/free **100%** · ONE-STEP-TF **88.9%** (n=4: 75.5%) → **fails** |
| V2 order dependence (BAG test LONG ≤ 70%) | 48.6% → passes |
| V3 answerability (blind judge ≥ 95%) | 54/54 and 54/54 → passes |
| V4 rule learnable at length (SYMBOLIC/snap val_seen LONG ≥ 95%) | 100% → passes |
| V5 leak check | passes in every run |

**V1 fails, so the registered run is invalid, not negative.** ONE-STEP-TF fits its training stories perfectly
(100%) but does not reach 95% on new event combinations of the same lengths. The registered primary comparison
(P1–P4, unseen-domain test split) is also uninterpretable for a second, independent reason: every model that reads
sentences is near the floor on unseen domains (below).

## What the run shows

### 1. Unseen domains: the bottleneck is recognising the sentences, not composing them
| test (unseen domains) | SHORT n=1–4 | LONG n=6–10 |
|---|---|---|
| SYMBOLIC (true sentence types given) | 100% | 100% |
| HOP-R896-d896 free (primary) | 30.2% | 19.2% |
| best HOP (F64-d64 snap) | 35.1% | 32.3% |
| ONE-STEP-TF | 31.8% | 21.5% |
| ONE-STEP-QWEN | 25.1% | 17.4% |
| BAG (true event counts, order-blind) | 89.9% | 48.6% |

- The rule itself is learned perfectly: the same cell given the true sentence types is 100% at every length.
- With Qwen sentence features, unseen-domain sentences are not recognised:
  - Nearest-class-mean recognition of the 13 sentence types in new domains is 41% (R896) and 69% (F64 funnel).
  - Even the initial state is right only ~50% of the time (per-step accuracy at t = 0).
  - Errors then compound with every event.
- Frozen Qwen2.5-0.5B mean-pooled features plus 16 training domains are not enough to map a new domain's
  sentences onto the abstract event types. That is the "analogy" step, and it is now the open problem.

### 2. Seen domains, longer and new chains: multi-step latent inference generalises, one-step models do not
The registered secondary measurement covers held-out stories in the 16 training domains, n = 1..10, trained on n ≤ 4.

| val_seen | SHORT | n=5 | LONG n=6–10 |
|---|---|---|---|
| HOP-R896-d896 free (state = only memory) | 100% | 99.8% | **99.8%** |
| HOP-R896-d64 free | 100% | 99.3% | 97.8% |
| HOP-R896-d64 / -d896 snap | 100% | 100% | 100% |
| ONE-STEP-TF (reads the whole sequence at once) | 88.9% | 57.6% | **48.3%** |
| ONE-STEP-QWEN (Qwen reads the whole story) | 77.5% | 50.2% | **32.0%** |
| BAG (order-blind, true event counts) | 90.0% | 72.7% | 48.6% |

- Trained only on 1–4 events, the recurrent hop tracks the state through 10 events almost perfectly.
- Both one-step models fall towards the order-blind level beyond the trained length. They are already well
  below the hop on new combinations at trained lengths (ONE-STEP-TF n=4: 75.5%).
- This is the advantage that multi-step inference was expected to have: generalisation to new combinations and
  longer chains with the same information. It is shown here within the seen domains only.

### 3. 64-d vs 896-d state, anchor calibration, funnel input (seen domains; descriptive)
- **State dimension:** an 896-d state is slightly more stable than a 64-d one (LONG 99.8% vs 97.8%, free). There is no
  sign that 64-d is more stable.
- **Snapping to the nearest anchor** (anchor calibration) removes the remaining drift: 100% for both R896 hops.
  It also helps most where drift is largest: F64-d64 rises from 90.1% to 96.8%.
- **Funnel input (F64)** is worse than raw R896 within the seen domains (d64 free: 90.1% vs 97.8%). Across domains
  it recognises sentences better (69% vs 41%), and on test it is the best input (+8.8 points LONG, P4), although
  every model is near the floor there.

## Registered quantities (test split; reported, not interpretable, see above)
| | value |
|---|---|
| P1: primary HOP − best one-step | −2.31 points |
| P2: 64-d − 896-d | −1.16 points |
| P3: snap − free | +3.8 to +6.8 points |
| P4: F64 − R896 | +8.84 points |

## Limits and next step
- The multi-step advantage is shown on seen domains only.
- The cross-domain claim needs a recogniser that maps new-domain sentences onto the abstract event types reliably,
  roughly ≥ 95% per sentence, before the unseen-domain test can measure composition. Candidates: a trained
  recognition head, a larger encoder or prompt-based embeddings, and more training domains.
- ONE-STEP-TF's V1 failure is a generalisation failure, not a training failure. A future registration should
  distinguish the two: training fit vs held-out combinations.
