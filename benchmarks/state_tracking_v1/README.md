# state_tracking_v1 — capacity-system state tracking (Lab 14 pilot data, issue #23)

Data for the pre-registered pilot "multi-step latent state tracking vs one-step (length generalisation)".
The task is one domain-general rule, told through 24 everyday systems in templated Simplified Chinese.

| file | what | sha256 |
|---|---|---|
| `templates.json` | 24 domains, hand-written sentences (16 train, 8 test) | `1bfd769efdf6dd3d61ded0e6aaff3eb70b394af2c266e2f02cd1ed9b598bca70` |
| `stories.json` | generated stories, `python tools/gen_state_tracking.py` | `ed6d16bd3e625198bba74d28e7b20f6bdba9c399a3eb94b9a0d1f113f11e3108` |

`stories.json` records the sha256 of the `templates.json` it was built from (`templates_sha256`); the
generator is deterministic (seed 20261031, plain Python, no torch) and reproduces the file byte for byte.

## Task
State: level ∈ {0, 1, 2, 3} (capacity 3 units) × safeguard ∈ {ok, broken}, plus an absorbing failure state.
Nine classes: `L0_ok L1_ok L2_ok L3_ok L0_broken L1_broken L2_broken L3_broken BURST`.

| event | effect |
|---|---|
| IN | level+1. At level 3: safeguard ok → the excess is diverted, level stays 3; safeguard broken → BURST |
| OUT | level−1; at 0 nothing happens |
| BREAK | safeguard := broken (no change if already broken) |
| FIX | safeguard := ok |
| NOOP | an irrelevant sentence; no change |
| any event after BURST | no change |

A story is one initial sentence, `rules + "；" + level[l] + "，" + safeguard[s] + "。"`, which states the
rules, the initial level and the safeguard status, followed by n event sentences. The question is the final
state (9-way, chance 11.1%). Labels are computed by the rule, never by hand.

## Domains
| split | domain (unit, safeguard) |
|---|---|
| train (16) | 水塔 (格水, 溢流管) · 水库 (亿方水, 泄洪闸) · 油罐 (车油, 回流阀) · 停车场 (辆车, 引导屏) · 便桥 (辆车, 限载栏杆) · 货车 (吨货, 称重报警器) · 急诊留观室 (位病人, 转诊通道) · 公交站台 (批乘客, 备用车) · 过山车排队区 (拨游客, 分流闸机) · 冷库 (柜冻品, 冷藏车) · 粮仓 (万斤粮, 备用粮囤) · 充电宝 (格电, 过充保护) · 手机存储 (个G, 云端备份) · 电动车充电棚 (辆电动车, 充电管理器) · 流浪狗收容所 (只狗, 寄养家庭) · 鱼缸 (条鱼, 水族店) |
| test (8) | 高压锅 (级气压, 限压阀) · 机场停机位 (架飞机, 备降调度) · 电梯 (位乘客, 超载报警) · 民宿 (组客人, 合作民宿) · 垃圾站 (车垃圾, 转运车) · 插线板 (台电器, 过载保护开关) · 服务器 (个任务, 自动限流) · 仓库 (托货, 临时仓) |

Both splits mix very everyday systems with less familiar ones, and every test domain has train domains with a
related mechanism (pressure relief, diversion of vehicles or people, overflow storage, electrical/digital
protection) but none with the same wording.

## Template conventions (`templates.json`)
Schema as in #23, plus two fields per domain used by the validator: `measure` (the measure word of one
unit, e.g. `格`, `辆`, `亿方`) and `safeguard_name` (the safeguard's name as it appears in BREAK/FIX).
- Levels are stated with explicit numbers (零/一/两/三 + unit). The rules also state the floor (nothing
  leaves an empty system) and that the failure is permanent.
- Each IN / OUT sentence contains "一" + measure exactly once (one unit); BREAK / FIX always name the
  safeguard; IN / OUT / NOOP never do; NOOP never mentions a unit.
- No event sentence contains a digit, a number word other than 一, or 满/空/半, so no event states a
  resulting level.
- A diverted unit always leaves the system (sent elsewhere, refused, vented), never waits in a queue that
  could come back. OUT sentences are entity-neutral ("一台电器", not "电饭煲"), so an OUT never refers to an
  object that was not put in, and in domains with a side store the OUT names the main store.
- 7 IN, 7 OUT, 6 BREAK, 6 FIX, 6 NOOP paraphrases per domain, all distinct; no string occurs in two
  domains. Sentence structure is varied across domains.

## Splits (`stories.json`)
| split | domains | n | stories per (domain, class, n) | per (class, n) | total |
|---|---|---|---|---|---|
| train | 16 train | 1–4 | 6 | 96 | 3,456 |
| val_seen | 16 train | 1–10 | 1 | 16 | 1,440 |
| test | 8 test | 1–10 | 2 | 16 | 1,440 |

Story fields: `id, domain, n, init_level, init_safeguard, init_text, events, event_texts, states, final`;
`states` has n+1 entries (after the initial sentence and after each event), `final == states[-1]`.

Generation, for each split and n:
1. Final classes are exactly balanced and spread evenly over the domains (table above).
2. Initial states are balanced too (train: 431–433 per state). P(final | init, n) under the event process is
   computed exactly; Sinkhorn scaling gives the (class × init) table with the class quotas as rows and equal
   columns that is closest in KL to the natural one; it is rounded to integers and dealt over the domains.
   Without this, L3_broken would be ~25% over-represented in train (it is the only n = 1 route to BURST).
3. Events are i.i.d. with P(IN, OUT, BREAK, FIX, NOOP) = (.30, .25, .15, .15, .15), rejected until the final
   state is the cell's class; a BURST-final story must reach BURST no earlier than event max(1, n − 3), so at
   most 3 events follow it (post-BURST events are 2–4% of all events).
4. Each event text is a uniformly random paraphrase of that event from that domain.
5. Stories are unique within a split on (domain, initial state, events, texts), and val_seen shares no story
   with train.

## Checks — `python tools/validate_state_tracking.py`
Errors (exit 1): template schema and counts (T1); sentence form, as in the conventions above (T2); no string
in two domains (T3); stories header and templates hash (S1); every story re-simulated with an independent
copy of the rule, its init text recomposed and every event text matched to that event's paraphrases (S2);
exact quotas per (split, n, class, domain) and split sizes (S3); uniqueness, no train/val_seen overlap, and no
string of a test domain inside any train/val_seen story or vice versa (S4).

Reported, with the result on this version (accuracy %, LONG = mean over n = 6..10, SHORT = n = 1..4):

| baseline | val_seen SHORT | val_seen LONG | test SHORT | test LONG |
|---|---|---|---|---|
| B1 order-blind lookup fitted on train (pre-registered early warning, warn > 60%) | 87.3 | 24.2 | 88.4 | 26.5 |
| B2 Bayes-optimal order-blind classifier, expected / realised | 89.9 / 90.8 | 60.7 / 61.0 | 89.9 / 89.9 | 60.7 / 61.0 |
| B3 last-event lookup fitted on train | 22.0 | 19.2 | 20.7 | 21.1 |

- **B1** maps (initial state, counts of the 5 event types) to the majority final class in train, falling back to
  (initial state, #IN, #OUT) and then the global majority. Train has n ≤ 4, so its full-count key never
  matches a LONG story (counts sum to n): half of the LONG items fall to the #IN/#OUT key, half to the global
  majority. It is therefore a weak warning for long stories.
- **B2** is the exact ceiling for any function of (initial state, event counts): P(final | init, counts)
  computed by dynamic programming under the generating process, weighted by each split's realised
  (class, init) mix. Per n on test (expected): 69.1, 64.6, 60.2, 56.4, 53.1 for n = 6..10. A BAG model
  cannot exceed ~61% on test LONG in expectation, so validity check V2 (BAG LONG ≤ 70%) holds by
  construction; a learned BAG trained on n = 1..4 should be lower still.
- Also printed: initial-state balance per split, the event mix, and the share of post-BURST events.

## Blind-judge sample (validity check V3)
`python tools/make_state_blind_sample.py ITEMS_DIR KEY_PATH` writes 54 test items with n = 6..10 (6 per final
class, each class over 6 different test domains, 6–7 items per domain, 10–11 per length) to
`ITEMS_DIR/items.json`: id, init_text, event_texts, the question, and the domain's 9 `state_desc` strings as
options A–I in shuffled order. The key goes to `KEY_PATH`, which must lie outside `ITEMS_DIR`. Bar (#23):
≥ 95% correct.
