# ==============================================================================
# Stage 2 / Lab 14: state-tracking pilot v1 (PRE-REGISTERED on issue #23; numbers only, NO verdict)
# Question: does a recurrent hop whose state is its ONLY memory generalise to longer chains (trained on n = 1..4,
#   tested on n = 1..10) and to unseen domains better than one-step models that receive the same information?
# Data: benchmarks/state_tracking_v1/stories.json. 9 state classes (L0..L3 x {ok, broken} + BURST); events IN / OUT /
#   BREAK / FIX / NOOP; train = seen domains, n = 1..4; val_seen = seen domains, n = 1..10; test = unseen domains,
#   n = 1..10. Every label is re-derived with the rule simulator `simulate` and must equal the file (data error).
# Features: frozen Qwen2.5-0.5B (bf16), last hidden state mean-pooled (Lab 12 extractor).
#   per sentence  every unique init (STATE PART only, [amendment 2026-11-01]) / event sentence, max_length 128;
#                 no truncation (asserted from token counts)
#   whole story   init state part + event texts concatenated with no separator (like Lab 12's premise s0+s1), max_length 512
#                 (asserted), for every TRAINING PREFIX and every val_seen / test story (ONE-STEP-QWEN only)
#   Train-split texts, val_seen-only texts and test texts are extracted in separate padded batches (never mixed).
#   R896 = normalize(h - mu); mu = mean feature of the unique TRAIN-split sentences.
#   F64  = normalize(funnel(h)); funnel 896 -> 256 -> 64 (GELU; Lab 12 struct_funnel) trained 400 steps (Lab 12
#          schedule) on the unique train-split sentences with Lab 12's s pull/push loss (closest-pair push term) over
#          13 sentence classes: 8 init classes (level x safeguard) + 5 event types. No d-branch.
# Models (all trained on the train split only; Lab 12 seed_all placement before every model init and training run):
#   HOP-<in>-d<D>  in in {R896, F64}, D in {64, 896}. h_0 learned (normalised). Step t: h = normalize(W2 GELU(W1
#                  [h; proj(x_t)] + b1) + b2), hidden 256, proj = Linear(D_in, D); step 1 reads the init sentence,
#                  steps 2..n+1 the events. Readout: 9 learned anchors A (normalised), logits = h A^T / 0.1, CE at every
#                  step against `states` (mean over all valid (story, step) pairs). AdamW 1e-3 -> 1e-4 cosine, 3000
#                  steps, full batch padded at the end (padding never reaches a valid step), clip 1.0, wd 0.01, BPTT
#                  through the free state. Inference with the same weights: free (h fed back as is) or snap (h <- its
#                  nearest anchor before it is fed back; a step's prediction is the same in both modes, only what is
#                  fed back differs). PRIMARY = HOP-R896-d896 / free.
#   SYMBOLIC       the HOP cell with D = 64 and proj(x_t) replaced by the one-hot (13) of the TRUE sentence type.
#   ONE-STEP-TF    Linear(896, 64) on R896, then 2 causal transformer layers (Lab 12 law_brain layer: d 64, nhead 4,
#                  ff 128, norm_first, PyTorch default dropout 0.1), NO positional embeddings, Linear(64, 9) at every
#                  position with CE at every position (= every prefix). Same optimiser and steps as HOP.
#   ONE-STEP-QWEN  Lab 12 MLP (896 -> 256 -> 64 -> 9) on the raw whole-text feature of every training prefix
#                  (label = state after the prefix); Lab 12 train_mlp recipe with 1000 steps.
#   BAG            the same MLP recipe on [one-hot init class (8); counts of IN, OUT, BREAK, FIX, NOOP] of every prefix.
#   LAST           majority lookup (init class, last event type) over the training prefixes; unseen key -> MAJORITY.
#   MAJORITY       majority class of the training prefix labels. Ties (here and in LAST) -> lowest class index.
#   "Every prefix" = k = 0..n events, n + 1 prefixes per training story: exactly the (story, step) set and weighting
#   of the per-step CE of HOP / SYMBOLIC / ONE-STEP-TF (asserted).
# Metrics: final-state accuracy per n = 1..10 on test (unseen domains) and on val_seen; SHORT = mean over n = 1..4,
#   LONG = mean over n = 6..10, n5 separately. HOP / SYMBOLIC per-step state accuracy on the test stories with n = 10
#   (t = number of events consumed, 0..10; t = 0 is the state after the init sentence). Training fit = accuracy over
#   every training prefix (= every training step).
# Validity numbers (printed, no verdict): V1 HOP-R896-d896/free and ONE-STEP-TF on val_seen n = 1-4; V2 BAG test LONG;
#   V4 SYMBOLIC/snap val_seen LONG; V5 leak check (asserted: no test-domain sentence or text in any training input or
#   loss). V3 (blind judge) is external.
# Machine lines: L14_ACC / L14_SUM / L14_STEP / L14_FIT / L14_PARAMS, then L14_CONFIG as the LAST line (run complete).
#   Aggregate: python tools/aggregate_lab14.py seed42.log seed1.log seed2.log
# Run:  AUDIT_SEED=42 python stage2_lab14_state_tracking.py   (seeds 42, 1, 2)
# Copied code is marked "(copied from Lab 12)"; "[lab14]" marks new or changed code.
# ==============================================================================
import os, sys, time, math, random, json, hashlib
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

SEED = int(os.environ.get("AUDIT_SEED", "42"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[l14] SEED={SEED} | device={DEVICE}", flush=True)
T_START = time.time()
# (copied from Lab 12; [lab14] log prefix) environment record (reproducibility)
print(f"[l14] python {sys.version.split()[0]} | torch {torch.__version__} | numpy {np.__version__} | "
      f"transformers {getattr(transformers, '__version__', '?')} ({getattr(transformers, '__file__', '?')})", flush=True)
try:
    with open(os.path.abspath(__file__), "rb") as _f:
        SCRIPT_SHA256 = hashlib.sha256(_f.read()).hexdigest()
except (NameError, OSError):
    SCRIPT_SHA256 = "unavailable"  # e.g. executed through a notebook kernel; the runner records the repo file's sha
print(f"[l14] script sha256={SCRIPT_SHA256}", flush=True)
ENV_TAG = f"torch={torch.__version__},transformers={getattr(transformers, '__version__', '?')},device={DEVICE}"
if DEVICE == "cuda":
    ENV_TAG += f",gpu={torch.cuda.get_device_name(0).replace(' ', '_')}"


# (copied from Lab 12)
def seed_all(seed: int = SEED):
    """(copied from Lab 09/11)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# (copied from Lab 12)
def cosine_lr(step: int, total: int, peak: float, floor: float) -> float:
    """Copied from Lab 01 (via Lab 09/11): cosine schedule from peak (step 0) to floor (step == total)."""
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * step / total))


# ---- [lab14] pre-registered configuration (issue #23) -------------------------------------------------------------
SEQ_STEPS, SEQ_LR, SEQ_LR_FLOOR, SEQ_WD = 3000, 1e-3, 1e-4, 0.01  # HOP / SYMBOLIC / ONE-STEP-TF
SEQ_LOG_STEPS = (1, 1000, 2000, 3000)
HEAD_STEPS, HEAD_LR, HEAD_LR_FLOOR = 400, 1e-3, 1e-4  # Lab 12 schedule (funnel; the MLPs use it with MLP_STEPS)
MLP_STEPS = 1000                                      # ONE-STEP-QWEN / BAG
CE_TAU = 0.1
HOP_HIDDEN = 256
SENT_MAXLEN, STORY_MAXLEN = 128, 512  # [amendment 2026-11-01] 64 -> 128 (state part of the init sentence <= ~40 tokens)
HOP_ALIGN = 1.0  # [amendment 2026-11-01] weight of the anchor-alignment term (1 - cos(h_t, A[y_t])) for HOP / SYMBOLIC
FEAT_BATCH = 64                                       # Lab 12 extraction batch size
SHORT_N, N5, LONG_N, ALL_N = (1, 2, 3, 4), 5, (6, 7, 8, 9, 10), tuple(range(1, 11))
STEP_N = 10                                           # per-step curves on the test stories with n = 10
LOSS_LOG = []                                         # (tag, step, message)

# ==============================================================================
# [0] Data ([lab14]; file handling as Lab 12)
# ==============================================================================
EXPECTED_STORIES_SHA256 = "ed6d16bd3e625198bba74d28e7b20f6bdba9c399a3eb94b9a0d1f113f11e3108"  # registered file (#23, commit d20be68)
_DATA_CANDIDATES = ["/content/stories.json", "benchmarks/state_tracking_v1/stories.json"]
try:
    _DATA_CANDIDATES.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks",
                                         "state_tracking_v1", "stories.json"))
except NameError:
    pass
DATA_PATH = next((p for p in _DATA_CANDIDATES if os.path.exists(p)), None)
if DATA_PATH is None:
    raise FileNotFoundError(f"stories.json not found in {_DATA_CANDIDATES}")
with open(DATA_PATH, "rb") as f:  # read bytes once: parse + sha256 of the exact file used (as Lab 12)
    _DATA_RAW = f.read()
DATA = json.loads(_DATA_RAW.decode("utf-8"))
DATA_SHA256 = hashlib.sha256(_DATA_RAW).hexdigest()
print(f"[l14] data = {DATA_PATH} ({DATA.get('version')}) sha256={DATA_SHA256} | generator seed={DATA.get('seed')} | "
      f"templates_sha256={DATA.get('templates_sha256')}", flush=True)
if EXPECTED_STORIES_SHA256 is not None:
    assert DATA_SHA256 == EXPECTED_STORIES_SHA256, f"stories.json is not the registered file: {DATA_SHA256}"
else:
    print("[l14] WARNING: EXPECTED_STORIES_SHA256 is None (not pinned yet): this cannot be the registered run",
          flush=True)
assert DATA.get("version") == "state_tracking_v1", DATA.get("version")

CLASSES = ["L0_ok", "L1_ok", "L2_ok", "L3_ok", "L0_broken", "L1_broken", "L2_broken", "L3_broken", "BURST"]
assert DATA["classes"] == CLASSES, f"DATA ERROR: classes {DATA['classes']}"
N_CLS = len(CLASSES)
CI = {c: i for i, c in enumerate(CLASSES)}
EVENTS = ("IN", "OUT", "BREAK", "FIX", "NOOP")
N_INIT = 8                         # init classes = the first 8 classes (level x safeguard); BURST is never initial
N_SCLS = N_INIT + len(EVENTS)      # 13 sentence classes (funnel loss, SYMBOLIC input)
SCLS_NAMES = CLASSES[:N_INIT] + [f"ev:{e}" for e in EVENTS]
CAPACITY = 3
SPLITS = ("train", "val_seen", "test")
EVAL_SPLITS = ("test", "val_seen")
N_RANGE = {"train": range(1, 5), "val_seen": range(1, 11), "test": range(1, 11)}
STORIES = {sp: DATA["splits"][sp] for sp in SPLITS}


def simulate(init_level, init_safeguard, events):
    """[lab14] The rule of issue #23 -> [class after init, after e1, ..., after en] (class names)."""
    lv, sg, burst = int(init_level), str(init_safeguard), False
    if not (0 <= lv <= CAPACITY and sg in ("ok", "broken")):
        raise ValueError(f"bad initial state {init_level!r} / {init_safeguard!r}")
    out = [f"L{lv}_{sg}"]
    for e in events:
        if e not in EVENTS:
            raise ValueError(f"unknown event {e!r}")
        if not burst:                       # any event after BURST: no change
            if e == "IN":
                if lv < CAPACITY:
                    lv += 1
                elif sg == "broken":        # at capacity, safeguard broken -> BURST
                    burst = True
                # at capacity, safeguard ok -> the excess is diverted; the level stays 3
            elif e == "OUT":
                lv = max(0, lv - 1)         # nothing happens at 0
            elif e == "BREAK":
                sg = "broken"
            elif e == "FIX":
                sg = "ok"
            # NOOP: no change
        out.append("BURST" if burst else f"L{lv}_{sg}")
    return out


def init_cls(st) -> int:
    name = f"L{int(st['init_level'])}_{st['init_safeguard']}"
    if name not in CLASSES[:N_INIT]:
        raise RuntimeError(f"DATA ERROR: story {st.get('id')}: initial class {name!r}")
    return CI[name]


# ---- story-level data checks (a failure is a data error) ----
_ids = set()
for sp in SPLITS:
    assert len(STORIES[sp]) > 0, f"DATA ERROR: split {sp} is empty"
    for st in STORIES[sp]:
        n, ev, et = int(st["n"]), st["events"], st["event_texts"]
        sid = st["id"]
        assert sid not in _ids, f"DATA ERROR: duplicate story id {sid}"
        _ids.add(sid)
        assert n in N_RANGE[sp], f"DATA ERROR: {sp} story {sid}: n={n}"
        assert len(ev) == n and len(et) == n and len(st["states"]) == n + 1, f"DATA ERROR: story {sid}: lengths"
        assert isinstance(st["init_text"], str) and st["init_text"], f"DATA ERROR: story {sid}: init_text"
        assert all(isinstance(t, str) and t for t in et), f"DATA ERROR: story {sid}: empty event text"
        sim = simulate(st["init_level"], st["init_safeguard"], ev)
        if st["states"] != sim:
            raise RuntimeError(f"DATA ERROR: story {sid}: states {st['states']} != rule {sim}")
        assert st["final"] == st["states"][-1], f"DATA ERROR: story {sid}: final != states[-1]"
        assert CLASSES[init_cls(st)] == st["states"][0], f"DATA ERROR: story {sid}: states[0] != initial class"
    for n in N_RANGE[sp]:
        assert any(int(st["n"]) == n for st in STORIES[sp]), f"DATA ERROR: split {sp} has no story with n={n}"
print(f"[l14] rule simulator reproduces `states` and `final` of all {len(_ids)} stories", flush=True)

DOMS = {sp: sorted({st["domain"] for st in STORIES[sp]}) for sp in SPLITS}
SEEN_DOMS, TEST_DOMS = set(DOMS["train"]), set(DOMS["test"])
assert set(DOMS["val_seen"]) <= SEEN_DOMS, f"DATA ERROR: val_seen domains not in train: {set(DOMS['val_seen']) - SEEN_DOMS}"
assert not SEEN_DOMS & TEST_DOMS, f"LEAK / DATA ERROR: domains in both train and test: {SEEN_DOMS & TEST_DOMS}"
for sp in SPLITS:
    by_n = defaultdict(Counter)
    for st in STORIES[sp]:
        by_n[int(st["n"])][st["final"]] += 1
    parts = []
    for n in sorted(by_n):
        cnt = [by_n[n][c] for c in CLASSES]
        parts.append(f"n{n}:{sum(cnt)}" + ("" if max(cnt) - min(cnt) <= 1 else f"(UNBALANCED {cnt})"))
    print(f"[l14] {sp:<8}: {len(STORIES[sp])} stories | {len(DOMS[sp])} domains | final classes per n: "
          + " ".join(parts), flush=True)
print(f"[l14] train domains: {', '.join(DOMS['train'])}\n[l14] test domains : {', '.join(DOMS['test'])}", flush=True)

# optional cross-check against templates.json (next to stories.json): same domain partition
_TPL_PATH = os.path.join(os.path.dirname(DATA_PATH), "templates.json")
if os.path.exists(_TPL_PATH):
    with open(_TPL_PATH, "rb") as f:
        _tpl_raw = f.read()
    _tpl_sha = hashlib.sha256(_tpl_raw).hexdigest()
    print(f"[l14] templates = {_TPL_PATH} sha256={_tpl_sha} "
          f"({'matches' if _tpl_sha == DATA.get('templates_sha256') else 'DIFFERS from'} stories.json templates_sha256)",
          flush=True)
    _tpl_split = {d["name"]: d["split"] for d in json.loads(_tpl_raw.decode("utf-8"))["domains"]}
    _lab_seen = {_tpl_split.get(d) for d in SEEN_DOMS}
    _lab_test = {_tpl_split.get(d) for d in TEST_DOMS}
    assert len(_lab_seen) == 1 and len(_lab_test) == 1 and _lab_seen != _lab_test and None not in _lab_seen | _lab_test, \
        f"DATA ERROR: templates.json domain splits disagree with stories.json: seen {_lab_seen}, test {_lab_test}"
    _unused = sorted(set(_tpl_split) - SEEN_DOMS - TEST_DOMS)
    print(f"[l14] templates.json: seen domains = split {_lab_seen.pop()!r}, test domains = split {_lab_test.pop()!r}"
          + (f" | domains without stories: {_unused}" if _unused else ""), flush=True)
else:
    print(f"[l14] templates.json not found next to stories.json (optional cross-check skipped)", flush=True)

# ---- sentences: one sentence class each (8 init classes + 5 event types) ----
SENT_CLS = {}               # sentence text -> sentence class 0..12
SENT_DOMS = defaultdict(set)
SENTS_OF = {sp: [] for sp in SPLITS}  # unique sentences per split, in order of first appearance


# [amendment 2026-11-01] expected state part of every init sentence, from templates.json (if present)
STATE_PART = {}
for _tp in ["/content/templates.json", "benchmarks/state_tracking_v1/templates.json"]:
    if os.path.exists(_tp):
        with open(_tp, encoding="utf-8") as _f:
            for _d in json.load(_f)["domains"]:
                for _lv, _lt in enumerate(_d["init"]["level"]):
                    for _sg, _st in _d["init"]["safeguard"].items():
                        STATE_PART[(_d["name"], _lv, _sg)] = _lt + "，" + _st + "。"
        print(f"[l14] init state parts checked against {_tp} ({len(STATE_PART)} entries)", flush=True)
        break


def init_state_text(st) -> str:
    """[amendment 2026-11-01] Model input for the init sentence: only its state part (level + safeguard), i.e. the text
    after the single '；'. The rules clause before it is identical in every story of a domain and carries no story
    information; at ~90 tokens it would dominate the mean-pooled feature. It is removed for EVERY method (per-sentence
    models and ONE-STEP-QWEN alike); the human judge still sees the full sentence. The generator builds
    init_text = rules + "；" + level + "，" + safeguard + "。" and the level / safeguard parts contain no '；', so the state
    part is the text after the LAST '；' (the rules clause itself may contain '；'). Checked against templates.json."""
    t = st["init_text"]
    if "；" not in t:
        raise RuntimeError(f"DATA ERROR: init_text has no '；': {t!r}")
    part = t.rsplit("；", 1)[1]
    want = STATE_PART.get((st["domain"], int(st["init_level"]), st["init_safeguard"]))
    if want is not None and part != want:
        raise RuntimeError(f"DATA ERROR: init state part {part!r} != templates {want!r}")
    return part


def story_sentences(st):
    """[lab14] [(text, sentence class)] for [init state part, e1, ..., en]."""
    return [(init_state_text(st), init_cls(st))] + [(t, N_INIT + EVENTS.index(e))
                                                for e, t in zip(st["events"], st["event_texts"])]


for sp in SPLITS:
    _seen = set()
    for st in STORIES[sp]:
        for text, c in story_sentences(st):
            if SENT_CLS.setdefault(text, c) != c:
                raise RuntimeError(f"DATA ERROR: sentence used as {SCLS_NAMES[SENT_CLS[text]]} and {SCLS_NAMES[c]}: {text!r}")
            SENT_DOMS[text].add(st["domain"])
            if text not in _seen:
                _seen.add(text)
                SENTS_OF[sp].append(text)
_multi = sum(len(d) > 1 for d in SENT_DOMS.values())
print(f"[l14] unique sentences: train {len(SENTS_OF['train'])} | val_seen {len(SENTS_OF['val_seen'])} | test "
      f"{len(SENTS_OF['test'])} | sentences shared across domains: {_multi}", flush=True)
TRAIN_SENT_SET = set(SENTS_OF["train"])
_train_scls = Counter(SENT_CLS[t] for t in SENTS_OF["train"])
assert all(_train_scls[c] > 0 for c in range(N_SCLS)), f"DATA ERROR: a sentence class has no train sentence: {_train_scls}"

# ---- V5 leak check (Lab 12 style): no test-domain sentence or test story text in any training input or loss ----
FORBIDDEN_SENT = set(SENTS_OF["test"])
assert not FORBIDDEN_SENT & (TRAIN_SENT_SET | set(SENTS_OF["val_seen"])), \
    f"LEAK: test sentence(s) also used in a seen domain: {sorted(FORBIDDEN_SENT & (TRAIN_SENT_SET | set(SENTS_OF['val_seen'])))[:5]}"


def story_text(st, k=None) -> str:
    """[lab14] Whole-story text: init state part ([amendment 2026-11-01]) + the first k event texts (all if k is None),
    no separator (Lab 12 premise)."""
    return init_state_text(st) + "".join(st["event_texts"] if k is None else st["event_texts"][:k])


# every training prefix (story index, k events), k = 0..n
TRAIN_PREFIXES = [(i, k) for i, st in enumerate(STORIES["train"]) for k in range(int(st["n"]) + 1)]
assert len(TRAIN_PREFIXES) == sum(int(st["n"]) + 1 for st in STORIES["train"])
PREFIX_TEXT = [story_text(STORIES["train"][i], k) for (i, k) in TRAIN_PREFIXES]
PREFIX_LABEL = [CI[STORIES["train"][i]["states"][k]] for (i, k) in TRAIN_PREFIXES]
WHOLE_TRAIN = list(dict.fromkeys(PREFIX_TEXT))                       # unique training prefix texts
_wrow = {t: j for j, t in enumerate(WHOLE_TRAIN)}
PREFIX_WROW = [_wrow[t] for t in PREFIX_TEXT]
WHOLE_EVAL, STORY_WROW = {}, {}
for sp in EVAL_SPLITS:
    _txt = [story_text(st) for st in STORIES[sp]]
    WHOLE_EVAL[sp] = list(dict.fromkeys(_txt))
    _wrow = {t: j for j, t in enumerate(WHOLE_EVAL[sp])}
    STORY_WROW[sp] = [_wrow[t] for t in _txt]
FORBIDDEN = FORBIDDEN_SENT | set(WHOLE_EVAL["test"])


def assert_not_forbidden(text: str):
    """(copied from Lab 12; [lab14] the substring loop runs over the test SENTENCES -- every test story text contains
    one, so it also covers the test story texts) No test-domain sentence (or test story text) may ever enter a loss."""
    if text in FORBIDDEN:
        raise RuntimeError(f"LEAK: test text in training data: {text!r}")
    for t in FORBIDDEN_SENT:
        if t in text:
            raise RuntimeError(f"LEAK: test sentence inside training text: {t!r}")


_train_texts = SENTS_OF["train"] + WHOLE_TRAIN  # funnel / mu / HOP / SYMBOLIC types / TF inputs; ONE-STEP-QWEN inputs
for _t_ in _train_texts:
    assert_not_forbidden(_t_)
print(f"[l14] V5 leak check passed for all {len(_train_texts)} training texts ({len(SENTS_OF['train'])} train sentences + "
      f"{len(WHOLE_TRAIN)} unique training prefix texts) against {len(FORBIDDEN_SENT)} test sentences and "
      f"{len(WHOLE_EVAL['test'])} test story texts", flush=True)
_tp = set(WHOLE_TRAIN)
_dup = Counter(int(st["n"]) for st in STORIES["val_seen"] if story_text(st) in _tp)
print(f"[l14] val_seen stories whose whole text equals a training prefix text (per n): "
      f"{dict(sorted(_dup.items())) if _dup else 'none'}", flush=True)
del _tp

# per-sentence rows: [train sentences | val_seen-only sentences | test sentences]
VAL_ONLY_SENTS = [t for t in SENTS_OF["val_seen"] if t not in TRAIN_SENT_SET]
ALL_SENTS = SENTS_OF["train"] + VAL_ONLY_SENTS + SENTS_OF["test"]
SROW = {t: i for i, t in enumerate(ALL_SENTS)}
assert len(SROW) == len(ALL_SENTS)
N_TRAIN_SENT = len(SENTS_OF["train"])
SENT_GROUPS = {"train": SENTS_OF["train"], "val_only": VAL_ONLY_SENTS, "test": SENTS_OF["test"]}
SCLS_ALL = torch.tensor([SENT_CLS[t] for t in ALL_SENTS])

# ==============================================================================
# [1] Qwen (copied from Lab 12): bf16, frozen
# ==============================================================================
MODEL_ID = "Qwen/Qwen2.5-0.5B"
print(f"=== [1] Loading '{MODEL_ID}' (frozen, bf16) ===", flush=True)
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.padding_side = "right"
qwen_base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map=DEVICE)
qwen_base.eval()
for p in qwen_base.parameters():
    p.requires_grad = False
_P0 = next(qwen_base.parameters(), None)  # [review N5] (the CPU test stub has no parameters and a FAKE version)
assert (_P0 is not None and _P0.dtype == torch.bfloat16) or "FAKE" in getattr(transformers, "__version__", "")
H_DIM = qwen_base.config.hidden_size  # 896
print(f"Loaded in {time.time()-t0:.1f}s | hidden={H_DIM} | layers={qwen_base.config.num_hidden_layers}", flush=True)
assert H_DIM == 896, H_DIM  # [lab14]


@torch.no_grad()
def extract_qwen_896_feature(text_list, max_length: int = 64):
    '''Copied from Lab 01: frozen Qwen last hidden state, mean-pooled, float32. [lab14] max_length argument (Lab 12: 64).'''
    enc = tokenizer(text_list, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(DEVICE)
    out = qwen_base.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
    last_h = out.last_hidden_state.float()
    mask_f = enc.attention_mask.unsqueeze(-1).float()
    pooled_896 = (last_h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
    return pooled_896  # (B, 896) float32


def extract_group(texts, max_length: int, tag: str):
    """[lab14] Features of ONE group of texts, in Lab 12's batches of 64. Train-split texts, val_seen-only texts and test
    texts are separate groups, so they are never in the same padded batch ([review N3] of Lab 12). No truncation:
    every text's token count (tokenised alone, no truncation) must be <= max_length (data error otherwise) and the
    padded, truncating batch tokenisation must keep exactly that many tokens per text."""
    out = torch.empty(len(texts), H_DIM, dtype=torch.float32, device=DEVICE)
    if not texts:
        return out, 0
    n_tok = [len(tokenizer(t).input_ids) for t in texts]
    j = int(np.argmax(n_tok))
    if n_tok[j] > max_length:
        raise RuntimeError(f"DATA ERROR: {tag}: {n_tok[j]} tokens > max_length {max_length} (would be truncated): "
                           f"{texts[j]!r}")
    for i in range(0, len(texts), FEAT_BATCH):
        chunk = texts[i:i + FEAT_BATCH]
        kept = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True,
                         max_length=max_length).attention_mask.sum(dim=1).tolist()
        assert [int(x) for x in kept] == n_tok[i:i + FEAT_BATCH], f"{tag}: batch tokenisation != per-text token counts"
        out[i:i + len(chunk)] = extract_qwen_896_feature(chunk, max_length)
    assert bool(torch.isfinite(out).all()), tag
    return out, n_tok[j]


t0 = time.time()
_parts, _maxtok = [], {}
for _g in ("train", "val_only", "test"):
    _h, _maxtok[f"sent/{_g}"] = extract_group(SENT_GROUPS[_g], SENT_MAXLEN, f"sentences/{_g}")
    _parts.append(_h)
H_SENT = torch.cat(_parts, dim=0)               # (N_sentences, 896), rows = ALL_SENTS
assert H_SENT.shape == (len(ALL_SENTS), H_DIM)
H_WHOLE_TRAIN, _maxtok["story/train_prefixes"] = extract_group(WHOLE_TRAIN, STORY_MAXLEN, "story/train_prefixes")
H_WHOLE_EVAL = {}
for sp in EVAL_SPLITS:
    H_WHOLE_EVAL[sp], _maxtok[f"story/{sp}"] = extract_group(WHOLE_EVAL[sp], STORY_MAXLEN, f"story/{sp}")
del _parts
print(f"[l14] extracted {H_SENT.shape[0]} sentence features (max_length {SENT_MAXLEN}) and "
      f"{H_WHOLE_TRAIN.shape[0]} + {sum(v.shape[0] for v in H_WHOLE_EVAL.values())} whole-story features (max_length "
      f"{STORY_MAXLEN}) in {time.time()-t0:.1f}s | max tokens: "
      + ", ".join(f"{k}={v}" for k, v in _maxtok.items()), flush=True)


def _t(lst):
    return torch.tensor(lst, device=DEVICE)


# ==============================================================================
# [2] Input spaces: R896 (centred on the train sentences, normalised) and F64 (funnel)
# ==============================================================================
with torch.no_grad():
    MU_TRAIN = H_SENT[:N_TRAIN_SENT].mean(dim=0)          # mean of the unique TRAIN-split sentence features only
    X_R896 = F.normalize(H_SENT - MU_TRAIN, p=2, dim=-1)  # (N_sentences, 896)

H_TRAIN_SENT = H_SENT[:N_TRAIN_SENT]                      # the only sentence features that enter the funnel loss
TRAIN_SCLS = SCLS_ALL[:N_TRAIN_SENT].to(DEVICE)
SCLS_OH = F.one_hot(TRAIN_SCLS, N_SCLS).float()           # (N_TRAIN_SENT, 13)
OFF_S = ~torch.eye(N_SCLS, device=DEVICE, dtype=torch.bool)


class Funnel(nn.Module):
    """[lab14] Lab 12 DualHead's struct_funnel (896 -> 256 -> 64, GELU) with the s-normalisation of extract_ds."""

    def __init__(self, qwen_hidden: int = 896, s_dim: int = 64):
        super().__init__()
        self.struct_funnel = nn.Sequential(nn.Linear(qwen_hidden, 256), nn.GELU(), nn.Linear(256, s_dim))

    def forward(self, h_896: torch.Tensor):
        return F.normalize(self.struct_funnel(h_896), p=2, dim=-1)


def funnel_loss(funnel):
    """(adapted from Lab 12 funnel_loss) s pull/push with the closest-pair term over the 13 sentence classes of the
    train sentences; [lab14] no d-branch."""
    s = funnel(H_TRAIN_SENT)
    s_mean = F.normalize(SCLS_OH.T @ s / SCLS_OH.sum(0).unsqueeze(-1), p=2, dim=-1)  # (13, 64)
    l_s_pull = (1.0 - (s * s_mean[TRAIN_SCLS]).sum(-1)).mean()
    _sp = F.relu((s_mean @ s_mean.T)[OFF_S] + 0.1)
    l_s_push = _sp.mean() + _sp.max()  # closest-pair term (from v3)
    return l_s_pull + l_s_push, (l_s_pull, l_s_push)


def train_funnel(tag: str = "FUNNEL-F64"):
    """(adapted from Lab 12 train_funnel_stage_a) 400 steps full batch, AdamW 1e-3 -> 1e-4 cosine, wd 0.01, clip 1.0."""
    seed_all(SEED)  # before model init
    fun = Funnel(H_DIM, 64).to(DEVICE).float()
    n_par = sum(p.numel() for p in fun.parameters())
    print(f"\n=== [2] Training {tag} (13 sentence classes, s pull/push): {n_par:,} params | {HEAD_STEPS} steps full "
          f"batch on {N_TRAIN_SENT} train sentences | AdamW lr {HEAD_LR:g} -> {HEAD_LR_FLOOR:g} cosine | clip 1.0 ===",
          flush=True)
    seed_all(SEED)  # before training run
    opt = torch.optim.AdamW(fun.parameters(), lr=HEAD_LR, weight_decay=0.01)
    fun.train()
    for step in range(1, HEAD_STEPS + 1):
        for pg in opt.param_groups:
            pg['lr'] = cosine_lr(step, HEAD_STEPS, HEAD_LR, HEAD_LR_FLOOR)
        opt.zero_grad()
        loss, (l_pull, l_push) = funnel_loss(fun)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fun.parameters(), 1.0)
        opt.step()
        if step in (1, 200, 400):
            msg = f"loss {loss.item():.4f} (s_pull {l_pull.item():.3f} s_push {l_push.item():.3f})"
            LOSS_LOG.append((tag, step, msg))
            print(f"  [{tag}] step {step:3d}/{HEAD_STEPS} | {msg}", flush=True)
    fun.eval()
    return fun, n_par


FUNNEL, FUNNEL_PARAMS = train_funnel()
with torch.no_grad():
    X_F64 = FUNNEL(H_SENT)                                 # (N_sentences, 64), unit norm
SPACE_X = {"R896": X_R896, "F64": X_F64}


@torch.no_grad()
def sentence_type_acc(X):
    """[lab14] diagnostic: nearest train-class-mean (13 sentence classes, cosine) accuracy per sentence group."""
    means = F.normalize(SCLS_OH.T @ X[:N_TRAIN_SENT] / SCLS_OH.sum(0).unsqueeze(-1), p=2, dim=-1)
    hit = ((X @ means.T).argmax(dim=-1).cpu() == SCLS_ALL).tolist()
    out, a = [], 0
    for g in ("train", "val_only", "test"):
        b = a + len(SENT_GROUPS[g])
        out.append(f"{g} {sum(hit[a:b])}/{b - a}")
        a = b
    return " | ".join(out)


for _sp_name, _X in SPACE_X.items():
    print(f"[l14] diagnostic: sentence type (13 classes) by nearest train-class mean in {_sp_name}: "
          f"{sentence_type_acc(_X)}", flush=True)


# ==============================================================================
# [3] Sequence tensors (HOP / SYMBOLIC / ONE-STEP-TF) and prefix examples (ONE-STEP-QWEN / BAG / LAST / MAJORITY)
# ==============================================================================
def seq_tensors(stories):
    """[lab14] rows (B, T) into ALL_SENTS for [init, e1..en] (padding: row 0), true sentence types (B, T) (padding: 0),
    targets (B, T) = class of `states` (-100 = padding), n (B,), final (B,). Padding is at the END only."""
    B, T = len(stories), max(int(st["n"]) for st in stories) + 1
    rows = torch.zeros(B, T, dtype=torch.long)
    types = torch.zeros(B, T, dtype=torch.long)
    tgt = torch.full((B, T), -100, dtype=torch.long)
    for b, st in enumerate(stories):
        for t, (text, c) in enumerate(story_sentences(st)):
            rows[b, t], types[b, t], tgt[b, t] = SROW[text], c, CI[st["states"][t]]
    n = torch.tensor([int(st["n"]) for st in stories])
    final = torch.tensor([CI[st["final"]] for st in stories])
    assert torch.equal(tgt[torch.arange(B), n], final)
    return {k: v.to(DEVICE) for k, v in dict(rows=rows, types=types, tgt=tgt, n=n, final=final).items()}


SEQ = {sp: seq_tensors(STORIES[sp]) for sp in SPLITS}
_valid = SEQ["train"]["tgt"] != -100
# V5: every sentence row read by a sequence model during training is a train-split sentence
assert int(SEQ["train"]["rows"][_valid].max()) < N_TRAIN_SENT, "LEAK: a non-train sentence in the sequence training input"
assert int(_valid.sum()) == len(TRAIN_PREFIXES), "per-step CE set != prefix set"
assert [int(SEQ["train"]["tgt"][i, k]) for (i, k) in TRAIN_PREFIXES] == PREFIX_LABEL


def seq_input(space: str, sp: str):
    """[lab14] (B, T, D_in): R896 / F64 sentence vectors, or SYM = one-hot (13) of the true sentence type."""
    if space == "SYM":
        return F.one_hot(SEQ[sp]["types"], N_SCLS).float()
    return SPACE_X[space][SEQ[sp]["rows"]]


def bag_features(init_c: int, events) -> list:
    """[lab14] BAG input: one-hot init class (8) + counts of IN, OUT, BREAK, FIX, NOOP (5)."""
    v = [0.0] * (N_INIT + len(EVENTS))
    v[init_c] = 1.0
    for e in events:
        v[N_INIT + EVENTS.index(e)] += 1.0
    return v


X_BAG_TRAIN = _t([bag_features(init_cls(STORIES["train"][i]), STORIES["train"][i]["events"][:k])
                  for (i, k) in TRAIN_PREFIXES]).float()
Y_PREFIX = _t(PREFIX_LABEL)
X_QWEN_TRAIN = H_WHOLE_TRAIN[_t(PREFIX_WROW)]            # (N_prefixes, 896): one row per (story, k), with multiplicity
assert X_BAG_TRAIN.shape[0] == X_QWEN_TRAIN.shape[0] == Y_PREFIX.shape[0] == len(TRAIN_PREFIXES)
print(f"[l14] training: {len(STORIES['train'])} stories = {len(TRAIN_PREFIXES)} prefixes / steps "
      f"({len(WHOLE_TRAIN)} unique prefix texts)", flush=True)


# ==============================================================================
# [4] Models
# ==============================================================================
class Hop(nn.Module):
    """[lab14] HOP (issue #23): the state h (unit sphere, D_state) is its ONLY memory.
    h_t = normalize(W2 GELU(W1 [h_{t-1}; proj(x_t)] + b1) + b2); logits_t = h_t A^T / tau (A = normalised anchors).
    symbolic=True: proj(x_t) is replaced by x_t itself (the one-hot sentence type)."""

    def __init__(self, d_in: int, d_state: int, symbolic: bool = False):
        super().__init__()
        self.proj = nn.Identity() if symbolic else nn.Linear(d_in, d_state)
        d_x = d_in if symbolic else d_state
        self.cell = nn.Sequential(nn.Linear(d_state + d_x, HOP_HIDDEN), nn.GELU(), nn.Linear(HOP_HIDDEN, d_state))
        self.h0 = nn.Parameter(torch.randn(d_state))
        self.anchors = nn.Parameter(torch.randn(N_CLS, d_state))

    def forward(self, x: torch.Tensor, mode: str = "free", return_fed: bool = False):
        """x (B, T, d_in) -> logits (B, T, 9) at every step. mode 'free': h is fed back as is; 'snap': h is replaced by
        its nearest anchor before it is fed back. return_fed: also the (B, T, D) states that were fed back."""
        if mode not in ("free", "snap"):
            raise ValueError(mode)
        B, T, _ = x.shape
        px = self.proj(x)
        A = F.normalize(self.anchors, p=2, dim=-1)
        h = F.normalize(self.h0, p=2, dim=-1).unsqueeze(0).expand(B, -1)
        logits, fed = [], []
        for t in range(T):
            h = F.normalize(self.cell(torch.cat([h, px[:, t]], dim=-1)), p=2, dim=-1)
            lg = (h @ A.T) / CE_TAU
            logits.append(lg)
            if mode == "snap":
                h = A[lg.argmax(dim=-1)]  # nearest anchor (unit vectors: largest cosine = largest logit)
            fed.append(h)
        out = torch.stack(logits, dim=1)
        return (out, torch.stack(fed, dim=1)) if return_fed else out


class OneStepTF(nn.Module):
    """[lab14] ONE-STEP-TF: Linear(896, 64), 2 causal transformer layers (Lab 12 law_brain layer), NO positional
    embeddings (order only through the causal mask), Linear(64, 9) at every position."""

    def __init__(self, d_in: int = 896, d_model: int = 64):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        # (copied from Lab 12 DualHead) d_model 64, nhead 4, ff 128, norm_first; PyTorch default dropout 0.1
        enc_l = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=128, batch_first=True, norm_first=True)
        self.law_brain = nn.TransformerEncoder(enc_l, num_layers=2, enable_nested_tensor=False)
        self.readout = nn.Linear(d_model, N_CLS)

    def forward(self, x: torch.Tensor):
        T = x.shape[1]
        c_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)  # (copied from Lab 12)
        return self.readout(self.law_brain(self.proj(x), mask=c_mask))


def n_params(m) -> int:
    return sum(p.numel() for p in m.parameters())


def train_seq(tag: str, build, X: torch.Tensor, tgt: torch.Tensor):
    """[lab14] HOP / SYMBOLIC / ONE-STEP-TF: CE at every valid step (mean over all valid (story, step) pairs), AdamW
    1e-3 -> 1e-4 cosine, SEQ_STEPS full batch, clip 1.0, wd 0.01. HOP trains with the free state (BPTT)."""
    seed_all(SEED)  # before model init
    model = build().to(DEVICE).float()
    n_par = n_params(model)
    valid = tgt != -100
    print(f"\n=== [4] Training {tag}: {n_par:,} params | {X.shape[0]} stories / {int(valid.sum())} steps | "
          f"{SEQ_STEPS} steps full batch | AdamW lr {SEQ_LR:g} -> {SEQ_LR_FLOOR:g} cosine | wd {SEQ_WD:g} | clip 1.0 ===",
          flush=True)
    t1 = time.time()
    seed_all(SEED)  # before training run (dropout in ONE-STEP-TF)
    opt = torch.optim.AdamW(model.parameters(), lr=SEQ_LR, weight_decay=SEQ_WD)
    model.train()
    for step in range(1, SEQ_STEPS + 1):
        for pg in opt.param_groups:
            pg['lr'] = cosine_lr(step, SEQ_STEPS, SEQ_LR, SEQ_LR_FLOOR)
        opt.zero_grad()
        if isinstance(model, Hop):  # HOP / SYMBOLIC: free mode; fed-back states = the free states h_t
            logits, H = model(X, return_fed=True)
            loss = F.cross_entropy(logits.reshape(-1, N_CLS), tgt.reshape(-1), ignore_index=-100)
            # [amendment 2026-11-01] anchor alignment: pull h_t onto its class anchor, so that snapping at inference
            # feeds back inputs of the kind seen in training (CE alone leaves h far from every anchor)
            _A = F.normalize(model.anchors, p=2, dim=-1)
            _cos = (H * _A[tgt.clamp(min=0)]).sum(-1)
            loss = loss + HOP_ALIGN * (1.0 - _cos[valid]).mean()
        else:
            logits = model(X)
            loss = F.cross_entropy(logits.reshape(-1, N_CLS), tgt.reshape(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in SEQ_LOG_STEPS:
            hit = (logits.argmax(-1) == tgt)[valid]
            msg = f"CE {loss.item():.4f} | train step acc {int(hit.sum())}/{hit.numel()} (train mode)"
            LOSS_LOG.append((tag, step, msg))
            print(f"  [{tag}] step {step:4d}/{SEQ_STEPS} | {msg}", flush=True)
    model.eval()
    print(f"  [{tag}] trained in {time.time()-t1:.1f}s", flush=True)
    return model, n_par


# (copied from Lab 12)
class MLP(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 256), nn.GELU(), nn.Linear(256, 64), nn.GELU(), nn.Linear(64, n_out))

    def forward(self, x):
        return self.net(x)


def train_mlp(tag: str, X: torch.Tensor, y: torch.Tensor, n_out: int, steps: int = MLP_STEPS):
    """(copied from Lab 12; [lab14] `steps` argument (Lab 12: HEAD_STEPS = 400), log steps 1 / half / end, returns the
    param count too)"""
    seed_all(SEED)  # before model init
    m = MLP(X.shape[1], n_out).to(DEVICE).float()
    n_par = sum(p.numel() for p in m.parameters())
    print(f"\n=== [4] Training {tag}: {n_par:,} params | {X.shape[0]} examples | {steps} steps full batch | "
          f"AdamW lr {HEAD_LR:g} -> {HEAD_LR_FLOOR:g} cosine | clip 1.0 ===", flush=True)
    seed_all(SEED)  # before training run
    opt = torch.optim.AdamW(m.parameters(), lr=HEAD_LR, weight_decay=0.01)
    m.train()
    for step in range(1, steps + 1):
        for pg in opt.param_groups:
            pg['lr'] = cosine_lr(step, steps, HEAD_LR, HEAD_LR_FLOOR)
        opt.zero_grad()
        logits = m(X)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step in (1, steps // 2, steps):
            acc = float((logits.argmax(-1) == y).float().mean().item())
            msg = f"CE {loss.item():.4f} | train acc {acc:.3f}"
            LOSS_LOG.append((tag, step, msg))
            print(f"  [{tag}] step {step:4d}/{steps} | {msg}", flush=True)
    m.eval()
    return m, n_par


@torch.no_grad()
def check_architectures():
    """[lab14] Architecture checks on freshly built models and TRAIN inputs, BEFORE any training (a failure is a code bug
    and costs nothing; every model is re-seeded before its own init, so this does not change any result). Tolerances
    are loose (1e-4) because a real violation gives O(1) differences."""
    seed_all(SEED)
    hop = Hop(H_DIM, 896).to(DEVICE).float().eval()
    rows = (torch.arange(16 * 11, device=DEVICE) % N_TRAIN_SENT).view(16, 11)  # train sentence rows only, no padding
    X = X_R896[rows]                                                           # (16, 11, 896)
    ls, fed_s = hop(X, mode="snap", return_fed=True)
    _lf, fed_f = hop(X, mode="free", return_fed=True)
    A = F.normalize(hop.anchors, p=2, dim=-1)
    # snap: every fed-back state IS an anchor, namely the arg-max of that step's logits; free: it is not
    assert torch.equal(fed_s, A[ls.argmax(-1)]), "snap must feed back the nearest anchor"
    assert not torch.allclose(fed_f, A[_lf.argmax(-1)], atol=1e-4), "free must feed back the state itself"
    # ONE-STEP-TF is causal: changing the inputs from position 2 on leaves positions 0..1 unchanged
    tf = OneStepTF(H_DIM, 64).to(DEVICE).float().eval()
    X2 = X.clone()
    X2[:, 2:] = X_R896[(rows[:, 2:] + 1) % N_TRAIN_SENT]                     # other train sentences from position 2 on
    o1, o2 = tf(X), tf(X2)
    assert torch.allclose(o1[:, :2], o2[:, :2], atol=1e-4, rtol=1e-4), "ONE-STEP-TF must be causal"
    assert not torch.allclose(o1[:, 2:], o2[:, 2:], atol=1e-4, rtol=1e-4), "ONE-STEP-TF must read later inputs"
    # ... and has no positional information: a constant input sequence gives the same output at every position
    assert not any("pos" in k for k, _ in tf.named_parameters()), "ONE-STEP-TF must have no position table"
    oc = tf(X[:, :1].expand(-1, 11, -1).contiguous())
    assert torch.allclose(oc, oc[:, :1].expand_as(oc), atol=1e-4, rtol=1e-4), "ONE-STEP-TF must have no positions"
    print("[l14] architecture checks passed (fresh models, train inputs): snap feeds back exactly the nearest anchor, "
          "free the state itself; ONE-STEP-TF is causal and position-free", flush=True)


check_architectures()


# ==============================================================================
# [5] Training (train split only)
# ==============================================================================
HOP_VARIANTS = [("HOP-R896-d896", "R896", 896), ("HOP-R896-d64", "R896", 64),
                ("HOP-F64-d64", "F64", 64), ("HOP-F64-d896", "F64", 896)]
SEQ_MODELS = {}   # method -> (model, input space)
PARAMS = {"FUNNEL-F64": FUNNEL_PARAMS}
for _name, _space, _d in HOP_VARIANTS:
    _din = SPACE_X[_space].shape[1]
    SEQ_MODELS[_name] = (train_seq(_name, lambda: Hop(_din, _d), seq_input(_space, "train"), SEQ["train"]["tgt"])[0],
                         _space)
SEQ_MODELS["SYMBOLIC"] = (train_seq("SYMBOLIC", lambda: Hop(N_SCLS, 64, symbolic=True), seq_input("SYM", "train"),
                                    SEQ["train"]["tgt"])[0], "SYM")
SEQ_MODELS["ONE-STEP-TF"] = (train_seq("ONE-STEP-TF", lambda: OneStepTF(H_DIM, 64), seq_input("R896", "train"),
                                       SEQ["train"]["tgt"])[0], "R896")
for _name, (_m, _s) in SEQ_MODELS.items():
    PARAMS[_name] = n_params(_m)
assert not any("pos" in k for k, _ in SEQ_MODELS["ONE-STEP-TF"][0].named_parameters()), "ONE-STEP-TF has no positions"

QWEN_MLP, PARAMS["ONE-STEP-QWEN"] = train_mlp("ONE-STEP-QWEN (whole-text Qwen feature, every prefix)",
                                              X_QWEN_TRAIN, Y_PREFIX, N_CLS)
BAG_MLP, PARAMS["BAG"] = train_mlp("BAG (init one-hot + event counts, every prefix)", X_BAG_TRAIN, Y_PREFIX, N_CLS)


def majority(labels) -> int:
    """[lab14] most frequent class; ties -> lowest class index."""
    cnt = Counter(labels)
    return max(range(N_CLS), key=lambda c: (cnt[c], -c))


MAJ_CLS = majority(PREFIX_LABEL)
_last_lab = defaultdict(list)
for (i, k), y in zip(TRAIN_PREFIXES, PREFIX_LABEL):
    st = STORIES["train"][i]
    _last_lab[(init_cls(st), st["events"][k - 1] if k > 0 else None)].append(y)
LAST_TABLE = {key: majority(v) for key, v in _last_lab.items()}
print(f"\n[l14] MAJORITY = {CLASSES[MAJ_CLS]} | LAST lookup: {len(LAST_TABLE)} (init class, last event) keys",
      flush=True)


def last_predict(st, k: int) -> int:
    return LAST_TABLE.get((init_cls(st), st["events"][k - 1] if k > 0 else None), MAJ_CLS)


with torch.no_grad():  # [lab14] reproducibility fingerprint (exact float sums; compare re-runs on the same hardware)
    _fp = lambda m: float(sum(p.double().sum() for p in m.state_dict().values()))
    print(f"[l14] fingerprint seed={SEED}: H_SENT={float(H_SENT.double().sum()).hex()} "
          f"H_WHOLE_TRAIN={float(H_WHOLE_TRAIN.double().sum()).hex()} funnel={_fp(FUNNEL).hex()} "
          + " ".join(f"{k}={_fp(m).hex()}" for k, (m, _s) in SEQ_MODELS.items())
          + f" qwen_mlp={_fp(QWEN_MLP).hex()} bag={_fp(BAG_MLP).hex()}", flush=True)


# ==============================================================================
# [6] Evaluation
# ==============================================================================
METHODS = [("HOP-R896-d896", "free"), ("HOP-R896-d896", "snap"), ("HOP-R896-d64", "free"), ("HOP-R896-d64", "snap"),
           ("HOP-F64-d64", "free"), ("HOP-F64-d64", "snap"), ("HOP-F64-d896", "free"), ("HOP-F64-d896", "snap"),
           ("ONE-STEP-TF", "-"), ("ONE-STEP-QWEN", "-"), ("SYMBOLIC", "free"), ("SYMBOLIC", "snap"),
           ("BAG", "-"), ("LAST", "-"), ("MAJORITY", "-")]
STEP_METHODS = [mm for mm in METHODS if mm[0].startswith("HOP-") or mm[0] == "SYMBOLIC"]
RES = {mm: {} for mm in METHODS}   # (method, mode) -> {split: {n: (correct, total)}, "fit": (correct, total)}
STEP = {mm: {} for mm in STEP_METHODS}  # (method, mode) -> {t: (correct, total)}


def per_n(pred_final, sp: str) -> dict:
    """pred_final (B,) predicted final class of every story of split sp -> {n: (correct, total)}."""
    hit = (pred_final.to(DEVICE) == SEQ[sp]["final"])
    n = SEQ[sp]["n"]
    return {k: (int(hit[n == k].sum()), int((n == k).sum())) for k in ALL_N}


@torch.no_grad()
def seq_logits(model, space: str, sp: str, mode: str):
    model.eval()
    X = seq_input(space, sp)
    return model(X, mode=mode) if isinstance(model, Hop) else model(X)


for (meth, mode) in METHODS:
    r = RES[(meth, mode)]
    if meth in SEQ_MODELS:
        model, space = SEQ_MODELS[meth]
        mode_arg = mode if isinstance(model, Hop) else None
        for sp in SPLITS:
            pred = seq_logits(model, space, sp, mode_arg).argmax(dim=-1)        # (B, T): prediction at every step
            if sp == "train":
                valid = SEQ[sp]["tgt"] != -100
                hit = (pred == SEQ[sp]["tgt"])[valid]
                r["fit"] = (int(hit.sum()), hit.numel())
                continue
            B = pred.shape[0]
            r[sp] = per_n(pred[torch.arange(B, device=DEVICE), SEQ[sp]["n"]], sp)
            if sp == "test" and (meth, mode) in STEP:
                sel = SEQ[sp]["n"] == STEP_N
                for t in range(STEP_N + 1):
                    h = pred[sel, t] == SEQ[sp]["tgt"][sel, t]
                    STEP[(meth, mode)][t] = (int(h.sum()), h.numel())
    elif meth in ("ONE-STEP-QWEN", "BAG"):
        m = QWEN_MLP if meth == "ONE-STEP-QWEN" else BAG_MLP
        with torch.no_grad():
            Xtr = X_QWEN_TRAIN if meth == "ONE-STEP-QWEN" else X_BAG_TRAIN
            hit = m(Xtr).argmax(-1) == Y_PREFIX
            r["fit"] = (int(hit.sum()), hit.numel())
            for sp in EVAL_SPLITS:
                if meth == "ONE-STEP-QWEN":
                    Xe = H_WHOLE_EVAL[sp][_t(STORY_WROW[sp])]
                else:
                    Xe = _t([bag_features(init_cls(st), st["events"]) for st in STORIES[sp]]).float()
                r[sp] = per_n(m(Xe).argmax(-1), sp)
    elif meth == "LAST":
        hit = [last_predict(STORIES["train"][i], k) == y for (i, k), y in zip(TRAIN_PREFIXES, PREFIX_LABEL)]
        r["fit"] = (sum(hit), len(hit))
        for sp in EVAL_SPLITS:
            r[sp] = per_n(_t([last_predict(st, int(st["n"])) for st in STORIES[sp]]), sp)
    elif meth == "MAJORITY":
        r["fit"] = (sum(int(y == MAJ_CLS) for y in PREFIX_LABEL), len(PREFIX_LABEL))
        for sp in EVAL_SPLITS:
            r[sp] = per_n(torch.full((len(STORIES[sp]),), MAJ_CLS), sp)
    else:
        raise ValueError(meth)

for (meth, mode) in METHODS:  # every method is evaluated on the same stories
    for sp in EVAL_SPLITS:
        assert [RES[(meth, mode)][sp][n][1] for n in ALL_N] == [RES[METHODS[0]][sp][n][1] for n in ALL_N]


# ==============================================================================
# [7] Report (numbers only; the pre-registered reading rules are applied by the reader / aggregator)
# ==============================================================================
def frac(ct) -> float:
    return ct[0] / ct[1]


def mean_over(r: dict, ns) -> float:
    return sum(frac(r[n]) for n in ns) / len(ns)


def pooled(r: dict, ns):
    return sum(r[n][0] for n in ns), sum(r[n][1] for n in ns)


def summ(r: dict):
    return mean_over(r, SHORT_N), frac(r[N5]), mean_over(r, LONG_N)


print("\n" + "=" * 150, flush=True)
for sp in EVAL_SPLITS:
    tot = [RES[METHODS[0]][sp][n][1] for n in ALL_N]
    print(f"=== RESULTS seed={SEED} split={sp} ({'unseen domains' if sp == 'test' else 'seen domains'}): final-state "
          f"accuracy (%) per n | stories per n: {tot} | SHORT = mean n=1-4, LONG = mean n=6-10 ===", flush=True)
    hdr = f"{'method':<16}{'mode':<6}" + "".join(f"{f'n={n}':>7}" for n in ALL_N) + f"{'SHORT':>8}{'n5':>7}{'LONG':>8}"
    print(hdr + "\n" + "-" * len(hdr), flush=True)
    for (meth, mode) in METHODS:
        r = RES[(meth, mode)][sp]
        s, f5, lg = summ(r)
        print(f"{meth:<16}{mode:<6}" + "".join(f"{100 * frac(r[n]):7.1f}" for n in ALL_N)
              + f"{100 * s:8.1f}{100 * f5:7.1f}{100 * lg:8.1f}", flush=True)
    print("", flush=True)

print(f"=== per-step state accuracy (%) on the test stories with n = {STEP_N} (t = events consumed; t = 0 = after the "
      f"init sentence) ===", flush=True)
hdr = f"{'method':<16}{'mode':<6}" + "".join(f"{f't={t}':>7}" for t in range(STEP_N + 1))
print(hdr + "\n" + "-" * len(hdr), flush=True)
for mm in STEP_METHODS:
    print(f"{mm[0]:<16}{mm[1]:<6}" + "".join(f"{100 * frac(STEP[mm][t]):7.1f}" for t in range(STEP_N + 1)), flush=True)

print(f"\n=== training fit (accuracy over all {len(TRAIN_PREFIXES)} training prefixes / steps, eval mode) ===", flush=True)
for mm in METHODS:
    c, t = RES[mm]["fit"]
    print(f"  {mm[0]:<16}{mm[1]:<6}{100 * c / t:7.2f}% ({c}/{t})", flush=True)

print("\n=== parameters ===", flush=True)
for k, v in PARAMS.items():
    print(f"  {k:<16}{v:>12,}", flush=True)
print(f"  {'LAST':<16}{len(LAST_TABLE):>12} lookup entries (+ MAJORITY fallback)", flush=True)


def vline(mm, sp, ns) -> str:
    r = RES[mm][sp]
    c, t = pooled(r, ns)
    return f"{mm[0]}/{mm[1]} = {100 * mean_over(r, ns):.2f}% ({c}/{t})"


print("\n=== validity numbers (issue #23; numbers only) ===", flush=True)
print(f"  V1 val_seen n=1-4 (mean over n; pooled count): {vline(('HOP-R896-d896', 'free'), 'val_seen', SHORT_N)} | "
      f"{vline(('ONE-STEP-TF', '-'), 'val_seen', SHORT_N)}", flush=True)
print(f"  V2 test LONG: {vline(('BAG', '-'), 'test', LONG_N)}", flush=True)
print(f"  V3 blind judge: external, not computed by this script", flush=True)
print(f"  V4 val_seen LONG: {vline(('SYMBOLIC', 'snap'), 'val_seen', LONG_N)}", flush=True)
print(f"  V5 leak check: passed (asserted): {len(FORBIDDEN_SENT)} test sentences / {len(WHOLE_EVAL['test'])} test "
      f"story texts absent from {N_TRAIN_SENT} train sentences and {len(WHOLE_TRAIN)} training prefix texts; sequence "
      f"training inputs read train sentence rows only; mu / funnel use train sentences only", flush=True)

print(f"\n=== Training losses (train mode) ===", flush=True)
for tag, step, msg in LOSS_LOG:
    print(f"  {tag:<56} step {step:4d} | {msg}", flush=True)

print("", flush=True)
for sp in EVAL_SPLITS:
    for (meth, mode) in METHODS:
        for n in ALL_N:
            c, t = RES[(meth, mode)][sp][n]
            print(f"L14_ACC seed={SEED} split={sp} method={meth} mode={mode} n={n} correct={c} total={t}", flush=True)
for sp in EVAL_SPLITS:
    for (meth, mode) in METHODS:
        s, f5, lg = summ(RES[(meth, mode)][sp])
        print(f"L14_SUM seed={SEED} split={sp} method={meth} mode={mode} short={s:.4f} n5={f5:.4f} long={lg:.4f}",
              flush=True)
for (meth, mode) in STEP_METHODS:
    for t in range(STEP_N + 1):
        c, tot = STEP[(meth, mode)][t]
        print(f"L14_STEP seed={SEED} method={meth} mode={mode} t={t} correct={c} total={tot}", flush=True)
for (meth, mode) in METHODS:
    c, t = RES[(meth, mode)]["fit"]
    print(f"L14_FIT seed={SEED} method={meth} mode={mode} acc={c / t:.4f} correct={c} total={t}", flush=True)
for k, v in PARAMS.items():
    print(f"L14_PARAMS method={k} params={v}", flush=True)
print(f"[l14] total time {time.time()-T_START:.1f}s", flush=True)
print(f"L14_CONFIG seed={SEED} data_sha={DATA_SHA256} script_sha={SCRIPT_SHA256} env={ENV_TAG}", flush=True)
