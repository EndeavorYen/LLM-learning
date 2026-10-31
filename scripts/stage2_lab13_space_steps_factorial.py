# ==============================================================================
# Stage 2 / Lab 13: state space x inference factorial on analogy v5 (EXPLORATORY; numbers only, NO verdict)
# Question: the Lab 12 gap HOP (BR 85.0% / DEC 91.2%) vs PREMISE-MLP (BR 88.9% / DEC 97.9%) mixes three factors:
#   (a) state space: supervised 64-d funnel vs raw 896-d Qwen features; (b) inference: 3-step rollout vs one
#   decision; (c) PREMISE-MLP lets Qwen read s0+s1 jointly as ONE text, the others use per-sentence vectors.
#   Lab 13 separates (a) from (b) with per-sentence vectors in every factorial cell; PREMISE-MLP is the reference for (c).
#   It also calibrates stress tests for later: input noise (T1), state perturbation during rollout (T2), long rollout (T3).
# Data: benchmarks/analogy_v5.json (sha256-pinned; 20 seen domains, 8 unseen; 48 unseen items = domain x law).
# State spaces, all from the same frozen per-sentence Qwen mean-pooled 896-d features H; anchors = normalised class
# means of the seen sentences in that space (27 state classes):
#   F64   Lab 12 stage-A funnel s (supervised, 64-d, unit norm)
#   P64   normalize(PCA_64(H - mu_seen)); PCA fitted on the centred seen-sentence features (unsupervised, no labels)
#   R896  normalize(H - mu_seen); mu_seen = mean of the seen-sentence features (no training)
# Methods, every one trained on the 20 seen domains only:
#   HOP          Lab 12 V4 recipe (2000 steps, CE over the 27 anchors of the space, tau 0.1, pos embeddings, AdamW
#                1e-3 -> 1e-4 cosine, clip 1.0, wd 0.01, Lab 12 seed_all placement) on the 120 seen chains of the space;
#                free rollout [x0, x1] -> x2 -> x3 -> x4.
#                F64 = Lab 12's head exactly. P64 = the same head (D = 64, no projections).
#                R896 (primary) = learned in_proj 896->64 + the same core (d_model 64) + next_law_proj 64->896,
#                output normalised; the 896-d output is what is fed back.
#                R896 HOP-wide (secondary) = d_model 896, nhead 8, ff 1792, 2 layers, next_law_proj 896->896.
#                The 64-d core (law_brain + pos) of F64 / P64 / R896 starts from the same seeded init.
#   ONE-STEP     MLP on concat[x0, x1] (2D -> 256 -> 64 -> 6 laws, GELU), Lab 12 train_mlp (400 steps).
#   X1-ONLY      MLP on x1 only: DEC exactly 50% (stage 1 is shared within a pair) -- asserted in every space.
#   PREMISE-MLP  exactly Lab 12 (Qwen feature of the joint text s0+s1 -> MLP), reference for factor (c).
# Metrics on the 48 unseen items, k = 2, 3, 4:
#   DEC-k  (primary, picker-free, Lab 12 definition) HOP: cos(x̂k, anchor(a,k)) vs cos(x̂k, anchor(b,k)) in its own
#          space; ONE-STEP / X1-ONLY / PREMISE-MLP: logit of a vs b. Pair order (a, b); ties -> a.
#   BR-k   the method's DEC-decided law at stage k -> Lab 12's sentence-MLP picker (mlp_picker): the same picker
#          for every method.
#   BRa-k  (HOP, secondary) anchor-mediated BR in the hop's own space = Lab 12's HOP BR.
#   A-k    (HOP) nearest of the 27 anchors to x̂k is (L, k).
#   seenDEC-k  DEC on the 120 seen training chains / premises (training fit).
# Stress tests on the 48 unseen items. One CPU torch.Generator seeded with AUDIT_SEED; all draws are made up front
# (T1 then T2) and every method gets identical draws; K = 5 draws per level; mean counts with 2 decimals.
#   T1 input noise: H' = H + alpha * sigma ⊙ eps on the rows of the test inputs s0, s1 and the unseen premises
#      (sigma = per-dimension std of the seen-sentence features). Noise is drawn per SENTENCE ROW, so the shared s1
#      of a pair gets the same draw for both of its items (X1-ONLY stays exactly 50%; asserted). Every space is
#      recomputed from H' (funnel(H'), centre+normalise(H'), PCA(H')); anchors stay fixed. Draw k uses the same eps at
#      every alpha (common random numbers). alpha in {0, 0.25, 0.5, 1, 2}; alpha = 0 must equal the clean result.
#   T2 state perturbation (HOP): after step 2, x̂2' = normalize(x̂2 + beta * m * u), fed back; DEC3, DEC4.
#      m = mean Euclidean distance from each anchor to its nearest other anchor in the space. u per item:
#      iso = uniform random unit vector in the full space; span = random unit vector in the span of the 27 anchors.
#      64-d spaces use the first 64 coordinates of the same 896-d Gaussian draw; R896 / R896-wide share all draws.
#   T3 long rollout (HOP, descriptive): steps 2..12 with a sliding window of the last 4 states (see rollout_long);
#      per step: mean cosine to the nearest anchor; fraction of items whose nearest anchor still belongs to the law
#      decided by DEC-2 (shared stage-1 classes belong to both laws of their pair).
# Machine lines: L13_RESULT / L13_NOISE / L13_PERT / L13_LONG / L13_CONFIG (+ L13_LOSS). -1 = n/a.
#   Aggregate: python tools/aggregate_lab13.py seed42.log seed1.log seed2.log
# Run:  AUDIT_SEED=42 python stage2_lab13_space_steps_factorial.py   (seeds 42, 1, 2)
#   On CUDA with AUDIT_SEED in {42, 1, 2}, F64 HOP (BR = BRa, DEC, A) and the PREMISE-MLP / common-picker numbers are
#   ASSERTED equal to results/analogy_v5/seed*.log (Colab L4). On CPU the comparison is printed, not asserted.
# Copied code is marked "(copied from Lab 12)"; "[lab13]" marks new or changed code.
# ==============================================================================
import os, sys, time, math, random, json, hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

SEED = int(os.environ.get("AUDIT_SEED", "42"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[l13] SEED={SEED} | device={DEVICE}", flush=True)
T_START = time.time()
# (copied from Lab 12; [lab13] log prefix) environment record (reproducibility)
print(f"[l13] python {sys.version.split()[0]} | torch {torch.__version__} | numpy {np.__version__} | "
      f"transformers {getattr(transformers, '__version__', '?')} ({getattr(transformers, '__file__', '?')})", flush=True)
try:
    with open(os.path.abspath(__file__), "rb") as _f:
        SCRIPT_SHA256 = hashlib.sha256(_f.read()).hexdigest()
except (NameError, OSError):
    SCRIPT_SHA256 = "unavailable"  # e.g. executed through a notebook kernel; the runner records the repo file's sha
print(f"[l13] script sha256={SCRIPT_SHA256}", flush=True)
ENV_TAG = f"torch={torch.__version__},transformers={getattr(transformers, '__version__', '?')},device={DEVICE}"
if DEVICE == "cuda":
    ENV_TAG += f",gpu={torch.cuda.get_device_name(0).replace(' ', '_')}"


def seed_all(seed: int = SEED):
    """(copied from Lab 09/11)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(step: int, total: int, peak: float, floor: float) -> float:
    """Copied from Lab 01 (via Lab 09/11): cosine schedule from peak (step 0) to floor (step == total)."""
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * step / total))


# (copied from Lab 11)
VARIANTS = {  # name: (pos embeddings, hop loss); V0 = Lab 09 joint training
    "V0": dict(pos=False, loss="cos"),
    "V1": dict(pos=False, loss="cos"),
    "V2": dict(pos=True, loss="cos"),
    "V3": dict(pos=False, loss="ce"),
    "V4": dict(pos=True, loss="ce"),
}
MODES = ("free", "snap")

# [lab13] HOP recipe = Lab 12's pre-registered HOP (HOP_VARIANT = "V4", HOP_MODE = "free"); no overrides in Lab 13
VARIANT, MODE = "V4", "free"
CFG = VARIANTS[VARIANT]
print(f"[l13] HOP recipe = Lab 12 {VARIANT} (pos={CFG['pos']}, loss={CFG['loss']}), {MODE} rollout", flush=True)


# ==============================================================================
# [0] Data (copied from Lab 12)
# ==============================================================================
_DATA_CANDIDATES = ["/content/analogy_v5.json", "benchmarks/analogy_v5.json"]
try:
    _DATA_CANDIDATES.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks", "analogy_v5.json"))
except NameError:
    pass
DATA_PATH = next((p for p in _DATA_CANDIDATES if os.path.exists(p)), None)
if DATA_PATH is None:
    raise FileNotFoundError(f"analogy_v5.json not found in {_DATA_CANDIDATES}")
with open(DATA_PATH, "rb") as f:  # [lab12] read bytes once: parse + sha256 of the exact file used
    _DATA_RAW = f.read()
DATA = json.loads(_DATA_RAW.decode("utf-8"))
DATA_SHA256 = hashlib.sha256(_DATA_RAW).hexdigest()
print(f"[v5] data = {DATA_PATH} ({DATA.get('version')}) sha256={DATA_SHA256}", flush=True)
assert DATA.get("version") == "analogy_v5", DATA.get("version")  # [lab12]
# [review S3] pin the exact file that passed the lexical check and the blind judgement (144/144)
EXPECTED_DATA_SHA256 = "d2a9e877595829b4ff1560be90cd04739acdb11be782b14a067c4b9c4d24f196"
assert DATA_SHA256 == EXPECTED_DATA_SHA256, f"analogy_v5.json is not the registered file: {DATA_SHA256}"

N_LAWS, N_STAGES, EVAL_STAGES = 6, 5, (2, 3, 4)
assert len(DATA["laws"]) == N_LAWS
LAW_NAMES = [l["name"] for l in DATA["laws"]]
PAIRS = [tuple(int(x) for x in p) for p in DATA["pairs"]]
assert len(PAIRS) == 3 and sorted(x for p in PAIRS for x in p) == list(range(N_LAWS)), PAIRS
PAIR_OF, PARTNER = {}, {}
for _p, (_a, _b) in enumerate(PAIRS):
    PAIR_OF[_a] = PAIR_OF[_b] = _p
    PARTNER[_a], PARTNER[_b] = _b, _a
for _i, _l in enumerate(DATA["laws"]):
    assert int(_l["id"]) == _i and int(_l["pair"]) == PAIR_OF[_i], f"law entry {_i} inconsistent with pairs"

# ---- 27 state classes: per pair (a, b): (a,0) (b,0) (pair,1) (a,2) (b,2) (a,3) (b,3) (a,4) (b,4) ----
CLASS_KEYS = []  # ("L", law, stage) or ("P", pair, 1)
for _p, (_a, _b) in enumerate(PAIRS):
    CLASS_KEYS += [("L", _a, 0), ("L", _b, 0), ("P", _p, 1)]
    for _k in EVAL_STAGES:
        CLASS_KEYS += [("L", _a, _k), ("L", _b, _k)]
N_CLS = len(CLASS_KEYS)
assert N_CLS == 27
CLS_INDEX = {k: i for i, k in enumerate(CLASS_KEYS)}


def cls_of(law: int, st: int) -> int:
    return CLS_INDEX[("P", PAIR_OF[law], 1)] if st == 1 else CLS_INDEX[("L", law, st)]


def law_of_class(c: int) -> int:
    """Law implied by a state class. Shared stage-1 classes: fixed tie rule = first law of the pair."""
    kind, idx, _st = CLASS_KEYS[c]
    return PAIRS[idx][0] if kind == "P" else idx


def clbl(c) -> str:
    if c is None:
        return "-"
    kind, idx, st = CLASS_KEYS[int(c)]
    return f"({'L' if kind == 'L' else 'P'}{idx},S{st})"


STAGE0_CLS = [cls_of(l, 0) for l in range(N_LAWS)]  # the 6 stage-0 anchors (copied from Lab 09)


def check_domain(dom):
    for law in range(N_LAWS):
        ch = dom["chains"][str(law)]
        assert len(ch) == N_STAGES and all(isinstance(t, str) and t for t in ch), f"{dom['name']} L{law}: bad chain"
    for (a, b) in PAIRS:
        if dom["chains"][str(a)][1] != dom["chains"][str(b)][1]:
            raise RuntimeError(f"{dom['name']}: stage-1 sentences of pair ({a},{b}) differ")


SEEN_DOMS = [d["name"] for d in DATA["seen_domains"]]
UNSEEN_DOMS = [d["name"] for d in DATA["unseen_domains"]]
N_SEEN_DOM, N_UNSEEN_DOM = len(SEEN_DOMS), len(UNSEEN_DOMS)
assert N_SEEN_DOM == 20 and N_UNSEEN_DOM == 8, (N_SEEN_DOM, N_UNSEEN_DOM)  # [lab12] v5: 20 seen + 8 unseen
assert len(set(SEEN_DOMS) | set(UNSEEN_DOMS)) == N_SEEN_DOM + N_UNSEEN_DOM, "domain names must be unique"  # [lab12]

ALL_TEXTS = []  # every text that gets a Qwen feature (shared stage-1 strings stored once per pair per domain)


def add_text(t: str) -> int:
    ALL_TEXTS.append(t)
    return len(ALL_TEXTS) - 1


def index_domain(dom):
    """-> {(law, stage): global idx}; the shared stage-1 sentence of a pair gets ONE global idx."""
    check_domain(dom)
    idx = {}
    for (a, b) in PAIRS:
        idx[(a, 1)] = idx[(b, 1)] = add_text(dom["chains"][str(a)][1])
    for law in range(N_LAWS):
        for st in (0, 2, 3, 4):
            idx[(law, st)] = add_text(dom["chains"][str(law)][st])
    return idx


SEEN = []      # (global idx, seen domain, state class)
SEEN_POS = {}  # (seen domain, law, stage) -> row in SEEN (stage 1 -> the shared row)
for di, dom in enumerate(DATA["seen_domains"]):
    gi = index_domain(dom)
    row_of_g = {}
    for (law, st), g in sorted(gi.items(), key=lambda kv: kv[1]):
        if g not in row_of_g:
            row_of_g[g] = len(SEEN)
            SEEN.append((g, di, cls_of(law, st)))
        SEEN_POS[(di, law, st)] = row_of_g[g]
N_SEEN_SENT = len(SEEN)
assert N_SEEN_SENT == N_SEEN_DOM * N_CLS, N_SEEN_SENT
SEEN_CLS_LIST = [c for (_g, _d, c) in SEEN]
assert all(SEEN_CLS_LIST.count(c) == N_SEEN_DOM for c in range(N_CLS))  # one sentence per class per seen domain

UNSEEN_G = {}  # (unseen domain, law, stage) -> global idx
for ui, dom in enumerate(DATA["unseen_domains"]):
    for (law, st), g in index_domain(dom).items():
        UNSEEN_G[(ui, law, st)] = g
ITEMS = [(ui, law) for ui in range(N_UNSEEN_DOM) for law in range(N_LAWS)]
N_ITEMS = len(ITEMS)
assert N_ITEMS == N_UNSEEN_DOM * N_LAWS == 48  # [lab12]


def premise_text(dom, law: int) -> str:
    return dom["chains"][str(law)][0] + dom["chains"][str(law)][1]


PREM_SEEN = []  # (global idx, seen domain, law): N_SEEN_DOM x N_LAWS seen premises s0+s1
for di, dom in enumerate(DATA["seen_domains"]):
    for law in range(N_LAWS):
        PREM_SEEN.append((add_text(premise_text(dom, law)), di, law))
PREM_UNSEEN_G = {}
for ui, dom in enumerate(DATA["unseen_domains"]):
    for law in range(N_LAWS):
        PREM_UNSEEN_G[(ui, law)] = add_text(premise_text(dom, law))

FORBIDDEN = {ALL_TEXTS[g] for g in UNSEEN_G.values()} | {ALL_TEXTS[g] for g in PREM_UNSEEN_G.values()}


def assert_not_forbidden(text: str):
    """No unseen-domain sentence (or unseen premise) may ever enter a loss."""
    if text in FORBIDDEN:
        raise RuntimeError(f"LEAK: unseen sentence in training data: {text!r}")
    for t in FORBIDDEN:
        if t in text:
            raise RuntimeError(f"LEAK: unseen sentence inside training text: {t!r}")


# Hop chains: every seen domain x law, positions 0..4 (stage 1 -> the shared row)
HOP_ROWS = [[SEEN_POS[(di, law, st)] for st in range(N_STAGES)] for di in range(N_SEEN_DOM) for law in range(N_LAWS)]
assert len(HOP_ROWS) == N_SEEN_DOM * N_LAWS == 120  # [lab12]

# Leak check over EVERY training text: seen sentences (funnel + sentence MLP + hop chains), seen premises (premise MLP)
_train_texts = [ALL_TEXTS[g] for (g, _d, _c) in SEEN] + [ALL_TEXTS[g] for (g, _d, _l) in PREM_SEEN]
_train_texts += ["".join(ALL_TEXTS[SEEN[r][0]] for r in row) for row in HOP_ROWS]
for _t in _train_texts:
    assert_not_forbidden(_t)
print(f"[v5] seen sentences={N_SEEN_SENT} ({N_SEEN_DOM} domains x {N_CLS} state classes) | seen premises={len(PREM_SEEN)} | "
      f"hop chains={len(HOP_ROWS)} | unseen items={N_ITEMS} | leak check passed for all {len(_train_texts)} training texts",
      flush=True)

# Per item: candidate laws in fixed pair order (a, b) -- NOT (L, partner), so ties never favour the true law
CAND_LAWS = [PAIRS[PAIR_OF[law]] for (_ui, law) in ITEMS]

# Seen-domain items: same order as PREM_SEEN (domain, law) and as HOP_ROWS (copied from Lab 11)
SEEN_ITEMS = [(di, law) for di in range(N_SEEN_DOM) for law in range(N_LAWS)]
N_SEEN_ITEMS = len(SEEN_ITEMS)
assert N_SEEN_ITEMS == N_SEEN_DOM * N_LAWS == 120 and [(d, l) for (_g, d, l) in PREM_SEEN] == SEEN_ITEMS  # [lab12]
assert all(HOP_ROWS[i][st] == SEEN_POS[(di, law, st)] for i, (di, law) in enumerate(SEEN_ITEMS) for st in range(N_STAGES))
# Hop-training targets: state class of positions 1..4 of every seen chain
HOP_TGT_CLS_LIST = [[SEEN_CLS_LIST[r] for r in row[1:]] for row in HOP_ROWS]
assert all(HOP_TGT_CLS_LIST[i][st - 1] == cls_of(law, st) for i, (_di, law) in enumerate(SEEN_ITEMS) for st in range(1, N_STAGES))

# ---- [lab13] leak check over the Lab 13 training inputs that Lab 12 did not have -----------------------------------
# Lab 12's check above covers every seen sentence (stage-A funnel, sentence-MLP picker, the hop chains of EVERY space,
# and the seen-sentence statistics mu_seen / PCA / sigma / anchors) and every seen premise (PREMISE-MLP). Lab 13 adds
# the per-sentence premise pairs (s0, s1) of ONE-STEP and the s1 inputs of X1-ONLY. The row-level check that no
# feature row feeding a loss or a statistic is an unseen text is in section [1].
_l13_texts = ["".join(ALL_TEXTS[SEEN[SEEN_POS[(di, law, st)]][0]] for st in (0, 1)) for (di, law) in SEEN_ITEMS]
_l13_texts += [ALL_TEXTS[SEEN[SEEN_POS[(di, law, 1)]][0]] for (di, law) in SEEN_ITEMS]
for _tx in _l13_texts:
    assert_not_forbidden(_tx)
print(f"[l13] leak check passed for the {len(_l13_texts)} additional Lab 13 training inputs "
      f"(ONE-STEP s0+s1 pairs, X1-ONLY s1)", flush=True)


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


@torch.no_grad()
def extract_qwen_896_feature(text_list):
    '''Copied from Lab 01: frozen Qwen last hidden state, mean-pooled, float32.'''
    enc = tokenizer(text_list, return_tensors="pt", padding=True, truncation=True, max_length=64).to(DEVICE)
    out = qwen_base.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
    last_h = out.last_hidden_state.float()
    mask_f = enc.attention_mask.unsqueeze(-1).float()
    pooled_896 = (last_h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
    return pooled_896  # (B, 896) float32


# [review N3] unseen-domain texts and training texts are never in the same padded batch, so no training feature
# depends on an unseen text even at bf16 rounding level (padding length).
_FORBID_IDX = [i for i, t in enumerate(ALL_TEXTS) if t in FORBIDDEN]
_TRAIN_IDX = [i for i, t in enumerate(ALL_TEXTS) if t not in FORBIDDEN]
H_ALL = torch.empty(len(ALL_TEXTS), H_DIM, dtype=torch.float32, device=DEVICE)
for _grp in (_TRAIN_IDX, _FORBID_IDX):
    for i in range(0, len(_grp), 64):
        _ix = _grp[i:i + 64]
        H_ALL[torch.tensor(_ix, device=DEVICE)] = extract_qwen_896_feature([ALL_TEXTS[j] for j in _ix])
assert H_ALL.shape == (len(ALL_TEXTS), H_DIM) and len(_FORBID_IDX) + len(_TRAIN_IDX) == len(ALL_TEXTS)
assert bool(torch.isfinite(H_ALL).all())
SEEN_G = torch.tensor([g for (g, _d, _c) in SEEN], device=DEVICE)
H_SEEN = H_ALL[SEEN_G]  # (N_SEEN_SENT, 896): the only sentence features that enter a head / sentence-MLP loss
PREM_SEEN_G = torch.tensor([g for (g, _d, _l) in PREM_SEEN], device=DEVICE)
H_PREM_SEEN = H_ALL[PREM_SEEN_G]  # (N_SEEN_ITEMS, 896): the only premise features that enter the premise-MLP loss
print(f"[v5] extracted {H_ALL.shape[0]} features ({H_DIM}-d, float32)", flush=True)

SEEN_CLS = torch.tensor(SEEN_CLS_LIST, device=DEVICE)
DOM_OF = torch.tensor([d for (_g, d, _c) in SEEN], device=DEVICE)
HOP_IDX = torch.tensor(HOP_ROWS, device=DEVICE)  # (N_SEEN_ITEMS, 5) rows into SEEN
CLS_OH = F.one_hot(SEEN_CLS, N_CLS).float()      # (N_SEEN_SENT, 27)


def _t(lst):
    return torch.tensor(lst, device=DEVICE)


# Item index tensors (N_ITEMS,) (copied from Lab 11)
IT_S0 = _t([UNSEEN_G[(ui, law, 0)] for (ui, law) in ITEMS])
IT_S0_FLIP = _t([UNSEEN_G[(ui, PARTNER[law], 0)] for (ui, law) in ITEMS])
IT_S1 = _t([UNSEEN_G[(ui, law, 1)] for (ui, law) in ITEMS])
IT_CAND = {k: _t([[UNSEEN_G[(ui, cl, k)] for cl in CAND_LAWS[i]] for i, (ui, _law) in enumerate(ITEMS)])
           for k in EVAL_STAGES}  # (N_ITEMS, 2)
IT_LAW = [law for (_ui, law) in ITEMS]
IT_TRUE_CLS = {k: _t([cls_of(law, k) for law in IT_LAW]) for k in EVAL_STAGES}
CAND_LAWS_T = _t([list(c) for c in CAND_LAWS])  # (N_ITEMS, 2)
IT_LAW_T = _t(IT_LAW)
# [lab12] DEC-k: the two anchor classes (a, k), (b, k) of the item's pair, in fixed pair order (N_ITEMS, 2);
# stage 0 is used by S0-RULE; stage 1 is the shared class (identical columns) -> LAST has no branch information
IT_DEC_CLS = {k: _t([[cls_of(c, k) for c in CAND_LAWS[i]] for i in range(N_ITEMS)]) for k in (0, 1) + EVAL_STAGES}
assert torch.equal(IT_DEC_CLS[1][:, 0], IT_DEC_CLS[1][:, 1])
assert all(bool((IT_DEC_CLS[k][:, 0] != IT_DEC_CLS[k][:, 1]).all()) for k in (0,) + EVAL_STAGES)

# Seen-domain item tensors (N_SEEN_ITEMS,): rows into SEEN (copied from Lab 11)
SI_S0 = _t([SEEN_POS[(di, law, 0)] for (di, law) in SEEN_ITEMS])
SI_S1 = _t([SEEN_POS[(di, law, 1)] for (di, law) in SEEN_ITEMS])
SI_LAW_T = _t([law for (_di, law) in SEEN_ITEMS])
SI_CL = _t([list(PAIRS[PAIR_OF[law]]) for (_di, law) in SEEN_ITEMS])  # (N_SEEN_ITEMS, 2) candidate laws, pair order
SI_CAND = {k: _t([[SEEN_POS[(di, c, k)] for c in PAIRS[PAIR_OF[law]]] for (di, law) in SEEN_ITEMS])
           for k in EVAL_STAGES}  # (N_SEEN_ITEMS, 2) rows into SEEN
HOP_TGT_CLS = _t(HOP_TGT_CLS_LIST)  # (N_SEEN_ITEMS, 4)
assert torch.equal(HOP_TGT_CLS, SEEN_CLS[HOP_IDX[:, 1:]])

# ---- [lab13] extra index tensors ----------------------------------------------------------------------------------
# seen-chain DEC-k: anchors (a, k), (b, k) of the seen item's pair, pair order (N_SEEN_ITEMS, 2)
SI_DEC_CLS = {k: _t([[cls_of(c, k) for c in PAIRS[PAIR_OF[law]]] for (_di, law) in SEEN_ITEMS]) for k in EVAL_STAGES}
assert torch.equal(SI_DEC_CLS[2], SEEN_CLS[SI_CAND[2]])
PM_ROWS = _t([PREM_UNSEEN_G[it] for it in ITEMS])  # (N_ITEMS,) rows of the unseen premises (PREMISE-MLP inputs)
# CLASS_HAS_LAW[c, l]: state class c belongs to law l ((L, l, st), or the shared stage-1 class of l's pair) -- T3
CLASS_HAS_LAW = torch.zeros(N_CLS, N_LAWS, dtype=torch.bool, device=DEVICE)
for _c, (_kind, _idx, _st) in enumerate(CLASS_KEYS):
    for _l in (PAIRS[_idx] if _kind == "P" else (_idx,)):
        CLASS_HAS_LAW[_c, _l] = True
assert int(CLASS_HAS_LAW.sum()) == N_CLS + len(PAIRS)

# [lab13] row-level leak check: every feature row that feeds a loss or a fitted statistic (stage-A funnel, hop chains
# of every space, ONE-STEP / X1-ONLY inputs, sentence-MLP picker, PREMISE-MLP, mu_seen, PCA, sigma, anchors) is a seen
# row, and no seen row is an unseen-domain text.
_ROWS_SEEN = set(SEEN_G.tolist())
assert not (_ROWS_SEEN | set(PREM_SEEN_G.tolist())) & set(_FORBID_IDX), "LEAK: a training row is an unseen text"
assert set(SEEN_G[HOP_IDX].flatten().tolist()) <= _ROWS_SEEN
assert set(SEEN_G[SI_S0].tolist()) | set(SEEN_G[SI_S1].tolist()) <= _ROWS_SEEN
# ... and every row the stress tests perturb is an unseen-domain row (test inputs only)
NOISE_ROWS = _t(sorted(set(IT_S0.tolist()) | set(IT_S1.tolist()) | set(PM_ROWS.tolist())))
assert set(NOISE_ROWS.tolist()) <= set(_FORBID_IDX)
print(f"[l13] row-level leak check passed | T1 noise rows (unseen s0, s1, premises) = {len(NOISE_ROWS)}", flush=True)

# (copied from Lab 11)
HEAD_STEPS, HEAD_LR, HEAD_LR_FLOOR = 400, 1e-3, 1e-4   # Lab 09 (V0 and stage A)
HOP_STEPS, HOP_LR, HOP_LR_FLOOR = 2000, 1e-3, 1e-4     # Lab 11 hop-only training (V1-V4)
HOP_LOG_STEPS = (1, 500, 1000, 2000)
CE_TAU = 0.1
LOSS_LOG = []  # (tag, step, description)


# ==============================================================================
# [2] Funnel + law_brain head with optional learned positional embeddings (copied from Lab 12)
# ==============================================================================
class DualHead(nn.Module):
    def __init__(self, qwen_hidden: int = 896, d_dim: int = 64, s_dim: int = 64, pos: bool = False,
                 max_len: int = N_STAGES):
        super().__init__()
        self.domain_funnel = nn.Sequential(nn.Linear(qwen_hidden, 256), nn.GELU(), nn.Linear(256, d_dim))
        self.struct_funnel = nn.Sequential(nn.Linear(qwen_hidden, 256), nn.GELU(), nn.Linear(256, s_dim))
        enc_l = nn.TransformerEncoderLayer(d_model=s_dim, nhead=4, dim_feedforward=128, batch_first=True, norm_first=True)
        self.law_brain = nn.TransformerEncoder(enc_l, num_layers=2, enable_nested_tensor=False)
        self.next_law_proj = nn.Linear(s_dim, s_dim)
        # [lab11] created LAST, so pos=False consumes the RNG exactly like Lab 09's DualHead
        if pos:
            self.pos = nn.Parameter(torch.empty(max_len, s_dim))
            nn.init.normal_(self.pos, mean=0.0, std=0.02)
        else:
            self.register_parameter("pos", None)

    def extract_ds(self, h_896: torch.Tensor):
        d = F.normalize(self.domain_funnel(h_896), p=2, dim=-1)
        s = F.normalize(self.struct_funnel(h_896), p=2, dim=-1)
        return d, s

    def hop_law(self, s_seq: torch.Tensor):
        B, T, _ = s_seq.shape
        if self.pos is not None:
            s_seq = s_seq + self.pos[:T].unsqueeze(0)  # only the first T rows
        c_mask = torch.triu(torch.ones(T, T, device=s_seq.device, dtype=torch.bool), diagonal=1)
        h = self.law_brain(s_seq, mask=c_mask)
        return F.normalize(self.next_law_proj(h), p=2, dim=-1)


def class_anchors(s_seen: torch.Tensor) -> torch.Tensor:
    """(copied from Lab 11) (N_SEEN_SENT, dim) seen s -> (27, dim) normalised class means (the 27 anchors)."""
    return F.normalize(CLS_OH.T @ s_seen / CLS_OH.sum(0).unsqueeze(-1), p=2, dim=-1)


DOM_OH = F.one_hot(DOM_OF, N_SEEN_DOM).float()
OFF_C = ~torch.eye(N_CLS, device=DEVICE, dtype=torch.bool)
OFF_D = ~torch.eye(N_SEEN_DOM, device=DEVICE, dtype=torch.bool)


def funnel_loss(head):
    """(copied from Lab 11) Lab 09 loss_dis (s pull/push with closest-pair term, d pull/push). Returns (loss_dis, parts, s);
    parts = the four component tensors (formatted with fmt_parts only at logging steps)."""
    d, s = head.extract_ds(H_SEEN)
    s_mean = F.normalize(CLS_OH.T @ s / CLS_OH.sum(0).unsqueeze(-1), p=2, dim=-1)  # (27, 64)
    d_mean = F.normalize(DOM_OH.T @ d / DOM_OH.sum(0).unsqueeze(-1), p=2, dim=-1)  # (N_SEEN_DOM, 64)
    l_s_pull = (1.0 - (s * s_mean[SEEN_CLS]).sum(-1)).mean()
    _sp = F.relu((s_mean @ s_mean.T)[OFF_C] + 0.1)
    l_s_push = _sp.mean() + _sp.max()  # closest-pair term (from v3)
    l_d_pull = (1.0 - (d * d_mean[DOM_OF]).sum(-1)).mean()
    l_d_push = F.relu((d_mean @ d_mean.T)[OFF_D] + 0.1).mean()
    loss_dis = l_s_pull + l_s_push + l_d_pull + l_d_push
    return loss_dis, (l_s_pull, l_s_push, l_d_pull, l_d_push), s


def fmt_parts(parts) -> str:
    l_s_pull, l_s_push, l_d_pull, l_d_push = parts
    return (f"s_pull {l_s_pull.item():.3f} s_push {l_s_push.item():.3f} "
            f"d_pull {l_d_pull.item():.3f} d_push {l_d_push.item():.3f}")


def hop_loss(pred, tgt_vec, anchors, kind: str):
    """(copied from Lab 11) pred / tgt_vec (N_SEEN_ITEMS, 4, dim); kind 'cos' = Lab 09 cosine regression,
    'ce' = CE over the 27 anchors (tau 0.1)."""
    if kind == "cos":
        return (1.0 - (pred * tgt_vec).sum(-1)).mean()
    if kind == "ce":
        logits = (pred @ anchors.T) / CE_TAU  # (N_SEEN_ITEMS, 4, 27)
        return F.cross_entropy(logits.reshape(-1, N_CLS), HOP_TGT_CLS.reshape(-1))
    raise ValueError(kind)


@torch.no_grad()
def hop_metrics(pred, tgt_vec, anchors) -> str:
    """(copied from Lab 11) Teacher-forced diagnostics on the seen chains: cosine loss, CE loss, next-state class accuracy."""
    cos_l = float((1.0 - (pred * tgt_vec).sum(-1)).mean().item())
    logits = (pred @ anchors.T) / CE_TAU
    ce_l = float(F.cross_entropy(logits.reshape(-1, N_CLS), HOP_TGT_CLS.reshape(-1)).item())
    hit = logits.argmax(-1) == HOP_TGT_CLS  # (N_SEEN_ITEMS, 4): targets at positions 1..4
    return (f"cos={cos_l:.4f} ce={ce_l:.4f} acc={int(hit.sum().item())}/{hit.numel()} "
            f"acc234={int(hit[:, 1:].sum().item())}/{hit[:, 1:].numel()}")


def train_funnel_stage_a(tag: str = "stageA"):
    """(copied from Lab 11) Stage A: funnel only (loss_dis, no hop), Lab 09's 400-step schedule. law_brain stays at its init."""
    seed_all(SEED)  # before model init (same init as V0's head)
    head = DualHead(qwen_hidden=H_DIM, d_dim=64, s_dim=64).to(DEVICE).float()
    params = list(head.domain_funnel.parameters()) + list(head.struct_funnel.parameters())
    n_par = sum(p.numel() for p in params)
    print(f"\n=== [2] Training {tag} (funnel only, no hop): {n_par:,} trainable params | {HEAD_STEPS} steps full batch | "
          f"AdamW lr {HEAD_LR:g} -> {HEAD_LR_FLOOR:g} cosine | clip 1.0 ===", flush=True)
    t1 = time.time()
    seed_all(SEED)  # before training run
    opt = torch.optim.AdamW(params, lr=HEAD_LR, weight_decay=0.01)
    head.train()
    for step in range(1, HEAD_STEPS + 1):
        for pg in opt.param_groups:
            pg['lr'] = cosine_lr(step, HEAD_STEPS, HEAD_LR, HEAD_LR_FLOOR)
        opt.zero_grad()
        loss_dis, parts, _s = funnel_loss(head)
        loss_dis.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step in (1, 200, 400):
            msg = f"dis {loss_dis.item():.4f} ({fmt_parts(parts)})"
            LOSS_LOG.append((tag, step, msg))
            print(f"  [{tag}] step {step:3d}/{HEAD_STEPS} | {msg}", flush=True)
    head.eval()
    print(f"  [{tag}] trained in {time.time()-t1:.1f}s", flush=True)
    return head


def build_variant_head(stage_a, pos: bool):
    """(copied from Lab 11) Fresh head (same init as stage A; + pos table if asked), stage-A weights loaded, funnel frozen."""
    seed_all(SEED)  # before model init: funnel/law_brain init identical to stage A's (pos is created last)
    head = DualHead(qwen_hidden=H_DIM, d_dim=64, s_dim=64, pos=pos).to(DEVICE).float()
    res = head.load_state_dict(stage_a.state_dict(), strict=False)
    assert list(res.unexpected_keys) == [] and list(res.missing_keys) == (["pos"] if pos else []), res
    for m in (head.domain_funnel, head.struct_funnel):
        for p in m.parameters():
            p.requires_grad = False
    return head


def train_hop(head, tag: str, kind: str):
    """(copied from Lab 11; HL_LOSS line renamed V5_LOSS; [lab13] renamed L13_LOSS) Stage B: law_brain + next_law_proj
    (+ pos) alone on the frozen s of the seen chains."""
    with torch.no_grad():
        _d, s = head.extract_ds(H_SEEN)   # frozen funnel -> fixed s
        anchors = class_anchors(s)         # 27 frozen anchors
    s_ch = s[HOP_IDX]                      # (N_SEEN_ITEMS, 5, 64)
    x, tgt = s_ch[:, :-1, :], s_ch[:, 1:, :]
    params = [p for p in head.parameters() if p.requires_grad]
    expect = list(head.law_brain.parameters()) + list(head.next_law_proj.parameters()) + \
        ([head.pos] if head.pos is not None else [])
    assert {id(p) for p in params} == {id(p) for p in expect}, "trainable set must be law_brain + next_law_proj (+ pos)"
    n_par = sum(p.numel() for p in params)
    print(f"\n=== [2] Training {tag} hop (funnel frozen; loss={kind}{f', tau {CE_TAU:g}' if kind == 'ce' else ''}; "
          f"pos={'yes' if head.pos is not None else 'no'}): {n_par:,} trainable params | {HOP_STEPS} steps full batch | "
          f"AdamW lr {HOP_LR:g} -> {HOP_LR_FLOOR:g} cosine | clip 1.0 ===", flush=True)
    t1 = time.time()
    seed_all(SEED)  # before hop training (dropout in law_brain)
    opt = torch.optim.AdamW(params, lr=HOP_LR, weight_decay=0.01)
    head.train()
    for step in range(1, HOP_STEPS + 1):
        for pg in opt.param_groups:
            pg['lr'] = cosine_lr(step, HOP_STEPS, HOP_LR, HOP_LR_FLOOR)
        opt.zero_grad()
        pred = head.hop_law(x)             # from positions 0..3, predicts positions 1..4
        loss = hop_loss(pred, tgt, anchors, kind)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step in HOP_LOG_STEPS:
            met = hop_metrics(pred.detach(), tgt, anchors)
            LOSS_LOG.append((tag, step, f"hop loss ({kind}) {loss.item():.4f} | {met}"))
            print(f"  [{tag}] step {step:4d}/{HOP_STEPS} | hop loss ({kind}) {loss.item():.4f} | {met}", flush=True)
            print(f"L13_LOSS seed={SEED} model={tag} step={step} loss={loss.item():.4f} {met}", flush=True)
    head.eval()
    print(f"  [{tag}] trained in {time.time()-t1:.1f}s", flush=True)
    return head


@torch.no_grad()
def eval_mode_hop_metrics(head) -> str:
    """(copied from Lab 11) Teacher-forced metrics on the seen chains in eval mode (dropout off)."""
    head.eval()
    _d, s = head.extract_ds(H_SEEN)
    s_ch = s[HOP_IDX]
    return hop_metrics(head.hop_law(s_ch[:, :-1, :]), s_ch[:, 1:, :], class_anchors(s))


# ---- [lab13] hop heads for the other spaces ------------------------------------------------------------------------
class ProjHop(nn.Module):
    """[lab13] R896 HOP (primary): learned in_proj (896 -> 64), then Lab 12's hop core (pos + law_brain, d_model 64,
    nhead 4, ff 128, 2 layers, causal), then next_law_proj (64 -> 896); output normalised. During rollout the 896-d
    output is what is fed back (through in_proj again)."""

    def __init__(self, x_dim: int = 896, s_dim: int = 64, max_len: int = N_STAGES):
        super().__init__()
        enc_l = nn.TransformerEncoderLayer(d_model=s_dim, nhead=4, dim_feedforward=128, batch_first=True, norm_first=True)
        self.law_brain = nn.TransformerEncoder(enc_l, num_layers=2, enable_nested_tensor=False)
        self.next_law_proj = nn.Linear(s_dim, x_dim)
        self.pos = nn.Parameter(torch.empty(max_len, s_dim))
        nn.init.normal_(self.pos, mean=0.0, std=0.02)
        self.in_proj = nn.Linear(x_dim, s_dim)

    def hop_law(self, x_seq: torch.Tensor):
        B, T, _ = x_seq.shape
        h = self.in_proj(x_seq) + self.pos[:T].unsqueeze(0)  # positional embeddings live in the 64-d core
        c_mask = torch.triu(torch.ones(T, T, device=x_seq.device, dtype=torch.bool), diagonal=1)
        h = self.law_brain(h, mask=c_mask)
        return F.normalize(self.next_law_proj(h), p=2, dim=-1)


class WideHop(nn.Module):
    """[lab13] R896 HOP-wide (secondary): Lab 12's hop architecture at d_model 896 (nhead 8, ff 1792, 2 layers,
    norm_first, causal, pos table 5 x 896 with std 0.02), next_law_proj 896 -> 896, no projections."""

    def __init__(self, x_dim: int = 896, nhead: int = 8, ff: int = 1792, max_len: int = N_STAGES):
        super().__init__()
        enc_l = nn.TransformerEncoderLayer(d_model=x_dim, nhead=nhead, dim_feedforward=ff, batch_first=True, norm_first=True)
        self.law_brain = nn.TransformerEncoder(enc_l, num_layers=2, enable_nested_tensor=False)
        self.next_law_proj = nn.Linear(x_dim, x_dim)
        self.pos = nn.Parameter(torch.empty(max_len, x_dim))
        nn.init.normal_(self.pos, mean=0.0, std=0.02)

    def hop_law(self, s_seq: torch.Tensor):  # the same computation as DualHead.hop_law
        B, T, _ = s_seq.shape
        s_seq = s_seq + self.pos[:T].unsqueeze(0)
        c_mask = torch.triu(torch.ones(T, T, device=s_seq.device, dtype=torch.bool), diagonal=1)
        h = self.law_brain(s_seq, mask=c_mask)
        return F.normalize(self.next_law_proj(h), p=2, dim=-1)


def hop_param_list(head):
    """[lab13] The hop's trainable parameters: law_brain + next_law_proj + pos (+ in_proj for ProjHop)."""
    ps = list(head.law_brain.parameters()) + list(head.next_law_proj.parameters()) + [head.pos]
    if isinstance(head, ProjHop):
        ps += list(head.in_proj.parameters())
    return ps


def n_hop_params(head) -> int:
    return sum(p.numel() for p in hop_param_list(head))


def hop_core_init_ref():
    """[lab13] law_brain / next_law_proj / pos of a freshly seeded DualHead(pos=True) = the init of Lab 12's HOP
    (build_variant_head: stage A leaves law_brain / next_law_proj at the seeded init, and pos is created last)."""
    seed_all(SEED)
    ref = DualHead(qwen_hidden=H_DIM, d_dim=64, s_dim=64, pos=True)
    return {k: v.detach().clone().to(DEVICE) for k, v in ref.state_dict().items() if "funnel" not in k}


def build_p64_hop():
    """[lab13] P64 HOP: the same head as F64 (DualHead, pos, d_model 64, no projections), built with the same seeding
    as build_variant_head, so its hop core starts from the identical init. The funnels are unused (P64 states come
    from the PCA) and frozen."""
    seed_all(SEED)  # before model init (as build_variant_head)
    head = DualHead(qwen_hidden=H_DIM, d_dim=64, s_dim=64, pos=CFG["pos"]).to(DEVICE).float()
    for m in (head.domain_funnel, head.struct_funnel):
        for p in m.parameters():
            p.requires_grad = False
    return head


def build_r896_hop():
    """[lab13] R896 HOP: law_brain + pos copied from the seeded reference init (= F64 / P64 core init); in_proj and
    next_law_proj initialised from the RNG stream right after the reference head."""
    core = hop_core_init_ref()  # seeds
    head = ProjHop(x_dim=H_DIM, s_dim=64).to(DEVICE).float()
    head.law_brain.load_state_dict({k[len("law_brain."):]: v for k, v in core.items() if k.startswith("law_brain.")})
    with torch.no_grad():
        head.pos.copy_(core["pos"])
    return head


def build_wide_hop():
    """[lab13] R896 HOP-wide: seeded before init (Lab 12 placement)."""
    seed_all(SEED)
    return WideHop(x_dim=H_DIM).to(DEVICE).float()


def core_matches_ref(head, ref) -> bool:
    """[lab13] law_brain (+ pos; + next_law_proj when it has the 64 -> 64 shape) equal to the reference init."""
    sd = head.state_dict()
    keys = [k for k in ref if k.startswith("law_brain.") or k == "pos"]
    if sd["next_law_proj.weight"].shape == ref["next_law_proj.weight"].shape:
        keys += ["next_law_proj.weight", "next_law_proj.bias"]
    return all(torch.equal(sd[k], ref[k]) for k in keys)


def train_hop_x(head, x_seen: torch.Tensor, tag: str, kind: str):
    """[lab13] Adapted from Lab 12 train_hop: the identical loop (seeding, AdamW, cosine schedule, clipping, logging);
    the frozen states of the seen sentences in the space are passed in as x_seen (N_SEEN_SENT, D) instead of
    head.extract_ds(H_SEEN), and the trainable set is the hop (hop_param_list). Its equivalence to train_hop is
    checked bit for bit on F64 in section [5]."""
    with torch.no_grad():
        s = x_seen                         # frozen states of the seen sentences in this space
        anchors = class_anchors(s)         # 27 frozen anchors of this space
    s_ch = s[HOP_IDX]                      # (N_SEEN_ITEMS, 5, D)
    x, tgt = s_ch[:, :-1, :], s_ch[:, 1:, :]
    params = [p for p in head.parameters() if p.requires_grad]
    expect = hop_param_list(head)
    assert {id(p) for p in params} == {id(p) for p in expect}, "trainable set must be the hop parameters only"
    n_par = sum(p.numel() for p in params)
    print(f"\n=== [2] Training {tag} hop (states frozen; loss={kind}{f', tau {CE_TAU:g}' if kind == 'ce' else ''}; "
          f"pos=yes; D={s.shape[1]}): {n_par:,} trainable params | {HOP_STEPS} steps full batch | "
          f"AdamW lr {HOP_LR:g} -> {HOP_LR_FLOOR:g} cosine | clip 1.0 ===", flush=True)
    t1 = time.time()
    seed_all(SEED)  # before hop training (dropout in law_brain)
    opt = torch.optim.AdamW(params, lr=HOP_LR, weight_decay=0.01)
    head.train()
    for step in range(1, HOP_STEPS + 1):
        for pg in opt.param_groups:
            pg['lr'] = cosine_lr(step, HOP_STEPS, HOP_LR, HOP_LR_FLOOR)
        opt.zero_grad()
        pred = head.hop_law(x)             # from positions 0..3, predicts positions 1..4
        loss = hop_loss(pred, tgt, anchors, kind)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step in HOP_LOG_STEPS:
            met = hop_metrics(pred.detach(), tgt, anchors)
            LOSS_LOG.append((tag, step, f"hop loss ({kind}) {loss.item():.4f} | {met}"))
            print(f"  [{tag}] step {step:4d}/{HOP_STEPS} | hop loss ({kind}) {loss.item():.4f} | {met}", flush=True)
            print(f"L13_LOSS seed={SEED} model={tag} step={step} loss={loss.item():.4f} {met}", flush=True)
    head.eval()
    print(f"  [{tag}] trained in {time.time()-t1:.1f}s", flush=True)
    return head


@torch.no_grad()
def tf_metrics_x(head, x_seen) -> str:
    """[lab13] eval_mode_hop_metrics for given seen states (teacher-forced, dropout off)."""
    head.eval()
    s_ch = x_seen[HOP_IDX]
    return hop_metrics(head.hop_law(s_ch[:, :-1, :]), s_ch[:, 1:, :], class_anchors(x_seen))


# ==============================================================================
# [3] Evaluation helpers (copied from Lab 12)
# ==============================================================================
@torch.no_grad()
def rollout(head, s0, s1, snap=None):
    """(copied from Lab 11) [s0, s1] -> hop 1/2/3 times feeding its own predictions -> {2: ŝ2, 3: ŝ3, 4: ŝ4}, each (B, dim).
    snap = (27, dim) anchors: each predicted vector is replaced by its nearest anchor before it is fed back
    (and before it is returned for picking). snap=None = Lab 09 free rollout."""
    seq = torch.stack([s0, s1], dim=1)  # (B, 2, dim)
    hats = {}
    for k in EVAL_STAGES:
        nxt = head.hop_law(seq)[:, -1, :]
        if snap is not None:
            nxt = snap[(nxt @ snap.T).argmax(dim=-1)]
        hats[k] = nxt
        seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)
    return hats


def pick_by(target, cand, cand_laws=None):
    """(copied from Lab 11) target (B, dim), cand (B, 2, dim) -> picked law per item (B,) (pair order; ties -> first law
    of pair). [lab12] DEC-k uses the same function with cand = the two anchors (a, k), (b, k)."""
    cl = CAND_LAWS_T if cand_laws is None else cand_laws
    j = (cand @ target.unsqueeze(-1)).squeeze(-1).argmax(dim=-1)  # (B,)
    return cl.gather(1, j.unsqueeze(1)).squeeze(1)


SCORE_KEYS = ("BR2", "BR3", "BR4", "DEC2", "DEC3", "DEC4", "A2", "A3", "A4", "FLIP2", "FLIP3", "FLIP4", "PREM")


def new_score():
    """[lab12] -1 = not applicable."""
    return {k: -1 for k in SCORE_KEYS}


@torch.no_grad()
def head_space(head):
    """(copied from Lab 11)"""
    _d, S = head.extract_ds(H_ALL)
    anchors = F.normalize(CLS_OH.T @ S[SEEN_G] / CLS_OH.sum(0).unsqueeze(-1), p=2, dim=-1)  # (27, dim)
    return S, anchors


def n_correct(pred_law) -> int:
    return int((pred_law == IT_LAW_T).sum().item())


@torch.no_grad()
def eval_hop(head, mode: str):
    """Hop method (HOP / RANDOM HOP / V0). Adapted from Lab 11 eval_head (BR-k, A-k, FLIP-k, seen-domain BR,
    copied) + [lab12] DEC-k and per-item picks."""
    assert mode in MODES
    head.eval()
    S, anchors = head_space(head)
    snap = anchors if mode == "snap" else None
    near = lambda v: (v @ anchors.T).argmax(dim=-1)  # (B,) nearest of the 27 anchors
    sc = new_score()
    # unseen items
    s0, s0f, s1 = S[IT_S0], S[IT_S0_FLIP], S[IT_S1]
    hats, hats_f = rollout(head, s0, s1, snap), rollout(head, s0f, s1, snap)
    picks = {"near": {0: near(s0).tolist()}, "br": {}, "dec": {}, "flip": {}}
    for k in EVAL_STAGES:
        cand = S[IT_CAND[k]]  # (N_ITEMS, 2, dim)
        na, naf = near(hats[k]), near(hats_f[k])
        pa, paf = pick_by(anchors[na], cand), pick_by(anchors[naf], cand)
        pdec = pick_by(hats[k], anchors[IT_DEC_CLS[k]])  # [lab12] cos(ŝk, anchor(a,k)) vs cos(ŝk, anchor(b,k)); ties -> a
        sc[f"BR{k}"] = n_correct(pa)
        sc[f"DEC{k}"] = n_correct(pdec)
        sc[f"A{k}"] = int((na == IT_TRUE_CLS[k]).sum().item())
        sc[f"FLIP{k}"] = int((pa != paf).sum().item())
        picks["near"][k], picks["br"][k] = na.tolist(), pa.tolist()
        picks["dec"][k], picks["flip"][k] = pdec.tolist(), paf.tolist()
    # seen-domain items (training chains)
    seen = new_score()
    Ss = S[SEEN_G]
    hats_s = rollout(head, Ss[SI_S0], Ss[SI_S1], snap)
    for k in EVAL_STAGES:
        cand = Ss[SI_CAND[k]]  # (N_SEEN_ITEMS, 2, dim)
        ns = near(hats_s[k])
        seen[f"BR{k}"] = int((pick_by(anchors[ns], cand, SI_CL) == SI_LAW_T).sum().item())
    return dict(score=sc, seen=seen, picks=picks)


# ==============================================================================
# [4] Stage A -> F64 HOP (copied from Lab 12; [lab13] tag HOP-F64, init check)
# ==============================================================================
STAGE_A = train_funnel_stage_a("stageA")
_STAGE_A_FUNNEL = {k: v.detach().clone() for k, v in STAGE_A.state_dict().items() if "funnel" in k}

# [lab12] check: stage A left law_brain / next_law_proj exactly at the seeded init (RANDOM HOP relies on it)
seed_all(SEED)
_REF_INIT = DualHead(qwen_hidden=H_DIM, d_dim=64, s_dim=64).to(DEVICE).float().state_dict()
_sa_sd = STAGE_A.state_dict()
assert set(_sa_sd) == set(_REF_INIT)
assert all(torch.equal(_sa_sd[k], _REF_INIT[k]) for k in _sa_sd if "funnel" not in k), "stage A changed law_brain"
del _REF_INIT, _sa_sd
FUNNEL_PARAMS = sum(p.numel() for m in (STAGE_A.domain_funnel, STAGE_A.struct_funnel) for p in m.parameters())  # [lab13]

HOP_TAG = "HOP-F64"  # [lab13] (Lab 12: f"HOP-{VARIANT}")
hop_head = build_variant_head(STAGE_A, pos=CFG["pos"])
CORE_REF = hop_core_init_ref()  # [lab13]
assert core_matches_ref(hop_head, CORE_REF), "F64 hop init is not the reference core init"  # [lab13]
hop_head = train_hop(hop_head, HOP_TAG, CFG["loss"])
_sd = hop_head.state_dict()
assert all(torch.equal(_sd[k], v) for k, v in _STAGE_A_FUNNEL.items()), f"{HOP_TAG}: funnel changed during hop training"

# (copied from Lab 12) reproducibility fingerprint (exact float sums; compare across re-runs on the same hardware)
with torch.no_grad():
    _fp = lambda m: float(sum(p.double().sum() for p in m.state_dict().values()))
    FINGERPRINT = dict(H_ALL=float(H_ALL.double().sum()).hex(), stageA=_fp(STAGE_A).hex(), hop=_fp(hop_head).hex())
print(f"[l13] fingerprint seed={SEED}: H_ALL={FINGERPRINT['H_ALL']} stageA={FINGERPRINT['stageA']} "
      f"hop={FINGERPRINT['hop']}", flush=True)
EV_HOP12 = eval_hop(hop_head, MODE)  # Lab 12's own evaluation of HOP (copied), for the equivalence checks below


# ==============================================================================
# [5] PREMISE-MLP + per-sentence 27-state picker (copied from Lab 12)
# ==============================================================================
class MLP(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 256), nn.GELU(), nn.Linear(256, 64), nn.GELU(), nn.Linear(64, n_out))

    def forward(self, x):
        return self.net(x)


def train_mlp(tag: str, X: torch.Tensor, y: torch.Tensor, n_out: int):
    """(copied from Lab 11 / Lab 09)"""
    seed_all(SEED)  # before model init
    m = MLP(X.shape[1], n_out).to(DEVICE).float()
    n_par = sum(p.numel() for p in m.parameters())
    print(f"\n=== [6] Training {tag}: {n_par:,} params | {X.shape[0]} examples | {HEAD_STEPS} steps full batch | "
          f"AdamW lr {HEAD_LR:g} -> {HEAD_LR_FLOOR:g} cosine | clip 1.0 ===", flush=True)
    seed_all(SEED)  # before training run
    opt = torch.optim.AdamW(m.parameters(), lr=HEAD_LR, weight_decay=0.01)
    m.train()
    for step in range(1, HEAD_STEPS + 1):
        for pg in opt.param_groups:
            pg['lr'] = cosine_lr(step, HEAD_STEPS, HEAD_LR, HEAD_LR_FLOOR)
        opt.zero_grad()
        logits = m(X)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step in (1, 200, 400):
            acc = float((logits.argmax(-1) == y).float().mean().item())
            msg = f"CE {loss.item():.4f} | train acc {acc:.3f}"
            LOSS_LOG.append((tag, step, msg))
            print(f"  [{tag}] step {step:3d}/{HEAD_STEPS} | {msg}", flush=True)
    m.eval()
    return m


prem_mlp = train_mlp("Premise MLP (6 laws)", H_PREM_SEEN, _t([l for (_g, _d, l) in PREM_SEEN]), N_LAWS)
sent_mlp = train_mlp("Sentence MLP (27 states)", H_SEEN, SEEN_CLS, N_CLS)
with torch.no_grad():
    PM_LOGITS = prem_mlp(H_ALL[_t([PREM_UNSEEN_G[it] for it in ITEMS])])  # (N_ITEMS, 6)  [lab12] logits kept for DEC
    PM_LAW = PM_LOGITS.argmax(-1).tolist()
    pm = new_score()
    pm["PREM"] = sum(int(pl == L) for pl, L in zip(PM_LAW, IT_LAW))
    for k in EVAL_STAGES:
        # [review] log_softmax per candidate: raw logits are not comparable across two different inputs
        lg = F.log_softmax(sent_mlp(H_ALL[IT_CAND[k]]), dim=-1)  # (N_ITEMS, 2, 27)
        col = _t([cls_of(l, k) for l in PM_LAW])                   # (N_ITEMS,)
        sc2 = lg.gather(2, col.view(-1, 1, 1).expand(-1, 2, 1)).squeeze(-1)  # (N_ITEMS, 2): logit of class (pred law, k)
        p = CAND_LAWS_T.gather(1, sc2.argmax(-1, keepdim=True)).squeeze(1)
        pm[f"BR{k}"] = int((p == IT_LAW_T).sum().item())
    # [lab12] DEC: premise-classifier logit of a vs b (pair order; ties -> a); the same decision for every k
    _lab = PM_LOGITS.gather(1, CAND_LAWS_T)  # (N_ITEMS, 2): logits of (a, b)
    PM_DEC = torch.where(_lab[:, 0] >= _lab[:, 1], CAND_LAWS_T[:, 0], CAND_LAWS_T[:, 1])
    for k in EVAL_STAGES:
        pm[f"DEC{k}"] = n_correct(PM_DEC)
    # seen-domain PREMISE-MLP (copied from Lab 11 / Lab 09 seen_domain_br)
    pl_seen = prem_mlp(H_PREM_SEEN).argmax(-1).tolist()  # PREM_SEEN is ordered (domain, law) like SEEN_ITEMS
    PM_SEEN = new_score()
    for k in EVAL_STAGES:
        lg = F.log_softmax(sent_mlp(H_SEEN[SI_CAND[k]]), dim=-1)
        col = _t([cls_of(l, k) for l in pl_seen])
        sc2 = lg.gather(2, col.view(-1, 1, 1).expand(-1, 2, 1)).squeeze(-1)
        PM_SEEN[f"BR{k}"] = int((SI_CL.gather(1, sc2.argmax(-1, keepdim=True)).squeeze(1) == SI_LAW_T).sum().item())


@torch.no_grad()
def mlp_picker(laws):
    """(copied from Lab 09) BR-k when the law is given by `laws` and the candidate is chosen with the sentence MLP
    (log-probs). [lab12] `laws` may also be a dict {k: laws at stage k}."""
    out = new_score()
    for k in EVAL_STAGES:
        lk = laws[k] if isinstance(laws, dict) else laws  # [lab12]
        lg = F.log_softmax(sent_mlp(H_ALL[IT_CAND[k]]), dim=-1)
        col = _t([cls_of(l, k) for l in lk])
        sc2 = lg.gather(2, col.view(-1, 1, 1).expand(-1, 2, 1)).squeeze(-1)
        out[f"BR{k}"] = int((CAND_LAWS_T.gather(1, sc2.argmax(-1, keepdim=True)).squeeze(1) == IT_LAW_T).sum().item())
    return out


# ==============================================================================
# [6] [lab13] Lab 13 metrics: DEC-k, BR-k (common picker), BRa-k, A-k, seen DEC
# ==============================================================================
L13_KEYS = ("DEC2", "DEC3", "DEC4", "BR2", "BR3", "BR4", "BRa2", "BRa3", "BRa4", "A2", "A3", "A4",
            "seenDEC2", "seenDEC3", "seenDEC4", "params")


def new_l13():
    return {k: -1 for k in L13_KEYS}  # -1 = n/a


def law_dec(logits, cand_laws):
    """Lab 12's PREMISE-MLP DEC rule (copied from Lab 12, as a function): logit of a vs b in pair order; ties -> a."""
    lab = logits.gather(1, cand_laws)  # (B, 2): logits of (a, b)
    return torch.where(lab[:, 0] >= lab[:, 1], cand_laws[:, 0], cand_laws[:, 1])


@torch.no_grad()
def hop_dec(hats, anchors, dec_cls=None, cand_laws=None):
    """Lab 12 hop DEC-k: cos(x̂k, anchor(a,k)) vs cos(x̂k, anchor(b,k)) in the hop's own space; ties -> a."""
    dc = IT_DEC_CLS if dec_cls is None else dec_cls
    return {k: pick_by(hats[k], anchors[dc[k]], cand_laws) for k in hats if k in dc}


@torch.no_grad()
def eval_hop_x(head, X_all, anchors):
    """[lab13] HOP in a space. X_all (N_texts, D): unit-norm states of every text; anchors (27, D). Unseen items:
    DEC-k, BR-k (DEC law -> common sentence-MLP picker), BRa-k and A-k (code as Lab 12 eval_hop); seen chains: DEC-k."""
    head.eval()
    near = lambda v: (v @ anchors.T).argmax(dim=-1)
    sc = new_l13()
    hats = rollout(head, X_all[IT_S0], X_all[IT_S1])
    dec = hop_dec(hats, anchors)
    for k in EVAL_STAGES:
        na = near(hats[k])
        sc[f"DEC{k}"] = n_correct(dec[k])
        sc[f"BRa{k}"] = n_correct(pick_by(anchors[na], X_all[IT_CAND[k]]))
        sc[f"A{k}"] = int((na == IT_TRUE_CLS[k]).sum().item())
    br = mlp_picker({k: dec[k].tolist() for k in EVAL_STAGES})
    Xs = X_all[SEEN_G]
    hats_s = rollout(head, Xs[SI_S0], Xs[SI_S1])
    dec_s = hop_dec(hats_s, anchors, SI_DEC_CLS, SI_CL)
    for k in EVAL_STAGES:
        sc[f"BR{k}"] = br[f"BR{k}"]
        sc[f"seenDEC{k}"] = int((dec_s[k] == SI_LAW_T).sum().item())
    sc["params"] = n_hop_params(head)
    return sc, dec, hats


def onestep_input(x0, x1, x1_only: bool):
    return x1 if x1_only else torch.cat([x0, x1], dim=-1)


@torch.no_grad()
def eval_onestep(m, X_all, x1_only: bool = False):
    """[lab13] ONE-STEP (concat[x0, x1]) or X1-ONLY (x1) law classifier: DEC = logit of a vs b (the same decision for
    every k); BR-k = DEC law -> common sentence-MLP picker (ONE-STEP only); seen DEC on the 120 seen premises."""
    m.eval()
    sc = new_l13()
    dec = law_dec(m(onestep_input(X_all[IT_S0], X_all[IT_S1], x1_only)), CAND_LAWS_T)
    br = None if x1_only else mlp_picker(dec.tolist())
    Xs = X_all[SEEN_G]
    seen = int((law_dec(m(onestep_input(Xs[SI_S0], Xs[SI_S1], x1_only)), SI_CL) == SI_LAW_T).sum().item())
    for k in EVAL_STAGES:
        sc[f"DEC{k}"] = n_correct(dec)
        sc[f"seenDEC{k}"] = seen
        if br is not None:
            sc[f"BR{k}"] = br[f"BR{k}"]
    sc["params"] = sum(p.numel() for p in m.parameters())
    return sc, dec


# ==============================================================================
# [7] [lab13] State spaces F64 / P64 / R896 and anchor geometry
# ==============================================================================
SPACES = ("F64", "P64", "R896")
PCA_DIM = 64
with torch.no_grad():
    MU_SEEN = H_SEEN.mean(dim=0)      # (896,) mean of the seen-sentence features (centring for R896 and P64)
    SIGMA_SEEN = H_SEEN.std(dim=0)    # (896,) per-dimension std of the seen-sentence features (T1 noise scale)
    _hc = (H_SEEN - MU_SEEN).double().cpu()                  # PCA on CPU / float64: device-independent basis
    _U, _Sv, _Vh = torch.linalg.svd(_hc, full_matrices=False)
    PCA_V = _Vh[:PCA_DIM].T.contiguous().float().to(DEVICE)  # (896, 64): top-64 principal directions
    PCA_EXPL = float((_Sv[:PCA_DIM] ** 2).sum() / (_Sv ** 2).sum())
    del _hc, _U, _Sv, _Vh


@torch.no_grad()
def embed(space: str, H: torch.Tensor) -> torch.Tensor:
    """[lab13] per-sentence features (N, 896) -> unit-norm states (N, D) of a space. Every space is a fixed function
    of H (the funnel / mu_seen / PCA basis are fitted on seen sentences only), so T1 recomputes it from noisy H."""
    if space == "F64":
        return hop_head.extract_ds(H)[1]  # the stage-A funnel s (hop_head's funnel = STAGE_A's, frozen)
    if space == "R896":
        return F.normalize(H - MU_SEEN, p=2, dim=-1)
    if space == "P64":
        return F.normalize((H - MU_SEEN) @ PCA_V, p=2, dim=-1)
    raise ValueError(space)


X_SP, ANCH = {}, {}
for _sp in SPACES:
    X_SP[_sp] = embed(_sp, H_ALL)                   # (N_texts, D), from the clean features
    ANCH[_sp] = class_anchors(X_SP[_sp][SEEN_G])    # (27, D): normalised class means of the seen sentences
_S12, _A12 = head_space(hop_head)
assert torch.equal(X_SP["F64"], _S12) and torch.equal(ANCH["F64"], _A12), "F64 space must be Lab 12's head_space"
del _S12, _A12


@torch.no_grad()
def anchor_geometry(anchors):
    """[lab13] max off-diagonal cosine; mean partner cosine cos(anchor(a,k), anchor(b,k)) at stages 0 and 2-4;
    m = mean Euclidean distance from each anchor to its nearest other anchor; rank and orthonormal basis of the
    span of the 27 anchors (CPU, float64)."""
    cos = anchors @ anchors.T
    off = cos.masked_fill(torch.eye(N_CLS, device=DEVICE, dtype=torch.bool), -9.0)
    a64 = anchors.double().cpu()
    dist = (a64[:, None, :] - a64[None, :, :]).norm(dim=-1)
    dist.fill_diagonal_(float("inf"))
    m = float(dist.min(dim=1).values.mean())
    part = [float(cos[cls_of(a, k), cls_of(b, k)]) for (a, b) in PAIRS for k in (0,) + EVAL_STAGES]
    U, Sv, _ = torch.linalg.svd(a64.T, full_matrices=False)  # a64.T (D, 27) = U diag(Sv) V^T
    r = int((Sv > Sv[0] * 1e-6).sum())
    return dict(max_off=float(off.max()), partner=float(np.mean(part)), m=m, rank=r,
                basis=U[:, :r].float().to(DEVICE))   # (D, r) orthonormal basis of the anchor span


GEOM = {sp: anchor_geometry(ANCH[sp]) for sp in SPACES}
print(f"\n[l13] spaces: PCA top-{PCA_DIM} explained variance of the centred seen features = {PCA_EXPL:.4f}", flush=True)
for _sp in SPACES:
    _g = GEOM[_sp]
    print(f"[l13] space {_sp:<4} D={X_SP[_sp].shape[1]:<3} | anchors: max off-diag cos {_g['max_off']:.3f} | mean partner "
          f"cos (stages 0,2,3,4) {_g['partner']:.3f} | m (mean nearest-anchor distance) {_g['m']:.4f} | span rank "
          f"{_g['rank']}", flush=True)


# ==============================================================================
# [8] [lab13] Lab 12 reproduction checks on F64 (and the reference) + adapted-code equivalence checks
# ==============================================================================
# Lab 12 logged numbers (results/analogy_v5/seed{42,1,2}.log, Colab L4, torch 2.11.0+cu130, transformers 5.21.0):
# V5_RESULT method=hop BR/DEC/A, V5_DIAG hop_dec_law_mlp_picker, V5_RESULT method=premise_mlp DEC,
# V5_DIAG premise_dec_law_mlp_picker, and the fingerprint line.
LAB12_LOGGED = {
    42: dict(hop_BR=(38, 40, 41), hop_DEC=(42, 42, 42), hop_A=(39, 39, 40), hop_dec_mlp=(38, 40, 40),
             pm_DEC=(47, 47, 47), pm_dec_mlp=(42, 44, 44),
             fp=dict(H_ALL="0x1.4b7abc716e954p+16", stageA="0x1.5474e6ff1a95cp+8", hop="0x1.3949c096b3af9p+8")),
    1: dict(hop_BR=(42, 43, 42), hop_DEC=(46, 45, 43), hop_A=(41, 41, 41), hop_dec_mlp=(42, 42, 41),
            pm_DEC=(47, 47, 47), pm_dec_mlp=(43, 44, 44),
            fp=dict(H_ALL="0x1.4b7abc716e954p+16", stageA="0x1.5bccc9dc21bb3p+8", hop="0x1.43db167241d0bp+8")),
    2: dict(hop_BR=(38, 43, 40), hop_DEC=(45, 44, 45), hop_A=(40, 41, 41), hop_dec_mlp=(42, 41, 44),
            pm_DEC=(47, 47, 47), pm_dec_mlp=(44, 44, 46),
            fp=dict(H_ALL="0x1.4b7abc716e954p+16", stageA="0x1.075fc783772b3p+8", hop="0x1.00c4a7dc9f57bp+8")),
}


def _lab12_log_values(seed: int):
    """Parse results/analogy_v5/seed{seed}.log if it is reachable (repo checkout); None otherwise."""
    cands = [f"/content/results/analogy_v5/seed{seed}.log", f"results/analogy_v5/seed{seed}.log"]
    try:
        cands.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "analogy_v5",
                                  f"seed{seed}.log"))
    except NameError:
        pass
    path = next((p for p in cands if os.path.exists(p)), None)
    if path is None:
        return None, None
    res, diag, fp = {}, {}, {}
    for line in open(path, encoding="utf-8", errors="ignore"):
        if line.startswith(("V5_RESULT ", "V5_DIAG ")):
            kv = dict(x.split("=", 1) for x in line.split()[1:])
            (res if line.startswith("V5_RESULT") else diag)[kv.get("method", kv.get("name"))] = kv
        elif line.startswith(f"[v5] fingerprint seed={seed}:"):
            fp = dict(x.split("=", 1) for x in line.split(":", 1)[1].split())
    t3 = lambda d, key: tuple(int(d[f"{key}{k}"]) for k in EVAL_STAGES)
    return dict(hop_BR=t3(res["hop"], "BR"), hop_DEC=t3(res["hop"], "DEC"), hop_A=t3(res["hop"], "A"),
                hop_dec_mlp=t3(diag["hop_dec_law_mlp_picker"], "BR"), pm_DEC=t3(res["premise_mlp"], "DEC"),
                pm_dec_mlp=t3(diag["premise_dec_law_mlp_picker"], "BR"), fp=fp), path


RES, DECS, HATS, MODELS = {}, {}, {}, {}  # keyed by (space, method)
tri = lambda sc, key: tuple(sc[f"{key}{k}"] for k in EVAL_STAGES)

# (i) Lab 13's evaluation of F64 HOP equals Lab 12's eval_hop (copied) on the same head
RES[("F64", "HOP")], DECS[("F64", "HOP")], HATS[("F64", "HOP")] = eval_hop_x(hop_head, X_SP["F64"], ANCH["F64"])
_e12 = EV_HOP12["score"]
assert tri(RES[("F64", "HOP")], "BRa") == tri(_e12, "BR") and tri(RES[("F64", "HOP")], "DEC") == tri(_e12, "DEC") \
    and tri(RES[("F64", "HOP")], "A") == tri(_e12, "A"), "eval_hop_x disagrees with Lab 12 eval_hop on F64"
assert all(DECS[("F64", "HOP")][k].tolist() == EV_HOP12["picks"]["dec"][k] for k in EVAL_STAGES)
# (ii) the PREMISE-MLP reference: law_dec == Lab 12's PM_DEC; BR-k = DEC law -> common picker
assert torch.equal(law_dec(PM_LOGITS, CAND_LAWS_T), PM_DEC)
_pm = new_l13()
_pm_br = mlp_picker(PM_DEC.tolist())
_pm_seen = int((law_dec(prem_mlp(H_PREM_SEEN), SI_CL) == SI_LAW_T).sum().item())
for _k in EVAL_STAGES:
    _pm[f"DEC{_k}"], _pm[f"BR{_k}"], _pm[f"seenDEC{_k}"] = pm[f"DEC{_k}"], _pm_br[f"BR{_k}"], _pm_seen
_pm["params"] = sum(p.numel() for p in prem_mlp.parameters())
RES[("JOINT896", "PREMISE-MLP")], DECS[("JOINT896", "PREMISE-MLP")] = _pm, PM_DEC
print(f"[l13] eval_hop_x == Lab 12 eval_hop on F64 HOP (BR = BRa, DEC, A, per-item DEC picks): OK", flush=True)

# (iii) reproduction of the logged Lab 12 numbers
REPRO_OK = None
if SEED in LAB12_LOGGED:
    _ref = LAB12_LOGGED[SEED]
    _logv, _logp = _lab12_log_values(SEED)
    if _logv is not None:  # guard against a typo in the table above
        assert {k: v for k, v in _logv.items() if k != "fp"} == {k: v for k, v in _ref.items() if k != "fp"} and \
            all(_logv["fp"].get(k) == v for k, v in _ref["fp"].items()), f"LAB12_LOGGED differs from {_logp}"
        print(f"[l13] LAB12_LOGGED[{SEED}] matches {_logp}", flush=True)
    _f, _p = RES[("F64", "HOP")], RES[("JOINT896", "PREMISE-MLP")]
    _rows = [("F64 HOP BR2-4 (Lab 13 BRa = Lab 12 anchor-mediated BR)", tri(_f, "BRa"), _ref["hop_BR"]),
             ("F64 HOP DEC2-4", tri(_f, "DEC"), _ref["hop_DEC"]),
             ("F64 HOP A2-4", tri(_f, "A"), _ref["hop_A"]),
             ("F64 HOP BR2-4 common picker (Lab 12 hop_dec_law_mlp_picker)", tri(_f, "BR"), _ref["hop_dec_mlp"]),
             ("PREMISE-MLP DEC2-4", tri(_p, "DEC"), _ref["pm_DEC"]),
             ("PREMISE-MLP BR2-4 common picker (Lab 12 premise_dec_law_mlp_picker)", tri(_p, "BR"), _ref["pm_dec_mlp"])]
    print(f"\n=== [l13] Lab 12 reproduction (seed {SEED}; logged run: Colab L4) ===", flush=True)
    for _name, _a, _b in _rows:
        print(f"[check] {_name:<70} lab13 {_a} | lab12 log {_b} | {'MATCH' if _a == _b else 'DIFF'}", flush=True)
    for _k, _v in _ref["fp"].items():
        print(f"[info]  fingerprint {_k:<7} lab13 {FINGERPRINT[_k]} | lab12 log {_v} | "
              f"{'MATCH' if FINGERPRINT[_k] == _v else 'DIFF'} (hardware-dependent, not asserted)", flush=True)
    REPRO_OK = all(a == b for _n, a, b in _rows)
    if DEVICE == "cuda":
        # [main review] exploratory run: a mismatch (e.g. a changed Colab image) is reported loudly, not fatal,
        # so the stress tests still run; the reader must treat a DIFF seed as not comparable to Lab 12.
        print(f"[check] Lab 12 reproduction on CUDA: {'OK' if REPRO_OK else '!!! DIFF -- NOT COMPARABLE TO LAB 12 !!!'}"
              f" ({ENV_TAG}; Lab 12 log: Colab L4, torch 2.11.0+cu130)", flush=True)
    else:
        print(f"[l13] CPU run: the Lab 12 reproduction is printed, not asserted", flush=True)
else:
    print(f"[l13] seed {SEED} has no Lab 12 log: reproduction check skipped", flush=True)

# (iv) train_hop_x (used for P64 / R896) reproduces Lab 12's train_hop bit for bit on F64
_chk = build_variant_head(STAGE_A, pos=CFG["pos"])
with torch.no_grad():
    _s_chk = _chk.extract_ds(H_SEEN)[1]
_chk = train_hop_x(_chk, _s_chk, "HOP-F64-check", CFG["loss"])
_sd_chk = _chk.state_dict()
assert set(_sd_chk) == set(_sd) and all(torch.equal(_sd_chk[k], v) for k, v in hop_head.state_dict().items()), \
    "train_hop_x does not reproduce Lab 12 train_hop on F64"
print(f"[check] train_hop_x == Lab 12 train_hop on F64 (bit-identical weights): OK", flush=True)
del _chk, _sd_chk, _s_chk


# ==============================================================================
# [9] [lab13] Training: HOP in P64 / R896 (+ HOP-wide), ONE-STEP and X1-ONLY in every space
# ==============================================================================
HOPS = {"F64": [("HOP", hop_head)]}
_p64 = build_p64_hop()
assert core_matches_ref(_p64, CORE_REF), "P64 hop init must equal the F64 core init"
HOPS["P64"] = [("HOP", train_hop_x(_p64, X_SP["P64"][SEEN_G], "HOP-P64", CFG["loss"]))]
_r896 = build_r896_hop()
assert core_matches_ref(_r896, CORE_REF), "R896 hop core init must equal the F64 core init"
_r896 = train_hop_x(_r896, X_SP["R896"][SEEN_G], "HOP-R896", CFG["loss"])
_wide = train_hop_x(build_wide_hop(), X_SP["R896"][SEEN_G], "HOP-wide-R896", CFG["loss"])
HOPS["R896"] = [("HOP", _r896), ("HOP-wide", _wide)]
for _sp in ("P64", "R896"):
    for _m, _h in HOPS[_sp]:
        RES[(_sp, _m)], DECS[(_sp, _m)], HATS[(_sp, _m)] = eval_hop_x(_h, X_SP[_sp], ANCH[_sp])
for _sp in SPACES:
    for _m, _h in HOPS[_sp]:
        MODELS[(_sp, _m)] = _h

ONE, X1 = {}, {}
for _sp in SPACES:
    _Xs = X_SP[_sp][SEEN_G]
    ONE[_sp] = train_mlp(f"ONE-STEP {_sp} (concat[x0, x1] -> 6 laws)", onestep_input(_Xs[SI_S0], _Xs[SI_S1], False),
                         SI_LAW_T, N_LAWS)
    X1[_sp] = train_mlp(f"X1-ONLY {_sp} (x1 -> 6 laws, sanity)", onestep_input(_Xs[SI_S0], _Xs[SI_S1], True),
                        SI_LAW_T, N_LAWS)
    RES[(_sp, "ONE-STEP")], DECS[(_sp, "ONE-STEP")] = eval_onestep(ONE[_sp], X_SP[_sp])
    RES[(_sp, "X1-ONLY")], DECS[(_sp, "X1-ONLY")] = eval_onestep(X1[_sp], X_SP[_sp], x1_only=True)
    MODELS[(_sp, "ONE-STEP")], MODELS[(_sp, "X1-ONLY")] = ONE[_sp], X1[_sp]

# sanity: X1-ONLY exactly 50% (stage 1 is shared within a pair), unseen items and seen premises, in every space
X1_OK = True
for _sp in SPACES:
    _sc = RES[(_sp, "X1-ONLY")]
    _ok = all(_sc[f"DEC{k}"] * 2 == N_ITEMS and _sc[f"seenDEC{k}"] * 2 == N_SEEN_ITEMS for k in EVAL_STAGES)
    X1_OK &= _ok
    print(f"[check] X1-ONLY {_sp}: DEC {tri(_sc, 'DEC')} == {N_ITEMS // 2}/{N_ITEMS}, seen DEC {_sc['seenDEC2']} == "
          f"{N_SEEN_ITEMS // 2}/{N_SEEN_ITEMS}: {'OK' if _ok else 'FAIL (design/indexing bug)'}", flush=True)
assert X1_OK, "X1-ONLY must be exactly 50% in every space"

TF_EVAL = {("F64", "HOP"): eval_mode_hop_metrics(hop_head)}
for _sp in ("P64", "R896"):
    for _m, _h in HOPS[_sp]:
        TF_EVAL[(_sp, _m)] = tf_metrics_x(_h, X_SP[_sp][SEEN_G])

ORDER = [("F64", "HOP"), ("F64", "ONE-STEP"), ("F64", "X1-ONLY"),
         ("P64", "HOP"), ("P64", "ONE-STEP"), ("P64", "X1-ONLY"),
         ("R896", "HOP"), ("R896", "HOP-wide"), ("R896", "ONE-STEP"), ("R896", "X1-ONLY"),
         ("JOINT896", "PREMISE-MLP")]
assert sorted(ORDER) == sorted(RES), sorted(RES)
HOP_KEYS = [(sp, m) for sp in SPACES for (m, _h) in HOPS[sp]]
DEC_METHODS = [key for key in ORDER if key[1] != "X1-ONLY"]  # methods reported in T1


# ==============================================================================
# [10] [lab13] Stress tests (unseen items only; one generator, identical draws for every method)
# ==============================================================================
NOISE_ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0)
PERT_BETAS = (0.0, 0.5, 1.0)
N_DRAWS = 5
LONG_LAST, LONG_WINDOW = 12, 4
GEN = torch.Generator(device="cpu").manual_seed(SEED)  # one generator seeded from AUDIT_SEED; CPU = device-independent
EPS_T1 = torch.randn(N_DRAWS, len(NOISE_ROWS), H_DIM, generator=GEN).to(DEVICE)  # T1: per noisy row, per draw
U_ISO_RAW = torch.randn(N_DRAWS, N_ITEMS, H_DIM, generator=GEN).to(DEVICE)       # T2 iso: per item, per draw
U_SPAN_RAW = torch.randn(N_DRAWS, N_ITEMS, N_CLS, generator=GEN).to(DEVICE)      # T2 span: coefficients


@torch.no_grad()
def dec_counts_all(Hn) -> dict:
    """[lab13] DEC-k counts of every T1 method on features Hn (N_texts, 896); fixed (clean) anchors."""
    out = {}
    for sp in SPACES:
        Xn = embed(sp, Hn)
        for m, head in HOPS[sp]:
            dec = hop_dec(rollout(head, Xn[IT_S0], Xn[IT_S1]), ANCH[sp])
            out[(sp, m)] = np.array([n_correct(dec[k]) for k in EVAL_STAGES], dtype=np.float64)
        d1 = law_dec(ONE[sp](onestep_input(Xn[IT_S0], Xn[IT_S1], False)), CAND_LAWS_T)
        out[(sp, "ONE-STEP")] = np.full(3, n_correct(d1), dtype=np.float64)
        dx = law_dec(X1[sp](onestep_input(Xn[IT_S0], Xn[IT_S1], True)), CAND_LAWS_T)
        assert n_correct(dx) * 2 == N_ITEMS, f"X1-ONLY {sp} is not 50% under T1 noise (noise must be per sentence row)"
    dp = law_dec(prem_mlp(Hn[PM_ROWS]), CAND_LAWS_T)
    out[("JOINT896", "PREMISE-MLP")] = np.full(3, n_correct(dp), dtype=np.float64)
    return out


# ---- T1: input noise ----
print(f"\n=== [10] T1 input noise: alpha in {NOISE_ALPHAS}, K={N_DRAWS} draws (common random numbers across alpha) ===",
      flush=True)
NOISE = {}  # (space, method) -> {alpha: mean DEC2-4 counts over draws (np.array(3))}
for _a in NOISE_ALPHAS:
    _acc = {key: np.zeros(3) for key in DEC_METHODS}
    for _d in range(N_DRAWS):
        _Hn = H_ALL.clone()
        _Hn[NOISE_ROWS] = _Hn[NOISE_ROWS] + _a * SIGMA_SEEN * EPS_T1[_d]
        _cnt = dec_counts_all(_Hn)
        for key in DEC_METHODS:
            _acc[key] += _cnt[key]
    for key in DEC_METHODS:
        NOISE.setdefault(key, {})[_a] = _acc[key] / N_DRAWS
for key in DEC_METHODS:  # alpha = 0 is the clean evaluation, exactly
    assert tuple(NOISE[key][0.0].tolist()) == tuple(float(v) for v in tri(RES[key], "DEC")), (key, NOISE[key][0.0])
print(f"[check] T1 alpha=0 equals the clean DEC for every method: OK | X1-ONLY 50% under every draw: OK", flush=True)


# ---- T2: state perturbation after step 2 ----
@torch.no_grad()
def rollout_pert(head, x0, x1, u, scale: float):
    """[lab13] T2: Lab 12 free rollout, except that after step 2 the state is replaced by normalize(x̂2 + scale * u)
    (scale = beta * m) before it is fed back. scale == 0 leaves x̂2 untouched (the clean rollout, bit for bit)."""
    seq = torch.stack([x0, x1], dim=1)
    hats = {}
    for k in EVAL_STAGES:
        nxt = head.hop_law(seq)[:, -1, :]
        if k == 2 and scale != 0.0:
            nxt = F.normalize(nxt + scale * u, p=2, dim=-1)
        hats[k] = nxt
        seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)
    return hats


print(f"\n=== [10] T2 state perturbation after step 2: beta in {PERT_BETAS}, kinds iso / span, K={N_DRAWS} ===", flush=True)
PERT = {}  # (space, method, kind) -> {beta: mean (DEC3, DEC4)}
for _sp in SPACES:
    _D = ANCH[_sp].shape[1]
    _B = GEOM[_sp]["basis"]  # (D, r)
    _U = {"iso": F.normalize(U_ISO_RAW[..., :_D], p=2, dim=-1),                       # (K, N_ITEMS, D)
          "span": F.normalize(U_SPAN_RAW[..., :_B.shape[1]] @ _B.T, p=2, dim=-1)}     # (K, N_ITEMS, D) in the span
    for _m, _h in HOPS[_sp]:
        for _kind in ("iso", "span"):
            for _b in PERT_BETAS:
                _acc = np.zeros(2)
                for _d in range(N_DRAWS):
                    _dec = hop_dec(rollout_pert(_h, X_SP[_sp][IT_S0], X_SP[_sp][IT_S1], _U[_kind][_d],
                                                _b * GEOM[_sp]["m"]), ANCH[_sp])
                    _acc += [n_correct(_dec[3]), n_correct(_dec[4])]
                PERT.setdefault((_sp, _m, _kind), {})[_b] = _acc / N_DRAWS
            assert tuple(PERT[(_sp, _m, _kind)][0.0].tolist()) == \
                (float(RES[(_sp, _m)]["DEC3"]), float(RES[(_sp, _m)]["DEC4"])), (_sp, _m, _kind)
print(f"[check] T2 beta=0 equals the clean DEC3/DEC4 for every hop: OK", flush=True)


# ---- T3: long rollout ----
@torch.no_grad()
def rollout_long(head, x0, x1, last: int = LONG_LAST, window: int = LONG_WINDOW):
    """[lab13] T3: free rollout from [x0, x1] to step `last`, feeding back the model's own states.
    Sliding window: the causal transformer always sees only the last `window` = 4 states, at positions 0..3. The hop
    was trained on 4-state inputs (teacher-forced chains, positions 0..3; the 5th row of the pos table is never
    trained), so a growing sequence would put states at untrained positions (4..10, and beyond the 5-row table from
    position 5). With the window, steps 2-4 are exactly the Lab 12 rollout; from step 5 on the oldest state drops out
    (x0 at step 5, x1 at step 6), so the law must then be carried by the model's own predicted states."""
    states, hats = [x0, x1], {}
    for t in range(2, last + 1):
        seq = torch.stack(states[-window:], dim=1)
        nxt = head.hop_law(seq)[:, -1, :]
        hats[t] = nxt
        states.append(nxt)
    return hats


print(f"\n=== [10] T3 long rollout to step {LONG_LAST} (sliding window of the last {LONG_WINDOW} states) ===", flush=True)
LONG = {}  # (space, method) -> {step: (mean cosine to nearest anchor, fraction keeping the DEC-2 law)}
for (_sp, _m) in HOP_KEYS:
    _h = MODELS[(_sp, _m)]
    _hl = rollout_long(_h, X_SP[_sp][IT_S0], X_SP[_sp][IT_S1])
    assert all(torch.equal(_hl[k], HATS[(_sp, _m)][k]) for k in EVAL_STAGES), f"T3 steps 2-4 differ from rollout ({_sp})"
    _law2 = DECS[(_sp, _m)][2]  # (N_ITEMS,) the law decided by DEC-2
    LONG[(_sp, _m)] = {}
    for _st in range(2, LONG_LAST + 1):
        _sim = _hl[_st] @ ANCH[_sp].T  # (N_ITEMS, 27)
        _keep = CLASS_HAS_LAW[_sim.argmax(dim=-1), _law2]
        LONG[(_sp, _m)][_st] = (float(_sim.max(dim=-1).values.mean()), float(_keep.float().mean()))
print(f"[check] T3 steps 2-4 equal the clean rollout for every hop: OK", flush=True)


# ==============================================================================
# [11] Report (numbers only, no verdict)
# ==============================================================================
def cell(v, n=N_ITEMS):
    """(copied from Lab 12)"""
    return f"{v:>2d}/{n}" if v >= 0 else "  -  "


def pct3(sc, key, n=N_ITEMS):
    vals = [sc[f"{key}{k}"] for k in EVAL_STAGES]
    return f"{100.0 * sum(vals) / (3 * n):6.2f}%" if all(v >= 0 for v in vals) else "     - "


print("\n" + "=" * 178, flush=True)
print(f"=== RESULTS seed={SEED} | {N_ITEMS} unseen items ({N_UNSEEN_DOM} domains x {N_LAWS} laws); chance {N_ITEMS // 2}/"
      f"{N_ITEMS} | DEC-k = picker-free pairwise decision (Lab 12) | BR-k = DEC law -> common sentence-MLP picker | "
      f"BRa-k = HOP anchor-mediated BR in its own space (= Lab 12 BR) | A-k = nearest anchor is (L,k) | seenDEC on "
      f"{N_SEEN_ITEMS} seen chains ===", flush=True)
hdr = (f"{'space':<9}{'method':<12}{'params':>11}" + "".join(f"{c:>7}" for c in ("DEC-2", "DEC-3", "DEC-4")) + f"{'DEC':>8}"
       + "".join(f"{c:>7}" for c in ("BR-2", "BR-3", "BR-4")) + f"{'BR':>8}"
       + "".join(f"{c:>7}" for c in ("BRa-2", "BRa-3", "BRa-4")) + f"{'BRa':>8}"
       + "".join(f"{c:>7}" for c in ("A-2", "A-3", "A-4")) + "".join(f"{c:>9}" for c in ("seenD-2", "seenD-3", "seenD-4")))
print(hdr, flush=True)
print("-" * len(hdr), flush=True)
for key in ORDER:
    sc = RES[key]
    print(f"{key[0]:<9}{key[1]:<12}{sc['params']:>11,}" + "".join(f"{cell(sc[f'DEC{k}']):>7}" for k in EVAL_STAGES)
          + f"{pct3(sc, 'DEC'):>8}" + "".join(f"{cell(sc[f'BR{k}']):>7}" for k in EVAL_STAGES) + f"{pct3(sc, 'BR'):>8}"
          + "".join(f"{cell(sc[f'BRa{k}']):>7}" for k in EVAL_STAGES) + f"{pct3(sc, 'BRa'):>8}"
          + "".join(f"{cell(sc[f'A{k}']):>7}" for k in EVAL_STAGES)
          + "".join(f"{cell(sc[f'seenDEC{k}'], N_SEEN_ITEMS):>9}" for k in EVAL_STAGES), flush=True)
print("-" * len(hdr), flush=True)
print(f"params = trainable parameters of the method itself (HOP: the hop only). Not included: the F64 stage-A funnel "
      f"({FUNNEL_PARAMS:,} params, trained once, shared by F64 HOP and F64 ONE-STEP) and the common sentence-MLP picker "
      f"({sum(p.numel() for p in sent_mlp.parameters()):,} params).", flush=True)
print(f"X1-ONLY exactly 50% in every space (unseen and seen): {'OK' if X1_OK else 'FAIL'} | Lab 12 reproduction: "
      f"{ {None: 'n/a', True: 'MATCH', False: 'DIFF'}[REPRO_OK] }{' (checked on CUDA)' if DEVICE == 'cuda' else ''}",
      flush=True)

print(f"\n=== STRESS T1: input noise H' = H + alpha*sigma*eps on the unseen s0 / s1 / premise rows; mean DEC-2/3/4 "
      f"counts over {N_DRAWS} draws (out of {N_ITEMS}); [mean of the 3 stages as %] ===", flush=True)
hdr = f"{'space':<9}{'method':<12}" + "".join(f"{f'alpha={a:g}':>30}" for a in NOISE_ALPHAS)
print(hdr + "\n" + "-" * len(hdr), flush=True)
for key in DEC_METHODS:
    print(f"{key[0]:<9}{key[1]:<12}" + "".join(
        f"{'/'.join(f'{v:.2f}' for v in NOISE[key][a]) + f' [{100.0 * NOISE[key][a].mean() / N_ITEMS:6.2f}%]':>30}"
        for a in NOISE_ALPHAS), flush=True)

print(f"\n=== STRESS T2: x̂2' = normalize(x̂2 + beta*m*u), fed back; mean DEC-3/DEC-4 counts over {N_DRAWS} draws ===",
      flush=True)
print("  m per space: " + ", ".join(f"{sp} {GEOM[sp]['m']:.4f} (anchor span rank {GEOM[sp]['rank']})" for sp in SPACES),
      flush=True)
hdr = f"{'space':<9}{'method':<12}{'u':<6}" + "".join(f"{f'beta={b:g}':>18}" for b in PERT_BETAS)
print(hdr + "\n" + "-" * len(hdr), flush=True)
for (sp, m) in HOP_KEYS:
    for kind in ("iso", "span"):
        print(f"{sp:<9}{m:<12}{kind:<6}" + "".join(
            f"{'/'.join(f'{v:.2f}' for v in PERT[(sp, m, kind)][b]):>18}" for b in PERT_BETAS), flush=True)

print(f"\n=== STRESS T3: long rollout (window {LONG_WINDOW}); per step: mean cosine to the nearest anchor | fraction of "
      f"items whose nearest anchor belongs to the DEC-2 law ===", flush=True)
hdr = f"{'space':<9}{'method':<12}{'':<10}" + "".join(f"{f't={s}':>8}" for s in range(2, LONG_LAST + 1))
print(hdr + "\n" + "-" * len(hdr), flush=True)
for (sp, m) in HOP_KEYS:
    print(f"{sp:<9}{m:<12}{'cos_near':<10}" + "".join(f"{LONG[(sp, m)][s][0]:>8.4f}" for s in range(2, LONG_LAST + 1)),
          flush=True)
    print(f"{'':<21}{'law_keep':<10}" + "".join(f"{LONG[(sp, m)][s][1]:>8.4f}" for s in range(2, LONG_LAST + 1)),
          flush=True)

print("\n=== Diagnostics ===", flush=True)
print(f"  PCA top-{PCA_DIM} explained variance (centred seen features): {PCA_EXPL:.4f}", flush=True)
for sp in SPACES:
    g = GEOM[sp]
    print(f"  anchors {sp:<4}: max off-diagonal cosine {g['max_off']:.3f} | mean partner cosine {g['partner']:.3f} | "
          f"m {g['m']:.4f} | span rank {g['rank']}", flush=True)
print("  eval-mode teacher-forced hop metrics on the seen chains (dropout off):", flush=True)
for key, met in TF_EVAL.items():
    print(f"    {key[0]:<5} {key[1]:<9} {met}", flush=True)
print(f"  Lab 12 eval_hop on F64 HOP: seen-domain BR {tri(EV_HOP12['seen'], 'BR')} / {N_SEEN_ITEMS}, FLIP "
      f"{tri(EV_HOP12['score'], 'FLIP')} / {N_ITEMS}", flush=True)

print(f"\n=== Training losses (train mode) ===", flush=True)
for tag, step, msg in LOSS_LOG:
    print(f"  {tag:<44} step {step:4d} | {msg}", flush=True)

print("", flush=True)
print(f"L13_CONFIG seed={SEED} data_sha={DATA_SHA256[:16]} script_sha={SCRIPT_SHA256[:16]} env={ENV_TAG}", flush=True)
for key in ORDER:
    sc = RES[key]
    print(f"L13_RESULT seed={SEED} space={key[0]} method={key[1]} " + " ".join(f"{k}={sc[k]}" for k in L13_KEYS),
          flush=True)
for key in DEC_METHODS:
    for a in NOISE_ALPHAS:
        v = NOISE[key][a]
        print(f"L13_NOISE seed={SEED} space={key[0]} method={key[1]} alpha={a:g} "
              f"DEC2={v[0]:.2f} DEC3={v[1]:.2f} DEC4={v[2]:.2f}", flush=True)
for (sp, m) in HOP_KEYS:
    for kind in ("iso", "span"):
        for b in PERT_BETAS:
            v = PERT[(sp, m, kind)][b]
            print(f"L13_PERT seed={SEED} space={sp} method={m} kind={kind} beta={b:g} DEC3={v[0]:.2f} DEC4={v[1]:.2f}",
                  flush=True)
for (sp, m) in HOP_KEYS:
    for s in range(2, LONG_LAST + 1):
        c, kp = LONG[(sp, m)][s]
        print(f"L13_LONG seed={SEED} space={sp} method={m} step={s} cos_near={c:.4f} law_keep={kp:.4f}", flush=True)
print(f"[l13] total time {time.time()-T_START:.1f}s", flush=True)
