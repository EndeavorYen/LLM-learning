# ==============================================================================
# Stage 2 / Lab 12: Association test v5 (CONFIRMATORY; pre-registered on issue #21 before it runs)
# Data: benchmarks/analogy_v5.json. Seen = all 20 v4 domains (verbatim), unseen = 8 brand-new domains.
# Laws, pairs, stages and the 27 state classes are those of v4. Every method is trained on the SEEN domains only.
# Methods:
#   HOP (primary)  Lab 11 two-stage training: stage A = funnel alone (Lab 09 loss_dis, 400 steps), then frozen;
#                  law_brain + next_law_proj (+ pos) trained alone for 2000 steps with the pre-registered variant
#                  HOP_VARIANT (V1-V4 exactly as in Lab 11) and evaluated with rollout mode HOP_MODE (free / snap).
#   RANDOM HOP     the same stage-A funnel; law_brain / next_law_proj (+ pos) at their seeded init, never trained;
#                  same rollout mode.
#   V0 joint       (secondary) Lab 09 joint training (Lab 11 train_head), free rollout.
#   LAST           s1 only, in the HOP head's (= stage-A funnel's) s-space and anchors; 50% by construction (asserted).
#   S0-RULE        nearest stage-0 anchor of s0 -> law, in the HOP head's (= stage-A funnel's) s-space and anchors.
#   PREMISE-MLP    frozen-Qwen feature of s0+s1 -> MLP law classifier; per-sentence 27-state MLP picks (log_softmax).
#   Qwen LM        mean log-prob ranking, zero-shot and few-shot (prefix = seen domain 0's six chains).
# Metrics on the unseen items (8 domains x 6 laws = 48), k = 2, 3, 4. Item = (domain, true law L), L in pair (a, b):
#   BR-k   branch choice between the stage-k sentences of a and b (fixed pair order; ties -> a).
#          Hop methods: anchor-mediated (nearest of the 27 anchors to ŝk, then the candidate closest to it).
#   DEC-k  pairwise branch decision, independent of the candidate sentences; ties -> a; correct if it equals L.
#          Hop methods: cos(ŝk, anchor(a,k)) vs cos(ŝk, anchor(b,k)). PREMISE-MLP: premise logit of a vs b (same
#          for every k). S0-RULE: cos(s0, anchor(a,0)) vs cos(s0, anchor(b,0)). LAST: always a (exactly 50%,
#          asserted). Qwen LM: DEC-k = BR-k.
#   A-k    (hop methods) nearest of the 27 anchors to ŝk is (L, k)
#   FLIP-k (hop methods) BR picks that change when s0 -> the partner law's s0 (same domain)
#   PREM   (PREMISE-MLP, S0-RULE) 6-way law of the premise correct
# Machine-readable lines: V5_RESULT / V5_DIAG / V5_CONFIG (+ V5_LOSS). This script prints numbers only; NO verdict.
# Run:  AUDIT_SEED=42 python stage2_lab12_association_v5.py   (seeds 42, 1, 2)
# Exploration only (never for the registered run): LAB12_VARIANT=V1..V4, LAB12_MODE=free|snap override the
# pre-registered HOP_VARIANT / HOP_MODE below; any difference is printed loudly and V5_CONFIG reports override=1.
# Copied code is marked "(copied from Lab 09)" / "(copied from Lab 11)"; "[lab12]" marks new or changed lines.
# ==============================================================================
import os, sys, time, math, random, json, hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- [lab12] Pre-registered HOP configuration ----------------------------------------------------------------------
HOP_VARIANT = "V4"   # set in the pre-registration on issue #21
HOP_MODE = "free"    # set in the pre-registration on issue #21

SEED = int(os.environ.get("AUDIT_SEED", "42"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[v5] SEED={SEED} | device={DEVICE}", flush=True)
T_START = time.time()
# [lab12] environment record (reproducibility)
print(f"[v5] python {sys.version.split()[0]} | torch {torch.__version__} | numpy {np.__version__} | "
      f"transformers {getattr(transformers, '__version__', '?')} ({getattr(transformers, '__file__', '?')})", flush=True)
try:
    with open(os.path.abspath(__file__), "rb") as _f:
        print(f"[v5] script sha256={hashlib.sha256(_f.read()).hexdigest()}", flush=True)
except (NameError, OSError):
    print("[v5] script sha256=unavailable (no __file__)", flush=True)


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

# ---- [lab12] resolve the HOP configuration (env overrides are for exploration only) -------------------------------
HOP_VARIANTS = ("V1", "V2", "V3", "V4")  # two-stage variants; V0 is the separate joint run
assert HOP_VARIANT in HOP_VARIANTS and HOP_MODE in MODES, (HOP_VARIANT, HOP_MODE)


def _env_choice(name: str, registered: str, allowed) -> str:
    v = os.environ.get(name)
    if v is None or v == "":
        return registered
    if v not in allowed:
        raise ValueError(f"{name}={v!r} is not one of {allowed}")
    if v == registered:
        print(f"[v5] {name}={v} is set but equals the pre-registered value", flush=True)
    return v


VARIANT = _env_choice("LAB12_VARIANT", HOP_VARIANT, HOP_VARIANTS)
MODE = _env_choice("LAB12_MODE", HOP_MODE, MODES)
CFG = VARIANTS[VARIANT]
OVERRIDE = (VARIANT, MODE) != (HOP_VARIANT, HOP_MODE)


def print_override_banner():
    if not OVERRIDE:
        return
    bar = "!" * 120
    print(f"{bar}\n!!! [v5] ENV OVERRIDE: HOP runs variant={VARIANT} mode={MODE}, but the PRE-REGISTERED values are "
          f"HOP_VARIANT={HOP_VARIANT} HOP_MODE={HOP_MODE}.\n!!! This run is EXPLORATORY. It is NOT the pre-registered "
          f"confirmatory test.\n{bar}", flush=True)


print(f"[v5] HOP variant={VARIANT} (pos={CFG['pos']}, loss={CFG['loss']}) mode={MODE} | pre-registered: "
      f"{HOP_VARIANT} / {HOP_MODE} | override={int(OVERRIDE)}", flush=True)
print_override_banner()


# ==============================================================================
# [0] Data (copied from Lab 11 / Lab 09; [lab12] v5 file and v5 counts)
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


# ==============================================================================
# [1] Qwen (copied from Lab 11 / Lab 09 / Lab 01): bf16, frozen
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


H_ALL = torch.cat([extract_qwen_896_feature(ALL_TEXTS[i:i + 64]) for i in range(0, len(ALL_TEXTS), 64)], dim=0)
assert H_ALL.shape == (len(ALL_TEXTS), H_DIM)
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

# (copied from Lab 11)
HEAD_STEPS, HEAD_LR, HEAD_LR_FLOOR = 400, 1e-3, 1e-4   # Lab 09 (V0 and stage A)
HOP_STEPS, HOP_LR, HOP_LR_FLOOR = 2000, 1e-3, 1e-4     # Lab 11 hop-only training (V1-V4)
HOP_LOG_STEPS = (1, 500, 1000, 2000)
CE_TAU = 0.1
LOSS_LOG = []  # (tag, step, description)


# ==============================================================================
# [2] Funnel + law_brain head with optional learned positional embeddings (copied from Lab 11)
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


def train_head(tag: str):
    """(copied from Lab 11; HL_LOSS line renamed V5_LOSS) V0: Lab 09 train_head unchanged (joint funnel + hop, 400 steps)."""
    seed_all(SEED)  # before model init
    head = DualHead(qwen_hidden=H_DIM, d_dim=64, s_dim=64).to(DEVICE).float()
    n_par = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"\n=== [2] Training {tag} (Lab 09 joint): {n_par:,} trainable params | {HEAD_STEPS} steps full batch | "
          f"AdamW lr {HEAD_LR:g} -> {HEAD_LR_FLOOR:g} cosine | clip 1.0 ===", flush=True)
    t1 = time.time()
    seed_all(SEED)  # before training run
    opt = torch.optim.AdamW(head.parameters(), lr=HEAD_LR, weight_decay=0.01)
    head.train()
    for step in range(1, HEAD_STEPS + 1):
        for pg in opt.param_groups:
            pg['lr'] = cosine_lr(step, HEAD_STEPS, HEAD_LR, HEAD_LR_FLOOR)
        opt.zero_grad()
        loss_dis, parts, s = funnel_loss(head)
        s_ch = s[HOP_IDX]                                   # (N_SEEN_ITEMS, 5, 64)
        pred = head.hop_law(s_ch[:, :-1, :])                # from positions 0..3, predicts positions 1..4
        loss_hop = (1.0 - (pred * s_ch[:, 1:, :].detach()).sum(-1)).mean()
        loss = loss_dis + 2.5 * loss_hop
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if step in (1, 200, 400):
            msg = (f"total {loss.item():.4f} | dis {loss_dis.item():.4f} ({fmt_parts(parts)}) | hop {loss_hop.item():.4f}")
            met = hop_metrics(pred.detach(), s_ch[:, 1:, :].detach(), class_anchors(s.detach()))
            LOSS_LOG.append((tag, step, msg + " | " + met))
            print(f"  [{tag}] step {step:3d}/{HEAD_STEPS} | {msg} | {met}", flush=True)
            print(f"V5_LOSS seed={SEED} model={tag} step={step} loss={loss_hop.item():.4f} {met}", flush=True)
    head.eval()
    print(f"  [{tag}] trained in {time.time()-t1:.1f}s", flush=True)
    return head


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
    """(copied from Lab 11; HL_LOSS line renamed V5_LOSS) Stage B: law_brain + next_law_proj (+ pos) alone on the
    frozen s of the seen chains."""
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
            print(f"V5_LOSS seed={SEED} model={tag} step={step} loss={loss.item():.4f} {met}", flush=True)
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


# ==============================================================================
# [3] Evaluation helpers
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


@torch.no_grad()
def eval_rules(head):
    """LAST and S0-RULE in the given head's s-space and anchors. BR-k / PREM: copied from Lab 09 eval_head;
    [lab12] DEC-k."""
    S, anchors = head_space(head)
    s0, s1 = S[IT_S0], S[IT_S1]
    near = lambda v: (v @ anchors.T).argmax(dim=-1)
    # LAST: s1 only. Law implied by s1's nearest anchor (shared stage-1 class -> first law of the pair).
    last = new_score()
    n1 = near(s1).tolist()
    last_law = [law_of_class(c) for c in n1]
    # S0-RULE: nearest of the 6 stage-0 anchors of s0 -> law
    s0r = new_score()
    s0_law = [law_of_class(STAGE0_CLS[j]) for j in (s0 @ anchors[_t(STAGE0_CLS)].T).argmax(dim=-1).tolist()]
    s0r["PREM"] = sum(int(pl == L) for pl, L in zip(s0_law, IT_LAW))
    for k in EVAL_STAGES:
        cand = S[IT_CAND[k]]
        for sc, pl in ((last, last_law), (s0r, s0_law)):
            tgt = anchors[_t([cls_of(l, k) for l in pl])]
            sc[f"BR{k}"] = n_correct(pick_by(tgt, cand))
    # [lab12] DEC. S0-RULE: cos(s0, anchor(a,0)) vs cos(s0, anchor(b,0)), ties -> a (the same decision for every k).
    s0_dec = pick_by(s0, anchors[IT_DEC_CLS[0]])
    # LAST: s1's class (pair, 1) is shared by a and b (asserted for IT_DEC_CLS[1]) -> no branch information,
    # tie rule -> always a.
    last_dec = CAND_LAWS_T[:, 0]
    for k in EVAL_STAGES:
        s0r[f"DEC{k}"] = n_correct(s0_dec)
        last[f"DEC{k}"] = n_correct(last_dec)
    return dict(last=last, s0r=s0r, n1=n1, s0_law=s0_law, s0_dec=s0_dec.tolist())


@torch.no_grad()
def anchor_picker(head, laws):
    """(copied from Lab 09) BR-k when the law is given by `laws` and the candidate is chosen with the head's anchor picker."""
    _d, S = head.extract_ds(H_ALL)
    anchors = F.normalize(CLS_OH.T @ S[SEEN_G] / CLS_OH.sum(0).unsqueeze(-1), p=2, dim=-1)
    out = new_score()
    for k in EVAL_STAGES:
        tgt = anchors[_t([cls_of(l, k) for l in laws])]
        out[f"BR{k}"] = int((pick_by(tgt, S[IT_CAND[k]]) == IT_LAW_T).sum().item())
    return out


@torch.no_grad()
def funnel_checks(head, tag: str):
    """(copied from Lab 11; [lab12] also returns the max off-diagonal anchor cosine) Lab 09 LAST sanity check
    (s1 only must be exactly 50%) + anchor geometry for this funnel."""
    S, anchors = head_space(head)
    n1 = (S[IT_S1] @ anchors.T).argmax(dim=-1).tolist()
    last_law = [law_of_class(c) for c in n1]
    ok_all = True
    for k in EVAL_STAGES:
        tgt = anchors[_t([cls_of(l, k) for l in last_law])]
        v = int((pick_by(tgt, S[IT_CAND[k]]) == IT_LAW_T).sum().item())
        ok = v * 2 == N_ITEMS
        ok_all &= ok
        print(f"[check] funnel={tag} LAST BR-{k} = {v}/{N_ITEMS} == {N_ITEMS // 2}/{N_ITEMS}: "
              f"{'OK' if ok else 'FAIL (design/indexing bug)'}", flush=True)
    off = (anchors @ anchors.T).masked_fill(torch.eye(N_CLS, device=DEVICE, dtype=torch.bool), -9.0)
    mi = int(off.argmax().item())
    print(f"[v5] funnel={tag} max off-diagonal anchor cosine = {float(off.max().item()):.3f} "
          f"({clbl(mi // N_CLS)} vs {clbl(mi % N_CLS)})", flush=True)
    return ok_all, (float(off.max().item()), mi // N_CLS, mi % N_CLS)


# ==============================================================================
# [4] Training: stage A -> HOP (pre-registered variant); RANDOM HOP; V0 joint
# ==============================================================================
# Stage A + HOP: the same calls as Lab 11's loop body for one variant (copied from Lab 11)
STAGE_A = train_funnel_stage_a("stageA")
_STAGE_A_FUNNEL = {k: v.detach().clone() for k, v in STAGE_A.state_dict().items() if "funnel" in k}

# [lab12] check: stage A left law_brain / next_law_proj exactly at the seeded init (RANDOM HOP relies on it)
seed_all(SEED)
_REF_INIT = DualHead(qwen_hidden=H_DIM, d_dim=64, s_dim=64).to(DEVICE).float().state_dict()
_sa_sd = STAGE_A.state_dict()
assert set(_sa_sd) == set(_REF_INIT)
assert all(torch.equal(_sa_sd[k], _REF_INIT[k]) for k in _sa_sd if "funnel" not in k), "stage A changed law_brain"
del _REF_INIT, _sa_sd

HOP_TAG = f"HOP-{VARIANT}"
hop_head = build_variant_head(STAGE_A, pos=CFG["pos"])
hop_head = train_hop(hop_head, HOP_TAG, CFG["loss"])
_sd = hop_head.state_dict()
assert all(torch.equal(_sd[k], v) for k, v in _STAGE_A_FUNNEL.items()), f"{HOP_TAG}: funnel changed during hop training"

# RANDOM HOP: same stage-A funnel; law_brain / next_law_proj (+ pos) at their seeded init; never trained
rand_head = build_variant_head(STAGE_A, pos=CFG["pos"])
rand_head.eval()
_rsd = rand_head.state_dict()
assert all(torch.equal(_rsd[k], v) for k, v in STAGE_A.state_dict().items()), "RANDOM HOP must be stage A + seeded init"
print(f"\n[v5] RANDOM HOP head: stage-A funnel + law_brain/next_law_proj{' + pos' if CFG['pos'] else ''} at the seed-{SEED} "
      f"init, untrained", flush=True)

# V0 joint (secondary): Lab 09 joint training (copied from Lab 11)
v0_head = train_head("V0")

# [lab12] reproducibility fingerprint (exact float sums; compare across re-runs on the same hardware)
with torch.no_grad():
    _fp = lambda m: float(sum(p.double().sum() for p in m.state_dict().values()))
    print(f"[v5] fingerprint seed={SEED}: H_ALL={float(H_ALL.double().sum()).hex()} stageA={_fp(STAGE_A).hex()} "
          f"hop={_fp(hop_head).hex()} v0={_fp(v0_head).hex()}", flush=True)

TF_EVAL = {HOP_TAG: eval_mode_hop_metrics(hop_head), "RANDOM HOP": eval_mode_hop_metrics(rand_head),
           "V0": eval_mode_hop_metrics(v0_head)}
FUNNEL_OK, MAX_OFF = {}, {}
for _fn, _h in (("stageA", STAGE_A), ("V0", v0_head)):
    FUNNEL_OK[_fn], MAX_OFF[_fn] = funnel_checks(_h, _fn)
assert all(FUNNEL_OK.values()), f"LAST is not exactly 50% (design/indexing bug): {FUNNEL_OK}"


# ==============================================================================
# [5] Evaluation of the head-based methods
# ==============================================================================
RESULTS = {}
EV_HOP = eval_hop(hop_head, MODE)
EV_RAND = eval_hop(rand_head, MODE)
EV_V0 = eval_hop(v0_head, "free")
RULES = eval_rules(hop_head)  # LAST / S0-RULE: the HOP head's (= stage-A funnel's) s-space and anchors
RESULTS["hop"], RESULTS["random_hop"], RESULTS["v0_joint"] = EV_HOP["score"], EV_RAND["score"], EV_V0["score"]
RESULTS["last"], RESULTS["s0_rule"] = RULES["last"], RULES["s0r"]
print(f"[v5] LAST and S0-RULE use the HOP head's s-space and anchors (stage-A funnel, frozen)", flush=True)

LAST_OK = True
for k in EVAL_STAGES:
    for key in (f"BR{k}", f"DEC{k}"):
        v = RESULTS["last"][key]
        ok = v * 2 == N_ITEMS
        LAST_OK &= ok
        print(f"[check] LAST {key} = {v}/{N_ITEMS} == {N_ITEMS // 2}/{N_ITEMS}: {'OK' if ok else 'FAIL (design/indexing bug)'}",
              flush=True)
assert LAST_OK, "LAST must be exactly 50% for BR and DEC at every stage"


# ==============================================================================
# [6] PREMISE-MLP: premise law classifier + per-sentence 27-state classifier (copied from Lab 11 / Lab 09)
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
RESULTS["premise_mlp"] = pm


# ==============================================================================
# [7] Diagnostics (secondary; not part of the registered criteria)
# ==============================================================================
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


DIAG = []  # (machine name, label, score, n)
DIAG.append(("true_law_anchor_picker", "true law -> anchor picker (stage-A funnel = HOP space) [ceiling]",
             anchor_picker(hop_head, IT_LAW), N_ITEMS))
DIAG.append(("true_law_anchor_picker_v0", "true law -> anchor picker (V0 funnel) [ceiling]",
             anchor_picker(v0_head, IT_LAW), N_ITEMS))
DIAG.append(("true_law_mlp_picker", "true law -> MLP picker [ceiling]", mlp_picker(IT_LAW), N_ITEMS))
DIAG.append(("premise_law_anchor_picker", "premise-MLP law -> anchor picker (stage-A funnel)",
             anchor_picker(hop_head, PM_LAW), N_ITEMS))
DIAG.append(("hop_dec_law_mlp_picker", "HOP DEC law at stage k -> MLP picker",
             mlp_picker({k: EV_HOP["picks"]["dec"][k] for k in EVAL_STAGES}), N_ITEMS))
DIAG.append(("premise_dec_law_mlp_picker", "premise-MLP DEC law (a vs b) -> MLP picker",
             mlp_picker(PM_DEC.tolist()), N_ITEMS))
DIAG.append(("seen_hop", f"seen-domain BR, HOP ({VARIANT}, {MODE})", EV_HOP["seen"], N_SEEN_ITEMS))
DIAG.append(("seen_v0_joint", "seen-domain BR, V0 joint (free)", EV_V0["seen"], N_SEEN_ITEMS))
DIAG.append(("seen_premise_mlp", "seen-domain BR, PREMISE-MLP", PM_SEEN, N_SEEN_ITEMS))


# ==============================================================================
# [8] Qwen LM ranking (copied from Lab 09; base model, no LoRA): mean log-prob of each candidate given the prompt
# ==============================================================================
@torch.no_grad()
def cand_mean_logprob(prompt: str, cand: str) -> float:
    # Tokenise prompt and candidate separately and concatenate (Lab 07 review fix).
    p_ids = tokenizer(prompt).input_ids
    c_ids = tokenizer(cand).input_ids
    f_ids = p_ids + c_ids
    n_c = len(c_ids)
    if not (1 <= n_c < len(f_ids)):
        raise RuntimeError(f"bad candidate span: {prompt!r} {cand!r} {n_c}")
    ids = torch.tensor([f_ids], device=DEVICE)
    lp = F.log_softmax(qwen_base(input_ids=ids).logits.float()[0, :-1, :], dim=-1)
    tok_lp = lp.gather(-1, ids[0, 1:].unsqueeze(-1))[:, 0]
    return float(tok_lp[-n_c:].mean().item())


def eval_lm(prefix: str, tag: str):
    """(copied from Lab 09) [lab12] DEC-k = BR-k."""
    sc = new_score()
    for k in EVAL_STAGES:
        sc[f"BR{k}"] = 0
    for i, (ui, law) in enumerate(ITEMS):
        prompt = prefix + ALL_TEXTS[PREM_UNSEEN_G[(ui, law)]]
        for k in EVAL_STAGES:
            cands = [ALL_TEXTS[UNSEEN_G[(ui, cl, k)]] for cl in CAND_LAWS[i]]
            lp = [cand_mean_logprob(prompt, t) for t in cands]
            j = 0 if lp[0] >= lp[1] else 1
            sc[f"BR{k}"] += int(CAND_LAWS[i][j] == law)
            prompt = prompt + cands[j]  # next stage conditioned on the method's own pick
    for k in EVAL_STAGES:
        sc[f"DEC{k}"] = sc[f"BR{k}"]  # [lab12]
    print(f"[v5] {tag}: BR2={sc['BR2']}/{N_ITEMS} BR3={sc['BR3']}/{N_ITEMS} BR4={sc['BR4']}/{N_ITEMS}", flush=True)
    return sc


FEWSHOT_PREFIX = "\n".join("".join(DATA["seen_domains"][0]["chains"][str(l)]) for l in range(N_LAWS)) + "\n"
for _line in FEWSHOT_PREFIX.split("\n"):
    if _line:
        assert_not_forbidden(_line)
assert_not_forbidden(FEWSHOT_PREFIX)  # [lab12] also the whole prefix
print(f"\n=== [8] Qwen LM ranking ===\n[v5] few-shot prefix (seen domain '{SEEN_DOMS[0]}'):\n{FEWSHOT_PREFIX}", flush=True)
RESULTS["qwen_zeroshot"] = eval_lm("", "Qwen zero-shot")
RESULTS["qwen_fewshot"] = eval_lm(FEWSHOT_PREFIX, "Qwen few-shot")


# ==============================================================================
# [9] Report (numbers only; the pre-registered criteria are applied by the reader)
# ==============================================================================
METHODS = [  # (machine tag, label)
    ("hop", f"HOP ({VARIANT}, {MODE}) [primary]"),
    ("random_hop", f"RANDOM HOP ({VARIANT}, {MODE})"),
    ("v0_joint", "V0 joint (free) [secondary]"),
    ("last", "LAST"),
    ("s0_rule", "S0-RULE"),
    ("premise_mlp", "PREMISE-MLP"),
    ("qwen_zeroshot", "Qwen zero-shot"),
    ("qwen_fewshot", "Qwen few-shot"),
]
assert [m for m, _ in METHODS] == list(RESULTS), list(RESULTS)
for _m, _ in METHODS:
    assert all(0 <= RESULTS[_m][k] <= N_ITEMS for k in ("BR2", "BR3", "BR4", "DEC2", "DEC3", "DEC4")), _m


def cell(v, n=N_ITEMS):
    return f"{v:>2d}/{n}" if v >= 0 else "  -  "


def pct_mean(sc, keys, n=N_ITEMS):
    vals = [sc[k] for k in keys]
    return f"{100.0 * sum(vals) / (len(vals) * n):5.1f}%" if all(v >= 0 for v in vals) else "   -  "


BR_K, DEC_K = ["BR2", "BR3", "BR4"], ["DEC2", "DEC3", "DEC4"]
A_K, FL_K = ["A2", "A3", "A4"], ["FLIP2", "FLIP3", "FLIP4"]
print("\n" + "=" * 170, flush=True)
print_override_banner()
print(f"=== RESULTS seed={SEED} | HOP = {VARIANT} ({'pos' if CFG['pos'] else 'no pos'}, {CFG['loss']} loss), {MODE} rollout "
      f"(pre-registered {HOP_VARIANT}/{HOP_MODE}) | {N_ITEMS} unseen items ({N_UNSEEN_DOM} domains x {N_LAWS} laws) ===",
      flush=True)
print(f"BR-k = branch choice at stage k (chance {N_ITEMS // 2}/{N_ITEMS}) | DEC-k = pairwise branch decision a vs b, "
      f"independent of the candidate sentences | A-k = nearest anchor of ŝk is (law, k) (27 anchors) | "
      f"FLIP-k = BR picks that change when s0 -> partner's s0 | PREM = 6-way premise law correct (chance {N_ITEMS // N_LAWS}/{N_ITEMS})",
      flush=True)
hdr = (f"{'method':<32}" + "".join(f"{c:>7}" for c in ("BR-2", "BR-3", "BR-4")) + f"{'BR':>8}"
       + "".join(f"{c:>7}" for c in ("DEC-2", "DEC-3", "DEC-4")) + f"{'DEC':>8}"
       + "".join(f"{c:>7}" for c in ("A-2", "A-3", "A-4", "FLIP-2", "FLIP-3", "FLIP-4")) + f"{'PREM':>7}")
print(hdr, flush=True)
print("-" * len(hdr), flush=True)
for m, label in METHODS:
    sc = RESULTS[m]
    print(f"{label:<32}" + "".join(f"{cell(sc[c]):>7}" for c in BR_K) + f"{pct_mean(sc, BR_K):>8}"
          + "".join(f"{cell(sc[c]):>7}" for c in DEC_K) + f"{pct_mean(sc, DEC_K):>8}"
          + "".join(f"{cell(sc[c]):>7}" for c in A_K + FL_K) + f"{cell(sc['PREM']):>7}", flush=True)
print("-" * len(hdr), flush=True)
print(f"LAST exactly 50% at every stage (BR and DEC): {'OK' if LAST_OK else 'FAIL'}", flush=True)
print("BR / DEC = mean of the three stages (%)", flush=True)

print(f"\n=== HOP ({VARIANT}, {MODE}) per-item detail (✓ = law correct; BR = anchor-mediated pick; DEC = pairwise "
      f"decision; flip = BR pick after s0 -> partner's s0) ===", flush=True)
P = EV_HOP["picks"]
print(f"{'domain':<10} {'law':>3} | {'s0 near':<9} | {'ŝ2 near':<9} {'ŝ3 near':<9} {'ŝ4 near':<9} | "
      f"{'BR 2/3/4':<20} | {'DEC 2/3/4':<20} | {'flip 2/3/4':<16}", flush=True)
for i, (ui, L) in enumerate(ITEMS):
    mk = lambda v: f"L{v}{'✓' if v == L else ' '}"
    print(f"{UNSEEN_DOMS[ui]:<10} {('L' + str(L)):>3} | {clbl(P['near'][0][i]):<9} | "
          + " ".join(f"{clbl(P['near'][k][i]):<9}" for k in EVAL_STAGES) + " | "
          + " ".join(f"{mk(P['br'][k][i]):<6}" for k in EVAL_STAGES) + "  | "
          + " ".join(f"{mk(P['dec'][k][i]):<6}" for k in EVAL_STAGES) + "  | "
          + " ".join(f"L{P['flip'][k][i]:<5}" for k in EVAL_STAGES), flush=True)

print(f"\n=== Predicted laws per item (S0-RULE 6-way / S0-RULE DEC / PREMISE-MLP 6-way / PREMISE-MLP DEC / LAST) ===",
      flush=True)
for i, (ui, L) in enumerate(ITEMS):
    print(f"  {UNSEEN_DOMS[ui]:<10} L{L} | S0-RULE L{RULES['s0_law'][i]} dec L{RULES['s0_dec'][i]} | "
          f"PREMISE-MLP L{PM_LAW[i]} dec L{int(PM_DEC[i])} | LAST s1 near {clbl(RULES['n1'][i])}", flush=True)

print("\n=== Diagnostics (secondary; NOT part of the registered criteria) ===", flush=True)
for name, label, sc, n in DIAG:
    print(f"  {label:<58} BR2={cell(sc['BR2'], n)} BR3={cell(sc['BR3'], n)} BR4={cell(sc['BR4'], n)} "
          f"mean {pct_mean(sc, BR_K, n)}", flush=True)
print("  eval-mode teacher-forced hop metrics on the seen chains (dropout off):", flush=True)
for name, met in TF_EVAL.items():
    print(f"    {name:<12} {met}", flush=True)
for fn in ("stageA", "V0"):
    v, a, b = MAX_OFF[fn]
    print(f"  anchor geometry: funnel={fn:<7} max off-diagonal anchor cosine = {v:.3f} ({clbl(a)} vs {clbl(b)})", flush=True)
print(f"  LAST exactly 50% (BR and DEC, HOP s-space): {'OK' if LAST_OK else 'FAIL'} | LAST BR check per funnel: "
      + ", ".join(f"{fn} {'OK' if ok else 'FAIL'}" for fn, ok in FUNNEL_OK.items()), flush=True)

print(f"\n=== Training losses (stageA / V0 / MLPs: steps 1, 200, 400; {HOP_TAG} hop: steps "
      f"{', '.join(map(str, HOP_LOG_STEPS))}; train mode) ===", flush=True)
for tag, step, msg in LOSS_LOG:
    print(f"  {tag:<26} step {step:4d} | {msg}", flush=True)

print("", flush=True)
print_override_banner()
print(f"V5_CONFIG seed={SEED} variant={VARIANT} mode={MODE} override={int(OVERRIDE)}", flush=True)
for m, _label in METHODS:
    sc = RESULTS[m]
    print(f"V5_RESULT seed={SEED} method={m} " + " ".join(f"{k}={sc[k]}" for k in SCORE_KEYS), flush=True)
for name, _label, sc, n in DIAG:
    print(f"V5_DIAG seed={SEED} name={name} " + " ".join(f"{k}={sc[k]}" for k in BR_K)
          + (f" n={n}" if n != N_ITEMS else ""), flush=True)
print(f"[v5] total time {time.time()-T_START:.1f}s", flush=True)
