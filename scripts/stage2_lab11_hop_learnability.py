# ==============================================================================
# Stage 2 / Lab 11: Hop learnability diagnosis (exploratory, after v4; spec = LEARN_SPEC)
# Derived from Lab 09 (v4). SAME data, SAME evaluation; only the way the hop (law_brain) is trained changes.
# Variants (all trained on SEEN domains only):
#   V0  Lab 09 unchanged: funnel + law_brain trained jointly, 400 steps, cosine-regression hop loss
#       (must reproduce Lab 09's HOP numbers for the same seed; checked against the Lab 09 logs below)
#   V1  two-stage: stage A = funnel only (loss_dis, no hop) 400 steps with Lab 09's schedule, then the funnel
#       is frozen and law_brain + next_law_proj are trained alone for 2000 steps (cosine regression on frozen s)
#   V2  V1 + learned positional embeddings (5 x 64, N(0, 0.02)) added to the s sequence before law_brain
#   V3  V1 but hop loss = cross-entropy over the 27 frozen anchors: logits = pred @ anchors.T / 0.1,
#       target = state class of the next position
#   V4  V3 + positional embeddings
# The stage-A funnel is trained once per seed and shared by V1-V4 (each variant re-seeds before its hop training).
# Inference modes for every variant:
#   free  Lab 09 rollout: the predicted vector is fed back as is
#   snap  after each hop the predicted vector is replaced by its nearest of the 27 anchors (before feeding it
#         back and before picking)
# Evaluation (Lab 09 eval_head logic): anchor-mediated pick between the two candidates of the pair; A-k; FLIP-k
# (s0 -> partner law's s0, same domain); seen-domain BR on the 72 training chains.
# Reference: PREMISE-MLP (Lab 09). Ceiling: true law -> anchor picker (per funnel).
# Machine-readable lines: HL_RESULT / HL_REF / HL_CEIL (+ HL_LOSS, HL_V0REPRO). This script prints NO verdict.
# Run:  AUDIT_SEED=42 python stage2_lab11_hop_learnability.py   (seeds 42, 1, 2)
# ==============================================================================
import os, sys, time, math, random, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

SEED = int(os.environ.get("AUDIT_SEED", "42"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[lab11] SEED={SEED} | device={DEVICE}", flush=True)
T_START = time.time()


def seed_all(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(step: int, total: int, peak: float, floor: float) -> float:
    """Copied from Lab 01: cosine schedule from peak (step 0) to floor (step == total)."""
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * step / total))


# ==============================================================================
# [0] Data (copied from Lab 09)
# ==============================================================================
_DATA_CANDIDATES = ["/content/analogy_v4.json", "benchmarks/analogy_v4.json"]
try:
    _DATA_CANDIDATES.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks", "analogy_v4.json"))
except NameError:
    pass
DATA_PATH = next((p for p in _DATA_CANDIDATES if os.path.exists(p)), None)
if DATA_PATH is None:
    raise FileNotFoundError(f"analogy_v4.json not found in {_DATA_CANDIDATES}")
with open(DATA_PATH, encoding="utf-8") as f:
    DATA = json.load(f)
print(f"[lab11] data = {DATA_PATH} ({DATA.get('version')})", flush=True)

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
assert N_SEEN_DOM == 12 and N_UNSEEN_DOM == 8, (N_SEEN_DOM, N_UNSEEN_DOM)

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
assert N_ITEMS == 48


def premise_text(dom, law: int) -> str:
    return dom["chains"][str(law)][0] + dom["chains"][str(law)][1]


PREM_SEEN = []  # (global idx, seen domain, law): 72 seen premises s0+s1
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
assert len(HOP_ROWS) == N_SEEN_DOM * N_LAWS == 72

# Leak check over EVERY training text: seen sentences (funnel + sentence MLP + hop chains), seen premises (premise MLP)
_train_texts = [ALL_TEXTS[g] for (g, _d, _c) in SEEN] + [ALL_TEXTS[g] for (g, _d, _l) in PREM_SEEN]
_train_texts += ["".join(ALL_TEXTS[SEEN[r][0]] for r in row) for row in HOP_ROWS]
for _t in _train_texts:
    assert_not_forbidden(_t)
print(f"[lab11] seen sentences={N_SEEN_SENT} ({N_SEEN_DOM} domains x {N_CLS} state classes) | seen premises={len(PREM_SEEN)} | "
      f"hop chains={len(HOP_ROWS)} | unseen items={N_ITEMS} | leak check passed for all {len(_train_texts)} training texts",
      flush=True)

# Per item: candidate laws in fixed pair order (a, b) -- NOT (L, partner), so ties never favour the true law
CAND_LAWS = [PAIRS[PAIR_OF[law]] for (_ui, law) in ITEMS]

# [lab11] Seen-domain items (72): same order as PREM_SEEN (domain, law) and as HOP_ROWS
SEEN_ITEMS = [(di, law) for di in range(N_SEEN_DOM) for law in range(N_LAWS)]
N_SEEN_ITEMS = len(SEEN_ITEMS)
assert N_SEEN_ITEMS == 72 and [(d, l) for (_g, d, l) in PREM_SEEN] == SEEN_ITEMS
assert all(HOP_ROWS[i][st] == SEEN_POS[(di, law, st)] for i, (di, law) in enumerate(SEEN_ITEMS) for st in range(N_STAGES))
# Hop-training targets: state class of positions 1..4 of every seen chain
HOP_TGT_CLS_LIST = [[SEEN_CLS_LIST[r] for r in row[1:]] for row in HOP_ROWS]
assert all(HOP_TGT_CLS_LIST[i][st - 1] == cls_of(law, st) for i, (_di, law) in enumerate(SEEN_ITEMS) for st in range(1, N_STAGES))


# ==============================================================================
# [1] Qwen (copied from Lab 01 / Lab 09): bf16, frozen
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
H_SEEN = H_ALL[SEEN_G]  # (324, 896): the only sentence features that enter a head / sentence-MLP loss
PREM_SEEN_G = torch.tensor([g for (g, _d, _l) in PREM_SEEN], device=DEVICE)
H_PREM_SEEN = H_ALL[PREM_SEEN_G]  # (72, 896): the only premise features that enter the premise-MLP loss
print(f"[lab11] extracted {H_ALL.shape[0]} features ({H_DIM}-d, float32)", flush=True)

SEEN_CLS = torch.tensor(SEEN_CLS_LIST, device=DEVICE)
DOM_OF = torch.tensor([d for (_g, d, _c) in SEEN], device=DEVICE)
HOP_IDX = torch.tensor(HOP_ROWS, device=DEVICE)  # (72, 5) rows into SEEN
CLS_OH = F.one_hot(SEEN_CLS, N_CLS).float()      # (324, 27)


def _t(lst):
    return torch.tensor(lst, device=DEVICE)


# Item index tensors (48,)
IT_S0 = _t([UNSEEN_G[(ui, law, 0)] for (ui, law) in ITEMS])
IT_S0_FLIP = _t([UNSEEN_G[(ui, PARTNER[law], 0)] for (ui, law) in ITEMS])
IT_S1 = _t([UNSEEN_G[(ui, law, 1)] for (ui, law) in ITEMS])
IT_CAND = {k: _t([[UNSEEN_G[(ui, cl, k)] for cl in CAND_LAWS[i]] for i, (ui, _law) in enumerate(ITEMS)])
           for k in EVAL_STAGES}  # (48, 2)
IT_LAW = [law for (_ui, law) in ITEMS]
IT_TRUE_CLS = {k: _t([cls_of(law, k) for law in IT_LAW]) for k in EVAL_STAGES}
CAND_LAWS_T = _t([list(c) for c in CAND_LAWS])  # (48, 2)
IT_LAW_T = _t(IT_LAW)

# [lab11] Seen-domain item tensors (72,): rows into SEEN (as in Lab 09 seen_domain_br)
SI_S0 = _t([SEEN_POS[(di, law, 0)] for (di, law) in SEEN_ITEMS])
SI_S1 = _t([SEEN_POS[(di, law, 1)] for (di, law) in SEEN_ITEMS])
SI_LAW_T = _t([law for (_di, law) in SEEN_ITEMS])
SI_CL = _t([list(PAIRS[PAIR_OF[law]]) for (_di, law) in SEEN_ITEMS])  # (72, 2) candidate laws, pair order
SI_CAND = {k: _t([[SEEN_POS[(di, c, k)] for c in PAIRS[PAIR_OF[law]]] for (di, law) in SEEN_ITEMS])
           for k in EVAL_STAGES}  # (72, 2) rows into SEEN
HOP_TGT_CLS = _t(HOP_TGT_CLS_LIST)  # (72, 4)
assert torch.equal(HOP_TGT_CLS, SEEN_CLS[HOP_IDX[:, 1:]])

HEAD_STEPS, HEAD_LR, HEAD_LR_FLOOR = 400, 1e-3, 1e-4   # Lab 09 (V0 and stage A)
HOP_STEPS, HOP_LR, HOP_LR_FLOOR = 2000, 1e-3, 1e-4     # [lab11] hop-only training (V1-V4)
HOP_LOG_STEPS = (1, 500, 1000, 2000)
CE_TAU = 0.1
VARIANTS = {  # name: (pos embeddings, hop loss); V0 = Lab 09 joint training
    "V0": dict(pos=False, loss="cos"),
    "V1": dict(pos=False, loss="cos"),
    "V2": dict(pos=True, loss="cos"),
    "V3": dict(pos=False, loss="ce"),
    "V4": dict(pos=True, loss="ce"),
}
MODES = ("free", "snap")
LOSS_LOG = []  # (tag, step, description)


# ==============================================================================
# [2] Funnel + law_brain head (Lab 09) with optional learned positional embeddings
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
    """(324, dim) seen s -> (27, dim) normalised class means (the 27 anchors)."""
    return F.normalize(CLS_OH.T @ s_seen / CLS_OH.sum(0).unsqueeze(-1), p=2, dim=-1)


DOM_OH = F.one_hot(DOM_OF, N_SEEN_DOM).float()
OFF_C = ~torch.eye(N_CLS, device=DEVICE, dtype=torch.bool)
OFF_D = ~torch.eye(N_SEEN_DOM, device=DEVICE, dtype=torch.bool)


def funnel_loss(head):
    """Lab 09 loss_dis (s pull/push with closest-pair term, d pull/push). Returns (loss_dis, parts, s);
    parts = the four component tensors (formatted with fmt_parts only at logging steps)."""
    d, s = head.extract_ds(H_SEEN)
    s_mean = F.normalize(CLS_OH.T @ s / CLS_OH.sum(0).unsqueeze(-1), p=2, dim=-1)  # (27, 64)
    d_mean = F.normalize(DOM_OH.T @ d / DOM_OH.sum(0).unsqueeze(-1), p=2, dim=-1)  # (12, 64)
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
    """pred / tgt_vec (72, 4, dim); kind 'cos' = Lab 09 cosine regression, 'ce' = CE over the 27 anchors (tau 0.1)."""
    if kind == "cos":
        return (1.0 - (pred * tgt_vec).sum(-1)).mean()
    if kind == "ce":
        logits = (pred @ anchors.T) / CE_TAU  # (72, 4, 27)
        return F.cross_entropy(logits.reshape(-1, N_CLS), HOP_TGT_CLS.reshape(-1))
    raise ValueError(kind)


@torch.no_grad()
def hop_metrics(pred, tgt_vec, anchors) -> str:
    """Teacher-forced diagnostics on the seen chains: cosine loss, CE loss, next-state class accuracy."""
    cos_l = float((1.0 - (pred * tgt_vec).sum(-1)).mean().item())
    logits = (pred @ anchors.T) / CE_TAU
    ce_l = float(F.cross_entropy(logits.reshape(-1, N_CLS), HOP_TGT_CLS.reshape(-1)).item())
    hit = logits.argmax(-1) == HOP_TGT_CLS  # (72, 4): targets at positions 1..4
    return (f"cos={cos_l:.4f} ce={ce_l:.4f} acc={int(hit.sum().item())}/{hit.numel()} "
            f"acc234={int(hit[:, 1:].sum().item())}/{hit[:, 1:].numel()}")


def train_head(tag: str):
    """V0: Lab 09 train_head unchanged (joint funnel + hop, 400 steps)."""
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
        s_ch = s[HOP_IDX]                                   # (72, 5, 64)
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
            print(f"HL_LOSS seed={SEED} variant={tag} step={step} loss={loss_hop.item():.4f} {met}", flush=True)
    head.eval()
    print(f"  [{tag}] trained in {time.time()-t1:.1f}s", flush=True)
    return head


def train_funnel_stage_a(tag: str = "stageA"):
    """V1-V4 stage A: funnel only (loss_dis, no hop), Lab 09's 400-step schedule. law_brain stays at its init."""
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
    """Fresh head (same init as stage A; + pos table if asked), stage-A weights loaded, funnel frozen."""
    seed_all(SEED)  # before model init: funnel/law_brain init identical to stage A's (pos is created last)
    head = DualHead(qwen_hidden=H_DIM, d_dim=64, s_dim=64, pos=pos).to(DEVICE).float()
    res = head.load_state_dict(stage_a.state_dict(), strict=False)
    assert list(res.unexpected_keys) == [] and list(res.missing_keys) == (["pos"] if pos else []), res
    for m in (head.domain_funnel, head.struct_funnel):
        for p in m.parameters():
            p.requires_grad = False
    return head


def train_hop(head, tag: str, kind: str):
    """V1-V4 stage B: law_brain + next_law_proj (+ pos) alone on the frozen s of the seen chains."""
    with torch.no_grad():
        _d, s = head.extract_ds(H_SEEN)   # frozen funnel -> fixed s
        anchors = class_anchors(s)         # 27 frozen anchors
    s_ch = s[HOP_IDX]                      # (72, 5, 64)
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
            print(f"HL_LOSS seed={SEED} variant={tag} step={step} loss={loss.item():.4f} {met}", flush=True)
    head.eval()
    print(f"  [{tag}] trained in {time.time()-t1:.1f}s", flush=True)
    return head


@torch.no_grad()
def eval_mode_hop_metrics(head) -> str:
    """Teacher-forced metrics on the seen chains in eval mode (dropout off)."""
    head.eval()
    _d, s = head.extract_ds(H_SEEN)
    s_ch = s[HOP_IDX]
    return hop_metrics(head.hop_law(s_ch[:, :-1, :]), s_ch[:, 1:, :], class_anchors(s))


# ==============================================================================
# [3] Evaluation (Lab 09 eval_head logic: anchor-mediated pick, A-k, FLIP-k; + seen-domain BR)
# ==============================================================================
@torch.no_grad()
def rollout(head, s0, s1, snap=None):
    """[s0, s1] -> hop 1/2/3 times feeding its own predictions -> {2: ŝ2, 3: ŝ3, 4: ŝ4}, each (B, dim).
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
    """target (B, dim), cand (B, 2, dim) -> picked law per item (B,) (pair order; ties -> first law of pair)."""
    cl = CAND_LAWS_T if cand_laws is None else cand_laws
    j = (cand @ target.unsqueeze(-1)).squeeze(-1).argmax(dim=-1)  # (B,)
    return cl.gather(1, j.unsqueeze(1)).squeeze(1)


def new_score():
    return {k: -1 for k in ("seenBR2", "seenBR3", "seenBR4", "BR2", "BR3", "BR4",
                            "A2", "A3", "A4", "FLIP2", "FLIP3", "FLIP4")}


@torch.no_grad()
def head_space(head):
    _d, S = head.extract_ds(H_ALL)
    anchors = F.normalize(CLS_OH.T @ S[SEEN_G] / CLS_OH.sum(0).unsqueeze(-1), p=2, dim=-1)  # (27, dim)
    return S, anchors


@torch.no_grad()
def eval_head(head, mode: str):
    assert mode in MODES
    head.eval()
    S, anchors = head_space(head)
    snap = anchors if mode == "snap" else None
    near = lambda v: (v @ anchors.T).argmax(dim=-1)  # (B,) nearest of the 27 anchors
    sc = new_score()
    # unseen items (48)
    s0, s0f, s1 = S[IT_S0], S[IT_S0_FLIP], S[IT_S1]
    hats, hats_f = rollout(head, s0, s1, snap), rollout(head, s0f, s1, snap)
    for k in EVAL_STAGES:
        cand = S[IT_CAND[k]]  # (48, 2, dim)
        na, naf = near(hats[k]), near(hats_f[k])
        pa, paf = pick_by(anchors[na], cand), pick_by(anchors[naf], cand)
        sc[f"BR{k}"] = int((pa == IT_LAW_T).sum().item())
        sc[f"A{k}"] = int((na == IT_TRUE_CLS[k]).sum().item())
        sc[f"FLIP{k}"] = int((pa != paf).sum().item())
    # seen-domain items (72; training chains)
    Ss = S[SEEN_G]
    hats_s = rollout(head, Ss[SI_S0], Ss[SI_S1], snap)
    for k in EVAL_STAGES:
        cand = Ss[SI_CAND[k]]  # (72, 2, dim)
        ns = near(hats_s[k])
        sc[f"seenBR{k}"] = int((pick_by(anchors[ns], cand, SI_CL) == SI_LAW_T).sum().item())
    return sc


@torch.no_grad()
def anchor_ceiling(head):
    """BR-k when the TRUE law is given and the candidate is chosen with the anchor picker (depends on the funnel only)."""
    S, anchors = head_space(head)
    Ss = S[SEEN_G]
    out = {}
    for k in EVAL_STAGES:
        tgt = anchors[_t([cls_of(l, k) for l in IT_LAW])]
        out[f"BR{k}"] = int((pick_by(tgt, S[IT_CAND[k]]) == IT_LAW_T).sum().item())
        tgt_s = anchors[_t([cls_of(law, k) for (_di, law) in SEEN_ITEMS])]
        out[f"seenBR{k}"] = int((pick_by(tgt_s, Ss[SI_CAND[k]], SI_CL) == SI_LAW_T).sum().item())
    return out


@torch.no_grad()
def funnel_checks(head, tag: str):
    """Lab 09 LAST sanity check (s1 only must be exactly 50%) + anchor geometry for this funnel."""
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
    print(f"[lab11] funnel={tag} max off-diagonal anchor cosine = {float(off.max().item()):.3f} "
          f"({clbl(mi // N_CLS)} vs {clbl(mi % N_CLS)})", flush=True)
    return ok_all


# ==============================================================================
# [4] Train + evaluate all variants
# ==============================================================================
HEADS, RES, CEIL, TF_EVAL = {}, {}, {}, {}

HEADS["V0"] = train_head("V0")
TF_EVAL["V0"] = eval_mode_hop_metrics(HEADS["V0"])
LAST_OK = funnel_checks(HEADS["V0"], "V0")
CEIL["V0"] = anchor_ceiling(HEADS["V0"])
for mode in MODES:
    RES[("V0", mode)] = eval_head(HEADS["V0"], mode)

STAGE_A = train_funnel_stage_a("stageA")
LAST_OK &= funnel_checks(STAGE_A, "stageA")
CEIL["stageA"] = anchor_ceiling(STAGE_A)
_STAGE_A_FUNNEL = {k: v.detach().clone() for k, v in STAGE_A.state_dict().items() if "funnel" in k}

for name in ("V1", "V2", "V3", "V4"):
    cfg = VARIANTS[name]
    head = build_variant_head(STAGE_A, pos=cfg["pos"])
    head = train_hop(head, name, cfg["loss"])
    sd = head.state_dict()
    assert all(torch.equal(sd[k], v) for k, v in _STAGE_A_FUNNEL.items()), f"{name}: funnel changed during hop training"
    HEADS[name] = head
    TF_EVAL[name] = eval_mode_hop_metrics(head)
    for mode in MODES:
        RES[(name, mode)] = eval_head(head, mode)


# ==============================================================================
# [5] PREMISE-MLP reference (copied from Lab 09): premise law classifier + per-sentence 27-state classifier
# ==============================================================================
class MLP(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 256), nn.GELU(), nn.Linear(256, 64), nn.GELU(), nn.Linear(64, n_out))

    def forward(self, x):
        return self.net(x)


def train_mlp(tag: str, X: torch.Tensor, y: torch.Tensor, n_out: int):
    seed_all(SEED)  # before model init
    m = MLP(X.shape[1], n_out).to(DEVICE).float()
    n_par = sum(p.numel() for p in m.parameters())
    print(f"\n=== [5] Training {tag}: {n_par:,} params | {X.shape[0]} examples | {HEAD_STEPS} steps full batch | "
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
    PM_LAW = prem_mlp(H_ALL[_t([PREM_UNSEEN_G[it] for it in ITEMS])]).argmax(-1).tolist()
    PM = {"PREM": sum(int(pl == L) for pl, L in zip(PM_LAW, IT_LAW))}
    for k in EVAL_STAGES:
        # [review] log_softmax per candidate: raw logits are not comparable across two different inputs
        lg = F.log_softmax(sent_mlp(H_ALL[IT_CAND[k]]), dim=-1)  # (48, 2, 27)
        col = _t([cls_of(l, k) for l in PM_LAW])                   # (48,)
        sc2 = lg.gather(2, col.view(-1, 1, 1).expand(-1, 2, 1)).squeeze(-1)  # (48, 2): logit of class (pred law, k)
        p = CAND_LAWS_T.gather(1, sc2.argmax(-1, keepdim=True)).squeeze(1)
        PM[f"BR{k}"] = int((p == IT_LAW_T).sum().item())
    # seen-domain PREMISE-MLP (Lab 09 seen_domain_br)
    pl_seen = prem_mlp(H_PREM_SEEN).argmax(-1).tolist()  # PREM_SEEN is ordered (domain, law) like SEEN_ITEMS
    PM_SEEN = {}
    for k in EVAL_STAGES:
        lg = F.log_softmax(sent_mlp(H_SEEN[SI_CAND[k]]), dim=-1)
        col = _t([cls_of(l, k) for l in pl_seen])
        sc2 = lg.gather(2, col.view(-1, 1, 1).expand(-1, 2, 1)).squeeze(-1)
        PM_SEEN[f"BR{k}"] = int((SI_CL.gather(1, sc2.argmax(-1, keepdim=True)).squeeze(1) == SI_LAW_T).sum().item())


# ==============================================================================
# [6] Report (numbers only; the exploratory criteria in LEARN_SPEC are applied by the reader)
# ==============================================================================
# Lab 09 numbers (scratchpad runs/v4_seed*.log, same data file): V0 free must reproduce them for the same seed.
LAB09_REF = {
    42: dict(BR2=26, BR3=24, BR4=24, A2=10, A3=6, A4=6, FLIP2=4, FLIP3=8, FLIP4=8, seenBR2=49, seenBR3=37, seenBR4=36,
             ceilBR2=29, ceilBR3=41, ceilBR4=39, pmBR2=32, pmBR3=39, pmBR4=39, pmseenBR2=72, pmseenBR3=72, pmseenBR4=72),
    1: dict(BR2=27, BR3=28, BR4=31, A2=11, A3=3, A4=14, FLIP2=6, FLIP3=12, FLIP4=18, seenBR2=43, seenBR3=51, seenBR4=48,
            ceilBR2=37, ceilBR3=42, ceilBR4=41, pmBR2=33, pmBR3=40, pmBR4=39, pmseenBR2=72, pmseenBR3=72, pmseenBR4=72),
    2: dict(BR2=28, BR3=26, BR4=30, A2=15, A3=9, A4=20, FLIP2=8, FLIP3=12, FLIP4=20, seenBR2=56, seenBR3=45, seenBR4=48,
            ceilBR2=36, ceilBR3=43, ceilBR4=37, pmBR2=34, pmBR3=39, pmBR4=39, pmseenBR2=72, pmseenBR3=72, pmseenBR4=72),
}


def pct(sc, keys, n):
    vals = [sc[k] for k in keys]
    return f"{100.0 * sum(vals) / (len(vals) * n):5.1f}%"


SEEN_K, BR_K = ["seenBR2", "seenBR3", "seenBR4"], ["BR2", "BR3", "BR4"]
A_K, FL_K = ["A2", "A3", "A4"], ["FLIP2", "FLIP3", "FLIP4"]
print("\n" + "=" * 150, flush=True)
print(f"=== RESULTS seed={SEED} | seen BR-k: {N_SEEN_ITEMS} training chains ({N_SEEN_DOM} seen domains x {N_LAWS} laws) | "
      f"BR-k / A-k / FLIP-k: {N_ITEMS} unseen items ({N_UNSEEN_DOM} domains x {N_LAWS} laws) | chance BR = 50% | "
      f"A-k = nearest anchor of ŝk is (law, k) | FLIP-k = picks that change when s0 -> partner's s0 ===", flush=True)
hdr = (f"{'variant':<8}{'mode':<6}" + "".join(f"{c:>9}" for c in ("sBR-2", "sBR-3", "sBR-4")) + f"{'seen':>8}"
       + "".join(f"{c:>8}" for c in ("BR-2", "BR-3", "BR-4")) + f"{'unseen':>8}"
       + "".join(f"{c:>8}" for c in ("A-2", "A-3", "A-4", "FLIP-2", "FLIP-3", "FLIP-4")))
print(hdr, flush=True)
print("-" * len(hdr), flush=True)
for name in VARIANTS:
    for mode in MODES:
        sc = RES[(name, mode)]
        print(f"{name:<8}{mode:<6}" + "".join(f"{sc[c]:>6d}/{N_SEEN_ITEMS}" for c in SEEN_K) + f"{pct(sc, SEEN_K, N_SEEN_ITEMS):>8}"
              + "".join(f"{sc[c]:>5d}/{N_ITEMS}" for c in BR_K) + f"{pct(sc, BR_K, N_ITEMS):>8}"
              + "".join(f"{sc[c]:>5d}/{N_ITEMS}" for c in A_K + FL_K), flush=True)
print("-" * len(hdr), flush=True)
print(f"{'PREMISE-MLP':<14}" + "".join(f"{PM_SEEN[c]:>6d}/{N_SEEN_ITEMS}" for c in BR_K) + f"{pct(PM_SEEN, BR_K, N_SEEN_ITEMS):>8}"
      + "".join(f"{PM[c]:>5d}/{N_ITEMS}" for c in BR_K) + f"{pct(PM, BR_K, N_ITEMS):>8}"
      + f"   (PREM = predicted law correct on unseen: {PM['PREM']}/{N_ITEMS})", flush=True)
for fn in ("V0", "stageA"):
    c = CEIL[fn]
    print(f"{'CEIL ' + fn:<14}" + "".join(f"{c[k]:>6d}/{N_SEEN_ITEMS}" for k in SEEN_K) + f"{pct(c, SEEN_K, N_SEEN_ITEMS):>8}"
          + "".join(f"{c[k]:>5d}/{N_ITEMS}" for k in BR_K) + f"{pct(c, BR_K, N_ITEMS):>8}"
          + "   (true law -> anchor picker)", flush=True)
print(f"LAST exactly 50% at every stage (both funnels): {'OK' if LAST_OK else 'FAIL'}", flush=True)

print(f"\n=== Training losses (V0 / stageA / MLPs: steps 1, 200, 400; V1-V4 hop: steps {', '.join(map(str, HOP_LOG_STEPS))}; "
      f"train mode; acc = teacher-forced next-state class over the 27 anchors) ===", flush=True)
for tag, step, msg in LOSS_LOG:
    print(f"  {tag:<26} step {step:4d} | {msg}", flush=True)
print("  eval-mode teacher-forced on the seen chains (dropout off):", flush=True)
for name in VARIANTS:
    print(f"    {name}: {TF_EVAL[name]}", flush=True)

print(f"\n=== V0 reproduction of Lab 09 (seed {SEED}) ===", flush=True)
v0 = RES[("V0", "free")]
V0_NOW = dict(**{k: v0[k] for k in BR_K + A_K + FL_K + SEEN_K},
              **{f"ceil{k}": CEIL["V0"][k] for k in BR_K}, **{f"pm{k}": PM[k] for k in BR_K},
              **{f"pmseen{k}": PM_SEEN[k] for k in BR_K})
print(f"HL_V0REPRO seed={SEED} " + " ".join(f"{k}={v}" for k, v in V0_NOW.items()), flush=True)
if SEED in LAB09_REF:
    diffs = [f"{k}: lab11 {V0_NOW[k]} vs lab09 {v}" for k, v in LAB09_REF[SEED].items() if V0_NOW[k] != v]
    print(f"[check] V0 (free) + ceiling + PREMISE-MLP vs Lab 09 seed {SEED}: "
          + ("identical" if not diffs else "DIFFERENT -> " + "; ".join(diffs)), flush=True)
else:
    print(f"[check] no Lab 09 reference numbers stored for seed {SEED}", flush=True)

print("", flush=True)
for name in VARIANTS:
    for mode in MODES:
        sc = RES[(name, mode)]
        print(f"HL_RESULT seed={SEED} variant={name} mode={mode} " + " ".join(f"{k}={sc[k]}" for k in SEEN_K + BR_K)
              + f" A2={sc['A2']} FLIP2={sc['FLIP2']} A3={sc['A3']} A4={sc['A4']} FLIP3={sc['FLIP3']} FLIP4={sc['FLIP4']}",
              flush=True)
print(f"HL_REF seed={SEED} name=premise_mlp " + " ".join(f"{k}={PM[k]}" for k in BR_K) + f" PREM={PM['PREM']}", flush=True)
print(f"HL_REF seed={SEED} name=seen_premise_mlp " + " ".join(f"{k}={PM_SEEN[k]}" for k in BR_K) + f" n={N_SEEN_ITEMS}",
      flush=True)
for fn in ("V0", "stageA"):
    print(f"HL_CEIL seed={SEED} funnel={fn} " + " ".join(f"{k}={CEIL[fn][k]}" for k in BR_K)
          + " " + " ".join(f"{k}={CEIL[fn][k]}" for k in SEEN_K), flush=True)
print(f"[lab11] total time {time.time()-T_START:.1f}s", flush=True)
