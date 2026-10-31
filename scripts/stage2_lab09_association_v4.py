# ==============================================================================
# Stage 2 / Lab 09: Association test v4 (pre-registered; spec = V4_SPEC "方法" / "判讀標準")
# Derived from Lab 07 (v3). v4 data: 6 laws in 3 pairs; within a domain the two laws of a pair share the
# IDENTICAL stage-1 sentence, so the branch at stages 2-4 is decided by stage 0 (the earlier context).
# Methods (all trained on SEEN domains only):
#   1 HOP          funnel + law_brain, hop 1/2/3 times from [s0, s1]; anchor-mediated pick (+ direct-cosine row)
#   2 LAST         s1 only (must be exactly 50% by construction; checked)
#   3 S0-RULE      nearest stage-0 anchor of s0 -> law -> candidate closest to anchor (law, k)
#   4 PREMISE-MLP  frozen-Qwen feature of s0+s1 -> MLP law classifier; per-sentence 27-state MLP picks candidate
#   5 Qwen LM      mean log-prob ranking, zero-shot and few-shot
#   6 RANDOM HOP   same DualHead, untrained (seeded)
#   FLIP           s0 replaced by the partner law's s0 (same domain); fraction of picks that change
# Method 7 (7B multiple-choice ceiling) is NOT run here.
# This script prints numbers only; it prints NO verdict.
# Run:  AUDIT_SEED=42 python stage2_lab09_association_v4.py   (seeds 42, 1, 2)
# ==============================================================================
import os, sys, time, math, random, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

SEED = int(os.environ.get("AUDIT_SEED", "42"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[v4] SEED={SEED} | device={DEVICE}", flush=True)


def seed_all(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(step: int, total: int, peak: float, floor: float) -> float:
    """Copied from Lab 01: cosine schedule from peak (step 0) to floor (step == total)."""
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * step / total))


# ==============================================================================
# [0] Data
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
print(f"[v4] data = {DATA_PATH} ({DATA.get('version')})", flush=True)

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


STAGE0_CLS = [cls_of(l, 0) for l in range(N_LAWS)]  # the 6 stage-0 anchors


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
print(f"[v4] seen sentences={N_SEEN_SENT} ({N_SEEN_DOM} domains x {N_CLS} state classes) | seen premises={len(PREM_SEEN)} | "
      f"hop chains={len(HOP_ROWS)} | unseen items={N_ITEMS} | leak check passed for all {len(_train_texts)} training texts",
      flush=True)

# Per item: candidate laws in fixed pair order (a, b) -- NOT (L, partner), so ties never favour the true law
CAND_LAWS = [PAIRS[PAIR_OF[law]] for (_ui, law) in ITEMS]


# ==============================================================================
# [1] Qwen (copied from Lab 01): bf16, frozen
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
print(f"[v4] extracted {H_ALL.shape[0]} features ({H_DIM}-d, float32)", flush=True)

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

HEAD_STEPS, HEAD_LR, HEAD_LR_FLOOR = 400, 1e-3, 1e-4
LOSS_LOG = []  # (model tag, step, description)


# ==============================================================================
# [2] Funnel + law_brain head (unchanged from Lab 07)
# ==============================================================================
class DualHead(nn.Module):
    def __init__(self, qwen_hidden: int = 896, d_dim: int = 64, s_dim: int = 64):
        super().__init__()
        self.domain_funnel = nn.Sequential(nn.Linear(qwen_hidden, 256), nn.GELU(), nn.Linear(256, d_dim))
        self.struct_funnel = nn.Sequential(nn.Linear(qwen_hidden, 256), nn.GELU(), nn.Linear(256, s_dim))
        enc_l = nn.TransformerEncoderLayer(d_model=s_dim, nhead=4, dim_feedforward=128, batch_first=True, norm_first=True)
        self.law_brain = nn.TransformerEncoder(enc_l, num_layers=2, enable_nested_tensor=False)
        self.next_law_proj = nn.Linear(s_dim, s_dim)

    def extract_ds(self, h_896: torch.Tensor):
        d = F.normalize(self.domain_funnel(h_896), p=2, dim=-1)
        s = F.normalize(self.struct_funnel(h_896), p=2, dim=-1)
        return d, s

    def hop_law(self, s_seq: torch.Tensor):
        B, T, _ = s_seq.shape
        c_mask = torch.triu(torch.ones(T, T, device=s_seq.device, dtype=torch.bool), diagonal=1)
        h = self.law_brain(s_seq, mask=c_mask)
        return F.normalize(self.next_law_proj(h), p=2, dim=-1)


def train_head(tag: str):
    seed_all(SEED)  # before model init
    head = DualHead(qwen_hidden=H_DIM, d_dim=64, s_dim=64).to(DEVICE).float()
    n_par = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"\n=== [2] Training {tag}: {n_par:,} trainable params | {HEAD_STEPS} steps full batch | "
          f"AdamW lr {HEAD_LR:g} -> {HEAD_LR_FLOOR:g} cosine | clip 1.0 ===", flush=True)
    seed_all(SEED)  # before training run
    dom_oh = F.one_hot(DOM_OF, N_SEEN_DOM).float()
    off_c = ~torch.eye(N_CLS, device=DEVICE, dtype=torch.bool)
    off_d = ~torch.eye(N_SEEN_DOM, device=DEVICE, dtype=torch.bool)
    opt = torch.optim.AdamW(head.parameters(), lr=HEAD_LR, weight_decay=0.01)
    head.train()
    for step in range(1, HEAD_STEPS + 1):
        for pg in opt.param_groups:
            pg['lr'] = cosine_lr(step, HEAD_STEPS, HEAD_LR, HEAD_LR_FLOOR)
        opt.zero_grad()
        d, s = head.extract_ds(H_SEEN)
        s_mean = F.normalize(CLS_OH.T @ s / CLS_OH.sum(0).unsqueeze(-1), p=2, dim=-1)  # (27, 64)
        d_mean = F.normalize(dom_oh.T @ d / dom_oh.sum(0).unsqueeze(-1), p=2, dim=-1)  # (12, 64)
        l_s_pull = (1.0 - (s * s_mean[SEEN_CLS]).sum(-1)).mean()
        _sp = F.relu((s_mean @ s_mean.T)[off_c] + 0.1)
        l_s_push = _sp.mean() + _sp.max()  # closest-pair term (from v3)
        l_d_pull = (1.0 - (d * d_mean[DOM_OF]).sum(-1)).mean()
        l_d_push = F.relu((d_mean @ d_mean.T)[off_d] + 0.1).mean()
        loss_dis = l_s_pull + l_s_push + l_d_pull + l_d_push
        s_ch = s[HOP_IDX]                                   # (72, 5, 64)
        pred = head.hop_law(s_ch[:, :-1, :])                # from positions 0..3, predicts positions 1..4
        loss_hop = (1.0 - (pred * s_ch[:, 1:, :].detach()).sum(-1)).mean()
        loss = loss_dis + 2.5 * loss_hop
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if step in (1, 200, 400):
            msg = (f"total {loss.item():.4f} | dis {loss_dis.item():.4f} (s_pull {l_s_pull.item():.3f} "
                   f"s_push {l_s_push.item():.3f} d_pull {l_d_pull.item():.3f} d_push {l_d_push.item():.3f}) | "
                   f"hop {loss_hop.item():.4f}")
            LOSS_LOG.append((tag, step, msg))
            print(f"  [{tag}] step {step:3d}/{HEAD_STEPS} | {msg}", flush=True)
    head.eval()
    return head


hop_head = train_head("HOP head")
seed_all(SEED)
rand_head = DualHead(qwen_hidden=H_DIM, d_dim=64, s_dim=64).to(DEVICE).float()
rand_head.eval()
print(f"\n[v4] RANDOM HOP head: untrained, init with seed {SEED}", flush=True)


# ==============================================================================
# [3] Evaluation of the head-based methods (HOP, LAST, S0-RULE, RANDOM HOP)
# ==============================================================================
def new_score():
    return {k: -1 for k in ("BR2", "BR3", "BR4", "A2", "A3", "A4", "FLIP2", "FLIP3", "FLIP4", "PREM")}


@torch.no_grad()
def rollout(head, s0, s1):
    """[s0, s1] -> hop 1/2/3 times feeding its own predictions -> {2: ŝ2, 3: ŝ3, 4: ŝ4}, each (B, dim)."""
    seq = torch.stack([s0, s1], dim=1)  # (B, 2, dim)
    hats = {}
    for k in EVAL_STAGES:
        nxt = head.hop_law(seq)[:, -1, :]
        hats[k] = nxt
        seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)
    return hats


def pick_by(target, cand):
    """target (B, dim), cand (B, 2, dim) -> picked law per item (B,) (pair order; ties -> first law of pair)."""
    j = (cand @ target.unsqueeze(-1)).squeeze(-1).argmax(dim=-1)  # (B,)
    return CAND_LAWS_T.gather(1, j.unsqueeze(1)).squeeze(1)


@torch.no_grad()
def eval_head(head):
    _d, S = head.extract_ds(H_ALL)
    anchors = F.normalize(CLS_OH.T @ S[SEEN_G] / CLS_OH.sum(0).unsqueeze(-1), p=2, dim=-1)  # (27, dim)
    gram = anchors @ anchors.T
    off = gram.masked_fill(torch.eye(N_CLS, device=DEVICE, dtype=torch.bool), -9.0)
    mi = int(off.argmax().item())
    max_off = (float(off.max().item()), mi // N_CLS, mi % N_CLS)

    s0, s0f, s1 = S[IT_S0], S[IT_S0_FLIP], S[IT_S1]
    hats, hats_f = rollout(head, s0, s1), rollout(head, s0f, s1)
    near = lambda v: (v @ anchors.T).argmax(dim=-1)  # (B,) nearest of the 27 anchors
    hop_a, hop_d = new_score(), new_score()
    picks = {"anchor": {}, "direct": {}, "anchor_flip": {}, "direct_flip": {}, "near": {}}
    for k in EVAL_STAGES:
        cand = S[IT_CAND[k]]  # (48, 2, dim)
        na, naf = near(hats[k]), near(hats_f[k])
        pa, pd = pick_by(anchors[na], cand), pick_by(hats[k], cand)
        paf, pdf = pick_by(anchors[naf], cand), pick_by(hats_f[k], cand)
        a_k = int((na == IT_TRUE_CLS[k]).sum().item())
        for sc, p, pf in ((hop_a, pa, paf), (hop_d, pd, pdf)):
            sc[f"BR{k}"] = int((p == IT_LAW_T).sum().item())
            sc[f"A{k}"] = a_k
            sc[f"FLIP{k}"] = int((p != pf).sum().item())
        picks["anchor"][k], picks["direct"][k] = pa.tolist(), pd.tolist()
        picks["anchor_flip"][k], picks["direct_flip"][k] = paf.tolist(), pdf.tolist()
        picks["near"][k] = na.tolist()
    picks["near"][0] = near(s0).tolist()

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
            sc[f"BR{k}"] = int((pick_by(tgt, cand) == IT_LAW_T).sum().item())
    return dict(hop_a=hop_a, hop_d=hop_d, last=last, s0r=s0r, picks=picks, anchors=anchors, max_off=max_off,
                n1=n1, s0_law=s0_law)


RESULTS = {}
EV_HOP = eval_head(hop_head)
EV_RAND = eval_head(rand_head)
RESULTS["HOP (anchor)"] = EV_HOP["hop_a"]
RESULTS["HOP (direct cos) [secondary]"] = EV_HOP["hop_d"]
RESULTS["LAST"] = EV_HOP["last"]
RESULTS["S0-RULE"] = EV_HOP["s0r"]
print("[v4] LAST and S0-RULE use the trained HOP head's s-space and anchors", flush=True)

LAST_OK = True
for k in EVAL_STAGES:
    ok = RESULTS["LAST"][f"BR{k}"] * 2 == N_ITEMS
    LAST_OK &= ok
    print(f"[check] LAST BR-{k} = {RESULTS['LAST'][f'BR{k}']}/{N_ITEMS} == {N_ITEMS // 2}/{N_ITEMS}: "
          f"{'OK' if ok else 'FAIL (design/indexing bug)'}", flush=True)


# ==============================================================================
# [4] PREMISE-MLP: premise law classifier + per-sentence 27-state classifier (frozen Qwen features)
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
    print(f"\n=== [4] Training {tag}: {n_par:,} params | {X.shape[0]} examples | {HEAD_STEPS} steps full batch | "
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
    pm = new_score()
    pm["PREM"] = sum(int(pl == L) for pl, L in zip(PM_LAW, IT_LAW))
    for k in EVAL_STAGES:
        # [review] log_softmax per candidate: raw logits are not comparable across two different inputs
        lg = F.log_softmax(sent_mlp(H_ALL[IT_CAND[k]]), dim=-1)  # (48, 2, 27)
        col = _t([cls_of(l, k) for l in PM_LAW])                   # (48,)
        sc2 = lg.gather(2, col.view(-1, 1, 1).expand(-1, 2, 1)).squeeze(-1)  # (48, 2): logit of class (pred law, k)
        p = CAND_LAWS_T.gather(1, sc2.argmax(-1, keepdim=True)).squeeze(1)
        pm[f"BR{k}"] = int((p == IT_LAW_T).sum().item())
RESULTS["PREMISE-MLP"] = pm


# ==============================================================================
# [4b] Diagnostics requested by the independent review (secondary; not part of the registered criteria)
# ==============================================================================
DIAG = {}


@torch.no_grad()
def anchor_picker(head, laws):
    """BR-k when the law is given by `laws` and the candidate is chosen with the HOP head's anchor picker."""
    _d, S = head.extract_ds(H_ALL)
    anchors = F.normalize(CLS_OH.T @ S[SEEN_G] / CLS_OH.sum(0).unsqueeze(-1), p=2, dim=-1)
    out = new_score()
    for k in EVAL_STAGES:
        tgt = anchors[_t([cls_of(l, k) for l in laws])]
        out[f"BR{k}"] = int((pick_by(tgt, S[IT_CAND[k]]) == IT_LAW_T).sum().item())
    return out


@torch.no_grad()
def mlp_picker(laws):
    """BR-k when the law is given by `laws` and the candidate is chosen with the sentence MLP (log-probs)."""
    out = new_score()
    for k in EVAL_STAGES:
        lg = F.log_softmax(sent_mlp(H_ALL[IT_CAND[k]]), dim=-1)
        col = _t([cls_of(l, k) for l in laws])
        sc2 = lg.gather(2, col.view(-1, 1, 1).expand(-1, 2, 1)).squeeze(-1)
        out[f"BR{k}"] = int((CAND_LAWS_T.gather(1, sc2.argmax(-1, keepdim=True)).squeeze(1) == IT_LAW_T).sum().item())
    return out


DIAG["premise-law -> anchor picker"] = anchor_picker(hop_head, PM_LAW)   # separates law inference from picking
DIAG["true law -> anchor picker"] = anchor_picker(hop_head, IT_LAW)      # ceiling of the anchor picker
DIAG["true law -> MLP picker"] = mlp_picker(IT_LAW)                       # ceiling of the sentence-MLP picker


@torch.no_grad()
def seen_domain_br():
    """BR-k on the SEEN domains (training data): distinguishes 'did not learn' from 'does not generalise'."""
    _d, S = hop_head.extract_ds(H_ALL)
    Ss = S[SEEN_G]
    anchors = F.normalize(CLS_OH.T @ Ss / CLS_OH.sum(0).unsqueeze(-1), p=2, dim=-1)
    items = [(di, law) for di in range(N_SEEN_DOM) for law in range(N_LAWS)]
    s0 = Ss[_t([SEEN_POS[(di, law, 0)] for (di, law) in items])]
    s1 = Ss[_t([SEEN_POS[(di, law, 1)] for (di, law) in items])]
    true = _t([law for (_di, law) in items])
    cl = _t([list(PAIRS[PAIR_OF[law]]) for (_di, law) in items])
    hats = rollout(hop_head, s0, s1)
    pl = prem_mlp(H_PREM_SEEN).argmax(-1).tolist()  # PREM_SEEN is ordered (domain, law) like `items`
    hop_sc, pm_sc = new_score(), new_score()
    for k in EVAL_STAGES:
        cand_rows = _t([[SEEN_POS[(di, c, k)] for c in PAIRS[PAIR_OF[law]]] for (di, law) in items])
        cand = Ss[cand_rows]
        near = (hats[k] @ anchors.T).argmax(-1)
        j = (cand @ anchors[near].unsqueeze(-1)).squeeze(-1).argmax(-1)
        hop_sc[f"BR{k}"] = int((cl.gather(1, j.unsqueeze(1)).squeeze(1) == true).sum().item())
        lg = F.log_softmax(sent_mlp(H_SEEN[cand_rows]), dim=-1)
        col = _t([cls_of(l, k) for l in pl])
        sc2 = lg.gather(2, col.view(-1, 1, 1).expand(-1, 2, 1)).squeeze(-1)
        pm_sc[f"BR{k}"] = int((cl.gather(1, sc2.argmax(-1, keepdim=True)).squeeze(1) == true).sum().item())
    return hop_sc, pm_sc, len(items)


SEEN_HOP, SEEN_PM, N_SEEN_ITEMS = seen_domain_br()


@torch.no_grad()
def eval_mode_hop_loss(head):
    """Teacher-forced hop loss on the seen chains in eval mode (should match the last training values)."""
    head.eval()
    _d, s = head.extract_ds(H_SEEN)
    s_ch = s[HOP_IDX]
    pred = head.hop_law(s_ch[:, :-1, :])
    return float((1.0 - (pred * s_ch[:, 1:, :]).sum(-1)).mean().item())


EVAL_HOP_LOSS = eval_mode_hop_loss(hop_head)


# ==============================================================================
# [5] Qwen LM ranking (base model, no LoRA): mean log-prob of each candidate given the prompt
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
    print(f"[v4] {tag}: BR2={sc['BR2']}/{N_ITEMS} BR3={sc['BR3']}/{N_ITEMS} BR4={sc['BR4']}/{N_ITEMS}", flush=True)
    return sc


FEWSHOT_PREFIX = "\n".join("".join(DATA["seen_domains"][0]["chains"][str(l)]) for l in range(N_LAWS)) + "\n"
for _line in FEWSHOT_PREFIX.split("\n"):
    if _line:
        assert_not_forbidden(_line)
print(f"\n=== [5] Qwen LM ranking ===\n[v4] few-shot prefix (seen domain '{SEEN_DOMS[0]}'):\n{FEWSHOT_PREFIX}", flush=True)
RESULTS["Qwen zero-shot"] = eval_lm("", "Qwen zero-shot")
RESULTS["Qwen few-shot"] = eval_lm(FEWSHOT_PREFIX, "Qwen few-shot")

RESULTS["RANDOM HOP (anchor)"] = EV_RAND["hop_a"]
RESULTS["RANDOM HOP (direct cos) [secondary]"] = EV_RAND["hop_d"]


# ==============================================================================
# [6] Report (numbers only; the pre-registered criteria are applied by the reader)
# ==============================================================================
def cell(v, n=N_ITEMS):
    return f"{v:>2d}/{n}" if v >= 0 else "  -  "


def pct_mean(sc, keys):
    vals = [sc[k] for k in keys]
    return f"{100.0 * sum(vals) / (len(vals) * N_ITEMS):5.1f}%" if all(v >= 0 for v in vals) else "   -  "


BR_K, A_K, FL_K = ["BR2", "BR3", "BR4"], ["A2", "A3", "A4"], ["FLIP2", "FLIP3", "FLIP4"]
print("\n" + "=" * 150, flush=True)
print(f"=== RESULTS seed={SEED} | {N_ITEMS} items ({N_UNSEEN_DOM} unseen domains x {N_LAWS} laws) | "
      f"BR-k = branch choice at stage k (chance 24/48) | A-k = nearest anchor of ŝk is (law, k) (27 anchors) | "
      f"FLIP-k = picks that change when s0 -> partner's s0 | PREM = predicted law correct (chance 8/48) ===", flush=True)
hdr = (f"{'method':<26}" + "".join(f"{c:>8}" for c in ("BR-2", "BR-3", "BR-4")) + f"{'BR':>9}"
       + "".join(f"{c:>8}" for c in ("A-2", "A-3", "A-4", "FLIP-2", "FLIP-3", "FLIP-4")) + f"{'FLIP':>9}{'PREM':>8}")
print(hdr, flush=True)
print("-" * len(hdr), flush=True)
for name, sc in RESULTS.items():
    print(f"{name:<26}" + "".join(f"{cell(sc[c]):>8}" for c in BR_K) + f"{pct_mean(sc, BR_K):>9}"
          + "".join(f"{cell(sc[c]):>8}" for c in A_K + FL_K) + f"{pct_mean(sc, FL_K):>9}" + f"{cell(sc['PREM']):>8}",
          flush=True)
print("-" * len(hdr), flush=True)
print(f"LAST exactly 50% at every stage: {'OK' if LAST_OK else 'FAIL'}", flush=True)
print("BR = mean of BR-2..4 (%); FLIP = mean of FLIP-2..4 (%)", flush=True)

print(f"\n=== HOP per-item detail (anchor-mediated picks; ✓ = law correct; flip = pick after s0 -> partner's s0) ===", flush=True)
P = EV_HOP["picks"]
print(f"{'domain':<10} {'law':>3} | {'s0 near':<9} | {'ŝ2 near':<9} {'ŝ3 near':<9} {'ŝ4 near':<9} | "
      f"{'pick 2/3/4 (anchor)':<22} | {'pick 2/3/4 (direct)':<22} | {'flip 2/3/4 (anchor)':<20}", flush=True)
for i, (ui, L) in enumerate(ITEMS):
    mk = lambda v: f"L{v}{'✓' if v == L else ' '}"
    print(f"{UNSEEN_DOMS[ui]:<10} {('L' + str(L)):>3} | {clbl(P['near'][0][i]):<9} | "
          + " ".join(f"{clbl(P['near'][k][i]):<9}" for k in EVAL_STAGES) + " | "
          + " ".join(f"{mk(P['anchor'][k][i]):<6}" for k in EVAL_STAGES) + "    | "
          + " ".join(f"{mk(P['direct'][k][i]):<6}" for k in EVAL_STAGES) + "    | "
          + " ".join(f"L{P['anchor_flip'][k][i]:<5}" for k in EVAL_STAGES), flush=True)

print(f"\n=== Predicted laws per item (S0-RULE / PREMISE-MLP / LAST) ===", flush=True)
for i, (ui, L) in enumerate(ITEMS):
    print(f"  {UNSEEN_DOMS[ui]:<10} L{L} | S0-RULE L{EV_HOP['s0_law'][i]} | PREMISE-MLP L{PM_LAW[i]} | "
          f"LAST s1 near {clbl(EV_HOP['n1'][i])}", flush=True)

print(f"\n=== Training losses (steps 1 / 200 / 400) ===", flush=True)
for tag, step, msg in LOSS_LOG:
    print(f"  {tag:<26} step {step:3d} | {msg}", flush=True)

print(f"\n=== Anchor geometry (27 anchors) ===", flush=True)
for name, ev in (("HOP head", EV_HOP), ("RANDOM HOP head", EV_RAND)):
    v, a, b = ev["max_off"]
    print(f"  {name:<16} max off-diagonal anchor cosine = {v:.3f}  ({clbl(a)} vs {clbl(b)})", flush=True)
with torch.no_grad():
    _gram = (EV_HOP["anchors"] @ EV_HOP["anchors"].T).cpu().numpy()
print("  HOP head anchor cosine matrix (rows/cols in class order):", flush=True)
print("          " + " ".join(f"{clbl(c):>8}" for c in range(N_CLS)), flush=True)
for i in range(N_CLS):
    print(f"  {clbl(i):>8} " + " ".join(f"{_gram[i, j]:>8.2f}" for j in range(N_CLS)), flush=True)

print("\n=== [review] diagnostics (secondary; not in the registered criteria) ===", flush=True)
for name, sc in DIAG.items():
    print(f"  {name:<32} BR2={cell(sc['BR2'])} BR3={cell(sc['BR3'])} BR4={cell(sc['BR4'])} mean {pct_mean(sc, ['BR2', 'BR3', 'BR4'])}", flush=True)
print(f"  seen-domain BR (training data, {N_SEEN_ITEMS} items): HOP " + " ".join(f"BR{k}={SEEN_HOP[f'BR{k}']}/{N_SEEN_ITEMS}" for k in EVAL_STAGES)
      + " | PREMISE-MLP " + " ".join(f"BR{k}={SEEN_PM[f'BR{k}']}/{N_SEEN_ITEMS}" for k in EVAL_STAGES), flush=True)
print(f"  eval-mode teacher-forced hop loss on seen chains: {EVAL_HOP_LOSS:.4f}", flush=True)
for name, sc in DIAG.items():
    print(f"V4_DIAG seed={SEED} name={name.replace(' ', '_')} BR2={sc['BR2']} BR3={sc['BR3']} BR4={sc['BR4']}", flush=True)
print(f"V4_DIAG seed={SEED} name=seen_hop " + " ".join(f"BR{k}={SEEN_HOP[f'BR{k}']}" for k in EVAL_STAGES) + f" n={N_SEEN_ITEMS}", flush=True)
print(f"V4_DIAG seed={SEED} name=seen_premise_mlp " + " ".join(f"BR{k}={SEEN_PM[f'BR{k}']}" for k in EVAL_STAGES) + f" n={N_SEEN_ITEMS}", flush=True)

METHOD_TAGS = {"HOP (anchor)": "hop", "HOP (direct cos) [secondary]": "hop_direct", "LAST": "last", "S0-RULE": "s0_rule",
               "PREMISE-MLP": "premise_mlp", "Qwen zero-shot": "qwen_zeroshot", "Qwen few-shot": "qwen_fewshot",
               "RANDOM HOP (anchor)": "random_hop", "RANDOM HOP (direct cos) [secondary]": "random_hop_direct"}
print("", flush=True)
for name, sc in RESULTS.items():
    print(f"V4_RESULT seed={SEED} method={METHOD_TAGS[name]} BR2={sc['BR2']} BR3={sc['BR3']} BR4={sc['BR4']} "
          f"A2={sc['A2']} A3={sc['A3']} A4={sc['A4']} FLIP2={sc['FLIP2']} FLIP3={sc['FLIP3']} FLIP4={sc['FLIP4']} "
          f"PREM={sc['PREM']}", flush=True)
