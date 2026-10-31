# ==============================================================================
# Stage 2 / Lab 04: Analogy test v2 (pre-registered; spec = issue #7 "類比測試 v2")
# Frozen Qwen2.5-0.5B + funnel / law_brain head, trained on SEEN domains only.
# Evaluated on 36 unseen items (12 unseen domains x 3 laws) + 9 cue-swap items,
# against 7 controls. This script prints numbers only; it prints NO verdict.
# Run:  AUDIT_SEED=42 python stage2_lab04_analogy_v2.py   (seeds 42, 1, 2)
# ==============================================================================
import os, sys, time, math, random, json, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

SEED = int(os.environ.get("AUDIT_SEED", "42"))
print(f"[v2] SEED={SEED}", flush=True)


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
_DATA_CANDIDATES = ["/content/analogy_v2.json", "benchmarks/analogy_v2.json"]
try:
    _DATA_CANDIDATES.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks", "analogy_v2.json"))
except NameError:
    pass
DATA_PATH = next((p for p in _DATA_CANDIDATES if os.path.exists(p)), None)
if DATA_PATH is None:
    raise FileNotFoundError(f"analogy_v2.json not found in {_DATA_CANDIDATES}")
with open(DATA_PATH, encoding="utf-8") as f:
    DATA = json.load(f)
print(f"[v2] data = {DATA_PATH} ({DATA.get('version')})", flush=True)

N_LAWS, N_STAGES = 3, 4
assert len(DATA["laws"]) == N_LAWS
SEEN_DOMS = [d["name"] for d in DATA["seen_domains"]]
UNSEEN_DOMS = [d["name"] for d in DATA["unseen_domains"]]
N_SEEN_DOM = len(SEEN_DOMS)
N_PHR = 2

ALL_TEXTS = []  # every sentence that gets a Qwen feature
SEEN = []       # (global_idx, dom, law, stage, phrasing)
SEEN_IDX = {}   # (dom, law, stage, phr) -> global idx
for di, dom in enumerate(DATA["seen_domains"]):
    for law in range(N_LAWS):
        cells = dom["chains"][str(law)]
        assert len(cells) == N_STAGES
        for st in range(N_STAGES):
            assert len(cells[st]) >= N_PHR
            for ph in range(N_PHR):
                g = len(ALL_TEXTS)
                ALL_TEXTS.append(cells[st][ph])
                SEEN.append((g, di, law, st, ph))
                SEEN_IDX[(di, law, st, ph)] = g
N_SEEN_SENT = len(SEEN)
assert N_SEEN_SENT == N_SEEN_DOM * N_LAWS * N_STAGES * N_PHR == 72

UNSEEN_IDX = {}  # (unseen dom, law, stage) -> global idx
ITEMS = []       # (unseen dom, law)
for ui, dom in enumerate(DATA["unseen_domains"]):
    for law in range(N_LAWS):
        chain = dom["chains"][str(law)]
        assert len(chain) == N_STAGES
        for st in range(N_STAGES):
            UNSEEN_IDX[(ui, law, st)] = len(ALL_TEXTS)
            ALL_TEXTS.append(chain[st])
        ITEMS.append((ui, law))
N_ITEMS = len(ITEMS)
assert N_ITEMS == 36

CUE = []  # (global idx, domain, law, stage, lexically_resembles_law)
for c in DATA["cue_swap"]:
    g = len(ALL_TEXTS)
    ALL_TEXTS.append(c["text"])
    CUE.append((g, c["domain"], int(c["law"]), int(c["stage"]), int(c["lexically_resembles"][0])))
N_CUE = len(CUE)

FORBIDDEN = {ALL_TEXTS[g] for g in UNSEEN_IDX.values()} | {ALL_TEXTS[c[0]] for c in CUE}


def assert_not_forbidden(text: str):
    """No unseen-domain or cue-swap sentence may ever enter a loss."""
    if text in FORBIDDEN:
        raise RuntimeError(f"LEAK: unseen/cue-swap sentence in training data: {text!r}")
    for t in FORBIDDEN:
        if t in text:
            raise RuntimeError(f"LEAK: unseen/cue-swap sentence inside training text: {t!r}")


for (g, *_rest) in SEEN:
    assert_not_forbidden(ALL_TEXTS[g])
print(f"[v2] seen sentences={N_SEEN_SENT} | unseen items={N_ITEMS} | cue-swap={N_CUE} | "
      f"leak check passed for all {N_SEEN_SENT} seen training sentences", flush=True)

ANCHOR_KEYS = [(l, s) for l in range(N_LAWS) for s in range(N_STAGES)]  # row = law*4 + stage


def lbl(key):
    return f"(L{key[0]},S{key[1]})"


# ==============================================================================
# [1] Qwen (copied from Lab 01): bf16, frozen
# ==============================================================================
MODEL_ID = "Qwen/Qwen2.5-0.5B"
print(f"=== [1] Loading '{MODEL_ID}' (frozen, bf16) ===", flush=True)
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.padding_side = "right"
qwen_base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
qwen_base.eval()
for p in qwen_base.parameters():
    p.requires_grad = False
H_DIM = qwen_base.config.hidden_size  # 896
PAD_ID = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
print(f"Loaded in {time.time()-t0:.1f}s | hidden={H_DIM} | layers={qwen_base.config.num_hidden_layers}", flush=True)


@torch.no_grad()
def extract_qwen_896_feature(text_list):
    '''Copied from Lab 01: frozen Qwen last hidden state, mean-pooled, float32.'''
    enc = tokenizer(text_list, return_tensors="pt", padding=True, truncation=True, max_length=64).to("cuda")
    out = qwen_base.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
    last_h = out.last_hidden_state.float()
    mask_f = enc.attention_mask.unsqueeze(-1).float()
    pooled_896 = (last_h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
    return pooled_896  # (B, 896) float32


# ---- LoRA (copied from Lab 01; wrapped now, disabled until the LoRA competitor) ----
class LoRALinear(nn.Module):
    def __init__(self, orig_linear: nn.Linear, r: int = 16, alpha: float = 32.0):
        super().__init__()
        self.orig = orig_linear
        self.scale = alpha / r
        in_f, out_f = orig_linear.in_features, orig_linear.out_features
        # [review] float32 master weights: bf16 parameters drop AdamW updates once the LR decays
        self.lora_A = nn.Parameter(torch.randn(in_f, r, device='cuda', dtype=torch.float32) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(r, out_f, device='cuda', dtype=torch.float32))
        self.enabled = True

    def forward(self, x):
        res = self.orig(x)
        if self.enabled:
            res = res + (x @ self.lora_A.to(x.dtype) @ self.lora_B.to(x.dtype)) * self.scale
        return res


seed_all(SEED)  # before LoRA init
lora_params = []
lora_modules = []
for layer_idx in range(18, 24):
    lyr = qwen_base.model.layers[layer_idx]
    for attr_path in [("self_attn", "q_proj"), ("self_attn", "v_proj"), ("mlp", "up_proj"), ("mlp", "down_proj")]:
        parent = getattr(lyr, attr_path[0])
        orig_mod = getattr(parent, attr_path[1])
        wrapped = LoRALinear(orig_mod, r=16, alpha=32.0)
        setattr(parent, attr_path[1], wrapped)
        lora_params.extend([wrapped.lora_A, wrapped.lora_B])
        lora_modules.append(wrapped)


def set_lora_enabled(flag: bool):
    for m in lora_modules:
        m.enabled = flag


set_lora_enabled(False)
n_lora_trainable = sum(p.numel() for p in lora_params)

# ---- Features of every sentence (frozen Qwen, LoRA disabled) ----
set_lora_enabled(False)
H_ALL = torch.cat([extract_qwen_896_feature(ALL_TEXTS[i:i + 64]) for i in range(0, len(ALL_TEXTS), 64)], dim=0)
assert H_ALL.shape == (len(ALL_TEXTS), H_DIM)
SEEN_G = torch.tensor([g for g, *_ in SEEN], device="cuda")
H_SEEN = H_ALL[SEEN_G]  # (72, 896): the ONLY features that ever enter a head loss
print(f"[v2] extracted {H_ALL.shape[0]} features ({H_DIM}-d, float32)", flush=True)


# ==============================================================================
# [2] Funnel + law_brain head (from Lab 01 DualBrainGraftedQwen; decoders/slots/humility dropped)
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


# Hop chains: each seen domain x law x every phrasing combination (2^4 = 16) -> 144 x 4 indices into H_SEEN
_pos_in_seen = {g: p for p, (g, *_r) in enumerate(SEEN)}
HOP_IDX = torch.tensor([[_pos_in_seen[SEEN_IDX[(di, law, st, combo[st])]] for st in range(N_STAGES)]
                        for di in range(N_SEEN_DOM) for law in range(N_LAWS)
                        for combo in itertools.product(range(N_PHR), repeat=N_STAGES)], device="cuda")
assert HOP_IDX.shape == (N_SEEN_DOM * N_LAWS * N_PHR ** N_STAGES, N_STAGES)

HEAD_STEPS, HEAD_LR, HEAD_LR_FLOOR = 400, 1e-3, 1e-4
TRUE_LAW = [law for (_g, _d, law, _s, _p) in SEEN]
STAGE_OF = torch.tensor([st for (_g, _d, _l, st, _p) in SEEN], device="cuda")
DOM_OF = torch.tensor([d for (_g, d, _l, _s, _p) in SEEN], device="cuda")


def train_head(tag: str, law_labels):
    seed_all(SEED)  # before model init
    head = DualHead(qwen_hidden=H_DIM, d_dim=64, s_dim=64).cuda().float()
    n_par = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"\n=== [2] Training {tag}: {n_par:,} trainable params | {HEAD_STEPS} steps full batch | "
          f"AdamW lr {HEAD_LR:g} -> {HEAD_LR_FLOOR:g} cosine | clip 1.0 ===", flush=True)
    seed_all(SEED)  # before training run
    cls = torch.tensor(law_labels, device="cuda") * N_STAGES + STAGE_OF  # (72,) class id = law*4 + stage
    cls_oh = F.one_hot(cls, N_LAWS * N_STAGES).float()                    # (72, 12)
    dom_oh = F.one_hot(DOM_OF, N_SEEN_DOM).float()                        # (72, 3)
    assert (cls_oh.sum(0) > 0).all()
    off12 = ~torch.eye(N_LAWS * N_STAGES, device="cuda", dtype=torch.bool)
    off3 = ~torch.eye(N_SEEN_DOM, device="cuda", dtype=torch.bool)
    opt = torch.optim.AdamW(head.parameters(), lr=HEAD_LR, weight_decay=0.01)
    head.train()
    for step in range(1, HEAD_STEPS + 1):
        for pg in opt.param_groups:
            pg['lr'] = cosine_lr(step, HEAD_STEPS, HEAD_LR, HEAD_LR_FLOOR)
        opt.zero_grad()
        d, s = head.extract_ds(H_SEEN)
        s_mean = F.normalize(cls_oh.T @ s / cls_oh.sum(0).unsqueeze(-1), p=2, dim=-1)  # (12, 64)
        d_mean = F.normalize(dom_oh.T @ d / dom_oh.sum(0).unsqueeze(-1), p=2, dim=-1)  # (3, 64)
        l_s_pull = (1.0 - (s * s_mean[cls]).sum(-1)).mean()
        l_s_push = F.relu((s_mean @ s_mean.T)[off12] + 0.1).mean()
        l_d_pull = (1.0 - (d * d_mean[DOM_OF]).sum(-1)).mean()
        l_d_push = F.relu((d_mean @ d_mean.T)[off3] + 0.1).mean()
        loss_dis = l_s_pull + l_s_push + l_d_pull + l_d_push
        s_ch = s[HOP_IDX]                                   # (144, 4, 64)
        pred = head.hop_law(s_ch[:, :-1, :])                # predicts positions 1..3
        loss_hop = (1.0 - (pred * s_ch[:, 1:, :].detach()).sum(-1)).mean()
        loss = loss_dis + 2.5 * loss_hop
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if step in (1, 200, 400):
            print(f"  [{tag}] step {step:3d}/{HEAD_STEPS} | total {loss.item():.4f} | dis {loss_dis.item():.4f} "
                  f"(s_pull {l_s_pull.item():.3f} s_push {l_s_push.item():.3f} d_pull {l_d_pull.item():.3f} "
                  f"d_push {l_d_push.item():.3f}) | hop {loss_hop.item():.4f}", flush=True)
    head.eval()
    return head, n_par


# ---- Dual head (true labels) ----
dual_head, n_dual_trainable = train_head("Dual head", TRUE_LAW)

# ---- Shuffled-label head: per seen domain, a random permutation of law labels applied to the whole chain ----
# [review] Latin square: independent random permutations often agreed across domains (e.g. seed 2 gave two
# identical permutations), which kept a majority cross-domain law mapping. With a cyclic shift per domain,
# every training label maps to a different true law in each seen domain, so no law correspondence survives.
_off = random.Random(SEED).randrange(N_LAWS)
SHUF_PERM = [[(l + di + _off) % N_LAWS for l in range(N_LAWS)] for di in range(N_SEEN_DOM)]
for _lbl in range(N_LAWS):
    _true = [SHUF_PERM[di].index(_lbl) for di in range(N_SEEN_DOM)]
    if len(set(_true)) != N_SEEN_DOM:
        raise RuntimeError(f"shuffled-label control keeps a law correspondence for label {_lbl}: {_true}")
print(f"\n[v2] shuffled-label permutations (true law -> training label), per seen domain: "
      + "; ".join(f"{SEEN_DOMS[di]}: {SHUF_PERM[di]}" for di in range(N_SEEN_DOM)), flush=True)
SHUF_LAW = [SHUF_PERM[d][law] for (_g, d, law, _s, _p) in SEEN]
shuf_head, _ = train_head("Shuffled-label head", SHUF_LAW)
print("[v2] note: the shuffled-label head shuffles the funnel's law labels only; its hop loss still trains on real seen chains", flush=True)

# ---- Random head: same architecture, untrained ----
seed_all(SEED)
rand_head = DualHead(qwen_hidden=H_DIM, d_dim=64, s_dim=64).cuda().float()
rand_head.eval()
print(f"\n[v2] Random head: untrained, init with seed {SEED}", flush=True)
print(f"[v2] trainable params: Dual head = {n_dual_trainable:,} | LoRA = {n_lora_trainable:,}", flush=True)


# ==============================================================================
# [3] Evaluation helpers
# ==============================================================================
def new_score():
    return dict(D1=0, D1x=0, D2=0, D2x=0, A1=0, A1x=0, A2=0, A2x=0, R1=0, B1=0, B2=0, C_mean=0, C_cue=0)


@torch.no_grad()
def eval_space(S: torch.Tensor, hop_fn=None):
    """S: (N_all, dim) L2-normalised representations of ALL_TEXTS.
    hop_fn given  -> head mode (A1/A2; B by cos(ŝ, s(cand))).
    hop_fn None   -> raw kNN mode (B by cos(raw cand, raw anchor(pred_law, stage)))."""
    sc = new_score()
    anchors = torch.stack([S[torch.tensor([g for (g, _d, l, s, _p) in SEEN if l == L and s == St], device="cuda")].mean(0)
                           for (L, St) in ANCHOR_KEYS])
    anchors = F.normalize(anchors, p=2, dim=-1)  # (12, dim)

    def nearest(v):
        return ANCHOR_KEYS[int((anchors @ v).argmax().item())]

    rows = []
    for (ui, law) in ITEMS:
        v = [S[UNSEEN_IDX[(ui, law, st)]] for st in range(N_STAGES)]
        n1 = nearest(v[1])
        n2t = nearest(v[2])
        sc['D1'] += int(n1[0] == law); sc['D1x'] += int(n1 == (law, 1))
        sc['D2'] += int(n2t[0] == law); sc['D2x'] += int(n2t == (law, 2))
        sc['R1'] += int(n1[0] == law)
        cand2 = torch.stack([S[UNSEEN_IDX[(ui, l, 2)]] for l in range(N_LAWS)])
        cand3 = torch.stack([S[UNSEEN_IDX[(ui, l, 3)]] for l in range(N_LAWS)])
        if hop_fn is not None:
            seq = torch.stack([v[0], v[1]]).unsqueeze(0)               # (1, 2, dim)
            s2h = hop_fn(seq)[:, -1, :]                                # (1, dim)
            s3h = hop_fn(torch.cat([seq, s2h.unsqueeze(1)], dim=1))[:, -1, :]
            na1, na2 = nearest(s2h[0]), nearest(s3h[0])
            sc['A1'] += int(na1[0] == law); sc['A1x'] += int(na1 == (law, 2))
            sc['A2'] += int(na2[0] == law); sc['A2x'] += int(na2 == (law, 3))
            b1 = int((cand2 @ s2h[0]).argmax().item())
            b2 = int((cand3 @ s3h[0]).argmax().item())
        else:
            na1 = na2 = None
            pl = n1[0]
            b1 = int((cand2 @ anchors[pl * N_STAGES + 2]).argmax().item())
            b2 = int((cand3 @ anchors[pl * N_STAGES + 3]).argmax().item())
        sc['B1'] += int(b1 == law); sc['B2'] += int(b2 == law)
        rows.append(dict(dom=UNSEEN_DOMS[ui], law=law, n1=n1, na1=na1, b1=b1, b2=b2))
    cue_preds = []
    for (g, _dn, law, st, lex_law) in CUE:
        rows_st = [l * N_STAGES + st for l in range(N_LAWS)]
        pred = int((anchors[rows_st] @ S[g]).argmax().item())
        sc['C_mean'] += int(pred == law); sc['C_cue'] += int(pred == lex_law)
        cue_preds.append(pred)
    return sc, rows, cue_preds, anchors


def head_space(head):
    with torch.no_grad():
        _d, s = head.extract_ds(H_ALL)
    return s


set_lora_enabled(False)
RESULTS = {}
dual_sc, dual_rows, dual_cue, dual_anchors = eval_space(head_space(dual_head), dual_head.hop_law)
RESULTS["Dual head"] = dual_sc
shuf_sc, _, shuf_cue, _ = eval_space(head_space(shuf_head), shuf_head.hop_law)
RESULTS["Shuffled-label head"] = shuf_sc
rand_sc, _, rand_cue, _ = eval_space(head_space(rand_head), rand_head.hop_law)
RESULTS["Random head"] = rand_sc
raw_sc, _, raw_cue, _ = eval_space(F.normalize(H_ALL, p=2, dim=-1), None)
for k in ("A1", "A1x", "A2", "A2x"):
    raw_sc[k] = -1
RESULTS["Raw-Qwen kNN"] = raw_sc
# [review] extra row: raw features centred on the mean of the SEEN features only (no unseen statistics)
_mu_seen = H_ALL[torch.tensor([g for (g, *_r) in SEEN], device=H_ALL.device)].mean(dim=0, keepdim=True)
rawc_sc, _, rawc_cue, _ = eval_space(F.normalize(H_ALL - _mu_seen, p=2, dim=-1), None)
for k in ("A1", "A1x", "A2", "A2x"):
    rawc_sc[k] = -1
RESULTS["Raw-Qwen kNN (centred)"] = rawc_sc

# ---- Lexical control (pure Python, char unigram Jaccard; same tokenisation as tools/validate_analogy_v2.py) ----
_PUNCT = set("，。,.；;：:！!？? 、")


def grams1(text):
    return set(c for c in text if c not in _PUNCT)


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 0.0


SEEN_G1 = {(L, St): [grams1(ALL_TEXTS[g]) for (g, _d, l, s, _p) in SEEN if l == L and s == St] for (L, St) in ANCHOR_KEYS}


def class_sim(text, key):
    g = grams1(text)
    sims = [jaccard(g, x) for x in SEEN_G1[key]]
    return sum(sims) / len(sims)


lex_sc = new_score()
for k in ("D1", "D1x", "D2x", "A1", "A1x", "A2", "A2x"):
    lex_sc[k] = -1
for (ui, law) in ITEMS:
    c = [ALL_TEXTS[UNSEEN_IDX[(ui, law, st)]] for st in range(N_STAGES)]
    premise = c[0] + c[1]
    pl = max([(l, s) for l in range(N_LAWS) for s in (0, 1)], key=lambda k: class_sim(premise, k))[0]
    lex_sc['R1'] += int(pl == law)
    cand2 = [ALL_TEXTS[UNSEEN_IDX[(ui, l, 2)]] for l in range(N_LAWS)]
    cand3 = [ALL_TEXTS[UNSEEN_IDX[(ui, l, 3)]] for l in range(N_LAWS)]
    b1 = max(range(N_LAWS), key=lambda i: class_sim(cand2[i], (pl, 2)))
    b2 = max(range(N_LAWS), key=lambda i: class_sim(cand3[i], (pl, 3)))
    lex_sc['B1'] += int(b1 == law); lex_sc['B2'] += int(b2 == law)
    d2 = max(range(N_LAWS), key=lambda l: class_sim(c[2], (l, 2)))
    lex_sc['D2'] += int(d2 == law)
lex_cue = []
for (g, _dn, law, st, lex_law) in CUE:
    pred = max(range(N_LAWS), key=lambda l: class_sim(ALL_TEXTS[g], (l, st)))
    lex_sc['C_mean'] += int(pred == law); lex_sc['C_cue'] += int(pred == lex_law)
    lex_cue.append(pred)
RESULTS["Lexical"] = lex_sc


# ---- Qwen scoring controls: mean log-prob of the candidate tokens given the prompt ----
@torch.no_grad()
def cand_mean_logprob(prompt: str, cand: str) -> float:
    # [review] Tokenise prompt and candidate separately and concatenate: tokenising prompt+cand lets Qwen's
    # pre-tokenizer glue the prompt's final "，" onto the candidate's first characters, which shifts the span.
    p_ids = tokenizer(prompt).input_ids
    c_ids = tokenizer(cand).input_ids
    f_ids = p_ids + c_ids
    n_c = len(c_ids)
    if not (1 <= n_c < len(f_ids)):
        raise RuntimeError(f"bad candidate span: {prompt!r} {cand!r} {n_c}")
    ids = torch.tensor([f_ids], device="cuda")
    lp = F.log_softmax(qwen_base(input_ids=ids).logits.float()[0, :-1, :], dim=-1)
    tok_lp = lp.gather(-1, ids[0, 1:].unsqueeze(-1))[:, 0]
    return float(tok_lp[-n_c:].mean().item())


def eval_lm(prefix: str, tag: str):
    sc = new_score()
    for k in ("D1", "D1x", "D2", "D2x", "A1", "A1x", "A2", "A2x", "R1", "C_mean", "C_cue"):
        sc[k] = -1
    for (ui, law) in ITEMS:
        c = [ALL_TEXTS[UNSEEN_IDX[(ui, law, st)]] for st in range(N_STAGES)]
        prompt = prefix + c[0] + c[1]
        cand2 = [ALL_TEXTS[UNSEEN_IDX[(ui, l, 2)]] for l in range(N_LAWS)]
        cand3 = [ALL_TEXTS[UNSEEN_IDX[(ui, l, 3)]] for l in range(N_LAWS)]
        lp2 = [cand_mean_logprob(prompt, t) for t in cand2]
        b1 = max(range(N_LAWS), key=lambda i: lp2[i])
        prompt3 = prompt + cand2[b1]  # B2 conditioned on the method's own B1 pick
        lp3 = [cand_mean_logprob(prompt3, t) for t in cand3]
        b2 = max(range(N_LAWS), key=lambda i: lp3[i])
        sc['B1'] += int(b1 == law); sc['B2'] += int(b2 == law)
    print(f"[v2] {tag}: B1={sc['B1']}/36 B2={sc['B2']}/36", flush=True)
    return sc


FEWSHOT_PREFIX = "\n".join("".join(DATA["seen_domains"][0]["chains"][str(l)][st][0] for st in range(N_STAGES))
                           for l in range(N_LAWS)) + "\n"
for t in FEWSHOT_PREFIX.split("\n"):
    if t:
        assert_not_forbidden(t)
print(f"\n=== [3] Qwen scoring controls (LoRA disabled) ===\n[v2] few-shot prefix:\n{FEWSHOT_PREFIX}", flush=True)
set_lora_enabled(False)
RESULTS["Qwen zero-shot"] = eval_lm("", "Qwen zero-shot")
RESULTS["Qwen few-shot"] = eval_lm(FEWSHOT_PREFIX, "Qwen few-shot")

# ==============================================================================
# [4] LoRA competitor: causal-LM SFT on seen chains (random phrasing per stage)
# ==============================================================================
LORA_STEPS, LORA_BS, LORA_LR, LORA_LR_FLOOR = 400, 16, 5e-4, 5e-5
print(f"\n=== [4] Training LoRA: {n_lora_trainable:,} trainable params | {LORA_STEPS} steps | batch {LORA_BS} | "
      f"lr {LORA_LR:g} -> {LORA_LR_FLOOR:g} cosine ===", flush=True)
seed_all(SEED)  # before training run
set_lora_enabled(True)
opt_lora = torch.optim.AdamW(lora_params, lr=LORA_LR)
for step in range(1, LORA_STEPS + 1):
    for pg in opt_lora.param_groups:
        pg['lr'] = cosine_lr(step, LORA_STEPS, LORA_LR, LORA_LR_FLOOR)
    batch = []
    for _ in range(LORA_BS):
        di, law = random.randrange(N_SEEN_DOM), random.randrange(N_LAWS)
        cells = DATA["seen_domains"][di]["chains"][str(law)]
        txt = "".join(cells[st][random.randrange(N_PHR)] for st in range(N_STAGES))
        assert_not_forbidden(txt)
        batch.append(txt)
    enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
    labels = enc.input_ids.clone()
    labels[enc.attention_mask == 0] = -100
    opt_lora.zero_grad()
    out = qwen_base(input_ids=enc.input_ids, attention_mask=enc.attention_mask, labels=labels)
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
    opt_lora.step()
    if step in (1, 200, 400):
        print(f"  [LoRA] step {step:3d}/{LORA_STEPS} | LM loss {out.loss.item():.4f}", flush=True)
set_lora_enabled(True)
RESULTS["LoRA"] = eval_lm("", "LoRA (zero-shot prompt, LoRA enabled)")
set_lora_enabled(False)

# ==============================================================================
# [5] Report (numbers only; the pre-registered criteria are applied by the reader)
# ==============================================================================
COLS = ["D1", "D2", "A1", "A2", "R1", "B1", "B2"]


def cell(v, n):
    return f"{v:>2d}/{n}" if v >= 0 else "  -  "


print("\n" + "=" * 100, flush=True)
print(f"=== RESULTS seed={SEED} | 36 items (12 unseen domains x 3 laws) | law-correct counts | chance per column = 12/36 (1/3) ===", flush=True)
print(f"{'method':<24}" + "".join(f"{c:>8}" for c in COLS), flush=True)
print("-" * (22 + 8 * len(COLS)), flush=True)
for name, sc in RESULTS.items():
    print(f"{name:<24}" + "".join(f"{cell(sc[c], 36):>8}" for c in COLS), flush=True)
print("-" * (22 + 8 * len(COLS)), flush=True)
print("exact (law, stage) nearest-anchor counts (chance 3/36 under 12 anchors):", flush=True)
for name in ("Dual head", "Shuffled-label head", "Random head", "Raw-Qwen kNN", "Raw-Qwen kNN (centred)"):
    sc = RESULTS[name]
    print(f"  {name:<22} D1x={cell(sc['D1x'], 36)}  D2x={cell(sc['D2x'], 36)}  A1x={cell(sc['A1x'], 36)}  A2x={cell(sc['A2x'], 36)}", flush=True)

print(f"\n=== CUE-SWAP (9 items; classify law among the 3 anchors/classes of the item's stage; chance 3/9) ===", flush=True)
print(f"{'method':<22}{'meaning-correct':>18}{'follows-cue':>14}", flush=True)
for name in ("Dual head", "Shuffled-label head", "Random head", "Raw-Qwen kNN", "Raw-Qwen kNN (centred)", "Lexical"):
    sc = RESULTS[name]
    print(f"{name:<22}{cell(sc['C_mean'], 9):>18}{cell(sc['C_cue'], 9):>14}", flush=True)
print("per item (true law / cue law | Dual / Raw / Lexical predicted law):", flush=True)
for i, (g, dn, law, st, lex_law) in enumerate(CUE):
    print(f"  [{dn} S{st}] true L{law} cue L{lex_law} | Dual L{dual_cue[i]} Raw L{raw_cue[i]} Lex L{lex_cue[i]} | {ALL_TEXTS[g]}", flush=True)

print(f"\n=== Dual head per-item detail (✓ = law correct) ===", flush=True)
print(f"{'domain':<10} {'law':>3} | {'s1 nearest':<10} | {'ŝ2 nearest':<10} | {'B1 pick':<8} | {'B2 pick':<8}", flush=True)
for r in dual_rows:
    L = r['law']
    print(f"{r['dom']:<10} {('L'+str(L)):>3} | {lbl(r['n1'])}{' ✓' if r['n1'][0] == L else '  '}  | "
          f"{lbl(r['na1'])}{' ✓' if r['na1'][0] == L else '  '}  | L{r['b1']}{' ✓' if r['b1'] == L else '  '}     | "
          f"L{r['b2']}{' ✓' if r['b2'] == L else '  '}", flush=True)

print(f"\n=== Dual head anchor cosine matrix (12 x 12, rows/cols = (law, stage), law-major) ===", flush=True)
with torch.no_grad():
    _gram = (dual_anchors @ dual_anchors.T).cpu().numpy()
print("         " + " ".join(f"{lbl(k):>8}" for k in ANCHOR_KEYS), flush=True)
for i, k in enumerate(ANCHOR_KEYS):
    print(f"{lbl(k):>8} " + " ".join(f"{_gram[i, j]:>8.2f}" for j in range(len(ANCHOR_KEYS))), flush=True)

METHOD_TAGS = {"Dual head": "dual", "Shuffled-label head": "shuffled", "Random head": "random",
               "Raw-Qwen kNN": "raw_knn", "Raw-Qwen kNN (centred)": "raw_knn_centred", "Lexical": "lexical", "Qwen zero-shot": "qwen_zeroshot",
               "Qwen few-shot": "qwen_fewshot", "LoRA": "lora"}
print("", flush=True)
for name, sc in RESULTS.items():
    print(f"V2_RESULT seed={SEED} method={METHOD_TAGS[name]} D1={sc['D1']} D2={sc['D2']} A1={sc['A1']} A2={sc['A2']} "
          f"R1={sc['R1']} B1={sc['B1']} B2={sc['B2']} C_mean={sc['C_mean']} C_cue={sc['C_cue']}", flush=True)
