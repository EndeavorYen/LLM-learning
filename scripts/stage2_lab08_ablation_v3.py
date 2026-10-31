# ==============================================================================
# Stage 2 / Lab 08: Ablation on the v3 analogy test (exploratory; design pre-registered before running)
# Derived from Lab 07 (v3). Data loading, leak checks, Qwen feature extraction, anchors, seeds,
# DualHead and the evaluation are copied from Lab 07. Changes:
#   - six variants trained on the same 216 seen sentences (H_SEEN only), 400 steps, AdamW,
#     cosine LR 1e-3 -> 1e-4, clip 1.0:
#       FULL     Lab 07 dual head (true labels)
#       NO_D     no d_pull / d_push (d branch unused)
#       NO_HOP   no hop loss (law_brain untrained; A1/A2/B1/B1a/B2a not reported)
#       V2_PUSH  s push loss = mean only (v2 loss, no closest-pair penalty)
#       LIN      Linear(896, 12) on L2-normalised H, cross-entropy on class law*4+stage
#       MLP      896-256-64-12 GELU MLP on L2-normalised H, cross-entropy
#   - new metric R1-B / R2-B: law of s1 -> candidate closest to anchor/class (law, 2) / (law, 3); no hop
#   - shuffled / random / LoRA / LM / lexical controls removed (reported in v3); raw-Qwen kNN kept
# This script prints numbers only; it prints NO verdict.
# Run:  AUDIT_SEED=42 python stage2_lab08_ablation_v3.py   (seeds 42, 1, 2)
# ==============================================================================
import os, sys, time, math, random, json, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

SEED = int(os.environ.get("AUDIT_SEED", "42"))
print(f"[v3] SEED={SEED}", flush=True)


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
_DATA_CANDIDATES = ["/content/analogy_v3.json", "benchmarks/analogy_v3.json"]
try:
    _DATA_CANDIDATES.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks", "analogy_v3.json"))
except NameError:
    pass
DATA_PATH = next((p for p in _DATA_CANDIDATES if os.path.exists(p)), None)
if DATA_PATH is None:
    raise FileNotFoundError(f"analogy_v3.json not found in {_DATA_CANDIDATES}")
with open(DATA_PATH, encoding="utf-8") as f:
    DATA = json.load(f)
print(f"[v3] data = {DATA_PATH} ({DATA.get('version')})", flush=True)

N_LAWS, N_STAGES = 3, 4
assert len(DATA["laws"]) == N_LAWS
SEEN_DOMS = [d["name"] for d in DATA["seen_domains"]]
UNSEEN_DOMS = [d["name"] for d in DATA["unseen_domains"]]
N_SEEN_DOM = len(SEEN_DOMS)
# [v3] seen cells hold 1 or 2 phrasings (3 original seen domains: 2; the former v2 unseen domains: 1)

ALL_TEXTS = []  # every sentence that gets a Qwen feature
SEEN = []       # (global_idx, dom, law, stage, phrasing)
SEEN_IDX = {}   # (dom, law, stage, phr) -> global idx
for di, dom in enumerate(DATA["seen_domains"]):
    for law in range(N_LAWS):
        cells = dom["chains"][str(law)]
        assert len(cells) == N_STAGES
        for st in range(N_STAGES):
            assert len(cells[st]) >= 1
            for ph in range(len(cells[st])):
                g = len(ALL_TEXTS)
                ALL_TEXTS.append(cells[st][ph])
                SEEN.append((g, di, law, st, ph))
                SEEN_IDX[(di, law, st, ph)] = g
N_SEEN_SENT = len(SEEN)
N_PHR_OF = {(di, law, st): len(DATA["seen_domains"][di]["chains"][str(law)][st])
            for di in range(N_SEEN_DOM) for law in range(N_LAWS) for st in range(N_STAGES)}
assert N_SEEN_SENT == sum(N_PHR_OF.values())

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
print(f"[v3] seen sentences={N_SEEN_SENT} | unseen items={N_ITEMS} | cue-swap={N_CUE} | "
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


# [ablation] LoRA wrapping removed (no LoRA competitor here); with LoRA disabled, Lab 07's features were the
# plain frozen-Qwen features, so the extraction below is unchanged.
# ---- Features of every sentence (frozen Qwen) ----
H_ALL = torch.cat([extract_qwen_896_feature(ALL_TEXTS[i:i + 64]) for i in range(0, len(ALL_TEXTS), 64)], dim=0)
assert H_ALL.shape == (len(ALL_TEXTS), H_DIM)
SEEN_G = torch.tensor([g for g, *_ in SEEN], device="cuda")
H_SEEN = H_ALL[SEEN_G]  # (N_SEEN_SENT, 896): the ONLY features that ever enter a head loss
print(f"[v3] extracted {H_ALL.shape[0]} features ({H_DIM}-d, float32)", flush=True)



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


# [v3] Hop chains: per seen domain x law, the "aligned" phrasing chains k = 0..max_phr-1 (phrasing
# min(k, n_cell-1) at each stage). Using every 2^4 combination would let the 3 two-phrasing domains
# outweigh the 12 one-phrasing domains 16:1 in the hop loss.
_pos_in_seen = {g: p for p, (g, *_r) in enumerate(SEEN)}
_hop_rows = []
for di in range(N_SEEN_DOM):
    for law in range(N_LAWS):
        kmax = max(N_PHR_OF[(di, law, st)] for st in range(N_STAGES))
        for k in range(kmax):
            _hop_rows.append([_pos_in_seen[SEEN_IDX[(di, law, st, min(k, N_PHR_OF[(di, law, st)] - 1))]]
                              for st in range(N_STAGES)])
HOP_IDX = torch.tensor(_hop_rows, device="cuda")
print(f"[v3] hop chains: {HOP_IDX.shape[0]} (seen domains {N_SEEN_DOM} x laws {N_LAWS} x aligned phrasings)", flush=True)

HEAD_STEPS, HEAD_LR, HEAD_LR_FLOOR = 400, 1e-3, 1e-4
TRUE_LAW = [law for (_g, _d, law, _s, _p) in SEEN]
STAGE_OF = torch.tensor([st for (_g, _d, _l, st, _p) in SEEN], device="cuda")
DOM_OF = torch.tensor([d for (_g, d, _l, _s, _p) in SEEN], device="cuda")



LOSS_LOG = {}  # [ablation] variant -> {step: total training loss}


def train_head(tag: str, law_labels, use_d: bool = True, use_hop: bool = True, push_max: bool = True):
    """[ablation] flags: use_d=False drops d_pull/d_push (NO_D); use_hop=False drops loss_hop (NO_HOP);
    push_max=False uses the v2 s push loss (mean only, V2_PUSH). Defaults reproduce Lab 07."""
    seed_all(SEED)  # before model init
    head = DualHead(qwen_hidden=H_DIM, d_dim=64, s_dim=64).cuda().float()
    n_par = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"\n=== [2] Training {tag}: {n_par:,} params in module | {HEAD_STEPS} steps full batch | "
          f"AdamW lr {HEAD_LR:g} -> {HEAD_LR_FLOOR:g} cosine | clip 1.0 | use_d={use_d} use_hop={use_hop} "
          f"push_max={push_max} ===", flush=True)
    seed_all(SEED)  # before training run
    cls = torch.tensor(law_labels, device="cuda") * N_STAGES + STAGE_OF  # (N_SEEN_SENT,) class id = law*4 + stage
    cls_oh = F.one_hot(cls, N_LAWS * N_STAGES).float()                    # (N_SEEN_SENT, 12)
    dom_oh = F.one_hot(DOM_OF, N_SEEN_DOM).float()                        # (N_SEEN_SENT, N_SEEN_DOM)
    assert (cls_oh.sum(0) > 0).all()
    off12 = ~torch.eye(N_LAWS * N_STAGES, device="cuda", dtype=torch.bool)
    off3 = ~torch.eye(N_SEEN_DOM, device="cuda", dtype=torch.bool)  # [v3] all seen domains
    # params without a gradient (unused d funnel / law_brain) are skipped by AdamW, incl. weight decay
    opt = torch.optim.AdamW(head.parameters(), lr=HEAD_LR, weight_decay=0.01)
    head.train()
    zero = torch.zeros((), device="cuda")
    LOSS_LOG[tag] = {}
    for step in range(1, HEAD_STEPS + 1):
        for pg in opt.param_groups:
            pg['lr'] = cosine_lr(step, HEAD_STEPS, HEAD_LR, HEAD_LR_FLOOR)
        opt.zero_grad()
        d, s = head.extract_ds(H_SEEN)
        s_mean = F.normalize(cls_oh.T @ s / cls_oh.sum(0).unsqueeze(-1), p=2, dim=-1)  # (12, 64)
        l_s_pull = (1.0 - (s * s_mean[cls]).sum(-1)).mean()
        _sp = F.relu((s_mean @ s_mean.T)[off12] + 0.1)
        if push_max:
            l_s_push = _sp.mean() + _sp.max()  # [v3] + closest-pair penalty: v2 seed 42 collapsed two anchors (cos 1.00)
        else:
            l_s_push = _sp.mean()  # [ablation] V2_PUSH: the v2 loss
        if use_d:
            d_mean = F.normalize(dom_oh.T @ d / dom_oh.sum(0).unsqueeze(-1), p=2, dim=-1)  # (N_SEEN_DOM, 64)
            l_d_pull = (1.0 - (d * d_mean[DOM_OF]).sum(-1)).mean()
            l_d_push = F.relu((d_mean @ d_mean.T)[off3] + 0.1).mean()
        else:
            l_d_pull = l_d_push = zero  # [ablation] NO_D
        loss_dis = l_s_pull + l_s_push + l_d_pull + l_d_push
        if use_hop:
            s_ch = s[HOP_IDX]                                   # (144, 4, 64)
            pred = head.hop_law(s_ch[:, :-1, :])                # predicts positions 1..3
            loss_hop = (1.0 - (pred * s_ch[:, 1:, :].detach()).sum(-1)).mean()
        else:
            loss_hop = zero  # [ablation] NO_HOP: law_brain untrained
        loss = loss_dis + 2.5 * loss_hop
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if step in (1, 200, 400):
            LOSS_LOG[tag][step] = loss.item()
            print(f"  [{tag}] step {step:3d}/{HEAD_STEPS} | total {loss.item():.4f} | dis {loss_dis.item():.4f} "
                  f"(s_pull {l_s_pull.item():.3f} s_push {l_s_push.item():.3f} d_pull {l_d_pull.item():.3f} "
                  f"d_push {l_d_push.item():.3f}) | hop {loss_hop.item():.4f}", flush=True)
    head.eval()
    return head, n_par


# ---- [ablation] classifier variants (LIN / MLP) on L2-normalised frozen Qwen features ----
HN_ALL = F.normalize(H_ALL, p=2, dim=-1)
HN_SEEN = HN_ALL[SEEN_G]  # only seen features enter the classifier loss


def make_lin():
    return nn.Linear(H_DIM, N_LAWS * N_STAGES)


def make_mlp():
    return nn.Sequential(nn.Linear(H_DIM, 256), nn.GELU(), nn.Linear(256, 64), nn.GELU(),
                         nn.Linear(64, N_LAWS * N_STAGES))


def train_clf(tag: str, make_fn, law_labels):
    seed_all(SEED)  # before model init
    clf = make_fn().cuda().float()
    n_par = sum(p.numel() for p in clf.parameters() if p.requires_grad)
    print(f"\n=== [2] Training {tag}: {n_par:,} trainable params | {HEAD_STEPS} steps full batch | "
          f"AdamW lr {HEAD_LR:g} -> {HEAD_LR_FLOOR:g} cosine | wd 0.01 | clip 1.0 | cross-entropy, 12 classes ===", flush=True)
    seed_all(SEED)  # before training run
    cls = torch.tensor(law_labels, device="cuda") * N_STAGES + STAGE_OF
    opt = torch.optim.AdamW(clf.parameters(), lr=HEAD_LR, weight_decay=0.01)
    clf.train()
    LOSS_LOG[tag] = {}
    for step in range(1, HEAD_STEPS + 1):
        for pg in opt.param_groups:
            pg['lr'] = cosine_lr(step, HEAD_STEPS, HEAD_LR, HEAD_LR_FLOOR)
        opt.zero_grad()
        logits = clf(HN_SEEN)
        loss = F.cross_entropy(logits, cls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
        opt.step()
        if step in (1, 200, 400):
            LOSS_LOG[tag][step] = loss.item()
            with torch.no_grad():
                acc = (logits.argmax(-1) == cls).float().mean().item()
            print(f"  [{tag}] step {step:3d}/{HEAD_STEPS} | CE {loss.item():.4f} | train acc (pre-step) {acc:.3f}", flush=True)
    clf.eval()
    return clf, n_par


# ==============================================================================
# [3] Evaluation helpers
# ==============================================================================
def new_score():
    return dict(D1=0, D1x=0, D2=0, D2x=0, A1=0, A1x=0, A2=0, A2x=0, R1=0, R1B=0, R2B=0,
                B1=0, B2=0, B1a=0, B2a=0, C_mean=0, C_cue=0)


@torch.no_grad()
def eval_space(S: torch.Tensor, hop_fn=None):
    """S: (N_all, dim) L2-normalised representations of ALL_TEXTS.
    hop_fn given  -> head mode (A1/A2; B by cos(ŝ, s(cand))).
    hop_fn None   -> raw kNN mode (B by cos(raw cand, raw anchor(pred_law, stage))).
    [ablation] R1B / R2B in both modes: pl = law of the anchor nearest to s1; candidate closest to anchor (pl, 2) / (pl, 3)."""
    sc = new_score()
    anchors = torch.stack([S[torch.tensor([g for (g, _d, l, s, _p) in SEEN if l == L and s == St], device="cuda")].mean(0)
                           for (L, St) in ANCHOR_KEYS])
    anchors = F.normalize(anchors, p=2, dim=-1)  # (12, dim)

    def nearest(v):
        return ANCHOR_KEYS[int((anchors @ v).argmax().item())]

    for (ui, law) in ITEMS:
        v = [S[UNSEEN_IDX[(ui, law, st)]] for st in range(N_STAGES)]
        n1 = nearest(v[1])
        n2t = nearest(v[2])
        sc['D1'] += int(n1[0] == law); sc['D1x'] += int(n1 == (law, 1))
        sc['D2'] += int(n2t[0] == law); sc['D2x'] += int(n2t == (law, 2))
        sc['R1'] += int(n1[0] == law)
        cand2 = torch.stack([S[UNSEEN_IDX[(ui, l, 2)]] for l in range(N_LAWS)])
        cand3 = torch.stack([S[UNSEEN_IDX[(ui, l, 3)]] for l in range(N_LAWS)])
        pl = n1[0]
        r1b = int((cand2 @ anchors[pl * N_STAGES + 2]).argmax().item())  # [ablation] R1-B: no hop
        r2b = int((cand3 @ anchors[pl * N_STAGES + 3]).argmax().item())  # [ablation] R2-B: no hop
        sc['R1B'] += int(r1b == law); sc['R2B'] += int(r2b == law)
        if hop_fn is not None:
            seq = torch.stack([v[0], v[1]]).unsqueeze(0)               # (1, 2, dim)
            s2h = hop_fn(seq)[:, -1, :]                                # (1, dim)
            s3h = hop_fn(torch.cat([seq, s2h.unsqueeze(1)], dim=1))[:, -1, :]
            na1, na2 = nearest(s2h[0]), nearest(s3h[0])
            sc['A1'] += int(na1[0] == law); sc['A1x'] += int(na1 == (law, 2))
            sc['A2'] += int(na2[0] == law); sc['A2x'] += int(na2 == (law, 3))
            b1 = int((cand2 @ s2h[0]).argmax().item())
            b2 = int((cand3 @ s3h[0]).argmax().item())
            # [v3] B1a / B2a: anchor-mediated -- the candidate closest to the anchor nearest to ŝ
            b1a = int((cand2 @ anchors[ANCHOR_KEYS.index(na1)]).argmax().item())
            b2a = int((cand3 @ anchors[ANCHOR_KEYS.index(na2)]).argmax().item())
        else:
            b1, b2 = r1b, r2b
            b1a, b2a = b1, b2  # raw kNN is already anchor-mediated
        sc['B1'] += int(b1 == law); sc['B2'] += int(b2 == law)
        sc['B1a'] += int(b1a == law); sc['B2a'] += int(b2a == law)
    for (g, _dn, law, st, lex_law) in CUE:
        rows_st = [l * N_STAGES + st for l in range(N_LAWS)]
        pred = int((anchors[rows_st] @ S[g]).argmax().item())
        sc['C_mean'] += int(pred == law); sc['C_cue'] += int(pred == lex_law)
    return sc, anchors


@torch.no_grad()
def eval_clf(clf):
    """[ablation] classifier variants: class id = law*4 + stage; no anchors, no hop."""
    sc = new_score()
    for k in ("A1", "A1x", "A2", "A2x", "B1", "B2", "B1a", "B2a"):
        sc[k] = -1
    Z = clf(HN_ALL)  # (N_all, 12) logits
    for (ui, law) in ITEMS:
        z1 = Z[UNSEEN_IDX[(ui, law, 1)]]
        z2 = Z[UNSEEN_IDX[(ui, law, 2)]]
        c1, c2 = int(z1.argmax().item()), int(z2.argmax().item())  # full 12-way argmax
        pl = c1 // N_STAGES
        sc['D1'] += int(pl == law); sc['D1x'] += int(c1 == law * N_STAGES + 1)
        sc['D2'] += int(c2 // N_STAGES == law); sc['D2x'] += int(c2 == law * N_STAGES + 2)
        sc['R1'] += int(pl == law)
        r1b = max(range(N_LAWS), key=lambda l: Z[UNSEEN_IDX[(ui, l, 2)], pl * N_STAGES + 2].item())
        r2b = max(range(N_LAWS), key=lambda l: Z[UNSEEN_IDX[(ui, l, 3)], pl * N_STAGES + 3].item())
        sc['R1B'] += int(r1b == law); sc['R2B'] += int(r2b == law)
    for (g, _dn, law, st, lex_law) in CUE:
        rows_st = [l * N_STAGES + st for l in range(N_LAWS)]
        pred = int(Z[g, rows_st].argmax().item())
        sc['C_mean'] += int(pred == law); sc['C_cue'] += int(pred == lex_law)
    return sc


def head_space(head):
    with torch.no_grad():
        _d, s = head.extract_ds(H_ALL)
    return s


# ==============================================================================
# [4] Train + evaluate every variant
# ==============================================================================
RESULTS = {}
ANCHOR_MAXCOS = {}
FUNNEL_VARIANTS = [("FULL", dict()), ("NO_D", dict(use_d=False)), ("NO_HOP", dict(use_hop=False)),
                   ("V2_PUSH", dict(push_max=False))]
for name, kw in FUNNEL_VARIANTS:
    head, _n = train_head(name, TRUE_LAW, **kw)
    use_hop = kw.get("use_hop", True)
    sc, anc = eval_space(head_space(head), head.hop_law if use_hop else None)
    if not use_hop:
        for k in ("A1", "A1x", "A2", "A2x", "B1", "B2", "B1a", "B2a"):
            sc[k] = -1
    RESULTS[name] = sc
    with torch.no_grad():
        _g = anc @ anc.T
        ANCHOR_MAXCOS[name] = _g[~torch.eye(len(ANCHOR_KEYS), device=_g.device, dtype=torch.bool)].max().item()
    del head

for name, fn in (("LIN", make_lin), ("MLP", make_mlp)):
    clf, _n = train_clf(name, fn, TRUE_LAW)
    RESULTS[name] = eval_clf(clf)
    del clf

raw_sc, _ = eval_space(HN_ALL, None)
for k in ("A1", "A1x", "A2", "A2x"):
    raw_sc[k] = -1
RESULTS["RAW_KNN"] = raw_sc

# ==============================================================================
# [5] Report (numbers only; the pre-registered reading is applied by the reader)
# ==============================================================================
COLS = [("D1", "D1"), ("D2", "D2"), ("A1", "A1"), ("R1", "R1"), ("R1-B", "R1B"), ("B1", "B1"), ("B1a", "B1a"),
        ("R2-B", "R2B"), ("B2a", "B2a"), ("C_mean", "C_mean")]


def cell(v, n):
    return f"{v:>2d}/{n}" if v >= 0 else "  -  "


print("\n" + "=" * 100, flush=True)
print(f"=== ABLATION seed={SEED} | 36 items (12 unseen domains x 3 laws) | law-correct counts | chance 1/3 | "
      f"C_mean = cue-swap meaning-correct out of 9 ===", flush=True)
print(f"{'variant':<10}" + "".join(f"{h:>8}" for h, _k in COLS), flush=True)
print("-" * (10 + 8 * len(COLS)), flush=True)
for name, sc in RESULTS.items():
    print(f"{name:<10}" + "".join(f"{cell(sc[k], 9 if k == 'C_mean' else 36):>8}" for _h, k in COLS), flush=True)
print("-" * (10 + 8 * len(COLS)), flush=True)
print("exact (law, stage) counts: " + " | ".join(f"{n} D1x={cell(sc['D1x'], 36)} D2x={cell(sc['D2x'], 36)}"
                                                for n, sc in RESULTS.items()), flush=True)
print("max off-diagonal anchor cosine (funnel variants): "
      + " | ".join(f"{n} {v:.3f}" for n, v in ANCHOR_MAXCOS.items()), flush=True)

print(f"\n=== training loss (total) at steps 1 / 200 / 400 ===", flush=True)
for name, log in LOSS_LOG.items():
    print(f"  {name:<10}" + "  ".join(f"step {s:3d}: {log[s]:.4f}" for s in (1, 200, 400)), flush=True)

TAGS = {"FULL": "full", "NO_D": "no_d", "NO_HOP": "no_hop", "V2_PUSH": "v2_push", "LIN": "lin", "MLP": "mlp",
        "RAW_KNN": "raw_knn"}
print("", flush=True)
for name, sc in RESULTS.items():
    print(f"ABL_RESULT seed={SEED} variant={TAGS[name]} D1={sc['D1']} D2={sc['D2']} A1={sc['A1']} R1={sc['R1']} "
          f"R1B={sc['R1B']} B1={sc['B1']} B1a={sc['B1a']} R2B={sc['R2B']} B2a={sc['B2a']} C_mean={sc['C_mean']}", flush=True)
