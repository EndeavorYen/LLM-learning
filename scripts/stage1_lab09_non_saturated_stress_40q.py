# ==============================================================================
# ORIGINAL EXECUTED SCRIPT EXTRACTED VERBATIM FROM TRANSCRIPT (AST-UNESCAPED)
# Transcript Step Index : 418
# Timestamp             : 2026-10-30T06:52:34Z
# Original Colab Session: stage1_stress_bench
# ==============================================================================

import os, sys, time, math, random, subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = int(os.environ.get("AUDIT_SEED", "42"))  # [audit #7] pre-registered seeds: 42, 1, 2 (each a full retrain)
print(f"[audit #7] SEED={SEED}", flush=True)

if not os.path.exists('/content/minimind'):
    # [audit #17] pinned: 233d00f (2026-10-27) is the upstream HEAD that predates every original run
    subprocess.run(['git', 'clone', '--filter=blob:none', 'https://github.com/jingyaogong/minimind.git', '/content/minimind'], check=True)
    subprocess.run(['git', '-C', '/content/minimind', 'checkout', '-q', '233d00fa068ec2b8c0d4f07b703934d6391a5a73'], check=True)
sys.path.insert(0, '/content/minimind')

from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

tokenizer = AutoTokenizer.from_pretrained('/content/minimind/model')
MASK_ID = 3
SLOT_PAD_ID = 4
BOS_ID = tokenizer.bos_token_id
EOS_ID = tokenizer.eos_token_id

# ==============================================================================
# NON-SATURATED STRESS BENCHMARK ([audit #16] now 36 scored tasks; file name kept for history)
# Tier 1 (16 tasks): High-Entropy Confusable Serial Numbers (differ by 1 char!)
# Tier 2 ( 8 tasks): Multi-Law Zero-Shot 1-Hop Analogy (4 Unseen Domains x 2 Competing Laws)
# Tier 3 ( 8 tasks): Long-Horizon 2-Hop Zero-Shot Chain (stage 3 from stage 0+1; [audit #16] was mislabelled 3-hop)
# Tier 4 ( 4 tasks): [audit #16] each of the 4 unseen-domain Emergency Codes scored ONCE (exact match).
#                    The old compound metric (Tier-2 hit AND code correct, /8) is printed as info only,
#                    because it re-counted Tier 2.
# ==============================================================================

# [audit #16] scoring helpers (verbatim from the audit spec)
import re as _re
_PUNCT = "，。,.；;：:！!？?"
def norm_text(s: str) -> str:
    """[audit #16] whitespace-free, full/half-width punctuation unified."""
    s = _re.sub(r"\s+", "", s or "")
    return s.replace(",", "，").replace(".", "。").replace(";", "；").replace(":", "：").replace("!", "！").replace("?", "？")
def first_clause(s: str) -> str:
    """[audit #16] text up to and including the first clause-ending punctuation (，。；！？)."""
    s = norm_text(s)
    m = _re.search(r"[，。；！？]", s)
    return s[: m.end()] if m else s
def clause_match(pred: str, gt: str) -> bool:
    """[audit #16] full-clause exact match (replaces the old first-5/6-character substring test)."""
    return first_clause(pred) == norm_text(gt)
def exact_answer_match(pred: str, gt: str) -> bool:
    """[audit #16] rote answers: prediction must start with the answer and then stop or hit punctuation."""
    p, g = norm_text(pred), norm_text(gt)
    return p.startswith(g) and (len(p) == len(g) or p[len(g)] in _PUNCT)

# 16 Highly Confusable Rote Serial Numbers (4 clusters x 4 near-identical nodes)
rote_serials = [
    ("查询节点A1密钥：", "SN-8841-902-A1"),
    ("查询节点A2密钥：", "SN-8841-902-A2"),
    ("查询节点A3密钥：", "SN-8841-902-A3"),
    ("查询节点A4密钥：", "SN-8841-902-A4"),
    ("查询节点B1密钥：", "SN-7739-105-B1"),
    ("查询节点B2密钥：", "SN-7739-105-B2"),
    ("查询节点B3密钥：", "SN-7739-105-B3"),
    ("查询节点B4密钥：", "SN-7739-105-B4"),
    ("查询节点C1密钥：", "SN-6628-441-C1"),
    ("查询节点C2密钥：", "SN-6628-441-C2"),
    ("查询节点C3密钥：", "SN-6628-441-C3"),
    ("查询节点C4密钥：", "SN-6628-441-C4"),
    ("查询节点D1密钥：", "SN-5517-339-D1"),
    ("查询节点D2密钥：", "SN-5517-339-D2"),
    ("查询节点D3密钥：", "SN-5517-339-D3"),
    ("查询节点D4密钥：", "SN-5517-339-D4"),
]

# Domain Emergency Codes for Tier 4 Hybrid Tasks
domain_emg_codes = [
    ("查询人体医学急救码：", "EMG-MED-119"),
    ("查询网络通讯抢修码：", "EMG-NET-404"),
    ("查询工厂产线维修码：", "EMG-FAC-502"),
    ("查询呼吸系统急救码：", "EMG-AIR-999"),
]

# 6 Domains x 2 Competing Laws x 4 Stages (s0 -> s1 -> s2 -> s3)
# Law 0: Narrowing -> Pressure Surge -> Blockage -> Relief Valve
# Law 1: Supply Cut -> Reserve Drain -> Starvation -> Backup Supply
multi_law_domains = [
    ("水利工程", True, {
        0: ["水流在水管稳定流动，", "当水管内壁结垢变窄时，", "管内水压就会急剧升高，", "最终导致水流堵塞中断。"],
        1: ["水库向管网正常供水，", "当上游水源突然断流时，", "管网储水就会迅速枯竭，", "最终导致末端供水停摆。"]
    }),
    ("交通工程", True, {
        0: ["车流在高速路正常行驶，", "当施工导致车道变窄时，", "路段交通压力急剧升高，", "最终导致车流瘫痪堵塞。"],
        1: ["巴士向干线正常发车，", "当总站发车突然中断时，", "干线运力就会迅速枯竭，", "最终导致站点运输停摆。"]
    }),
    ("人體醫學", False, {
        0: ["血液在血管中正常循环，", "当血管壁堆积胆固醇变窄时，", "血管内血压就会急剧升高，", "最终导致血液循环受阻。"],
        1: ["心脏向全身正常供血，", "当心室供血突然中断时，", "器官血氧就会迅速枯竭，", "最终导致组织机能停摆。"]
    }),
    ("網路通訊", False, {
        0: ["数据在光纤中高速传输，", "当网络带宽通道受限变窄时，", "网络延迟与拥塞急剧升高，", "最终导致数据包传输中断。"],
        1: ["核心网向基站正常发包，", "当主干路由突然断连时，", "基站缓存就会迅速枯竭，", "最终导致通信服务停摆。"]
    }),
    ("工廠產線", False, {
        0: ["零件在流水线顺畅运转，", "当关键工位故障产能变窄时，", "产线积压負荷急剧升高，", "最终导致工厂生产停摆。"],
        1: ["仓库向产线正常供料，", "当上游原料突然断供时，", "产线库存就会迅速枯竭，", "最终导致装配作业停摆。"]
    }),
    ("呼吸系統", False, {
        0: ["空气在气管中顺畅进出，", "当支气管发炎肿胀变窄时，", "呼吸道阻力就会急剧升高，", "最终导致呼吸困难窒息。"],
        1: ["肺部向血液正常供氧，", "当外界氧气突然断绝时，", "体内血氧就会迅速枯竭，", "最终导致呼吸代谢停摆。"]
    }),
]

# [audit #13] Lengths are derived from the data. The original hardcoded SLOT_LEN = 12 and
# PROMPT_LEN = 16/18 silently truncated targets (e.g. "当血管壁堆积胆固醇变窄时，" is 18 tokens)
# and premises (c0 + c1 is 23-27 tokens). Measured token lengths are listed in issue #13.
def _tok_len(txt):
    return len(tokenizer(txt, add_special_tokens=False).input_ids)
_slot_texts = ([q for q, _ in rote_serials + domain_emg_codes] + [a for _, a in rote_serials + domain_emg_codes]
               + [c for _, _, laws in multi_law_domains for cl in laws.values() for c in cl])
_prompt_texts = ([q for q, _ in rote_serials + domain_emg_codes]
                 + [cl[0] + cl[1] for _, _, laws in multi_law_domains for cl in laws.values()]
                 # [audit #16] Pure-Diffusion's Tier-3 second pass uses premise = stage1 + stage2 (seen-domain training pairs)
                 + [cl[1] + cl[2] for _, is_seen, laws in multi_law_domains if is_seen for cl in laws.values()])
SLOT_LEN = max(_tok_len(t) for t in _slot_texts) + 1      # +1 so every slot can end with [□]
PROMPT_LEN = max(_tok_len(t) for t in _prompt_texts) + 1  # +1 for BOS
AR_MAX_NEW_TOKENS = SLOT_LEN                              # AR may emit the longest answer in full
print(f"[audit #13] SLOT_LEN={SLOT_LEN} PROMPT_LEN={PROMPT_LEN} AR_MAX_NEW_TOKENS={AR_MAX_NEW_TOKENS}", flush=True)

def pad_slot(txt, length=None):
    length = SLOT_LEN if length is None else length
    ids = tokenizer(txt, add_special_tokens=False).input_ids
    if len(ids) > length:
        raise ValueError(f"[audit #13] pad_slot would truncate {len(ids)} tokens to {length}: {txt!r}")
    val = [1]*len(ids) + [0]*(length - len(ids))
    pad = ids + [SLOT_PAD_ID]*(length - len(ids))
    return torch.tensor([pad], device='cuda'), torch.tensor([val], device='cuda')

def pad_prompt(txt, length=None):
    length = PROMPT_LEN if length is None else length
    body = tokenizer(txt, add_special_tokens=False).input_ids
    if len(body) + 1 > length:
        raise ValueError(f"[audit #13] pad_prompt would truncate {len(body) + 1} tokens to {length}: {txt!r}")
    ids = [BOS_ID] + body
    ids = [SLOT_PAD_ID]*(length - len(ids)) + ids
    return torch.tensor([ids], device='cuda')

def pad_prompt_tail(txt):
    """[audit #16] like pad_prompt, but for model-generated premises: keeps the LAST PROMPT_LEN-1 tokens if too long."""
    body = tokenizer(txt, add_special_tokens=False).input_ids[-(PROMPT_LEN - 1):]
    ids = [BOS_ID] + body
    ids = [SLOT_PAD_ID]*(PROMPT_LEN - len(ids)) + ids
    return torch.tensor([ids], device='cuda')

def clean_str(t_ids):
    c = [tid for tid in t_ids.tolist() if tid not in (MASK_ID, SLOT_PAD_ID, BOS_ID, EOS_ID)]
    return tokenizer.decode(c)

all_rote = rote_serials + domain_emg_codes
rote_tensors = [(pad_prompt(q), pad_slot(a)[0], q, a) for q, a in all_rote]

# Structure states: 2 Laws x 4 Stages = 8 distinct universal structural states!
# Domain states: 6 distinct domains!
dom_law_tensors = []
for d_idx, (d_name, is_seen, laws_dict) in enumerate(multi_law_domains):
    for law_id in (0, 1):
        clauses = laws_dict[law_id]
        c_t = [pad_slot(c) for c in clauses]
        dom_law_tensors.append((d_idx, d_name, is_seen, law_id, clauses, c_t))

class PureDiffusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = MiniMindForCausalLM(config)
        for layer in self.backbone.model.layers:
            layer.self_attn.is_causal = False

    def forward_loss(self, prompt_ids, target_slots):
        B, L = target_slots.shape
        m = (torch.rand((B, L), device=target_slots.device) < 0.6)
        m[:, 0] = True
        canv = target_slots.clone(); canv[m] = MASK_ID
        lbl = target_slots.clone(); lbl[~m] = -100
        full_in = torch.cat([prompt_ids, canv], dim=1)
        logits = self.backbone(full_in).logits[:, prompt_ids.size(1):, :].contiguous()
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), lbl.reshape(-1), ignore_index=-100)

    @torch.no_grad()
    def generate_slots(self, prompt_ids, steps=2):
        B = prompt_ids.size(0)
        canv = torch.full((B, SLOT_LEN), MASK_ID, dtype=torch.long, device=prompt_ids.device)
        for s in range(steps):
            logits = self.backbone(torch.cat([prompt_ids, canv], dim=1)).logits[:, prompt_ids.size(1):, :]
            conf, pred = F.softmax(logits, dim=-1).max(dim=-1)
            still_m = (canv == MASK_ID)
            n_rem = still_m[0].sum().item()
            if n_rem == 0: break
            conf = conf.masked_fill(~still_m, -1.0)
            k = n_rem if s == steps - 1 else max(1, SLOT_LEN // steps)
            _, top_idx = torch.topk(conf, k=min(k, n_rem), dim=-1)
            for b in range(B):
                for idx in top_idx[b]:
                    if still_m[b, idx]: canv[b, idx] = pred[b, idx]
        return canv

class StressDualBrainModel(nn.Module):
    def __init__(self, config: MiniMindConfig, thought_dim: int = 256, pure_thought_only: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.thought_dim = thought_dim
        self.pure_thought_only = pure_thought_only

        self.backbone = MiniMindForCausalLM(config)
        for layer in self.backbone.model.layers:
            layer.self_attn.is_causal = False

        self.domain_proj = nn.Linear(config.hidden_size, thought_dim)
        self.struct_proj = nn.Linear(config.hidden_size, thought_dim)

        nhead = 4 if thought_dim >= 32 else 2
        enc_layer = nn.TransformerEncoderLayer(
            d_model=thought_dim, nhead=nhead, dim_feedforward=thought_dim*2,
            batch_first=True, norm_first=True
        )
        self.struct_brain = nn.TransformerEncoder(enc_layer, num_layers=2, enable_nested_tensor=False)
        self.next_struct_head = nn.Linear(thought_dim, thought_dim)

        self.thought_to_prefix = nn.Linear(thought_dim * 2, config.hidden_size * 2)
        self._active_prefix = None

        def _hook(module, inp, out):
            if self._active_prefix is not None:
                out = out.clone()
                out[:, :2, :] = self._active_prefix
            return out

        self.backbone.model.embed_tokens.register_forward_hook(_hook)

    def forward_fast_muscle(self, prompt_ids, canvas_ids):
        self._active_prefix = None
        P = prompt_ids.size(1)
        full_ids = torch.cat([prompt_ids, canvas_ids], dim=1)
        return self.backbone(full_ids).logits[:, P:, :].contiguous()

    def encode_thought(self, clause_ids, valid_mask, noise_std: float = 0.0):
        self._active_prefix = None
        h = self.backbone.model(clause_ids)[0]
        mask_f = valid_mask.unsqueeze(-1).type_as(h)
        pooled = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        d = F.normalize(self.domain_proj(pooled), p=2, dim=-1)
        s = F.normalize(self.struct_proj(pooled), p=2, dim=-1)
        if noise_std > 0.0:
            d = F.normalize(d + torch.randn_like(d) * noise_std, p=2, dim=-1)
            s = F.normalize(s + torch.randn_like(s) * noise_std, p=2, dim=-1)
        return d, s

    def forward_struct_hop(self, s_seq):
        B, T, _ = s_seq.shape
        c_mask = torch.triu(torch.ones(T, T, device=s_seq.device, dtype=torch.bool), diagonal=1)
        h = self.struct_brain(s_seq, mask=c_mask)
        return F.normalize(self.next_struct_head(h), p=2, dim=-1)

    def forward_thought_render(self, d, s, canvas_ids):
        B, L = canvas_ids.shape
        z = torch.cat([d, s], dim=-1)
        self._active_prefix = self.thought_to_prefix(z).reshape(B, 2, self.hidden_size)
        dummy_prefix = torch.full((B, 2), BOS_ID, dtype=torch.long, device=canvas_ids.device)
        full_ids = torch.cat([dummy_prefix, canvas_ids], dim=1)
        logits = self.backbone(full_ids).logits[:, 2:, :].contiguous()
        self._active_prefix = None
        return logits

    @torch.no_grad()
    def probe_muscle(self, prompt_ids, steps=2):
        B = prompt_ids.size(0)
        canv = torch.full((B, SLOT_LEN), MASK_ID, dtype=torch.long, device=prompt_ids.device)
        min_conf = 0.0
        for s in range(steps):
            logits = self.forward_fast_muscle(prompt_ids, canv)
            probs = F.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            if s == 0:
                non_pad = (pred != SLOT_PAD_ID)
                min_conf = conf[non_pad].min().item() if non_pad.any() else conf.min().item()
            still_m = (canv == MASK_ID)
            n_rem = still_m[0].sum().item()
            if n_rem == 0: break
            conf_m = conf.masked_fill(~still_m, -1.0)
            k = n_rem if s == steps - 1 else max(1, SLOT_LEN // steps)
            _, top_idx = torch.topk(conf_m, k=min(k, n_rem), dim=-1)
            for b in range(B):
                for idx in top_idx[b]:
                    if still_m[b, idx]: canv[b, idx] = pred[b, idx]
        return min_conf, canv

    @torch.no_grad()
    def render_from_ds(self, d, s, steps=2):
        B = d.size(0)
        canv = torch.full((B, SLOT_LEN), MASK_ID, dtype=torch.long, device=d.device)
        for step in range(steps):
            logits = self.forward_thought_render(d, s, canv)
            conf, pred = F.softmax(logits, dim=-1).max(dim=-1)
            still_m = (canv == MASK_ID)
            n_rem = still_m[0].sum().item()
            if n_rem == 0: break
            conf_m = conf.masked_fill(~still_m, -1.0)
            k = n_rem if step == steps - 1 else max(1, SLOT_LEN // steps)
            _, top_idx = torch.topk(conf_m, k=min(k, n_rem), dim=-1)
            for b in range(B):
                for idx in top_idx[b]:
                    if still_m[b, idx]: canv[b, idx] = pred[b, idx]
        return canv

    @torch.no_grad()
    def solve_multi_hop_analogy(self, c0_ids, c0_val, c1_ids, c1_val, hops: int = 1, hop_noise: float = 0.0):
        '''
        If hops=1: predicts s_2 from [s_0, s_1]
        If hops=2: predicts s_2 -> s_3 from [s_0, s_1]
        Always conditions on [s_0, s_1]; hops=k appends k predicted states. [audit #16] Tier 3 uses hops=2
        (the old "hops=3 from only s_0" mode was never implemented nor used).
        '''
        d0, s0 = self.encode_thought(c0_ids, c0_val)
        d1, s1 = self.encode_thought(c1_ids, c1_val)
        d_tgt = F.normalize(d0 + d1, p=2, dim=-1)
        curr_seq = torch.cat([s0, s1], dim=0).unsqueeze(0)  # (1, 2, D)
        for h_i in range(hops):
            next_s = self.forward_struct_hop(curr_seq)[:, -1, :]
            if hop_noise > 0.0:
                next_s = F.normalize(next_s + torch.randn_like(next_s) * hop_noise, p=2, dim=-1)
            curr_seq = torch.cat([curr_seq, next_s.unsqueeze(1)], dim=1)
        final_s = curr_seq[:, -1, :]
        return self.render_from_ds(d_tgt, final_s, steps=2)

# Instantiate 5 competing models across scales
cfg_512 = MiniMindConfig(hidden_size=512, num_hidden_layers=8, use_moe=False)
cfg_256 = MiniMindConfig(hidden_size=256, num_hidden_layers=4, use_moe=False)
cfg_128 = MiniMindConfig(hidden_size=128, num_hidden_layers=2, use_moe=False)

random.seed(SEED)  # [audit #7] also seeds AR mini-batch sampling (random.sample) so a seed is a reproducible run
torch.manual_seed(SEED)  # [audit #7]
m_ar = MiniMindForCausalLM(cfg_512).cuda().bfloat16()
torch.manual_seed(SEED)  # [audit #7]
m_diff = PureDiffusionModel(cfg_512).cuda().bfloat16()
torch.manual_seed(SEED)  # [audit #7]
# Pure-Thought LCM compresses all sentences (including 16 confusable serials!) into a single continuous thought vector without fast token attention!
m_lcm = StressDualBrainModel(cfg_512, thought_dim=24, pure_thought_only=True).cuda().bfloat16()
torch.manual_seed(SEED)  # [audit #7]
m_micro_dual = StressDualBrainModel(cfg_128, thought_dim=32, pure_thought_only=False).cuda().bfloat16()
torch.manual_seed(SEED)  # [audit #7]
m_tiny_dual = StressDualBrainModel(cfg_256, thought_dim=128, pure_thought_only=False).cuda().bfloat16()
torch.manual_seed(SEED)  # [audit #7]
m_full_dual = StressDualBrainModel(cfg_512, thought_dim=256, pure_thought_only=False).cuda().bfloat16()

def count_m(m): return sum(p.numel() for p in m.parameters()) / 1e6
print("Models instantiated:", flush=True)
for nm, md in [("AR-512", m_ar), ("Diff-512", m_diff), ("PureThought-LCM-512", m_lcm), ("Micro-Dual-128", m_micro_dual), ("Tiny-Dual-256", m_tiny_dual), ("Full-Dual-512", m_full_dual)]:
    print(f"  {nm:<22}: {count_m(md):6.2f}M params", flush=True)

# 1. Train AR-512 and Pure-Diffusion-512
_n_held_total = sum(2 * 2 for _, is_seen, _ in multi_law_domains if not is_seen)  # 2 laws x stages 2,3
print(f"[audit #14] held out {_n_held_total} unseen answer clauses from every loss", flush=True)
ar_seqs, diff_p, diff_a = [], [], []
for pt, at, q_s, a_s in rote_tensors:
    ar_seqs.append([BOS_ID] + tokenizer(q_s + a_s, add_special_tokens=False).input_ids + [EOS_ID])
    diff_p.append(pt[0]); diff_a.append(at[0])

for d_idx, d_name, is_seen, law_id, clauses, c_t in dom_law_tensors:
    if is_seen:
        ar_seqs.append([BOS_ID] + tokenizer("".join(clauses), add_special_tokens=False).input_ids + [EOS_ID])
        diff_p.append(pad_prompt(clauses[0] + clauses[1])[0]); diff_a.append(c_t[2][0][0])
        # [audit #16] seen-domain (stage1+stage2 -> stage3) pairs so Pure-Diffusion's Tier-3 second pass
        # (premise = stage1 + its own stage-2 prediction) is an in-distribution query.
        diff_p.append(pad_prompt(clauses[1] + clauses[2])[0]); diff_a.append(c_t[3][0][0])
    else:
        # [audit #14] unseen domain: only the PREMISE (stage 0 + 1) is visible, with NO EOS after it
        # (EOS here would teach AR to stop exactly where the answer should begin). Answer clauses
        # (stage 2, 3) were previously in the AR data as standalone sequences -> leak, now removed.
        ar_seqs.append([BOS_ID] + tokenizer(clauses[0] + clauses[1], add_special_tokens=False).input_ids)
        # [audit #14] same premise visibility for Pure-Diffusion: (stage 0 -> stage 1) pair only.
        diff_p.append(pad_prompt(clauses[0])[0]); diff_a.append(c_t[1][0][0])

# [audit #16] equal budget for all 6 models: TRAIN_STEPS optimizer steps, cosine LR 3e-4 -> 3e-5
# (AR / Pure-Diffusion previously got 160 steps at constant LR; the thought models got 220 cosine steps).
TRAIN_STEPS = 220
LR_MAX, LR_MIN = 3e-4, 3e-5
def cosine_lr(step, steps):
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * step / steps))
LOG_STEPS = {1, TRAIN_STEPS // 2, TRAIN_STEPS}  # [audit #16] log losses so training failures are visible
for nm in ("AR-512", "Diff-512", "PureThought-LCM-512", "Micro-Dual-128", "Tiny-Dual-256", "Full-Dual-512"):
    print(f"[audit #16] budget {nm:<22}: {TRAIN_STEPS} steps, cosine LR {LR_MAX:g}->{LR_MIN:g}", flush=True)

opt_ar = torch.optim.AdamW(m_ar.parameters(), lr=LR_MAX)
opt_diff = torch.optim.AdamW(m_diff.parameters(), lr=LR_MAX)
t_dp = torch.stack(diff_p); t_da = torch.stack(diff_a)

for step in range(1, TRAIN_STEPS + 1):
    lr = cosine_lr(step, TRAIN_STEPS)  # [audit #16]
    for opt_x in (opt_ar, opt_diff):
        for pg in opt_x.param_groups: pg['lr'] = lr
    opt_ar.zero_grad()
    # Sample mini-batch of 16 sequences for AR speed
    batch_s = random.sample(ar_seqs, min(16, len(ar_seqs)))
    loss_a = sum(m_ar(torch.tensor([s], device='cuda'), labels=torch.tensor([s], device='cuda')).loss for s in batch_s) / len(batch_s)
    loss_a.backward(); torch.nn.utils.clip_grad_norm_(m_ar.parameters(), 1.0); opt_ar.step()

    opt_diff.zero_grad()
    loss_d = m_diff.forward_loss(t_dp, t_da)
    loss_d.backward(); torch.nn.utils.clip_grad_norm_(m_diff.parameters(), 1.0); opt_diff.step()
    if step in LOG_STEPS:  # [audit #16]
        print(f"  [AR-512]   step {step:4d}/{TRAIN_STEPS} loss={loss_a.item():.4f} lr={lr:.2e}", flush=True)
        print(f"  [Diff-512] step {step:4d}/{TRAIN_STEPS} loss={loss_d.item():.4f} lr={lr:.2e}", flush=True)

# 2. Train PureThought-LCM, Micro-Dual-128, Tiny-Dual-256, Full-Dual-512
def train_stress_model(mod, steps=TRAIN_STEPS, tag="model"):
    opt_m = torch.optim.AdamW(mod.parameters(), lr=LR_MAX)
    mod.train()
    # [audit #14] Stack only ALLOWED clauses: every seen-domain clause + unseen-domain premise clauses
    # (stage 0, 1). Unseen answer clauses (stage 2, 3) never enter any loss. Explicit index lists replace
    # the old .view(6, 2, 4, D) grid, which needed all 48 clauses.
    all_ct, all_cv, k_dom, k_law, k_stage = [], [], [], [], []
    row_of = {}  # (d_idx, law_id, stage) -> row in b_ct
    n_held = 0
    for d_idx, _, is_seen, law_id, _, c_t in dom_law_tensors:
        for st, (ct, cv) in enumerate(c_t):
            if (not is_seen) and st >= 2:
                n_held += 1
                continue
            row_of[(d_idx, law_id, st)] = len(all_ct)
            all_ct.append(ct[0]); all_cv.append(cv[0])
            k_dom.append(d_idx); k_law.append(law_id); k_stage.append(st)
    b_ct = torch.stack(all_ct); b_cv = torch.stack(all_cv)  # (32, SLOT_LEN)
    k_dom = torch.tensor(k_dom, device='cuda')
    k_state = torch.tensor(k_law, device='cuda') * 4 + torch.tensor(k_stage, device='cuda')  # 0..7
    n_dom = len(multi_law_domains)
    dom_anchor_rows = torch.tensor([row_of[(d, 0, 0)] for d in range(n_dom)], device='cuda')       # (6,)
    struct_anchor_rows = torch.tensor([row_of[(0, lw, st)] for lw in (0, 1) for st in range(4)], device='cuda')  # (8,) domain 0 (seen)
    seen_doms = [d for d, (_, is_seen, _) in enumerate(multi_law_domains) if is_seen]
    hop_rows = torch.tensor([[row_of[(d, lw, st)] for st in range(4)] for d in seen_doms for lw in (0, 1)], device='cuda')  # (4, 4)
    print(f"[audit #14] held out {n_held} unseen answer clauses from every loss ({tag}; {len(all_ct)} clauses used)", flush=True)

    rt_p = torch.stack([x[0][0] for x in rote_tensors])
    rt_a = torch.stack([x[1][0] for x in rote_tensors])
    rt_qt = torch.stack([pad_slot(x[2])[0][0] for x in rote_tensors])
    rt_qv = torch.stack([pad_slot(x[2])[1][0] for x in rote_tensors])

    for step in range(1, steps + 1):
        lr = cosine_lr(step, steps)  # [audit #16] same schedule as AR / Pure-Diffusion
        for pg in opt_m.param_groups: pg['lr'] = lr
        opt_m.zero_grad()
        loss_tot = 0.0

        if not mod.pure_thought_only:
            # Train System 1 Fast Muscle Memory on Rote Serials + Emergency Codes
            m1 = (torch.rand_like(rt_a, dtype=torch.float) < 0.5); m1[:, 0] = True
            m2 = torch.ones_like(rt_a, dtype=torch.bool)
            for m in (m1, m2):
                c = rt_a.clone(); c[m] = MASK_ID
                l = rt_a.clone(); l[~m] = -100
                lg = mod.forward_fast_muscle(rt_p, c)
                loss_tot = loss_tot + F.cross_entropy(lg.reshape(-1, lg.size(-1)), l.reshape(-1), ignore_index=-100)
        else:
            # Pure Thought LCM: Must squeeze all 20 Rote Serials through continuous thought vector with realistic latent noise!
            dq_r, sq_r = mod.encode_thought(rt_qt, rt_qv, noise_std=0.04)
            canv_r = torch.full_like(rt_a, MASK_ID)
            lg_r = mod.forward_thought_render(dq_r, sq_r, canv_r)
            loss_tot = loss_tot + F.cross_entropy(lg_r.reshape(-1, lg_r.size(-1)), rt_a.reshape(-1), ignore_index=-100)

        # [audit #14] Encode the allowed clauses only (seen: all stages; unseen: stage 0/1)
        d_vec, s_vec = mod.encode_thought(b_ct, b_cv)

        loss_dis = 0.0
        # 1. Domain invariance: every allowed clause of domain i shares d_i (anchor = law 0, stage 0)
        d_anchor = d_vec[dom_anchor_rows]  # (6, D)
        loss_dis = loss_dis + (1.0 - (d_vec * d_anchor[k_dom]).sum(dim=-1)).mean()
        # Domain orthogonality across the 6 domains
        d_gram = torch.matmul(d_anchor, d_anchor.T)
        eye_6 = torch.eye(n_dom, device='cuda', dtype=torch.bool)
        loss_dis = loss_dis + F.relu(d_gram[~eye_6] + 0.1).mean()

        # 2. Structural isomorphism: every allowed clause with (law_id, stage_j) shares s_{law, stage} (anchor = Domain 0, seen)
        s_anchor = s_vec[struct_anchor_rows]  # (8, D)
        loss_dis = loss_dis + (1.0 - (s_vec * s_anchor[k_state]).sum(dim=-1)).mean()
        # Orthogonal repulsion across all 8 distinct structural states (2 laws x 4 stages = 8 states)
        s_gram = torch.matmul(s_anchor, s_anchor.T)
        eye_8 = torch.eye(8, device='cuda', dtype=torch.bool)
        loss_dis = loss_dis + F.relu(s_gram[~eye_8] + 0.1).mean()

        # 3. Train Structural Hopping ONLY on the 2 Seen Domains (Water & Highway, both laws: 4 chains of length 4)!
        s_seen_chains = s_vec[hop_rows]  # (4, 4, D) [audit #14] explicit seen-domain index list
        pred_s = mod.forward_struct_hop(s_seen_chains[:, :-1, :])
        loss_hop = (1.0 - (pred_s * s_seen_chains[:, 1:, :].detach()).sum(dim=-1)).mean()

        # 4. Train Thought-to-Language Block-Diffusion Mapper on the allowed clauses ([audit #14] was all 48)
        for mask_p in (0.5, 1.0):
            m = (torch.rand_like(b_ct, dtype=torch.float) < mask_p); m[:, 0] = True
            c = b_ct.clone(); c[m] = MASK_ID
            l = b_ct.clone(); l[~m] = -100
            lg_s2 = mod.forward_thought_render(d_vec, s_vec, c)
            loss_tot = loss_tot + F.cross_entropy(lg_s2.reshape(-1, lg_s2.size(-1)), l.reshape(-1), ignore_index=-100)

        loss_tot = loss_tot + 2.5 * loss_hop + 0.8 * loss_dis
        loss_tot.backward()
        torch.nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
        opt_m.step()
        if step in {1, steps // 2, steps}:  # [audit #16] make training failures (e.g. Micro Dual-Brain) visible
            print(f"  [{tag}] step {step:4d}/{steps} loss={loss_tot.item():.4f} hop={loss_hop.item():.4f} dis={loss_dis.item():.4f} lr={lr:.2e}", flush=True)
    mod.eval()

print("Training competing models on the 36-Task Multi-Law & High-Entropy Stress Suite...", flush=True)
train_stress_model(m_lcm, steps=TRAIN_STEPS, tag="PureThought-LCM-512")
train_stress_model(m_micro_dual, steps=TRAIN_STEPS, tag="Micro-Dual-128")
train_stress_model(m_tiny_dual, steps=TRAIN_STEPS, tag="Tiny-Dual-256")
train_stress_model(m_full_dual, steps=TRAIN_STEPS, tag="Full-Dual-512")
m_ar.eval(); m_diff.eval()

# [audit #13] Fix the RNG before evaluation: Tier 3 injects random drift noise per hop.
random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)  # [audit #7]

# ==============================================================================
# EVALUATE ON THE 36-TASK NON-SATURATED STRESS BENCHMARK  [audit #16] (was 40: Tier 4 double-counted Tier 2)
# ==============================================================================
def second_clause(s: str) -> str:
    """[audit #16] AR Tier-3: the text after the first generated clause (i.e. the 2nd generated clause onward)."""
    s = norm_text(s)
    m = _re.search(r"[，。；！？]", s)
    return s[m.end():] if m else ""

def predict_rote(mod, q_s):
    """[audit #16] one rote-answer prediction path per model family (shared by Tier 1 and Tier 4)."""
    if mod is m_ar:
        inp = torch.tensor([[BOS_ID] + tokenizer(q_s, add_special_tokens=False).input_ids], device='cuda')
        out = mod.generate(inp, max_new_tokens=AR_MAX_NEW_TOKENS, do_sample=False, use_cache=True)
        return clean_str(out[0, inp.size(1):])
    elif mod is m_diff:
        return clean_str(mod.generate_slots(pad_prompt(q_s), steps=2)[0])
    elif mod.pure_thought_only:
        qt, qv = pad_slot(q_s)
        dq, sq = mod.encode_thought(qt, qv, noise_std=0.04)
        return clean_str(mod.render_from_ds(dq, sq, steps=2)[0])
    else:
        _, out = mod.probe_muscle(pad_prompt(q_s), steps=2)
        return clean_str(out[0])

N_T1, N_T2, N_T3, N_T4, N_TOT = 16, 8, 8, 4, 36  # [audit #16]
print("\n" + "=" * 150, flush=True)
print("=== [NON-SATURATED STRESS BENCHMARK] 36 SCORED TASKS ACROSS 4 TIERS (Tier 3 = 2 latent hops; Tier4-compound is info only, NOT in total) ===", flush=True)
print("=" * 150, flush=True)
print(f"{'Model Architecture':<30} | {'Params':>7} | {'Tier1:16 Serials':>16} | {'Tier2:8 MultiLaw':>16} | {'Tier3:8 2-Hop':>15} | {'Tier4:4 EmgCode':>15} | {'Total (36)':>13} | {'IQ Density':>12} | {'[info] T2&Code /8':>17}", flush=True)
print("-" * 150, flush=True)

for name_m, mod in [
    ("1. AR-MiniMind (8L,512D)", m_ar),
    ("2. Pure-Diffusion (8L,512D)", m_diff),
    ("3. Pure-Thought LCM (512D)", m_lcm),
    ("4. Micro Dual-Brain (2L,128D)", m_micro_dual),
    ("5. Tiny Dual-Brain (4L,256D)", m_tiny_dual),
    ("6. Full Dual-Brain (8L,512D)", m_full_dual),
]:
    t1_ok, t2_ok, t3_ok, t4_ok, t4_compound = 0, 0, 0, 0, 0

    # --- Tier 1: 16 Confusable High-Entropy Serial Numbers ---
    for q_s, gt_a in rote_serials:
        pred = predict_rote(mod, q_s)
        if exact_answer_match(pred, gt_a): t1_ok += 1  # [audit #16]

    # --- Tier 4: [audit #16] each of the 4 domain Emergency Codes scored ONCE (exact_answer_match) ---
    code_ok = []
    for emg_q, emg_gt in domain_emg_codes:
        code_ok.append(exact_answer_match(predict_rote(mod, emg_q), emg_gt))
    t4_ok = sum(code_ok)

    # --- Tier 2: 8 Multi-Law 1-Hop Unseen Analogies (4 Unseen Domains x 2 Laws, predicting Stage 2 from Stage 0+1) ---
    # --- Tier 3: 8 Multi-Hop Unseen Analogies (predicting Stage 3 from Stage 0+1 after 2 latent hops + drift noise) ---
    # [audit #16] Tier-3 baseline fairness: AR and Pure-Diffusion used to re-use their stage-2 output as the
    # stage-3 answer (could never be right). Now:
    #   * AR keeps generating (2 x AR_MAX_NEW_TOKENS) and is scored on the SECOND generated clause;
    #   * Pure-Diffusion runs a second time with premise = clauses[1] + its own predicted stage-2 clause
    #     (it was trained on seen-domain (stage1+stage2 -> stage3) pairs for exactly this query).
    unseen_idx = 0
    for d_idx, d_name, is_seen, law_id, clauses, c_t in dom_law_tensors:
        if is_seen: continue
        p_str = clauses[0] + clauses[1]
        gt_stage2 = clauses[2]
        gt_stage3 = clauses[3]

        if mod is m_ar:
            inp = torch.tensor([[BOS_ID] + tokenizer(p_str, add_special_tokens=False).input_ids], device='cuda')
            out = mod.generate(inp, max_new_tokens=2 * AR_MAX_NEW_TOKENS, do_sample=False, use_cache=True)
            gen = clean_str(out[0, inp.size(1):])
            pred_s2 = gen
            pred_s3 = second_clause(gen)  # [audit #16]
        elif mod is m_diff:
            pred_s2 = clean_str(mod.generate_slots(pad_prompt(p_str), steps=2)[0])
            pred_s3 = clean_str(mod.generate_slots(pad_prompt_tail(clauses[1] + first_clause(pred_s2)), steps=2)[0])  # [audit #16]
        else:
            # 1-hop analogy for Stage 2
            out_s2 = mod.solve_multi_hop_analogy(c_t[0][0], c_t[0][1], c_t[1][0], c_t[1][1], hops=1, hop_noise=0.02)
            pred_s2 = clean_str(out_s2[0])
            # 2-hop analogy for Stage 3 (with latent drift noise=0.08 per hop)
            out_s3 = mod.solve_multi_hop_analogy(c_t[0][0], c_t[0][1], c_t[1][0], c_t[1][1], hops=2, hop_noise=0.08)
            pred_s3 = clean_str(out_s3[0])

        hit_s2 = clause_match(pred_s2, gt_stage2)  # [audit #16] full clause, was gt[:5] substring
        hit_s3 = clause_match(pred_s3, gt_stage3)  # [audit #16]
        if hit_s2: t2_ok += 1
        if hit_s3: t3_ok += 1

        # [audit #16] old compound metric (Tier-2 hit AND domain code correct, /8): informational only, NOT in total
        if hit_s2 and code_ok[unseen_idx // 2]:
            t4_compound += 1
        unseen_idx += 1

    tot = t1_ok + t2_ok + t3_ok + t4_ok
    pm = count_m(mod)
    dens = tot / pm  # [audit #16] density over the 36 scored tasks
    print(f"{name_m:<30} | {pm:6.2f}M | {t1_ok:7d}/{N_T1} ({t1_ok/N_T1*100:3.0f}%) | {t2_ok:7d}/{N_T2}  ({t2_ok/N_T2*100:3.0f}%) | {t3_ok:8d}/{N_T3}  ({t3_ok/N_T3*100:3.0f}%) | {t4_ok:8d}/{N_T4}  ({t4_ok/N_T4*100:3.0f}%) | {tot:4d}/{N_TOT} ({tot/N_TOT*100:3.0f}%) | {dens:7.2f} t/M | {t4_compound:12d}/8", flush=True)

# ==============================================================================
# [audit #7] MECHANISM TEST (pre-registered, see issue #7)
# Runs AFTER the benchmark above (which is left untouched). Everything is inference-only:
# models in eval mode, torch.no_grad, NO latent / hop noise (noise_std=0, hop_noise=0).
#   A  : premise s0,s1 (unseen domain) -> struct_brain hop -> s^2 (1 hop), s^3 (2 hops) -> nearest anchor
#   D1 : s0, s1 themselves -> nearest anchor (funnel localisation, 16 items)
#   D2 : encode the HELD-OUT true c2 (inference only) -> nearest anchor (answer localisation)
#   B  : rank the same 8 candidates (unseen domain's 2 laws x 4 stages) per item; chance 1/8
# Anchors: 8 (law, stage) anchors = L2-normalised mean of s over SEEN-domain clauses.
# ==============================================================================
P7_THOUGHT_MODELS = [("PureThought-LCM-512", m_lcm), ("Micro-Dual-128", m_micro_dual),
                     ("Tiny-Dual-256", m_tiny_dual), ("Full-Dual-512", m_full_dual)]  # [audit #7]
P7_ALL_MODELS = [("AR-512", m_ar), ("Diff-512", m_diff)] + P7_THOUGHT_MODELS        # [audit #7]
P7_LABELS = [(lw, st) for lw in (0, 1) for st in range(4)]                          # [audit #7] anchor index = lw*4+st
P7_INSPECT = "Tiny-Dual-256"                                                        # [audit #7]

def p7_lbl(i):  # [audit #7]
    lw, st = P7_LABELS[i]
    return f"L{lw}S{st}"

# [audit #7] the 8 probe items: (domain name, law, clauses, c_t) for unseen domains, same order as Tier 2/3
p7_items = [(d_name, law_id, clauses, c_t) for _, d_name, is_seen, law_id, clauses, c_t in dom_law_tensors if not is_seen]
assert len(p7_items) == 8
# [audit #7] candidate set per unseen domain: its 8 clauses (law-major, stage-minor), shared by ALL models
p7_cands = {}
for _, d_name, is_seen, law_id, clauses, c_t in dom_law_tensors:
    if is_seen: continue
    p7_cands.setdefault(d_name, [None] * 8)
    for st in range(4):
        p7_cands[d_name][law_id * 4 + st] = (clauses[st], c_t[st])

@torch.no_grad()
def p7_encode(mod, ct):  # [audit #7] same code path as solve_multi_hop_analogy, noise 0
    d, s = mod.encode_thought(ct[0], ct[1], noise_std=0.0)
    return d, s  # (1, D) each, model dtype

@torch.no_grad()
def p7_hops(mod, s0, s1, hops):  # [audit #7] identical loop to solve_multi_hop_analogy, hop_noise = 0
    curr_seq = torch.cat([s0, s1], dim=0).unsqueeze(0)
    for _ in range(hops):
        next_s = mod.forward_struct_hop(curr_seq)[:, -1, :]
        curr_seq = torch.cat([curr_seq, next_s.unsqueeze(1)], dim=1)
    return curr_seq[:, -1, :]

@torch.no_grad()
def p7_anchors(mod):  # [audit #7] (8, D) float32, L2-normalised mean s over seen-domain clauses
    acc = {i: [] for i in range(8)}
    for _, _, is_seen, law_id, _, c_t in dom_law_tensors:
        if not is_seen: continue
        for st in range(4):
            acc[law_id * 4 + st].append(p7_encode(mod, c_t[st])[1].float()[0])
    return F.normalize(torch.stack([torch.stack(acc[i]).mean(dim=0) for i in range(8)]), p=2, dim=-1)

def p7_nearest(anchors, s):  # [audit #7] cosine-nearest anchor index (anchors and s are unit vectors)
    return int((anchors @ F.normalize(s.float()[0], p=2, dim=-1)).argmax().item())

def p7_cos(a, b):  # [audit #7]
    return F.cosine_similarity(a.float(), b.float(), dim=-1).item()

@torch.no_grad()
def p7_ar_score(prem, cand):  # [audit #7] log p(cand | BOS + premise) under AR (premise tokens are common to all candidates)
    p_ids = [BOS_ID] + tokenizer(prem, add_special_tokens=False).input_ids
    c_ids = tokenizer(cand, add_special_tokens=False).input_ids
    inp = torch.tensor([p_ids + c_ids], device='cuda')
    lp = F.log_softmax(m_ar(inp).logits[0].float(), dim=-1)
    pos = torch.arange(len(p_ids) - 1, len(p_ids) + len(c_ids) - 1, device='cuda')
    return lp[pos, torch.tensor(c_ids, device='cuda')].sum().item()

@torch.no_grad()
def p7_diff_score(prem, cand):  # [audit #7] single step, fully-masked slot, sum log-prob of the padded candidate slot
    canv = torch.full((1, SLOT_LEN), MASK_ID, dtype=torch.long, device='cuda')
    prompt = pad_prompt_tail(prem)
    lg = m_diff.backbone(torch.cat([prompt, canv], dim=1)).logits[:, prompt.size(1):, :]
    lp = F.log_softmax(lg[0].float(), dim=-1)
    tgt = pad_slot(cand)[0][0]
    return lp[torch.arange(SLOT_LEN, device='cuda'), tgt].sum().item()

# [audit #7, post-hoc] Length-normalised variants. Summed log-likelihood systematically prefers shorter
# candidates; lm-evaluation-harness reports a length-normalised accuracy for this reason. Added AFTER the
# first seed-42 run showed AR at 0/8 on every seed; reported alongside (not instead of) the registered sums.
@torch.no_grad()
def p7_ar_score_norm(prem, cand):
    n = len(tokenizer(cand, add_special_tokens=False).input_ids)
    return p7_ar_score(prem, cand) / max(1, n)

@torch.no_grad()
def p7_diff_score_norm(prem, cand):  # mean log-prob over the candidate's real tokens only (pad slots excluded)
    canv = torch.full((1, SLOT_LEN), MASK_ID, dtype=torch.long, device='cuda')
    prompt = pad_prompt_tail(prem)
    lg = m_diff.backbone(torch.cat([prompt, canv], dim=1)).logits[:, prompt.size(1):, :]
    lp = F.log_softmax(lg[0].float(), dim=-1)
    tgt, val = pad_slot(cand)
    tgt, val = tgt[0], val[0].bool()
    return lp[torch.arange(SLOT_LEN, device='cuda'), tgt][val].mean().item()

def p7_argmax(scores):  # [audit #7]
    return max(range(len(scores)), key=lambda i: scores[i])

p7_results = {}
p7_inspect_rows = []
with torch.no_grad():
    for nm7, mod7 in P7_ALL_MODELS:
        mod7.eval()
        r = dict(A1=-1, A2=-1, A1law=-1, A1stage=-1, A2law=-1, A2stage=-1, D1=-1, D2=-1,
                 B1=0, B2=0, B1s=-1, B2s=-1, B2direct=-1, B1n=-1, B2n=-1)
        is_thought = isinstance(mod7, StressDualBrainModel)
        if is_thought:
            anchors = p7_anchors(mod7)
            for k in ("A1", "A2", "A1law", "A1stage", "A2law", "A2stage", "D1", "D2", "B1s", "B2s"): r[k] = 0
        else:
            r["B2direct"] = 0
            r["B1n"] = 0; r["B2n"] = 0  # [audit #7, post-hoc]
        for d_name, law_id, clauses, c_t in p7_items:
            cands = p7_cands[d_name]
            gold2, gold3 = law_id * 4 + 2, law_id * 4 + 3
            if is_thought:
                d0, s0 = p7_encode(mod7, c_t[0]); d1, s1 = p7_encode(mod7, c_t[1])
                s_hat2 = p7_hops(mod7, s0, s1, 1); s_hat3 = p7_hops(mod7, s0, s1, 2)
                n0, n1 = p7_nearest(anchors, s0), p7_nearest(anchors, s1)
                n2, n3 = p7_nearest(anchors, s_hat2), p7_nearest(anchors, s_hat3)
                nc2 = p7_nearest(anchors, p7_encode(mod7, c_t[2])[1])  # D2: held-out c2, inference only
                # Test A
                r["A1"] += int(n2 == gold2); r["A2"] += int(n3 == gold3)
                r["A1law"] += int(P7_LABELS[n2][0] == law_id); r["A1stage"] += int(P7_LABELS[n2][1] == 2)
                r["A2law"] += int(P7_LABELS[n3][0] == law_id); r["A2stage"] += int(P7_LABELS[n3][1] == 3)
                # D1 / D2
                r["D1"] += int(n0 == law_id * 4 + 0) + int(n1 == law_id * 4 + 1)
                r["D2"] += int(nc2 == gold2)
                # Test B: cos(s^, s_c) + cos(d_premise, d_c); d_premise as in solve_multi_hop_analogy
                d_prem = F.normalize(d0 + d1, p=2, dim=-1)
                enc_c = [p7_encode(mod7, ct) for _, ct in cands]
                b_picks = []
                for s_hat, gold, kf, ks in ((s_hat2, gold2, "B1", "B1s"), (s_hat3, gold3, "B2", "B2s")):
                    full = [p7_cos(s_hat, sc) + p7_cos(d_prem, dc) for dc, sc in enc_c]
                    sonly = [p7_cos(s_hat, sc) for _, sc in enc_c]
                    r[kf] += int(p7_argmax(full) == gold); r[ks] += int(p7_argmax(sonly) == gold)
                    b_picks.append(p7_lbl(p7_argmax(full)))
                if nm7 == P7_INSPECT:
                    p7_inspect_rows.append((d_name, law_id, p7_lbl(n0), p7_lbl(n1), p7_lbl(n2), p7_lbl(n3), p7_lbl(nc2), b_picks))
            else:
                score = p7_ar_score if mod7 is m_ar else p7_diff_score
                prem = clauses[0] + clauses[1]
                sc2 = [score(prem, c) for c, _ in cands]
                pick2 = p7_argmax(sc2)
                r["B1"] += int(pick2 == gold2)
                # B2 (scored): candidate follows the model's OWN B1 pick, mirroring Tier 3 (no gold stage-2 leak).
                #   AR: premise c0+c1+pick; Pure-Diffusion: premise c1+pick (the seen-domain stage1+stage2->stage3 pairs it was trained on).
                prem3 = (prem + cands[pick2][0]) if mod7 is m_ar else (clauses[1] + cands[pick2][0])
                sc3 = [score(prem3, c) for c, _ in cands]
                r["B2"] += int(p7_argmax(sc3) == gold3)
                # B2-direct (info only): candidate scored right after premise c0+c1
                r["B2direct"] += int(p7_argmax(sc2) == gold3)
                # [audit #7, post-hoc] length-normalised B1/B2 with the same B2 conditioning rule
                score_n = p7_ar_score_norm if mod7 is m_ar else p7_diff_score_norm
                sn2 = [score_n(prem, c) for c, _ in cands]
                pick2n = p7_argmax(sn2)
                r["B1n"] += int(pick2n == gold2)
                prem3n = (prem + cands[pick2n][0]) if mod7 is m_ar else (clauses[1] + cands[pick2n][0])
                r["B2n"] += int(p7_argmax([score_n(prem3n, c) for c, _ in cands]) == gold3)
        p7_results[nm7] = r

print("\n" + "=" * 150, flush=True)
print("=== [audit #7] MECHANISM TEST (pre-registered, see issue #7) ===", flush=True)
print("=" * 150, flush=True)
print(f"[audit #7] SEED={SEED}  |  8 items = 4 unseen domains x 2 laws  |  anchors = mean s over seen-domain clauses per (law, stage)  |  hop noise 0", flush=True)
for nm7, _ in P7_ALL_MODELS:
    r = p7_results[nm7]
    print(f"--- {nm7} ---", flush=True)
    if r["A1"] >= 0:
        print(f"  A1 (s^2 -> (law,2))   : {r['A1']}/8 (chance 1/8)   law-correct {r['A1law']}/8, stage-correct {r['A1stage']}/8", flush=True)
        print(f"  A2 (s^3 -> (law,3))   : {r['A2']}/8 (chance 1/8)   law-correct {r['A2law']}/8, stage-correct {r['A2stage']}/8", flush=True)
        print(f"  D1 (s0,s1 localised)  : {r['D1']}/16 (chance 1/8)   [diagnostic]", flush=True)
        print(f"  D2 (true c2 localised): {r['D2']}/8 (chance 1/8)   [diagnostic]", flush=True)
        print(f"  B1 full / s-only      : {r['B1']}/8 / {r['B1s']}/8 (chance 1/8)", flush=True)
        print(f"  B2 full / s-only      : {r['B2']}/8 / {r['B2s']}/8 (chance 1/8)", flush=True)
    else:
        print(f"  A1/A2/D1/D2           : n/a (no thought vector)", flush=True)
        print(f"  B1                    : {r['B1']}/8 (chance 1/8)", flush=True)
        print(f"  B2 (after own B1 pick): {r['B2']}/8 (chance 1/8)   [info] B2-direct after c0+c1: {r['B2direct']}/8", flush=True)
        print(f"  [post-hoc] length-normalised B1 / B2: {r['B1n']}/8 / {r['B2n']}/8 (chance 1/8)", flush=True)
print(f"--- nearest-anchor labels per item ({P7_INSPECT}; LxSy = law x, stage y) ---", flush=True)
print(f"  {'domain':<10} {'law':>3} | {'s0':>4} {'s1':>4} | {'s^2':>4} {'s^3':>4} | {'c2':>4} | {'B1pk':>4} {'B2pk':>4} | expected", flush=True)
for d_name, law_id, l0, l1, l2, l3, lc2, bp in p7_inspect_rows:
    print(f"  {d_name:<10} {law_id:>3} | {l0:>4} {l1:>4} | {l2:>4} {l3:>4} | {lc2:>4} | {bp[0]:>4} {bp[1]:>4} | L{law_id}S0 L{law_id}S1 L{law_id}S2 L{law_id}S3", flush=True)
for nm7, _ in P7_ALL_MODELS:
    r = p7_results[nm7]
    d1 = f"{r['D1']}/16" if r["D1"] >= 0 else "-1"
    print(f"PROBE_RESULT seed={SEED} model={nm7} A1={r['A1']} A2={r['A2']} D1={d1} D2={r['D2']} B1={r['B1']} B2={r['B2']} B1s={r['B1s']} B2s={r['B2s']} B1n={r['B1n']} B2n={r['B2n']}", flush=True)
