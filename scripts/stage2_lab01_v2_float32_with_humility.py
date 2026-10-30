# ==============================================================================
# ORIGINAL EXECUTED SCRIPT EXTRACTED VERBATIM FROM TRANSCRIPT (AST-UNESCAPED)
# Transcript Step Index : 458
# Timestamp             : 2026-10-30T07:32:27Z
# Original Colab Session: stage2_qwen_pk_v2
# ==============================================================================

import os, sys, time, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
SEED = int(os.environ.get("AUDIT_SEED", "42"))  # [audit #7] seed sweep (42, 1, 2) via env var
print(f"[audit #7] SEED={SEED}", flush=True)
random.seed(SEED)  # [audit #7] was random.seed(42)
torch.manual_seed(SEED)  # [audit #13] fixed seed (missing in the original run); [audit #7] was 42
from transformers import AutoTokenizer, AutoModelForCausalLM

# [audit #16] scoring helpers (inlined verbatim from the audit spec)
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

def cosine_lr(step: int, total: int, peak: float, floor: float) -> float:
    """[audit #16] one cosine schedule shared by every competitor (same shape as the original Dual schedule)."""
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * step / total))

MODEL_ID = "Qwen/Qwen2.5-0.5B"
print(f"=== [1] Loading Official Pre-trained '{MODEL_ID}' (494M Params, 24 Layers, 896-D Hidden State) ===", flush=True)
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
qwen_base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
qwen_base.eval()
for p in qwen_base.parameters():
    p.requires_grad = False  # 100% Freeze the 494M pre-trained weights!

H_DIM = qwen_base.config.hidden_size  # 896
VOCAB_SIZE = qwen_base.config.vocab_size
print(f"Loaded in {time.time()-t0:.1f}s! Hidden Dimension = {H_DIM}, Layers = {qwen_base.config.num_hidden_layers}, Vocab = {VOCAB_SIZE}", flush=True)

world_knowledge_checks = [
    ("中国的首都是", ["北京"]),
    ("地球围绕着什么天体公转？答：地球围绕着", ["太阳", "恒星"]),
    ("一年有四个季节，分别是春夏", ["秋冬"])
]

rote_secrets = [
    ("请问妈妈的手机号码是？", "0912-888-168"),
    ("请问爸爸的手机号码是？", "0933-777-999"),
    ("请问核心星舰的重启密钥是？", "OMEGA-7749-X9"),
    ("请问深海基地的通信频段是？", "FREQ-409.5-MHZ"),
    ("请问量子金库的通行密码是？", "VAULT-9901-Z7"),
    ("请问主控服务器的IP地址是？", "192.168.88.254")
]

domains_pk = [
    ("水利管網", True, {
        0: ["水流在主管道稳定输送，", "当管道内壁结垢导致通道变窄时，", "管道内部水压就会急剧飙升，", "最终触发高压泄流阀自动停机。"],
        1: ["水库向主管道持续供水，", "当上游水源泵站突然断流时，", "管道内部储水就会迅速枯竭，", "最终触发备用蓄水池紧急补水。"]
    }),
    ("高速交通", True, {
        0: ["车流在主干道稳定行驶，", "当道路施工导致车道变窄时，", "主干道车流密度与压力急剧飙升，", "最终触发匝道限流号志自动封路。"],
        1: ["总站向主干道持续发车，", "当总站调度中心突然断流时，", "主干道运力班次就会迅速枯竭，", "最终触发备用接驳车队紧急支援。"]
    }),
    ("反物質約束環", False, {
        0: ["反物质流在磁环内稳定束缚，", "当磁场畸变导致约束通道变窄时，", "磁环内部辐射压力就会急剧飙升，", "最终触发磁能泄压盾自动停机。"],
        1: ["注入器向磁环持续供能，", "当反物质注入源突然断流时，", "磁环内部等离子体就会迅速枯竭，", "最终触发备用聚变堆紧急补能。"]
    }),
    ("神經植入晶片", False, {
        0: ["神经讯号在晶片内稳定传输，", "当突触阻抗导致讯号通道变窄时，", "晶片内部电位负荷就会急剧飙升，", "最终触发神经保护闸自动断电。"],
        1: ["生物电池向晶片持续供电，", "当微电流供给源突然断流时，", "晶片内部电荷储备就会迅速枯竭，", "最终触发备用压电模组紧急供电。"]
    }),
    ("星際曲速引擎", False, {
        0: ["曲速等离子在导管稳定循环，", "当引力波干扰导致导管变窄时，", "导管内部曲速场压就会急剧飙升，", "最终触发空间折叠阀自动停机。"],
        1: ["暗物质舱向导管持续供料，", "当暗物质输送阀突然断流时，", "导管内部曲速燃料就会迅速枯竭，", "最终触发备用反重力引擎紧急接管。"]
    }),
    ("人體微血管網", False, {
        0: ["血液在微血管稳定循环，", "当脂质斑块导致血管通道变窄时，", "微血管内部血压就会急剧飙升，", "最终触发血管扩张反射自动降压。"],
        1: ["心室向微血管持续供血，", "当冠状动脉血流突然断流时，", "微血管内部血氧就会迅速枯竭，", "最终触发侧支循环血管紧急供血。"]
    }),
]

# [audit #13] SLOT_LEN is derived from the data. The original hardcoded 12 combined with
# `[:length]` silently truncated any answer or clause longer than 12 Qwen tokens.
_slot_texts = ([a for _, a in rote_secrets]
               + [c for _, _, laws in domains_pk for cl in laws.values() for c in cl])
SLOT_LEN = max(len(tokenizer.encode(t, add_special_tokens=False)) for t in _slot_texts) + 1  # +1 pad slot
print(f"[audit #13] SLOT_LEN={SLOT_LEN} (longest slot text + 1)", flush=True)
PAD_ID = tokenizer.eos_token_id

def encode_slots(txt, length=None):
    length = SLOT_LEN if length is None else length
    ids = tokenizer.encode(txt, add_special_tokens=False)
    if len(ids) > length:
        raise ValueError(f"[audit #13] encode_slots would truncate {len(ids)} tokens to {length}: {txt!r}")
    pad = ids + [PAD_ID] * (length - len(ids))
    return torch.tensor(pad, dtype=torch.long, device='cuda')

def clean_qwen_str(t_ids):
    ids = [x for x in t_ids.tolist() if x != PAD_ID]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()

@torch.no_grad()
def extract_qwen_896_feature(text_list):
    '''Passes text through FROZEN Qwen2.5-0.5B and extracts the 24th-layer 896-D hidden state in float32!'''
    enc = tokenizer(text_list, return_tensors="pt", padding=True, truncation=True, max_length=64).to("cuda")
    out = qwen_base.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
    last_h = out.last_hidden_state.float()  # (B, L, 896) cast to float32 for high-precision geometry!
    mask_f = enc.attention_mask.unsqueeze(-1).float()
    pooled_896 = (last_h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
    return pooled_896  # (B, 896) float32

# ==============================================================================
# COMPETITOR A: Standard Qwen2.5-0.5B + Causal LoRA Fine-Tuning (Industrial Baseline)
# ==============================================================================
class LoRALinear(nn.Module):
    def __init__(self, orig_linear: nn.Linear, r: int = 16, alpha: float = 32.0):
        super().__init__()
        self.orig = orig_linear
        self.scale = alpha / r
        in_f, out_f = orig_linear.in_features, orig_linear.out_features
        self.lora_A = nn.Parameter(torch.randn(in_f, r, device='cuda', dtype=torch.bfloat16) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(r, out_f, device='cuda', dtype=torch.bfloat16))
        self.enabled = True

    def forward(self, x):
        res = self.orig(x)
        if self.enabled:
            res = res + (x @ self.lora_A @ self.lora_B) * self.scale
        return res

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
    for m in lora_modules: m.enabled = flag

sft_texts = []
for q_s, a_s in rote_secrets:
    sft_texts.append(q_s + a_s)
for d_name, is_seen, laws_d in domains_pk:
    for law_id in (0, 1):
        c_list = laws_d[law_id]
        if is_seen:
            sft_texts.append("".join(c_list))
        else:
            # [audit #14] unseen systems: premise clauses only (stage 0/1); the answer clauses c2+c3 are held out.
            # No EOS is appended (Qwen's tokenizer adds none and pads are label -100), so the baseline is not
            # taught to stop where the answer should begin.
            sft_texts.append(c_list[0])
            sft_texts.append(c_list[1])

# [audit #16] equal budget: both competitors get STEPS optimizer steps on the same cosine schedule
# (largest step count previously used in this lab = Dual's 240; LoRA used 120 with a constant LR).
STEPS = 240
LR_FLOOR = 5e-5
LORA_LR = 5e-4
DUAL_LR = 1.2e-3
n_lora_trainable = sum(p.numel() for p in lora_params)
print(f"=== [2] Training Competitor A: Standard Qwen2.5-0.5B + LoRA SFT ({n_lora_trainable/1e6:.2f}M LoRA params) ===", flush=True)
print(f"[audit #16] Competitor A budget: {STEPS} steps, cosine LR {LORA_LR:g} -> {LR_FLOOR:g}, trainable params = {n_lora_trainable:,}", flush=True)
set_lora_enabled(True)
opt_lora = torch.optim.AdamW(lora_params, lr=LORA_LR)
enc_sft = tokenizer(sft_texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to("cuda")
sft_labels = enc_sft.input_ids.clone()
sft_labels[enc_sft.attention_mask == 0] = -100

for step in range(1, STEPS + 1):  # [audit #16] was range(1, 121)
    for pg in opt_lora.param_groups: pg['lr'] = cosine_lr(step, STEPS, LORA_LR, LR_FLOOR)  # [audit #16]
    opt_lora.zero_grad()
    out = qwen_base(input_ids=enc_sft.input_ids, attention_mask=enc_sft.attention_mask, labels=sft_labels)
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
    opt_lora.step()
    if step in (1, 80, 160, 240):  # [audit #16]
        print(f"  [Competitor A: Qwen2.5-0.5B + LoRA SFT] Step {step:3d}/{STEPS} | SFT Loss: {out.loss.item():.4f}", flush=True)

set_lora_enabled(False)

# ==============================================================================
# COMPETITOR B: Surgical Dual-Brain Qwen2.5-0.5B (in float32 for precision + OOD Humility Gate)
# ==============================================================================
class DualBrainGraftedQwen(nn.Module):
    def __init__(self, qwen_hidden: int = 896, d_dim: int = 64, s_dim: int = 64, slot_len: int = SLOT_LEN):
        super().__init__()
        self.qwen_hidden = qwen_hidden
        self.d_dim = d_dim
        self.s_dim = s_dim
        self.slot_len = slot_len

        self.domain_funnel = nn.Sequential(nn.Linear(qwen_hidden, 256), nn.GELU(), nn.Linear(256, d_dim))
        self.struct_funnel = nn.Sequential(nn.Linear(qwen_hidden, 256), nn.GELU(), nn.Linear(256, s_dim))

        enc_l = nn.TransformerEncoderLayer(d_model=s_dim, nhead=4, dim_feedforward=128, batch_first=True, norm_first=True)
        self.law_brain = nn.TransformerEncoder(enc_l, num_layers=2, enable_nested_tensor=False)
        self.next_law_proj = nn.Linear(s_dim, s_dim)

        self.fast_muscle_slots = nn.Sequential(
            nn.Linear(qwen_hidden, qwen_hidden),
            nn.GELU(),
            nn.Linear(qwen_hidden, slot_len * qwen_hidden)
        )

        self.thought_render_slots = nn.Sequential(
            nn.Linear(d_dim + s_dim, qwen_hidden),
            nn.GELU(),
            nn.Linear(qwen_hidden, slot_len * qwen_hidden)
        )

    def extract_ds(self, h_896: torch.Tensor):
        d = F.normalize(self.domain_funnel(h_896), p=2, dim=-1)
        s = F.normalize(self.struct_funnel(h_896), p=2, dim=-1)
        return d, s

    def hop_law(self, s_seq: torch.Tensor):
        B, T, _ = s_seq.shape
        c_mask = torch.triu(torch.ones(T, T, device=s_seq.device, dtype=torch.bool), diagonal=1)
        h = self.law_brain(s_seq, mask=c_mask)
        return F.normalize(self.next_law_proj(h), p=2, dim=-1)

    def decode_fast_muscle(self, h_prompt_896: torch.Tensor, lm_head: nn.Module):
        B = h_prompt_896.size(0)
        slot_h = self.fast_muscle_slots(h_prompt_896).view(B, self.slot_len, self.qwen_hidden)
        return lm_head(slot_h.to(torch.bfloat16)).float()

    def decode_thought(self, d: torch.Tensor, s: torch.Tensor, lm_head: nn.Module):
        B = d.size(0)
        z_128 = torch.cat([d, s], dim=-1)
        slot_h = self.thought_render_slots(z_128).view(B, self.slot_len, self.qwen_hidden)
        return lm_head(slot_h.to(torch.bfloat16)).float()

dual_head = DualBrainGraftedQwen(qwen_hidden=H_DIM, d_dim=64, s_dim=64, slot_len=SLOT_LEN).cuda().float()
n_dual_trainable = sum(p.numel() for p in dual_head.parameters() if p.requires_grad)
print(f"\n=== [3] Training Competitor B: Dual-Brain Grafted Qwen2.5-0.5B (128-D Funnel + Fast/Slow Heads = {n_dual_trainable/1e6:.2f}M params) ===", flush=True)
print(f"[audit #16] Competitor B budget: {STEPS} steps, cosine LR {DUAL_LR:g} -> {LR_FLOOR:g}, trainable params = {n_dual_trainable:,}", flush=True)

rote_q_txts = [q for q, _ in rote_secrets]
rote_a_slots = torch.stack([encode_slots(a) for _, a in rote_secrets])

for d_name, is_seen, laws_d in domains_pk:
    if is_seen:  # [audit #14] muscle memory uses seen-domain answers only; unseen answers never enter loss_s1
        for law_id in (0, 1):
            c_list = laws_d[law_id]
            rote_q_txts.append(c_list[0] + c_list[1])
            rote_a_slots = torch.cat([rote_a_slots, encode_slots(c_list[2]).unsqueeze(0)], dim=0)

h_rote_prompts = extract_qwen_896_feature(rote_q_txts)  # (10, 896)

all_clause_txts = []
for d_name, is_seen, laws_d in domains_pk:
    for law_id in (0, 1):
        for c in laws_d[law_id]:
            all_clause_txts.append(c)

h_all_clauses = extract_qwen_896_feature(all_clause_txts)  # (48, 896)
target_all_slots = torch.stack([encode_slots(c) for c in all_clause_txts])  # (48, SLOT_LEN)

# [audit #14] Explicit clause bookkeeping. Unseen systems: stage 0/1 = premise (allowed), stage 2/3 = answer
# (held out of EVERY loss). Replaces the old `.view(6, 2, 4, 64)` grid that used all 48 clauses.
clause_meta = []  # (dom_idx, law_id, stage, is_seen), aligned with all_clause_txts
for _i, (_dn, _seen, _laws) in enumerate(domains_pk):
    for _j in (0, 1):
        for _k in range(len(_laws[_j])):
            clause_meta.append((_i, _j, _k, _seen))
_gidx = {(m[0], m[1], m[2]): g for g, m in enumerate(clause_meta)}
allowed_idx = [g for g, (_i, _j, _k, _seen) in enumerate(clause_meta) if _seen or _k in (0, 1)]
_pos = {g: p for p, g in enumerate(allowed_idx)}
N_DOM = len(domains_pk)
allowed_t = torch.tensor(allowed_idx, dtype=torch.long, device='cuda')
h_train_clauses = h_all_clauses[allowed_t]          # only allowed clauses ever reach a loss
target_train_slots = target_all_slots[allowed_t]
dom_of_train = torch.tensor([clause_meta[g][0] for g in allowed_idx], device='cuda')
d_anchor_pos = torch.tensor([_pos[_gidx[(i, 0, 0)]] for i in range(N_DOM)], device='cuda')  # (dom, law1, stage0) premise
s_anchor_pos = torch.tensor([_pos[_gidx[(0, clause_meta[g][1], clause_meta[g][2])]] for g in allowed_idx], device='cuda')  # water (law, stage)
water_pos = torch.tensor([_pos[_gidx[(0, j, k)]] for j in (0, 1) for k in range(4)], device='cuda')  # 8 water clauses (seen)
seen_doms = [i for i, (_dn, _seen, _l) in enumerate(domains_pk) if _seen]
seen_hop_pos = torch.tensor([[_pos[_gidx[(i, j, k)]] for k in range(4)] for i in seen_doms for j in (0, 1)], device='cuda')
print(f"[audit #14] held out {len(clause_meta) - len(allowed_idx)} unseen answer clauses from every loss", flush=True)

opt_dual = torch.optim.AdamW(dual_head.parameters(), lr=DUAL_LR, weight_decay=0.01)
for step in range(1, STEPS + 1):  # [audit #16] STEPS shared with Competitor A
    lr = cosine_lr(step, STEPS, DUAL_LR, LR_FLOOR)
    for pg in opt_dual.param_groups: pg['lr'] = lr
    opt_dual.zero_grad()

    # 1. System 1 Muscle Memory Loss + OOD Humility Regularization (on perturbed/interpolated vectors not in rote memory!)
    logits_s1 = dual_head.decode_fast_muscle(h_rote_prompts, qwen_base.lm_head)
    loss_s1 = F.cross_entropy(logits_s1.reshape(-1, VOCAB_SIZE), rote_a_slots.reshape(-1))

    # OOD Humility: random convex combinations of isolated single clauses should have low max confidence in Fast Muscle!
    # [audit #14] OOD samples drawn only from allowed clauses (no unseen answer clauses)
    perm = torch.randperm(h_train_clauses.size(0), device='cuda')[:8]
    h_ood = F.normalize(h_train_clauses[perm] + 0.35 * torch.randn_like(h_train_clauses[perm]), p=2, dim=-1) * h_rote_prompts.norm(dim=-1).mean()
    logits_ood = dual_head.decode_fast_muscle(h_ood, qwen_base.lm_head)[:, :4, :]  # check first 4 slots
    probs_ood = F.softmax(logits_ood, dim=-1)
    max_conf_ood = probs_ood.max(dim=-1).values
    loss_humility = F.relu(max_conf_ood - 0.35).mean()

    # 2. 896-D -> 128-D (64D d _|_ 64D s) Narrow Orthogonal Funnel Loss
    # [audit #14] explicit index lists over allowed clauses replace the 6x2x4 grid view
    d_vec, s_vec = dual_head.extract_ds(h_train_clauses)

    loss_dis = 0.0
    d_anchor = d_vec[d_anchor_pos]  # (6, 64): each domain's (law 1, stage 0) premise
    loss_dis = loss_dis + (1.0 - (d_vec * d_anchor[dom_of_train]).sum(dim=-1)).mean()
    d_gram = torch.matmul(d_anchor, d_anchor.T)
    loss_dis = loss_dis + F.relu(d_gram[~torch.eye(N_DOM, device='cuda', dtype=torch.bool)] + 0.1).mean()

    # each allowed clause's s aligns to the water clause with the same (law, stage); unseen domains only stage 0/1
    loss_dis = loss_dis + (1.0 - (s_vec * s_vec[s_anchor_pos]).sum(dim=-1)).mean()
    s_flat8 = s_vec[water_pos]  # (8, 64)
    s_gram = torch.matmul(s_flat8, s_flat8.T)
    loss_dis = loss_dis + F.relu(s_gram[~torch.eye(8, device='cuda', dtype=torch.bool)] + 0.1).mean()

    # 3. Train Law Hopping ONLY on the 2 Seen Domains (Water & Highway)! NEVER on the 4 Unseen Systems!
    s_seen = s_vec[seen_hop_pos]  # [audit #14] (4, 4, 64) via explicit indices
    pred_s = dual_head.hop_law(s_seen[:, :-1, :])
    loss_hop = (1.0 - (pred_s * s_seen[:, 1:, :].detach()).sum(dim=-1)).mean()

    # 4. System 2 Thought-to-Language Rendering Loss
    logits_s2 = dual_head.decode_thought(d_vec, s_vec, qwen_base.lm_head)
    loss_s2 = F.cross_entropy(logits_s2.reshape(-1, VOCAB_SIZE), target_train_slots.reshape(-1))  # [audit #14]

    loss = loss_s1 + 2.0 * loss_humility + loss_s2 + 2.5 * loss_hop + 0.8 * loss_dis
    loss.backward()
    torch.nn.utils.clip_grad_norm_(dual_head.parameters(), 1.0)
    opt_dual.step()

    if step in (1, 80, 160, 240):
        print(f"  [Competitor B: Dual-Brain Qwen2.5-0.5B] Step {step:3d}/{STEPS} | Fast-Muscle: {loss_s1.item():.4f} | Thought-Render: {loss_s2.item():.4f} | Law-Hop: {loss_hop.item():.4f} | Funnel-Ortho: {loss_dis.item():.4f}", flush=True)

dual_head.eval()

# ==============================================================================
# HEAD-TO-HEAD PK EVALUATION: Standard LoRA Qwen2.5-0.5B vs Dual-Brain Qwen2.5-0.5B
# ==============================================================================
print("\n====================================================================================================================", flush=True)
print("=== [HEAD-TO-HEAD PK] Competitor A (Qwen2.5-0.5B + Standard LoRA) vs Competitor B (Dual-Brain Grafted Qwen2.5-0.5B) ===", flush=True)
print("====================================================================================================================", flush=True)

# 1. Test 18T Pre-trained World Knowledge Retention
set_lora_enabled(True)
wk_lora, wk_dual = 0, 0
for w_q, w_gts in world_knowledge_checks:
    inp = tokenizer(w_q, return_tensors="pt").to("cuda")
    out_a = qwen_base.generate(**inp, max_new_tokens=6, do_sample=False, pad_token_id=PAD_ID)
    ans_a = tokenizer.decode(out_a[0, inp.input_ids.size(1):], skip_special_tokens=True).strip()
    if any(g in ans_a for g in w_gts): wk_lora += 1

set_lora_enabled(False)
for w_q, w_gts in world_knowledge_checks:
    inp = tokenizer(w_q, return_tensors="pt").to("cuda")
    out_b = qwen_base.generate(**inp, max_new_tokens=6, do_sample=False, pad_token_id=PAD_ID)
    ans_b = tokenizer.decode(out_b[0, inp.input_ids.size(1):], skip_special_tokens=True).strip()
    if any(g in ans_b for g in w_gts): wk_dual += 1  # [audit #16] base Qwen (LoRA off), Dual head not involved

# 2. Test 6 Rote Secrets
rote_lora, rote_dual = 0, 0
for q_s, gt_a in rote_secrets:
    set_lora_enabled(True)
    inp = tokenizer(q_s, return_tensors="pt").to("cuda")
    out_a = qwen_base.generate(**inp, max_new_tokens=max(10, SLOT_LEN), do_sample=False, pad_token_id=PAD_ID)
    pred_a = tokenizer.decode(out_a[0, inp.input_ids.size(1):], skip_special_tokens=True).strip()
    if exact_answer_match(pred_a, gt_a): rote_lora += 1  # [audit #16]

    set_lora_enabled(False)
    h_q = extract_qwen_896_feature([q_s])
    logits_b = dual_head.decode_fast_muscle(h_q, qwen_base.lm_head)
    pred_b = clean_qwen_str(logits_b.argmax(dim=-1)[0])
    if exact_answer_match(pred_b, gt_a): rote_dual += 1  # [audit #16]

# 3. Test 4 Seen System Chains & 8 UNSEEN System Chains
seen_lora, seen_dual = 0, 0
unseen_lora, unseen_dual = 0, 0
pk_comparisons = []

for d_name, is_seen, laws_d in domains_pk:
    for law_id in (0, 1):
        c_list = laws_d[law_id]
        prompt_str = c_list[0] + c_list[1]
        gt_stage2 = c_list[2]
        gt_stage3 = c_list[3]

        set_lora_enabled(True)
        inp = tokenizer(prompt_str, return_tensors="pt").to("cuda")
        out_a = qwen_base.generate(**inp, max_new_tokens=max(18, SLOT_LEN), do_sample=False, pad_token_id=PAD_ID)
        pred_a = tokenizer.decode(out_a[0, inp.input_ids.size(1):], skip_special_tokens=True).strip()

        set_lora_enabled(False)
        h_p = extract_qwen_896_feature([prompt_str])
        with torch.no_grad():
            fast_logits = dual_head.decode_fast_muscle(h_p, qwen_base.lm_head)
            probs = F.softmax(fast_logits, dim=-1)
            conf, fast_pred = probs.max(dim=-1)
            non_pad = (fast_pred != PAD_ID)
            min_conf = conf[non_pad].min().item() if non_pad.any() else conf.min().item()

            if min_conf >= 0.85:
                route_tag = f"⚡ Fast Muscle (MinConf={min_conf*100:.1f}%)"
                pred_b = clean_qwen_str(fast_pred[0])
            else:
                route_tag = f"🧠 128-D Thought Funnel (MinConf={min_conf*100:.1f}% < 85%)"
                h_c01 = extract_qwen_896_feature([c_list[0], c_list[1]])
                d01, s01 = dual_head.extract_ds(h_c01)
                d_tgt = F.normalize(d01[0:1] + d01[1:2], p=2, dim=-1)
                s_seq = s01.unsqueeze(0)
                s2_hat = dual_head.hop_law(s_seq)[:, -1, :]
                s3_hat = dual_head.hop_law(torch.cat([s_seq, s2_hat.unsqueeze(1)], dim=1))[:, -1, :]
                t_logits_2 = dual_head.decode_thought(d_tgt, s2_hat, qwen_base.lm_head)
                t_logits_3 = dual_head.decode_thought(d_tgt, s3_hat, qwen_base.lm_head)
                pred_b = clean_qwen_str(t_logits_2.argmax(dim=-1)[0]) + clean_qwen_str(t_logits_3.argmax(dim=-1)[0])

        hit_a = clause_match(pred_a, gt_stage2)  # [audit #16] full stage-2 clause, not first 5 chars
        hit_b = clause_match(pred_b, gt_stage2)
        if is_seen:
            if hit_a: seen_lora += 1
            if hit_b: seen_dual += 1
        else:
            if hit_a: unseen_lora += 1
            if hit_b: unseen_dual += 1
            pk_comparisons.append((f"{d_name}-規律{law_id+1}", prompt_str, gt_stage2 + gt_stage3, pred_a, route_tag, pred_b, gt_stage2))  # [audit #16] keep stage-2 gt for clause_match

print(f"{'Evaluation Category':<42} | {'Qwen2.5-0.5B + Standard LoRA':>28} | {'Dual-Brain Qwen2.5-0.5B (Ours)':>30}", flush=True)
print("-" * 106, flush=True)
# [audit #16] the 'Dual' world-knowledge number is base Qwen with LoRA disabled; it is not a Dual-Brain result
print(f"{'1. 18T World Knowledge Retention (3)':<42} | {wk_lora:24d} / 3 | {wk_dual:2d} / 3 (base Qwen, Dual head not involved)", flush=True)
print(f"{'2. Private Rote Secrets & Keys (6)':<42} | {rote_lora:24d} / 6 | {rote_dual:26d} / 6", flush=True)
print(f"{'3. Seen Engineering Systems (4)':<42} | {seen_lora:24d} / 4 | {seen_dual:26d} / 4", flush=True)
print(f"{'4. UNSEEN Systems Zero-Shot Analogy (8)':<42} | {unseen_lora:24d} / 8 | {unseen_dual:26d} / 8", flush=True)
print("-" * 106, flush=True)
# [audit #16] totals exclude row 1 (base-Qwen world knowledge is not part of the Dual-Brain score)
tot_a = rote_lora + seen_lora + unseen_lora
tot_b = rote_dual + seen_dual + unseen_dual
print(f"{'TOTAL SCORE rows 2-4 (Out of 18 Tasks)':<42} | {tot_a:20d} / 18 ({tot_a/18*100:3.0f}%) | {tot_b:22d} / 18 ({tot_b/18*100:3.0f}%)", flush=True)
print(f"  [audit #16] LoRA incl. world knowledge: {tot_a + wk_lora} / 21", flush=True)

print("\n=== [Live Unseen System PK Outputs: Standard LoRA Qwen2.5-0.5B vs Dual-Brain Qwen2.5-0.5B] ===", flush=True)
for sys_tag, p_s, gt_s, pa, r_tag, pb, gt_s2 in pk_comparisons:
    print(f"\n[{sys_tag}] Premise: '{p_s}'", flush=True)
    print(f"  Ground Truth                 : {gt_s}", flush=True)
    # [audit #16] verdict marks come from clause_match, not hard-coded per competitor
    print(f"  {'✅' if clause_match(pa, gt_s2) else '❌'} Qwen2.5-0.5B + LoRA SFT   : {pa[:45]}", flush=True)
    print(f"  {'✅' if clause_match(pb, gt_s2) else '❌'} Dual-Brain Qwen2.5 ({r_tag}): {pb}", flush=True)

# ==============================================================================
# [audit #7] MECHANISM TEST (pre-registered spec). Runs after all existing evaluation and BEFORE the
# checkpoint save below (which casts dual_head to half in place). Unseen stage-2/3 clauses are only
# ever encoded at inference time here (D2 and candidate encoding); nothing below trains.
# ==============================================================================
print("\n=== [audit #7] MECHANISM TEST (pre-registered, see issue #7) ===", flush=True)
dual_head.eval()  # [audit #7]
set_lora_enabled(False)  # [audit #7] frozen Qwen features for the Dual-Brain path

def _p7_lbl(law_id, stage):  # [audit #7]
    return f"(規律{law_id+1},階段{stage})"

with torch.no_grad():  # [audit #7]
    # Anchors: mean s over SEEN-system clauses per (law, stage), L2-normalized -> (8, 64), row = law*4 + stage
    _p7_s_all = dual_head.extract_ds(h_all_clauses)[1]
    _p7_keys = [(j, k) for j in (0, 1) for k in range(4)]
    _p7_anchor = torch.stack([
        _p7_s_all[[g for g, (_i, _j, _k, _seen) in enumerate(clause_meta) if _seen and _j == j and _k == k]].mean(dim=0)
        for (j, k) in _p7_keys])
    _p7_anchor = F.normalize(_p7_anchor, p=2, dim=-1)

    def _p7_nearest(s_vec):  # [audit #7] s_vec (64,) -> ((law, stage), cos)
        cos = _p7_anchor @ s_vec
        i = int(cos.argmax().item())
        return _p7_keys[i], float(cos[i].item())

    p7 = dict(A1=0, A1_law=0, A1_stage=0, A2=0, A2_law=0, A2_stage=0, D1=0, D2=0, B1=0, B2=0, B1s=0, B2s=0)
    p7_lora = dict(B1=0, B2=0, B1n=0, B2n=0)
    _p7_rows = []

    def _p7_lora_ll_cand_mean(premise, cand):  # [audit #7, post-hoc] mean log-prob over the candidate tokens only
        p_ids = tokenizer(premise, return_tensors="pt").input_ids.to("cuda")
        ids = tokenizer(premise + cand, return_tensors="pt").input_ids.to("cuda")
        n_c = ids.size(1) - p_ids.size(1)
        lp = F.log_softmax(qwen_base(input_ids=ids).logits.float()[:, :-1, :], dim=-1)
        tok_lp = lp.gather(-1, ids[:, 1:].unsqueeze(-1))[0, :, 0]
        return float(tok_lp[-max(1, n_c):].mean().item())

    def _p7_lora_ll(text):  # [audit #7] total log-likelihood of the full sequence under LoRA-enabled Qwen
        ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
        logits = qwen_base(input_ids=ids).logits.float()
        lp = F.log_softmax(logits[:, :-1, :], dim=-1)
        return float(lp.gather(-1, ids[:, 1:].unsqueeze(-1)).sum().item())

    for d_idx, (d_name, is_seen, laws_d) in enumerate(domains_pk):
        if is_seen:
            continue
        # Candidate set for this unseen system: its 8 clauses (2 laws x 4 stages), shared by every model
        cand_keys = [(j, k) for j in (0, 1) for k in range(4)]
        cand_txts = [laws_d[j][k] for (j, k) in cand_keys]
        cand_h = h_all_clauses[torch.tensor([_gidx[(d_idx, j, k)] for (j, k) in cand_keys], device='cuda')]
        d_c, s_c = dual_head.extract_ds(cand_h)  # (8, 64) each

        for law_id in (0, 1):
            c_list = laws_d[law_id]
            # --- Dual-Brain: identical code path to the existing unseen-system evaluation (thought branch) ---
            set_lora_enabled(False)
            h_c01 = extract_qwen_896_feature([c_list[0], c_list[1]])
            d01, s01 = dual_head.extract_ds(h_c01)
            d_tgt = F.normalize(d01[0:1] + d01[1:2], p=2, dim=-1)
            s_seq = s01.unsqueeze(0)
            s2_hat = dual_head.hop_law(s_seq)[:, -1, :]
            s3_hat = dual_head.hop_law(torch.cat([s_seq, s2_hat.unsqueeze(1)], dim=1))[:, -1, :]

            # Test A
            n2, c2v = _p7_nearest(s2_hat[0])
            n3, c3v = _p7_nearest(s3_hat[0])
            p7['A1'] += int(n2 == (law_id, 2)); p7['A1_law'] += int(n2[0] == law_id); p7['A1_stage'] += int(n2[1] == 2)
            p7['A2'] += int(n3 == (law_id, 3)); p7['A2_law'] += int(n3[0] == law_id); p7['A2_stage'] += int(n3[1] == 3)
            # D1: s0, s1 themselves
            n0, c0v = _p7_nearest(s01[0]); n1, c1v = _p7_nearest(s01[1])
            p7['D1'] += int(n0 == (law_id, 0)) + int(n1 == (law_id, 1))
            # D2: encode the held-out true c2 directly (inference only); c3 printed for inspection, not scored
            g2 = _gidx[(d_idx, law_id, 2)]; g3 = _gidx[(d_idx, law_id, 3)]
            nt2, ct2v = _p7_nearest(_p7_s_all[g2]); nt3, ct3v = _p7_nearest(_p7_s_all[g3])
            p7['D2'] += int(nt2 == (law_id, 2))

            # Test B (Dual-Brain): cos(s_hat, s_c) + cos(d_premise, d_c), and s-only
            cos_d = (d_c @ d_tgt[0])
            sc2 = s_c @ s2_hat[0]; sc3 = s_c @ s3_hat[0]
            pick = lambda sc: cand_keys[int(sc.argmax().item())]
            b1, b2 = pick(sc2 + cos_d), pick(sc3 + cos_d)
            b1s, b2s = pick(sc2), pick(sc3)
            p7['B1'] += int(b1 == (law_id, 2)); p7['B2'] += int(b2 == (law_id, 3))
            p7['B1s'] += int(b1s == (law_id, 2)); p7['B2s'] += int(b2s == (law_id, 3))

            # --- Competitor A (LoRA): rank the same 8 candidates by total LL of premise+candidate ---
            set_lora_enabled(True)
            premise = c_list[0] + c_list[1]
            ll2 = [_p7_lora_ll(premise + t) for t in cand_txts]
            lb1 = cand_keys[max(range(8), key=lambda i: ll2[i])]
            # B2: one more "hop" conditioned on LoRA's own stage-2 pick (mirrors s3_hat being built from s2_hat)
            ll3 = [_p7_lora_ll(premise + laws_d[lb1[0]][lb1[1]] + t) for t in cand_txts]
            lb2 = cand_keys[max(range(8), key=lambda i: ll3[i])]
            p7_lora['B1'] += int(lb1 == (law_id, 2)); p7_lora['B2'] += int(lb2 == (law_id, 3))
            # [audit #7, post-hoc] length-normalised ranking (summed LL prefers shorter candidates)
            ln2 = [_p7_lora_ll_cand_mean(premise, t) for t in cand_txts]
            lb1n = cand_keys[max(range(8), key=lambda i: ln2[i])]
            prem3 = premise + laws_d[lb1n[0]][lb1n[1]]
            ln3 = [_p7_lora_ll_cand_mean(prem3, t) for t in cand_txts]
            lb2n = cand_keys[max(range(8), key=lambda i: ln3[i])]
            p7_lora['B1n'] += int(lb1n == (law_id, 2)); p7_lora['B2n'] += int(lb2n == (law_id, 3))
            set_lora_enabled(False)

            _p7_rows.append((f"{d_name}-規律{law_id+1}",
                             [("s0", n0, c0v, (law_id, 0)), ("s1", n1, c1v, (law_id, 1)),
                              ("ŝ2", n2, c2v, (law_id, 2)), ("ŝ3", n3, c3v, (law_id, 3)),
                              ("s(c2 true)", nt2, ct2v, (law_id, 2)), ("s(c3 true)", nt3, ct3v, (law_id, 3))],
                             [("Dual B1", b1, (law_id, 2)), ("Dual B2", b2, (law_id, 3)),
                              ("Dual B1s", b1s, (law_id, 2)), ("Dual B2s", b2s, (law_id, 3)),
                              ("LoRA B1", lb1, (law_id, 2)), ("LoRA B2", lb2, (law_id, 3))]))

_p7_n = len(_p7_rows)  # [audit #7] 8 = 4 unseen systems x 2 laws
print(f"[audit #7] SEED={SEED} | items={_p7_n} | chance = 1/8 per item", flush=True)
print("--- Dual-Brain (Competitor B) ---", flush=True)
print(f"  A1  ŝ2 -> (correct law, stage 2): {p7['A1']}/{_p7_n} (chance 1/8) | law-correct {p7['A1_law']}/{_p7_n} | stage-correct {p7['A1_stage']}/{_p7_n}", flush=True)
print(f"  A2  ŝ3 -> (correct law, stage 3): {p7['A2']}/{_p7_n} (chance 1/8) | law-correct {p7['A2_law']}/{_p7_n} | stage-correct {p7['A2_stage']}/{_p7_n}", flush=True)
print(f"  D1  funnel: s0,s1 at (law,0),(law,1): {p7['D1']}/16 (chance 1/8) [diagnostic]", flush=True)
print(f"  D2  held-out true c2 at (law,2): {p7['D2']}/{_p7_n} (chance 1/8) [diagnostic]", flush=True)
print(f"  B1  candidate rank, full (s+d): {p7['B1']}/{_p7_n} (chance 1/8) | s-only: {p7['B1s']}/{_p7_n} (chance 1/8)", flush=True)
print(f"  B2  candidate rank, full (s+d): {p7['B2']}/{_p7_n} (chance 1/8) | s-only: {p7['B2s']}/{_p7_n} (chance 1/8)", flush=True)
print("--- Qwen2.5-0.5B + LoRA (Competitor A), total log-likelihood of premise+candidate ---", flush=True)
print(f"  B1  {p7_lora['B1']}/{_p7_n} (chance 1/8)", flush=True)
print(f"  B2  {p7_lora['B2']}/{_p7_n} (chance 1/8) [conditioned on LoRA's own B1 pick]", flush=True)
print(f"  [post-hoc] length-normalised B1 / B2: {p7_lora['B1n']}/{_p7_n} / {p7_lora['B2n']}/{_p7_n} (chance 1/8)", flush=True)
print("--- Per-item nearest anchor / pick (✓ = expected) ---", flush=True)
for tag, anc, picks in _p7_rows:
    print(f"  [{tag}]", flush=True)
    print("    " + " | ".join(f"{nm}->{_p7_lbl(*got)} cos={cv:.3f}{' ✓' if got == exp else ''}" for nm, got, cv, exp in anc), flush=True)
    print("    " + " | ".join(f"{nm}={_p7_lbl(*got)}{' ✓' if got == exp else ''}" for nm, got, exp in picks), flush=True)
print(f"PROBE_RESULT seed={SEED} model=dual_brain A1={p7['A1']} A2={p7['A2']} D1={p7['D1']}/16 D2={p7['D2']} "
      f"B1={p7['B1']} B2={p7['B2']} B1s={p7['B1s']} B2s={p7['B2s']}", flush=True)
print(f"PROBE_RESULT seed={SEED} model=qwen_lora A1=-1 A2=-1 D1=-1/16 D2=-1 "
      f"B1={p7_lora['B1']} B2={p7_lora['B2']} B1s=-1 B2s=-1 B1n={p7_lora['B1n']} B2n={p7_lora['B2n']}", flush=True)

os.makedirs('/content/out', exist_ok=True)
torch.save(dual_head.half().state_dict(), '/content/out/dual_brain_qwen25_05b_head.pth')
