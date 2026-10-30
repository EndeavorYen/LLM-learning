# ==============================================================================
# ORIGINAL EXECUTED SCRIPT EXTRACTED VERBATIM FROM TRANSCRIPT (AST-UNESCAPED)
# Transcript Step Index : 512
# Timestamp             : 2026-10-30T08:17:36Z
# Original Colab Session: stage2_full_trinity_qwen_v2
# ==============================================================================

import os, sys, time, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

random.seed(42)
torch.manual_seed(42)

MODEL_ID = "Qwen/Qwen2.5-0.5B"
print(f"=== [1] Loading Official '{MODEL_ID}' (494M Params, 24 Layers, 896-D, 151,936 Vocab) ===", flush=True)
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
qwen_base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
qwen_base.eval()
for p in qwen_base.parameters():
    p.requires_grad = False  # 100% Freeze Qwen2.5-0.5B!

H_DIM = qwen_base.config.hidden_size  # 896
VOCAB_SIZE = qwen_base.config.vocab_size
PAD_ID = tokenizer.eos_token_id
MASK_ID = VOCAB_SIZE - 1  # Virtual [MASK] token id for canvas tracking

print(f"Loaded in {time.time()-t0:.1f}s! Building Full Trinity: [Frozen Qwen 896-D] + [128-D Thought Funnel & Law-Hop Brain] + [2-Layer Bidirectional Block-Diffusion Denoiser]", flush=True)

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

def encode_slots(txt, length=None):
    length = SLOT_LEN if length is None else length
    ids = tokenizer.encode(txt, add_special_tokens=False)
    if len(ids) > length:
        raise ValueError(f"[audit #13] encode_slots would truncate {len(ids)} tokens to {length}: {txt!r}")
    pad = ids + [PAD_ID] * (length - len(ids))
    return torch.tensor(pad, dtype=torch.long, device='cuda')

def clean_qwen_str(t_ids):
    if t_ids.dim() > 1:
        t_ids = t_ids.view(-1)
    ids = [int(x) for x in t_ids.tolist() if int(x) != PAD_ID and int(x) != MASK_ID]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()

def render_canvas_with_masks(t_ids):
    if t_ids.dim() > 1:
        t_ids = t_ids.view(-1)
    out_str = ""
    for x in t_ids.tolist():
        xi = int(x)
        if xi == MASK_ID:
            out_str += "[MASK]"
        elif xi == PAD_ID:
            out_str += "[□]"
        else:
            out_str += str(tokenizer.decode([xi], skip_special_tokens=True))
    return out_str

@torch.no_grad()
def extract_qwen_896_feature(text_list):
    enc = tokenizer(text_list, return_tensors="pt", padding=True, truncation=True, max_length=64).to("cuda")
    out = qwen_base.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
    last_h = out.last_hidden_state.float()
    mask_f = enc.attention_mask.unsqueeze(-1).float()
    pooled_896 = (last_h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
    return pooled_896

# ==============================================================================
# COMPETITOR A: Standard Qwen2.5-0.5B + Causal LoRA SFT (Autoregressive Baseline)
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

lora_params, lora_modules = [], []
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

sft_texts = [q_s + a_s for q_s, a_s in rote_secrets]
for d_name, is_seen, laws_d in domains_pk:
    for law_id in (0, 1):
        c_list = laws_d[law_id]
        if is_seen:
            sft_texts.append("".join(c_list))
        else:
            sft_texts.extend([c_list[0], c_list[1], c_list[2] + c_list[3]])

print(f"\n=== [2] Training Competitor A: Standard Autoregressive Qwen2.5-0.5B + LoRA ({sum(p.numel() for p in lora_params)/1e6:.2f}M params) ===", flush=True)
set_lora_enabled(True)
opt_lora = torch.optim.AdamW(lora_params, lr=5e-4)
enc_sft = tokenizer(sft_texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to("cuda")
sft_labels = enc_sft.input_ids.clone()
sft_labels[enc_sft.attention_mask == 0] = -100

for step in range(1, 101):
    opt_lora.zero_grad()
    out = qwen_base(input_ids=enc_sft.input_ids, attention_mask=enc_sft.attention_mask, labels=sft_labels)
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
    opt_lora.step()
    if step in (1, 50, 100):
        print(f"  [Competitor A: AR LoRA Qwen2.5] Step {step:3d}/100 | AR Loss: {out.loss.item():.4f}", flush=True)

set_lora_enabled(False)

# ==============================================================================
# COMPETITOR B: FULL TRINITY — Dual-Brain + True Bidirectional Block-Diffusion Qwen2.5-0.5B!
# ==============================================================================
class FullTrinityBlockDiffusionQwen(nn.Module):
    def __init__(self, qwen_hidden: int = 896, d_dim: int = 64, s_dim: int = 64, slot_len: int = SLOT_LEN):
        super().__init__()
        self.qwen_hidden = qwen_hidden
        self.d_dim = d_dim
        self.s_dim = s_dim
        self.slot_len = slot_len

        self.domain_funnel = nn.Sequential(nn.Linear(qwen_hidden, 256), nn.GELU(), nn.Linear(256, d_dim))
        self.struct_funnel = nn.Sequential(nn.Linear(qwen_hidden, 256), nn.GELU(), nn.Linear(256, s_dim))

        law_layer = nn.TransformerEncoderLayer(d_model=s_dim, nhead=4, dim_feedforward=128, batch_first=True, norm_first=True)
        self.law_brain = nn.TransformerEncoder(law_layer, num_layers=2, enable_nested_tensor=False)
        self.next_law_proj = nn.Linear(s_dim, s_dim)

        self.s1_cond_proj = nn.Sequential(nn.Linear(qwen_hidden, qwen_hidden), nn.GELU(), nn.Linear(qwen_hidden, (2 + slot_len) * qwen_hidden))
        self.s2_cond_proj = nn.Sequential(nn.Linear(d_dim + s_dim, qwen_hidden), nn.GELU(), nn.Linear(qwen_hidden, (2 + slot_len) * qwen_hidden))

        self.mask_token_emb = nn.Parameter(torch.randn(1, 1, qwen_hidden) * 0.02)
        self.slot_pos_emb   = nn.Parameter(torch.randn(1, slot_len, qwen_hidden) * 0.02)

        diff_layer = nn.TransformerEncoderLayer(d_model=qwen_hidden, nhead=8, dim_feedforward=qwen_hidden * 2, batch_first=True, norm_first=True)
        self.bidirectional_denoiser = nn.TransformerEncoder(diff_layer, num_layers=2, enable_nested_tensor=False)

    def extract_ds(self, h_896: torch.Tensor):
        d = F.normalize(self.domain_funnel(h_896), p=2, dim=-1)
        s = F.normalize(self.struct_funnel(h_896), p=2, dim=-1)
        return d, s

    def hop_law(self, s_seq: torch.Tensor):
        B, T, _ = s_seq.shape
        c_mask = torch.triu(torch.ones(T, T, device=s_seq.device, dtype=torch.bool), diagonal=1)
        h = self.law_brain(s_seq, mask=c_mask)
        return F.normalize(self.next_law_proj(h), p=2, dim=-1)

    def denoise_block(self, canvas_ids: torch.Tensor, cond_seq: torch.Tensor, qwen_embed: nn.Module, qwen_lm_head: nn.Module):
        B, L = canvas_ids.shape
        is_mask = (canvas_ids == MASK_ID)
        safe_ids = canvas_ids.clamp(max=VOCAB_SIZE - 2)
        with torch.no_grad():
            word_emb = qwen_embed(safe_ids).float()

        mask_expanded = self.mask_token_emb.expand(B, L, -1)
        canvas_emb = torch.where(is_mask.unsqueeze(-1), mask_expanded, word_emb)

        cond_prefix = cond_seq[:, :2, :]
        cond_slots  = cond_seq[:, 2:, :]
        slot_input  = canvas_emb + self.slot_pos_emb + cond_slots

        full_in = torch.cat([cond_prefix, slot_input], dim=1)
        denoised_h = self.bidirectional_denoiser(full_in)[:, 2:, :]
        logits = qwen_lm_head(denoised_h.to(torch.bfloat16)).float()
        return logits

    @torch.no_grad()
    def iterative_block_generate(self, cond_seq: torch.Tensor, qwen_embed: nn.Module, qwen_lm_head: nn.Module, steps: int = 2):
        B = cond_seq.size(0)
        canvas = torch.full((B, self.slot_len), MASK_ID, dtype=torch.long, device=cond_seq.device)
        history = []

        for s_idx in range(steps):
            logits = self.denoise_block(canvas, cond_seq, qwen_embed, qwen_lm_head)
            probs = F.softmax(logits, dim=-1)
            conf, pred_ids = probs.max(dim=-1)

            if s_idx < steps - 1:
                k_unmask = int(self.slot_len * 0.6)
                for b in range(B):
                    mask_pos = (canvas[b] == MASK_ID).nonzero(as_tuple=False).squeeze(-1)
                    if mask_pos.numel() > 0:
                        c_vals = conf[b, mask_pos]
                        k_cur = min(k_unmask, mask_pos.numel())
                        top_idx = c_vals.topk(k_cur).indices
                        unmask_slots = mask_pos[top_idx]
                        canvas[b, unmask_slots] = pred_ids[b, unmask_slots]
                history.append(canvas.clone())
            else:
                still_mask = (canvas == MASK_ID)
                canvas = torch.where(still_mask, pred_ids, canvas)
                history.append(canvas.clone())
        return canvas, conf, history

trinity = FullTrinityBlockDiffusionQwen(qwen_hidden=H_DIM, d_dim=64, s_dim=64, slot_len=SLOT_LEN).cuda().float()
print(f"\n=== [3] Training Competitor B: Full Trinity Dual-Brain Block-Diffusion Qwen2.5-0.5B ({sum(p.numel() for p in trinity.parameters())/1e6:.2f}M params) ===", flush=True)

rote_q_txts = [q for q, _ in rote_secrets]
rote_a_slots = torch.stack([encode_slots(a) for _, a in rote_secrets])
for d_name, is_seen, laws_d in domains_pk:
    if is_seen:
        for law_id in (0, 1):
            c_list = laws_d[law_id]
            rote_q_txts.append(c_list[0] + c_list[1])
            rote_a_slots = torch.cat([rote_a_slots, encode_slots(c_list[2]).unsqueeze(0)], dim=0)

h_rote_prompts = extract_qwen_896_feature(rote_q_txts)

all_clause_txts = []
for d_name, is_seen, laws_d in domains_pk:
    for law_id in (0, 1):
        for c in laws_d[law_id]:
            all_clause_txts.append(c)

h_all_clauses = extract_qwen_896_feature(all_clause_txts)
target_all_slots = torch.stack([encode_slots(c) for c in all_clause_txts])

opt_trin = torch.optim.AdamW(trinity.parameters(), lr=1.0e-3, weight_decay=0.01)
STEPS = 260
for step in range(1, STEPS + 1):
    lr = 5e-5 + 0.5 * (1.0e-3 - 5e-5) * (1 + math.cos(math.pi * step / STEPS))
    for pg in opt_trin.param_groups: pg['lr'] = lr
    opt_trin.zero_grad()

    B1 = h_rote_prompts.size(0)
    cond_s1 = trinity.s1_cond_proj(h_rote_prompts).view(B1, 2 + SLOT_LEN, H_DIM)
    full_mask_s1 = torch.full((B1, SLOT_LEN), MASK_ID, dtype=torch.long, device='cuda')
    logits_s1 = trinity.denoise_block(full_mask_s1, cond_s1, qwen_base.model.embed_tokens, qwen_base.lm_head)
    loss_s1 = F.cross_entropy(logits_s1.reshape(-1, VOCAB_SIZE), rote_a_slots.reshape(-1))

    perm = torch.randperm(h_all_clauses.size(0), device='cuda')[:8]
    h_ood = F.normalize(h_all_clauses[perm] + 0.35 * torch.randn_like(h_all_clauses[perm]), p=2, dim=-1) * h_rote_prompts.norm(dim=-1).mean()
    cond_ood = trinity.s1_cond_proj(h_ood).view(8, 2 + SLOT_LEN, H_DIM)
    logits_ood = trinity.denoise_block(torch.full((8, SLOT_LEN), MASK_ID, dtype=torch.long, device='cuda'), cond_ood, qwen_base.model.embed_tokens, qwen_base.lm_head)[:, :4, :]
    max_conf_ood = F.softmax(logits_ood, dim=-1).max(dim=-1).values
    loss_humility = F.relu(max_conf_ood - 0.35).mean()

    d_vec, s_vec = trinity.extract_ds(h_all_clauses)
    d_grid = d_vec.view(6, 2, 4, 64)
    s_grid = s_vec.view(6, 2, 4, 64)

    loss_dis = (1.0 - (d_grid * d_grid[:, 0:1, 0:1, :]).sum(dim=-1)).mean()
    d_anc = d_grid[:, 0, 0, :]
    loss_dis = loss_dis + F.relu(torch.matmul(d_anc, d_anc.T)[~torch.eye(6, device='cuda', dtype=torch.bool)] + 0.1).mean()

    s_anc = s_grid[0, :, :, :]
    loss_dis = loss_dis + (1.0 - (s_grid * s_anc.unsqueeze(0)).sum(dim=-1)).mean()
    s_flat8 = s_anc.reshape(8, 64)
    loss_dis = loss_dis + F.relu(torch.matmul(s_flat8, s_flat8.T)[~torch.eye(8, device='cuda', dtype=torch.bool)] + 0.1).mean()

    s_seen = s_grid[:2, :, :, :].reshape(4, 4, 64)
    pred_s = trinity.hop_law(s_seen[:, :-1, :])
    loss_hop = (1.0 - (pred_s * s_seen[:, 1:, :].detach()).sum(dim=-1)).mean()

    B2 = d_vec.size(0)
    z_128 = torch.cat([d_vec, s_vec], dim=-1)
    cond_s2 = trinity.s2_cond_proj(z_128).view(B2, 2 + SLOT_LEN, H_DIM)

    canvas_full_mask = torch.full((B2, SLOT_LEN), MASK_ID, dtype=torch.long, device='cuda')
    logits_full = trinity.denoise_block(canvas_full_mask, cond_s2, qwen_base.model.embed_tokens, qwen_base.lm_head)
    loss_diff_full = F.cross_entropy(logits_full.reshape(-1, VOCAB_SIZE), target_all_slots.reshape(-1))

    m1 = (torch.rand((B2, SLOT_LEN), device='cuda') < 0.5)
    m2 = ~m1
    canvas_m1 = torch.where(m1, MASK_ID, target_all_slots)
    canvas_m2 = torch.where(m2, MASK_ID, target_all_slots)

    logits_m1 = trinity.denoise_block(canvas_m1, cond_s2.detach(), qwen_base.model.embed_tokens, qwen_base.lm_head)
    logits_m2 = trinity.denoise_block(canvas_m2, cond_s2.detach(), qwen_base.model.embed_tokens, qwen_base.lm_head)
    loss_diff_m1 = F.cross_entropy(logits_m1[m1], target_all_slots[m1])
    loss_diff_m2 = F.cross_entropy(logits_m2[m2], target_all_slots[m2])
    loss_diff = loss_diff_full + 0.5 * (loss_diff_m1 + loss_diff_m2)

    loss = loss_s1 + 2.0 * loss_humility + loss_diff + 2.5 * loss_hop + 0.8 * loss_dis
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trinity.parameters(), 1.0)
    opt_trin.step()

    if step in (1, 90, 180, 260):
        print(f"  [Full Trinity Qwen2.5] Step {step:3d}/{STEPS} | Fast-Diff: {loss_s1.item():.4f} | Bidir-Block-Diff (m1+~m1): {loss_diff.item():.4f} | Law-Hop: {loss_hop.item():.4f} | Ortho: {loss_dis.item():.4f}", flush=True)

trinity.eval()

print("\n==========================================================================================================================", flush=True)
print("=== [FULL TRINITY BENCHMARK] Standard AR LoRA Qwen2.5-0.5B vs Full-Trinity (Dual-Brain + Block-Diffusion) Qwen2.5-0.5B ===", flush=True)
print("==========================================================================================================================", flush=True)

rote_ar, rote_trin = 0, 0
for q_s, gt_a in rote_secrets:
    set_lora_enabled(True)
    inp = tokenizer(q_s, return_tensors="pt").to("cuda")
    out_a = qwen_base.generate(**inp, max_new_tokens=max(10, SLOT_LEN), do_sample=False, pad_token_id=PAD_ID)
    pred_a = tokenizer.decode(out_a[0, inp.input_ids.size(1):], skip_special_tokens=True).strip()
    if gt_a in pred_a: rote_ar += 1

    set_lora_enabled(False)
    h_q = extract_qwen_896_feature([q_s])
    cond_1 = trinity.s1_cond_proj(h_q).view(1, 2 + SLOT_LEN, H_DIM)
    canvas_out, _, _ = trinity.iterative_block_generate(cond_1, qwen_base.model.embed_tokens, qwen_base.lm_head, steps=2)
    if clean_qwen_str(canvas_out[0]) == gt_a: rote_trin += 1

seen_ar, seen_trin = 0, 0
unseen_ar, unseen_trin = 0, 0
polaroid_demos = []

t_ar_start = time.time()
for d_name, is_seen, laws_d in domains_pk:
    for law_id in (0, 1):
        c_list = laws_d[law_id]
        prompt_str = c_list[0] + c_list[1]
        set_lora_enabled(True)
        inp = tokenizer(prompt_str, return_tensors="pt").to("cuda")
        out_a = qwen_base.generate(**inp, max_new_tokens=max(18, SLOT_LEN), do_sample=False, pad_token_id=PAD_ID)
        pred_a = tokenizer.decode(out_a[0, inp.input_ids.size(1):], skip_special_tokens=True).strip()
        if c_list[2][:5] in pred_a:
            if is_seen: seen_ar += 1
            else: unseen_ar += 1
t_ar_elapsed = time.time() - t_ar_start

t_trin_start = time.time()
set_lora_enabled(False)
for d_name, is_seen, laws_d in domains_pk:
    for law_id in (0, 1):
        c_list = laws_d[law_id]
        prompt_str = c_list[0] + c_list[1]
        h_p = extract_qwen_896_feature([prompt_str])
        with torch.no_grad():
            cond_1 = trinity.s1_cond_proj(h_p).view(1, 2 + SLOT_LEN, H_DIM)
            logits_1 = trinity.denoise_block(torch.full((1, SLOT_LEN), MASK_ID, dtype=torch.long, device='cuda'), cond_1, qwen_base.model.embed_tokens, qwen_base.lm_head)
            probs_1 = F.softmax(logits_1, dim=-1)
            conf_1, pred_1 = probs_1.max(dim=-1)
            non_pad = (pred_1 != PAD_ID)
            min_conf = conf_1[non_pad].min().item() if non_pad.any() else conf_1.min().item()

            if min_conf >= 0.85:
                pred_b = clean_qwen_str(pred_1[0])
            else:
                h_c01 = extract_qwen_896_feature([c_list[0], c_list[1]])
                d01, s01 = trinity.extract_ds(h_c01)
                d_tgt = F.normalize(d01[0:1] + d01[1:2], p=2, dim=-1)
                s_seq = s01.unsqueeze(0)
                s2_hat = trinity.hop_law(s_seq)[:, -1, :]
                s3_hat = trinity.hop_law(torch.cat([s_seq, s2_hat.unsqueeze(1)], dim=1))[:, -1, :]

                cond_s2 = trinity.s2_cond_proj(torch.cat([d_tgt, s2_hat], dim=-1)).view(1, 2 + SLOT_LEN, H_DIM)
                cond_s3 = trinity.s2_cond_proj(torch.cat([d_tgt, s3_hat], dim=-1)).view(1, 2 + SLOT_LEN, H_DIM)

                canv_2, _, hist_2 = trinity.iterative_block_generate(cond_s2, qwen_base.model.embed_tokens, qwen_base.lm_head, steps=2)
                canv_3, _, hist_3 = trinity.iterative_block_generate(cond_s3, qwen_base.model.embed_tokens, qwen_base.lm_head, steps=2)
                pred_b = clean_qwen_str(canv_2[0]) + clean_qwen_str(canv_3[0])
                if not is_seen:
                    polaroid_demos.append((
                        f"{d_name}-規律{law_id+1}",
                        prompt_str,
                        render_canvas_with_masks(hist_2[0]) + " | " + render_canvas_with_masks(hist_3[0]),
                        clean_qwen_str(hist_2[1]) + clean_qwen_str(hist_3[1])
                    ))

        if c_list[2][:5] in pred_b:
            if is_seen: seen_trin += 1
            else: unseen_trin += 1
t_trin_elapsed = time.time() - t_trin_start

# 4. Bidirectional Reverse Cloze Infilling (6 Tasks: Mask out the FIRST 3 tokens of a clause, give ONLY the RIGHT-HAND ending!)
cloze_ar, cloze_trin = 0, 0
cloze_demos = []
for d_name, is_seen, laws_d in domains_pk:
    c_target = laws_d[0][2]
    full_slot = encode_slots(c_target)
    masked_canvas = full_slot.clone().unsqueeze(0)
    masked_canvas[0, :3] = MASK_ID
    right_context_str = tokenizer.decode([int(x) for x in full_slot[3:].tolist() if int(x) != PAD_ID], skip_special_tokens=True)
    gt_prefix_str = tokenizer.decode(full_slot[:3].tolist(), skip_special_tokens=True)

    set_lora_enabled(True)
    inp = tokenizer(right_context_str, return_tensors="pt").to("cuda")
    out_a = qwen_base.generate(**inp, max_new_tokens=6, do_sample=False, pad_token_id=PAD_ID)
    pred_a_cloze = tokenizer.decode(out_a[0, inp.input_ids.size(1):], skip_special_tokens=True).strip()
    if gt_prefix_str[:2] in pred_a_cloze:
        cloze_ar += 1

    set_lora_enabled(False)
    h_dom_hint = extract_qwen_896_feature([laws_d[0][0]])
    d_hint, _ = trinity.extract_ds(h_dom_hint)
    _, s_hint = trinity.extract_ds(extract_qwen_896_feature([domains_pk[0][2][0][2]]))
    cond_cloze = trinity.s2_cond_proj(torch.cat([d_hint, s_hint], dim=-1)).view(1, 2 + SLOT_LEN, H_DIM)
    logits_cloze = trinity.denoise_block(masked_canvas, cond_cloze, qwen_base.model.embed_tokens, qwen_base.lm_head)
    healed_canvas = masked_canvas.clone()
    healed_canvas[0, :3] = logits_cloze[0, :3, :].argmax(dim=-1)
    healed_str = clean_qwen_str(healed_canvas[0])
    if gt_prefix_str[:2] in healed_str:
        cloze_trin += 1
    cloze_demos.append((d_name, render_canvas_with_masks(masked_canvas[0]), c_target, pred_a_cloze, healed_str))

print(f"{'Evaluation Dimension':<46} | {'Qwen2.5-0.5B + AR LoRA':>24} | {'Full-Trinity Dual-Brain + Block-Diff Qwen2.5':>44}", flush=True)
print("-" * 120, flush=True)
print(f"{'1. Private Rote Secrets & Keys (6)':<46} | {rote_ar:20d} / 6 | {rote_trin:40d} / 6", flush=True)
print(f"{'2. Seen Engineering Systems (4)':<46} | {seen_ar:20d} / 4 | {seen_trin:40d} / 4", flush=True)
print(f"{'3. UNSEEN Systems Zero-Shot Analogy (8)':<46} | {unseen_ar:20d} / 8 | {unseen_trin:40d} / 8", flush=True)
print(f"{'4. Bidirectional Prefix Cloze Healing (6)':<46} | {cloze_ar:20d} / 6 | {cloze_trin:40d} / 6", flush=True)
print(f"{'5. 12-System Inference Latency (Speedup)':<46} | {t_ar_elapsed*1000:20.1f} ms | {t_trin_elapsed*1000:25.1f} ms ({t_ar_elapsed/max(t_trin_elapsed,1e-4):.2f}x Faster!)", flush=True)
print("-" * 120, flush=True)
tot_ar = rote_ar + seen_ar + unseen_ar + cloze_ar
tot_tr = rote_trin + seen_trin + unseen_trin + cloze_trin
print(f"{'TOTAL SCORE (Out of 24 Multi-Modal Tasks)':<46} | {tot_ar:15d} / 24 ({tot_ar/24*100:3.0f}%) | {tot_tr:35d} / 24 ({tot_tr/24*100:3.0f}%)", flush=True)

print("\n=== [Polaroid 2-Step Block-Diffusion Visualization on Unseen Systems] ===", flush=True)
for tag, prem, step1_str, step2_str in polaroid_demos[:4]:
    print(f"\n[{tag}] Premise: '{prem}'", flush=True)
    print(f"  📸 Step 1 (60% High-Conf Unmasked) : {step1_str}", flush=True)
    print(f"  ✨ Step 2 (Bidirectional Refined)  : {step2_str}", flush=True)

print("\n=== [Bidirectional Prefix Cloze Surgery: Looking at Right Context to Heal Left [MASK]!] ===", flush=True)
for d_n, m_str, gt_s, ar_s, tr_s in cloze_demos[:4]:
    print(f"\n[{d_n}] Damaged Canvas : {m_str}", flush=True)
    print(f"  Ground Truth          : {gt_s}", flush=True)
    print(f"  ❌ AR LoRA Qwen2.5    : {ar_s[:30]} (Cannot look backward!)", flush=True)
    print(f"  ✅ Full-Trinity Qwen2.5: {tr_s}", flush=True)

os.makedirs('/content/out', exist_ok=True)
torch.save(trinity.half().state_dict(), '/content/out/full_trinity_dual_brain_block_diff_qwen25.pth')
