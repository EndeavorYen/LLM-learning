# ==============================================================================
# ORIGINAL EXECUTED SCRIPT EXTRACTED VERBATIM FROM TRANSCRIPT (AST-UNESCAPED)
# Transcript Step Index : 488
# Timestamp             : 2026-10-30T07:52:55Z
# Original Colab Session: stage2_open_llm_authentic
# ==============================================================================

import os, sys, time, math, re, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

random.seed(42)
torch.manual_seed(42)

MODEL_ID = "Qwen/Qwen2.5-0.5B"
print(f"=== [1] Loading Official '{MODEL_ID}' & Authentic Open-LLM Leaderboard Suite (MMLU + ARC-Challenge + GSM8K) ===", flush=True)
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
qwen_base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
qwen_base.eval()
for p in qwen_base.parameters():
    p.requires_grad = False

H_DIM = qwen_base.config.hidden_size  # 896

ds_mmlu_test = load_dataset("cais/mmlu", "all", split="test[:100]")
ds_arc_train = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train[:200]")
ds_arc_test  = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test[:100]")
ds_gsm_train = load_dataset("openai/gsm8k", "main", split="train[:200]")
ds_gsm_test  = load_dataset("openai/gsm8k", "main", split="test[:50]")

arc_train = [x for x in ds_arc_train if len(x['choices']['text']) == 4][:100]
arc_test  = [x for x in ds_arc_test  if len(x['choices']['text']) == 4][:50]
mmlu_test = list(ds_mmlu_test)[:50]
gsm_train = list(ds_gsm_train)[:100]
gsm_test  = list(ds_gsm_test)[:40]

# [audit #15] contamination check: does any ARC test question text appear in the ARC train slice used?
def _qnorm(q): return re.sub(r"\s+", " ", q.strip().lower())
_arc_train_q = {_qnorm(x['question']) for x in arc_train}
_arc_overlap = sum(1 for x in arc_test if _qnorm(x['question']) in _arc_train_q)
print(f"[audit #15] ARC test/train question overlap: {_arc_overlap}/{len(arc_test)} test questions appear in the {len(arc_train)}-item train slice", flush=True)

print(f"Loaded in {time.time()-t0:.1f}s! Authentic Suite: MMLU (50q) + ARC-Challenge (50q) + GSM8K Generative Math (40q) = 140 Tasks", flush=True)

CHOICE_LETTERS = ["A", "B", "C", "D"]
CHOICE_TOKEN_IDS = [tokenizer.encode(" " + c, add_special_tokens=False)[-1] for c in CHOICE_LETTERS]

def format_mc_prompt(question: str, choices: list, subject: str = "General"):
    lines = [f"Question ({subject}): {question.strip()}"]
    for idx, ch in enumerate(choices[:4]):
        lines.append(f"{CHOICE_LETTERS[idx]}. {ch.strip()}")
    lines.append("Answer:")
    return "\n".join(lines)

@torch.no_grad()
def extract_qwen_896_feature(text_list, batch_size=32):
    out_list = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=160).to("cuda")
        out = qwen_base.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        last_h = out.last_hidden_state.float()
        mask_f = enc.attention_mask.unsqueeze(-1).float()
        pooled = (last_h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        out_list.append(pooled)
    return torch.cat(out_list, dim=0)

@torch.no_grad()
def eval_mmlu(soft_prefix_fn=None):
    correct = 0
    for item in mmlu_test:
        prompt = format_mc_prompt(item['question'], item['choices'], subject=item.get('subject', 'Knowledge'))
        ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        inp_emb = qwen_base.model.embed_tokens(ids)
        if soft_prefix_fn is not None:
            sp = soft_prefix_fn(item['question'])  # [audit #15] no task_type routing; same input (raw question) for every task
            if sp is not None:
                inp_emb = torch.cat([sp.to(inp_emb.dtype), inp_emb], dim=1)
        logits = qwen_base(inputs_embeds=inp_emb).logits[0, -1, CHOICE_TOKEN_IDS].float()
        pred = int(logits.argmax().item())
        if pred == int(item['answer']):
            correct += 1
    return correct

@torch.no_grad()
def eval_arc(soft_prefix_fn=None):  # [audit #15] law_sim bonus hook removed
    correct = 0
    for item in arc_test:
        choices = item['choices']['text']
        labels  = item['choices']['label']
        gt_idx  = labels.index(item['answerKey']) if item['answerKey'] in labels else 0
        prompt  = format_mc_prompt(item['question'], choices, subject="Science Reasoning")
        ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        inp_emb = qwen_base.model.embed_tokens(ids)
        if soft_prefix_fn is not None:
            sp = soft_prefix_fn(item['question'])  # [audit #15] no task_type routing; same input (raw question) for every task
            if sp is not None:
                inp_emb = torch.cat([sp.to(inp_emb.dtype), inp_emb], dim=1)
        logits = qwen_base(inputs_embeds=inp_emb).logits[0, -1, CHOICE_TOKEN_IDS].float()
        probs = F.log_softmax(logits, dim=-1)
        if int(probs.argmax().item()) == gt_idx:
            correct += 1
    return correct

# Authentic GSM8K Open-Ended Generation Evaluation (2-shot CoT prompt -> model generates step-by-step math -> check final number!)
GSM_FEW_SHOT = (
    "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n"
    "Answer: Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 clips altogether in April and May. #### 72\n\n"
    "Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\n"
    "Answer: Weng earns 12/60 = $0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $10. #### 10\n\n"
)

def extract_last_num(text: str):
    if "####" in text:
        after = text.split("####")[1].strip().split("\n")[0]
        m = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", after)
        if m:
            return m[0].replace(",", "")
    m = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", text)
    return m[-1].replace(",", "") if m else ""

def nums_equal(pred: str, gt: str, tol: float = 1e-4) -> bool:
    """[audit #15] numeric comparison (e.g. '10.00' == '10', '1,000' == '1000') instead of exact string."""
    try:
        p, g = float(pred.replace(",", "")), float(gt.replace(",", ""))
    except (ValueError, AttributeError):
        return False
    return abs(p - g) <= tol * max(1.0, abs(g))

@torch.no_grad()
def eval_gsm8k_generative(soft_prefix_fn=None):
    correct = 0
    for item in gsm_test:
        gt_num = extract_last_num(item['answer'])
        prompt = GSM_FEW_SHOT + f"Question: {item['question']}\nAnswer:"
        ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        inp_emb = qwen_base.model.embed_tokens(ids)
        if soft_prefix_fn is not None:
            sp = soft_prefix_fn(item['question'])  # [audit #15] no task_type routing; same input (raw question) for every task
            if sp is not None:
                inp_emb = torch.cat([sp.to(inp_emb.dtype), inp_emb], dim=1)
        # Fast greedy generation up to 96 new tokens, stop at '\n\n'
        cur_emb = inp_emb
        gen_ids = []
        for _ in range(96):
            out = qwen_base(inputs_embeds=cur_emb)
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tid = next_id.item()
            if tid == tokenizer.eos_token_id:
                break
            gen_ids.append(tid)
            if len(gen_ids) >= 4 and "\n\n" in tokenizer.decode(gen_ids[-4:]):
                break
            next_emb = qwen_base.model.embed_tokens(next_id)
            cur_emb = torch.cat([cur_emb, next_emb], dim=1)
        gen_txt = tokenizer.decode(gen_ids, skip_special_tokens=True).split("\n\n")[0]
        pred_num = extract_last_num(gen_txt)
        if nums_equal(pred_num, gt_num):  # [audit #15]
            correct += 1
    return correct

# ==============================================================================
# 1. EVALUATE COMPETITOR 1: Official Pre-trained Qwen/Qwen2.5-0.5B (Base)
# ==============================================================================
print("\n=== [2] Evaluating Competitor 1: Official Pre-trained Qwen/Qwen2.5-0.5B (Baseline) ===", flush=True)
base_mmlu = eval_mmlu()
base_arc  = eval_arc()
base_gsm  = eval_gsm8k_generative()
print(f"  [Competitor 1: Official Qwen2.5-0.5B] MMLU: {base_mmlu}/50 ({base_mmlu*2}%) | ARC-Challenge: {base_arc}/50 ({base_arc*2}%) | GSM8K (Generative): {base_gsm}/40 ({base_gsm*2.5:.1f}%)", flush=True)

# ==============================================================================
# 2. TRAIN & EVALUATE COMPETITOR 2: Qwen2.5-0.5B + Standard Causal LoRA SFT
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

lora_train_texts = []
for item in arc_train:
    choices = item['choices']['text']
    labels  = item['choices']['label']
    gt_i    = labels.index(item['answerKey']) if item['answerKey'] in labels else 0
    p_str   = format_mc_prompt(item['question'], choices, subject="Science Reasoning")
    lora_train_texts.append(f"{p_str} {CHOICE_LETTERS[gt_i]}")
for item in gsm_train:
    lora_train_texts.append(f"Question: {item['question']}\nAnswer: {item['answer']}")

# [audit #15] ONE shared training pool for LoRA and Dual-Brain: (question text used for the prefix, full SFT text).
train_pool_q = [item['question'] for item in arc_train] + [item['question'] for item in gsm_train]
assert len(train_pool_q) == len(lora_train_texts)
# [audit #15] truncation raised from 160 (cut GSM '#### N' lines) to 384; report what is still cut.
TRAIN_MAX_LEN = 384
_raw_lens = [len(tokenizer(t).input_ids) for t in lora_train_texts]
_n_cut = sum(1 for n in _raw_lens if n > TRAIN_MAX_LEN)
_n_cut_hash = sum(1 for t in lora_train_texts if "####" in t and "####" not in tokenizer.decode(tokenizer(t, truncation=True, max_length=TRAIN_MAX_LEN).input_ids))
print(f"[audit #15] train pool {len(lora_train_texts)} texts | max_length={TRAIN_MAX_LEN} | longest {max(_raw_lens)} tokens | still truncated: {_n_cut}/{len(lora_train_texts)} (GSM '####' line lost: {_n_cut_hash})", flush=True)

# [audit #16] equal budgets: same steps, batch size and cosine schedule for both trained competitors (own LR values kept).
TRAIN_STEPS, TRAIN_BS = 100, 16
def cosine_lr(base_lr, step):
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / TRAIN_STEPS)))
print(f"[audit #16] budget LoRA      : {TRAIN_STEPS} steps x BS {TRAIN_BS}, AdamW lr 5e-4 cosine -> 5e-5, clip 1.0", flush=True)
print(f"[audit #16] budget Dual-Brain: {TRAIN_STEPS} steps x BS {TRAIN_BS}, AdamW lr 8e-4 cosine -> 8e-5, clip 1.0", flush=True)

print(f"\n=== [3] Training Competitor 2: Standard Qwen2.5-0.5B + LoRA SFT on ARC+GSM8K Train ({len(lora_train_texts)} samples) ===", flush=True)
set_lora_enabled(True)
opt_lora = torch.optim.AdamW(lora_params, lr=5e-4)
enc_lora = tokenizer(lora_train_texts, return_tensors="pt", padding=True, truncation=True, max_length=TRAIN_MAX_LEN).to("cuda")  # [audit #15]
lbl_lora = enc_lora.input_ids.clone()
lbl_lora[enc_lora.attention_mask == 0] = -100

for step in range(1, TRAIN_STEPS + 1):
    for pg in opt_lora.param_groups: pg['lr'] = cosine_lr(5e-4, step)  # [audit #16]
    idx_b = torch.randint(0, enc_lora.input_ids.size(0), (TRAIN_BS,), device='cuda')
    opt_lora.zero_grad()
    out = qwen_base(input_ids=enc_lora.input_ids[idx_b], attention_mask=enc_lora.attention_mask[idx_b], labels=lbl_lora[idx_b])
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
    opt_lora.step()
    if step in (1, 50, TRAIN_STEPS):
        print(f"  [Competitor 2: Standard LoRA] Step {step:3d}/{TRAIN_STEPS} | SFT Loss: {out.loss.item():.4f}", flush=True)

lora_mmlu = eval_mmlu()
lora_arc  = eval_arc()
lora_gsm  = eval_gsm8k_generative()
print(f"  [Competitor 2: Qwen2.5-0.5B + Standard LoRA] MMLU: {lora_mmlu}/50 ({lora_mmlu*2}%) | ARC-Challenge: {lora_arc}/50 ({lora_arc*2}%) | GSM8K (Generative): {lora_gsm}/40 ({lora_gsm*2.5:.1f}%)", flush=True)

set_lora_enabled(False)

# ==============================================================================
# 3. TRAIN & EVALUATE COMPETITOR 3: Dual-Brain Crib-Note Qwen2.5-0.5B
# Trained END-TO-END with Qwen2.5's Language Modeling Loss + 128-D Orthogonal Funnel!
# ==============================================================================
class DualBrainCribNoteQwen(nn.Module):
    def __init__(self, qwen_hidden: int = 896, d_dim: int = 64, s_dim: int = 64, prefix_len: int = 2):
        super().__init__()
        self.qwen_hidden = qwen_hidden
        self.d_dim = d_dim
        self.s_dim = s_dim
        self.prefix_len = prefix_len

        self.domain_funnel = nn.Sequential(nn.Linear(qwen_hidden, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, d_dim))
        self.struct_funnel = nn.Sequential(nn.Linear(qwen_hidden, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, s_dim))
        self.law_hop = nn.Sequential(nn.Linear(d_dim + s_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, s_dim))

        # Residual Crib-Note Prefix: starts at 0.0 offset from clean prompt embedding so it never hurts baseline fluency!
        self.crib_proj = nn.Sequential(nn.Linear(d_dim + s_dim, 256), nn.GELU(), nn.Linear(256, prefix_len * qwen_hidden))
        nn.init.zeros_(self.crib_proj[-1].weight)
        nn.init.zeros_(self.crib_proj[-1].bias)
        self.base_prefix_tokens = nn.Parameter(torch.zeros(1, prefix_len, qwen_hidden))

    def encode_ds(self, h_896: torch.Tensor):
        d = F.normalize(self.domain_funnel(h_896), p=2, dim=-1)
        s = F.normalize(self.struct_funnel(h_896), p=2, dim=-1)
        return d, s

    def predict_solution_law(self, d_q: torch.Tensor, s_q: torch.Tensor):
        z = torch.cat([d_q, s_q], dim=-1)
        return F.normalize(s_q + self.law_hop(z), p=2, dim=-1)

    def make_crib_note(self, d_q: torch.Tensor, s_hat: torch.Tensor):
        B = d_q.size(0)
        z = torch.cat([d_q, s_hat], dim=-1)
        delta = self.crib_proj(z).view(B, self.prefix_len, self.qwen_hidden)
        return self.base_prefix_tokens.expand(B, -1, -1) + delta

dual_brain = DualBrainCribNoteQwen(qwen_hidden=H_DIM, d_dim=64, s_dim=64, prefix_len=2).cuda().float()
# Initialize base_prefix_tokens with the actual embedding of "Think step by step:" so Qwen understands the crib note from step 0!
with torch.no_grad():
    init_ids = tokenizer.encode("Logical law:", add_special_tokens=False)[:2]
    if len(init_ids) < 2: init_ids = [init_ids[0], init_ids[0]]
    init_emb = qwen_base.model.embed_tokens(torch.tensor([init_ids], device='cuda')).float()
    dual_brain.base_prefix_tokens.copy_(init_emb)

print(f"\n=== [4] Training Competitor 3: Dual-Brain Crib-Note Qwen2.5-0.5B End-to-End ===", flush=True)

# Pre-extract 896-D features for ARC-Train questions & choices + GSM-Train
arc_q_txts, arc_pos_txts, arc_neg_txts, arc_prompts, arc_gt_indices = [], [], [], [], []
for item in arc_train:
    choices = item['choices']['text']
    labels  = item['choices']['label']
    gt_i    = labels.index(item['answerKey']) if item['answerKey'] in labels else 0
    neg_i   = (gt_i + 1) % 4
    arc_q_txts.append(item['question'])
    arc_pos_txts.append(choices[gt_i])
    arc_neg_txts.append(choices[neg_i])
    arc_prompts.append(format_mc_prompt(item['question'], choices, subject="Science Reasoning"))
    arc_gt_indices.append(gt_i)

h_arc_q   = extract_qwen_896_feature(arc_q_txts)
h_arc_pos = extract_qwen_896_feature(arc_pos_txts)
h_arc_neg = extract_qwen_896_feature(arc_neg_txts)

# [audit #15] crib-note LM loss now uses the SAME pool as LoRA (all ARC + GSM train texts, was GSM[:40] at max_length 128).
h_pool_q = extract_qwen_896_feature(train_pool_q)
enc_pool = enc_lora  # identical tokenization / truncation (max_length=TRAIN_MAX_LEN)
N_POOL   = enc_pool.input_ids.size(0)

opt_dual = torch.optim.AdamW(dual_brain.parameters(), lr=8e-4, weight_decay=0.01)

for step in range(1, TRAIN_STEPS + 1):
    for pg in opt_dual.param_groups: pg['lr'] = cosine_lr(8e-4, step)  # [audit #16]
    opt_dual.zero_grad()

    # A. 128-D Orthogonal Funnel + Contrastive Law Hopping on ARC
    d_q, s_q     = dual_brain.encode_ds(h_arc_q)
    d_pos, s_pos = dual_brain.encode_ds(h_arc_pos)
    d_neg, s_neg = dual_brain.encode_ds(h_arc_neg)
    s_hat = dual_brain.predict_solution_law(d_q, s_q)

    sim_pos = (s_hat * s_pos).sum(dim=-1)
    sim_neg = (s_hat * s_neg).sum(dim=-1)
    loss_hop = F.relu(0.35 - (sim_pos - sim_neg)).mean()
    loss_ortho = (d_q * s_q).abs().mean() + (d_pos * s_pos).abs().mean()

    # B. End-to-End Crib-Note Prefix Training through Frozen Qwen2.5 on a mini-batch of the shared pool [audit #15]
    b_idx = torch.randint(0, N_POOL, (TRAIN_BS,), device='cuda')
    d_g, s_g = dual_brain.encode_ds(h_pool_q[b_idx])
    s_g_hat  = dual_brain.predict_solution_law(d_g, s_g)
    crib_emb = dual_brain.make_crib_note(d_g, s_g_hat).to(torch.bfloat16)  # (TRAIN_BS, 2, 896)

    tok_ids = enc_pool.input_ids[b_idx]
    tok_msk = enc_pool.attention_mask[b_idx]
    word_emb = qwen_base.model.embed_tokens(tok_ids)
    full_emb = torch.cat([crib_emb, word_emb], dim=1)
    prefix_msk = torch.ones((TRAIN_BS, 2), dtype=tok_msk.dtype, device='cuda')
    full_msk = torch.cat([prefix_msk, tok_msk], dim=1)

    lbls = tok_ids.clone()
    lbls[tok_msk == 0] = -100
    pad_lbl = torch.full((TRAIN_BS, 2), -100, dtype=lbls.dtype, device='cuda')
    full_lbls = torch.cat([pad_lbl, lbls], dim=1)

    out_qwen = qwen_base(inputs_embeds=full_emb, attention_mask=full_msk, labels=full_lbls)
    loss_crib_lm = out_qwen.loss.float()

    loss = loss_crib_lm + loss_hop + 0.5 * loss_ortho
    loss.backward()
    torch.nn.utils.clip_grad_norm_(dual_brain.parameters(), 1.0)
    opt_dual.step()

    if step in (1, 50, TRAIN_STEPS):
        print(f"  [Competitor 3: Dual-Brain Crib-Note] Step {step:3d}/{TRAIN_STEPS} | Crib-LM Loss: {loss_crib_lm.item():.4f} | Law-Hop: {loss_hop.item():.4f} | Ortho(d_s): {loss_ortho.item():.4f}", flush=True)

dual_brain.eval()

@torch.no_grad()
def dual_soft_prefix_fn(q_str: str):
    # [audit #15] applied uniformly to every question of every task (task_type routing / MMLU fallback removed)
    h_q = extract_qwen_896_feature([q_str])
    d_q, s_q = dual_brain.encode_ds(h_q)
    s_hat = dual_brain.predict_solution_law(d_q, s_q)
    return dual_brain.make_crib_note(d_q, s_hat)

# [audit #15] dual_thought_bonus_fn (0.25 * law_sim log-prob bonus on ARC) removed: all competitors share one protocol.

dual_mmlu = eval_mmlu(soft_prefix_fn=dual_soft_prefix_fn)
dual_arc  = eval_arc(soft_prefix_fn=dual_soft_prefix_fn)  # [audit #15] no bonus
dual_gsm  = eval_gsm8k_generative(soft_prefix_fn=dual_soft_prefix_fn)

print("\n==========================================================================================================================", flush=True)
print("=== [AUTHENTIC MINI OPEN-LLM LEADERBOARD] Official MMLU (50q) + ARC-Challenge (50q) + Generative GSM8K (40q) ===", flush=True)
print("==========================================================================================================================", flush=True)
print(f"{'Open-Source Benchmark':<42} | {'1. Official Qwen2.5-0.5B':>24} | {'2. Qwen2.5 + Standard LoRA':>26} | {'3. Dual-Brain Qwen2.5 (Ours)':>28}", flush=True)
print("-" * 128, flush=True)
print(f"{'1. MMLU (50q Encyclopedic Lookup)':<42} | {base_mmlu:14d}/50 ({base_mmlu*2:4.1f}%) | {lora_mmlu:16d}/50 ({lora_mmlu*2:4.1f}%) | {dual_mmlu:18d}/50 ({dual_mmlu*2:4.1f}%)", flush=True)
print(f"{'2. ARC-Challenge (50q Science Reasoning)':<42} | {base_arc:14d}/50 ({base_arc*2:4.1f}%) | {lora_arc:16d}/50 ({lora_arc*2:4.1f}%) | {dual_arc:18d}/50 ({dual_arc*2:4.1f}%)", flush=True)
print(f"{'3. GSM8K (40q Step-by-Step Math Gen)':<42} | {base_gsm:14d}/40 ({base_gsm*2.5:4.1f}%) | {lora_gsm:16d}/40 ({lora_gsm*2.5:4.1f}%) | {dual_gsm:18d}/40 ({dual_gsm*2.5:4.1f}%)", flush=True)
print("-" * 128, flush=True)
tot_base = base_mmlu + base_arc + base_gsm
tot_lora = lora_mmlu + lora_arc + lora_gsm
tot_dual = dual_mmlu + dual_arc + dual_gsm
print(f"[audit #15] all three columns: identical prompts and scoring per task; Dual-Brain prefix applied to all 140 questions; ARC train/test overlap {_arc_overlap}/{len(arc_test)}", flush=True)
print(f"{'OVERALL SCORE (Out of 140 Official Tasks)':<42} | {tot_base:13d}/140 ({tot_base/140*100:4.1f}%) | {tot_lora:15d}/140 ({tot_lora/140*100:4.1f}%) | {tot_dual:17d}/140 ({tot_dual/140*100:4.1f}%)", flush=True)

os.makedirs('/content/out', exist_ok=True)
torch.save(dual_brain.state_dict(), '/content/out/dual_brain_qwen25_open_llm_cribnote.pth')
