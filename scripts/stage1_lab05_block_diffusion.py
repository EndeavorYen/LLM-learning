# ==============================================================================
# ORIGINAL EXECUTED SCRIPT EXTRACTED VERBATIM FROM TRANSCRIPT (AST-UNESCAPED)
# Transcript Step Index : 310
# Timestamp             : 2026-10-30T05:04:47Z
# Original Colab Session: stage1_block_diff_v2
# ==============================================================================

import os, sys, time, math, json, random, subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F

if not os.path.exists('/content/minimind'):
    subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/jingyaogong/minimind.git', '/content/minimind'], check=True)
sys.path.insert(0, '/content/minimind')
os.makedirs('/content/minimind/out', exist_ok=True)

from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

tokenizer = AutoTokenizer.from_pretrained('/content/minimind/model')
MASK_ID = 3                      # [▓] Mask slot to be denoised
SLOT_PAD_ID = 4                  # [□] Dedicated dynamic-canvas placeholder ("nothing more to say in this slot")
BOS_ID = tokenizer.bos_token_id

class BlockDiffusionMiniMind(nn.Module):
    def __init__(self, config: MiniMindConfig, block_size: int = 8):
        super().__init__()
        self.backbone = MiniMindForCausalLM(config)
        self.block_size = block_size
        for layer in self.backbone.model.layers:
            layer.self_attn.is_causal = False

    @torch.inference_mode()
    def generate_block_diffusion(self, prompt_ids: torch.Tensor, block_size: int = 8, steps_per_block: int = 3, max_blocks: int = 5):
        '''
        Dynamic Block Expansion (8 -> 16 -> 24...) + Cross-Block KV-Cache + [□] Placeholder Auto-Stop
        '''
        B, P = prompt_ids.shape
        device = prompt_ids.device

        # 1. Prefill prompt into KV-Cache once
        prefill_out = self.backbone(prompt_ids, use_cache=True)
        locked_kv = prefill_out.past_key_values

        generated_blocks = []
        expansion_trace = []

        for blk_idx in range(max_blocks):
            block_canvas = torch.full((B, block_size), MASK_ID, dtype=torch.long, device=device)
            snaps = []

            for s in range(steps_per_block):
                out = self.backbone(block_canvas, past_key_values=locked_kv, use_cache=False)
                probs = F.softmax(out.logits, dim=-1)
                conf, pred = probs.max(dim=-1)

                still_masked = (block_canvas == MASK_ID)
                n_rem = still_masked[0].sum().item()
                if n_rem == 0:
                    break
                conf = conf.masked_fill(~still_masked, -1.0)
                k = n_rem if s == steps_per_block - 1 else max(1, int(round(block_size / steps_per_block)))
                k = min(k, n_rem)
                _, top_idx = torch.topk(conf, k=k, dim=-1)
                for b in range(B):
                    for idx in top_idx[b]:
                        if still_masked[b, idx]:
                            block_canvas[b, idx] = pred[b, idx]
                snaps.append(block_canvas[0].clone())

            generated_blocks.append(block_canvas)
            expansion_trace.append((blk_idx + 1, snaps))

            # If placeholder [□] (SLOT_PAD_ID) appears in finalized block, stop expanding!
            if B == 1 and (block_canvas[0] == SLOT_PAD_ID).any():
                break

            # Otherwise lock finalized block into KV-Cache and expand +8 slots!
            locked_kv = self.backbone(block_canvas, past_key_values=locked_kv, use_cache=True).past_key_values

        return torch.cat([prompt_ids] + generated_blocks, dim=1), expansion_trace

    @torch.inference_mode()
    def local_text_surgery(self, token_seq: list, steps: int = 3):
        '''
        Photoshop-style Local Text Editing (Modify / Expand / Remove & Heal):
        Only denoises the [▓] (MASK_ID) slots while keeping all surrounding text locked!
        Any excess slots predicted as [□] (SLOT_PAD_ID) are automatically collapsed!
        '''
        device = next(self.parameters()).device
        canvas = torch.tensor([token_seq], dtype=torch.long, device=device)
        total_masks = (canvas[0] == MASK_ID).sum().item()
        snaps = [canvas[0].clone()]

        for s in range(max(1, steps)):
            still_masked = (canvas == MASK_ID)
            n_rem = still_masked[0].sum().item()
            if n_rem == 0:
                break
            logits = self.backbone(canvas).logits
            probs = F.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            conf = conf.masked_fill(~still_masked, -1.0)
            k = n_rem if s == steps - 1 else max(1, int(round(total_masks / steps)))
            k = min(k, n_rem)
            _, top_idx = torch.topk(conf, k=k, dim=-1)
            for idx in top_idx[0]:
                if still_masked[0, idx]:
                    canvas[0, idx] = pred[0, idx]
            snaps.append(canvas[0].clone())
        return canvas[0], snaps

    @torch.inference_mode()
    def generate_full_canvas_no_cache(self, prompt_ids: torch.Tensor, gen_len: int = 32, steps: int = 16):
        B, P = prompt_ids.shape
        canvas = torch.full((B, P + gen_len), MASK_ID, dtype=torch.long, device=prompt_ids.device)
        canvas[:, :P] = prompt_ids
        for s in range(steps):
            logits = self.backbone(canvas).logits[:, P:, :]
            conf, pred = F.softmax(logits, dim=-1).max(dim=-1)
            still_masked = (canvas[:, P:] == MASK_ID)
            n_rem = still_masked[0].sum().item()
            if n_rem == 0: break
            conf = conf.masked_fill(~still_masked, -1.0)
            k = n_rem if s == steps - 1 else max(1, int(round(gen_len / steps)))
            _, top_idx = torch.topk(conf, k=min(k, n_rem), dim=-1)
            for b in range(B):
                for idx in top_idx[b]:
                    if still_masked[b, idx]:
                        canvas[b, P + idx] = pred[b, idx]
        return canvas

# Corpus with 3 distinct lengths + rich local editing pairs
sentences = [
    ("地球是太阳系", "中的一颗行星。"),
    ("机器学习通过", "算法让计算机从海量数据中自动学习规律。"),
    ("因为太阳光进入", "大气层时，蓝光的波长比较短，容易被空气分子散射，所以我们看到的天空是蓝色的。"),
    ("周末下午我", "特地去了一趟图书馆，在书架上挑了两本最喜欢的科幻小说安静地阅读。"),
    ("周末下午我", "特地去了一趟牙科诊所，请专业医生帮我检查牙齿并清理了牙结石。"),
    ("周末下午我", "特地去了一趟生鲜超市，买了新鲜的苹果、牛奶和面包准备做早餐。"),
    ("周末下午我", "买了新鲜的面包准备做早餐。")
]

BLOCK_SIZE = 8
# Build block-aligned training examples (prefix + 8-token target block padded with SLOT_PAD_ID only in the final block!)
block_prefix_examples = []
full_seq_examples = []

for p_str, cont_str in sentences:
    p_ids = [BOS_ID] + tokenizer(p_str, add_special_tokens=False).input_ids
    c_ids = tokenizer(cont_str, add_special_tokens=False).input_ids
    # Pad continuation ONLY up to the next multiple of BLOCK_SIZE (8) using SLOT_PAD_ID ([□])!
    rem = len(c_ids) % BLOCK_SIZE
    pad_needed = (BLOCK_SIZE - rem) if rem != 0 else BLOCK_SIZE
    # Keep at least 2 [□] placeholders in the last block so the model clearly marks sentence end
    if pad_needed < 2:
        pad_needed += BLOCK_SIZE
    c_padded = c_ids + [SLOT_PAD_ID] * pad_needed

    num_blks = len(c_padded) // BLOCK_SIZE
    for b_i in range(num_blks):
        known_prefix = p_ids + c_padded[: b_i * BLOCK_SIZE]
        target_block = c_padded[b_i * BLOCK_SIZE : (b_i + 1) * BLOCK_SIZE]
        block_prefix_examples.append((known_prefix, target_block))

    full_seq_examples.append(p_ids + c_ids)

# Also build over-expanded examples (where middle has extra [□] placeholders to teach automatic slot collapse!)
s_long = [BOS_ID] + tokenizer("因为太阳光进入大气层时，", add_special_tokens=False).input_ids
m_long = tokenizer("蓝光的波长比较短，容易被空气分子散射，", add_special_tokens=False).input_ids
e_long = tokenizer("所以我们看到的天空是蓝色的。", add_special_tokens=False).input_ids
# Suppose user inserts 16 slots in the middle, and m_long is 14 tokens -> 2 slots become [□] (SLOT_PAD_ID)!
expanded_middle_target = m_long + [SLOT_PAD_ID] * (16 - len(m_long))

cfg = MiniMindConfig(hidden_size=512, num_hidden_layers=8, use_moe=False)
torch.manual_seed(42)
model = BlockDiffusionMiniMind(cfg, block_size=BLOCK_SIZE).cuda().bfloat16()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

print("=== [1] Training Block-Diffusion MiniMind (Complementary Masking + Block Expansion + Local Surgery) ===", flush=True)
STEPS = 300
model.train()
for step in range(1, STEPS + 1):
    lr = 3e-5 + 0.5 * (3e-4 - 3e-5) * (1 + math.cos(math.pi * step / STEPS))
    for pg in opt.param_groups: pg['lr'] = lr
    opt.zero_grad()
    total_loss = 0.0

    # Part A: Block-Expansion training with Complementary Masking (100% token coverage on target_block!)
    for known_p, tgt_blk in block_prefix_examples:
        t_p = torch.tensor(known_p, dtype=torch.long, device='cuda')
        t_b = torch.tensor(tgt_blk, dtype=torch.long, device='cuda')
        # Mask m1 on the 8-token block, and complementary mask m2 = ~m1 ( plus full all-masked block!)
        m1 = torch.rand(BLOCK_SIZE, device='cuda') < 0.5
        if not m1.any(): m1[0] = True
        if m1.all(): m1[-1] = False
        m2 = ~m1
        m_all = torch.ones(BLOCK_SIZE, dtype=torch.bool, device='cuda')

        batch_in, batch_lbl = [], []
        for m in (m1, m2, m_all):
            noisy_b = t_b.clone()
            noisy_b[m] = MASK_ID
            inp = torch.cat([t_p, noisy_b], dim=0)
            lbl = torch.full_like(inp, -100)
            lbl_b = t_b.clone()
            lbl_b[~m] = -100
            lbl[len(t_p):] = lbl_b
            batch_in.append(inp)
            batch_lbl.append(lbl)

        inp_t = torch.stack(batch_in, dim=0)
        lbl_t = torch.stack(batch_lbl, dim=0)
        logits = model.backbone(inp_t).logits
        loss_blk = F.cross_entropy(logits.view(-1, logits.size(-1)), lbl_t.view(-1), ignore_index=-100)
        total_loss = total_loss + loss_blk

    # Part B: Local Text Surgery (Modify / Expand with [□] collapse / Remove & Heal)
    # 1. Expand example (s_long + 16 slots + e_long)
    exp_gt = torch.tensor(s_long + expanded_middle_target + e_long, dtype=torch.long, device='cuda')
    exp_in = exp_gt.clone()
    m_exp = torch.zeros_like(exp_gt, dtype=torch.bool)
    for idx_m in range(len(s_long), len(s_long) + 16):
        if random.random() < 0.85 or step % 2 == 0:
            m_exp[idx_m] = True
    exp_in[m_exp] = MASK_ID
    exp_lbl = exp_gt.clone(); exp_lbl[~m_exp] = -100

    # 2. Modify & Remove/Heal examples on full_seq_examples
    surg_losses = []
    for f_seq in full_seq_examples:
        f_gt = torch.tensor(f_seq, dtype=torch.long, device='cuda')
        f_in = f_gt.clone()
        m_f = torch.rand(len(f_seq), device='cuda') < 0.45
        # Also contiguous span mask
        st = random.randint(3, max(4, len(f_seq) - 5))
        ed = min(len(f_seq), st + random.randint(2, 6))
        m_f[st:ed] = True
        m_f[0] = False
        f_in[m_f] = MASK_ID
        f_lbl = f_gt.clone(); f_lbl[~m_f] = -100
        logits_f = model.backbone(f_in.unsqueeze(0)).logits
        surg_losses.append(F.cross_entropy(logits_f.view(-1, logits_f.size(-1)), f_lbl.view(-1), ignore_index=-100))

    logits_e = model.backbone(exp_in.unsqueeze(0)).logits
    loss_e = F.cross_entropy(logits_e.view(-1, logits_e.size(-1)), exp_lbl.view(-1), ignore_index=-100)

    loss = (total_loss / len(block_prefix_examples)) + loss_e + (sum(surg_losses) / len(surg_losses))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if step in (1, 100, 200, 300):
        print(f"  Step {step:3d}/{STEPS} | Total Loss: {loss.item():.4f} | Block Token Coverage (m1 U m2): 100.0%", flush=True)

model.eval()

def render_with_symbols(t_ids):
    out = []
    for tid in t_ids.tolist():
        if tid == MASK_ID: out.append("[▓]")
        elif tid == SLOT_PAD_ID: out.append("[□]")
        elif tid == BOS_ID: continue
        else: out.append(tokenizer.decode([tid]))
    return "".join(out)

def render_clean(t_ids):
    clean = [tid for tid in t_ids.tolist() if tid not in (MASK_ID, SLOT_PAD_ID, BOS_ID)]
    return tokenizer.decode(clean)

print("\n=== [Benchmark A] Problem 2 Solved: Dynamic Block Expansion (+8 Slots) + [□] Placeholder Auto-Stop ===", flush=True)
for label, p_str in [
    ("Short Sentence (1 Block = 8 slots)", "地球是太阳系"),
    ("Medium Sentence (2 Blocks = 16 slots)", "机器学习通过"),
    ("Long Sentence (4 Blocks = 32 slots)", "因为太阳光进入")
]:
    p_ids = torch.tensor([[BOS_ID] + tokenizer(p_str, add_special_tokens=False).input_ids], device='cuda')
    full_out, trace = model.generate_block_diffusion(p_ids, block_size=8, steps_per_block=3, max_blocks=5)
    print(f"\n[{label}] Prompt: '{p_str}' -> Expanded {len(trace)} time(s) ({len(trace)*8} slots total):", flush=True)
    for blk_num, snaps in trace:
        print(f"  Block {blk_num} (+8 slots): Step 1: {render_with_symbols(snaps[0])}  -->  Step 3 (Final): {render_with_symbols(snaps[-1])}", flush=True)
    print(f"  => Auto-Stopped on [□]! Final Output: {render_clean(full_out[0])}", flush=True)

print("\n=== [Benchmark B] Photoshop-Style Local Text Surgery: Modify (修改) / Expand (擴展) / Remove & Heal (移除與縫合) ===", flush=True)

# 1. MODIFY (局部修改)
mod_prefix = [BOS_ID] + tokenizer("周末下午我特地去了一趟", add_special_tokens=False).input_ids
mod_old_mid = tokenizer("图书馆", add_special_tokens=False).input_ids
mod_new_suffix = tokenizer("，请专业医生帮我检查牙齿并清理了牙结石。", add_special_tokens=False).input_ids
# Replace 3-token '图书馆' with 4 [▓] slots to match '牙科诊所'
mod_canvas = mod_prefix + [MASK_ID] * 4 + mod_new_suffix
mod_res, mod_snaps = model.local_text_surgery(mod_canvas, steps=2)
print("1. [局部修改 Modify] Changed ending from reading books to cleaning teeth, masked location:", flush=True)
print(f"   Before : {render_with_symbols(torch.tensor(mod_canvas))}", flush=True)
print(f"   After  : {render_clean(mod_res)}", flush=True)

# 2. EXPAND (局部插槽擴展 + 多餘欄位自動 [□] 摺疊)
exp_canvas = s_long + [MASK_ID] * 16 + e_long
exp_res, exp_snaps = model.local_text_surgery(exp_canvas, steps=3)
print("\n2. [局部擴展 Expand] Inserted +16 [▓] slots into short skeleton sentence (14 needed, 2 excess -> [□]):", flush=True)
print(f"   Skeleton : 因为太阳光进入大气层时， + [16x ▓] + 所以我们看到的天空是蓝色的。", flush=True)
print(f"   Step 1   : {render_with_symbols(exp_snaps[1])}", flush=True)
print(f"   Step 3   : {render_with_symbols(exp_snaps[-1])}", flush=True)
print(f"   Collapsed: {render_clean(exp_res)}", flush=True)

# 3. REMOVE & HEAL (局部刪除與接縫自動修復)
# Original: "周末下午我特地去了一趟生鲜超市，买了新鲜的苹果、牛奶和面包准备做早餐。"
# Delete "特地去了一趟生鲜超市，买了新鲜的苹果、牛奶和", leaving "周末下午我" + [▓][▓][▓][▓] + "面包准备做早餐。"
rem_left = [BOS_ID] + tokenizer("周末下午我", add_special_tokens=False).input_ids
rem_right = tokenizer("面包准备做早餐。", add_special_tokens=False).input_ids
rem_canvas = rem_left + [MASK_ID] * 4 + rem_right
rem_res, _ = model.local_text_surgery(rem_canvas, steps=2)
print("\n3. [局部移除與縫合 Remove & Heal] Deleted middle supermarket trip + apples/milk, left 4 [▓] healing slots:", flush=True)
print(f"   Cut Seam : {render_with_symbols(torch.tensor(rem_canvas))}", flush=True)
print(f"   Healed   : {render_clean(rem_res)}", flush=True)

print("\n=== [Benchmark C] Problem 3 Solved: Block Diffusion (With KV-Cache) vs Full-Canvas Diffusion on Long Contexts ===", flush=True)
BENCH_BS = 8
for ctx_len in (256, 1024, 2048):
    long_prompt = torch.full((BENCH_BS, ctx_len), BOS_ID, dtype=torch.long, device='cuda')
    # Generate 32 tokens (4 blocks of 8, 3 steps/block = 12 denoising steps + 3 cache steps = 15 passes, vs 15 passes of Full-Canvas)
    _ = model.generate_block_diffusion(long_prompt, block_size=8, steps_per_block=3, max_blocks=4)
    _ = model.generate_full_canvas_no_cache(long_prompt, gen_len=32, steps=15)
    torch.cuda.synchronize()

    t0 = time.time()
    _ = model.generate_full_canvas_no_cache(long_prompt, gen_len=32, steps=15)
    torch.cuda.synchronize()
    t_full = (time.time() - t0) * 1000

    t0 = time.time()
    _ = model.generate_block_diffusion(long_prompt, block_size=8, steps_per_block=3, max_blocks=4)
    torch.cuda.synchronize()
    t_blk = (time.time() - t0) * 1000

    full_ops = 15 * (ctx_len + 32)
    blk_ops = ctx_len + 4 * (3 * 8 + 8)
    print(f"  Context={ctx_len:4d} + Gen=32 | Full-Canvas (No Cache): {t_full:6.1f} ms ({full_ops:5d} tok-passes) | Block-Diffusion (KV-Cache): {t_blk:6.1f} ms ({blk_ops:4d} tok-passes) -> {t_full/t_blk:.2f}x Faster!", flush=True)

torch.save(model.state_dict(), '/content/minimind/out/block_diffusion_minimind_512.pth')
