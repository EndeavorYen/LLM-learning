# ==============================================================================
# ORIGINAL EXECUTED SCRIPT EXTRACTED VERBATIM FROM TRANSCRIPT (AST-UNESCAPED)
# Transcript Step Index : 288
# Timestamp             : 2026-10-30T04:49:10Z
# Original Colab Session: stage1_diff_v2
# ==============================================================================

import os, sys, time, math, json, random, subprocess, requests
import torch
import torch.nn as nn
import torch.nn.functional as F

if not os.path.exists('/content/minimind'):
    # [audit #17] pinned: 233d00f (2026-10-27) is the upstream HEAD that predates every original run
    subprocess.run(['git', 'clone', '--filter=blob:none', 'https://github.com/jingyaogong/minimind.git', '/content/minimind'], check=True)
    subprocess.run(['git', '-C', '/content/minimind', 'checkout', '-q', '233d00fa068ec2b8c0d4f07b703934d6391a5a73'], check=True)
sys.path.insert(0, '/content/minimind')
os.makedirs('/content/minimind/out', exist_ok=True)

from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

tokenizer = AutoTokenizer.from_pretrained('/content/minimind/model')
MASK_ID = 3  # Dedicated [MASK] token id

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

class DiffusionMiniMind(nn.Module):
    def __init__(self, config: MiniMindConfig, mask_token_id: int = MASK_ID):
        super().__init__()
        self.backbone = MiniMindForCausalLM(config)
        self.mask_token_id = mask_token_id
        # Surgery 1: Disable Causal Mask -> Enable Full Bidirectional Attention
        for layer in self.backbone.model.layers:
            layer.self_attn.is_causal = False

    def forward_loss(self, input_ids: torch.Tensor, pad_id: int = 0):
        B, L = input_ids.shape
        device = input_ids.device
        # Surgery 2: LLaDA Hybrid Masking (Uniform random masking t~U(0.15, 1.0) + Suffix/Span masking)
        t = torch.empty((B, 1), device=device).uniform_(0.15, 1.0)
        rand_mask = (torch.rand((B, L), device=device) < t)
        # For 30% of samples in batch, mask a contiguous span or suffix (teaches full suffix & infill)
        for b in range(B):
            if random.random() < 0.35:
                start = random.randint(4, max(5, L - 8))
                end = random.randint(start + 2, L)
                rand_mask[b, :] = False
                rand_mask[b, start:end] = True
        rand_mask = rand_mask & (input_ids != tokenizer.bos_token_id)

        noisy_ids = input_ids.clone()
        noisy_ids[rand_mask] = self.mask_token_id

        logits = self.backbone(noisy_ids).logits
        labels = input_ids.clone()
        labels[~rand_mask] = -100
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return loss, rand_mask.float().mean().item()

    @torch.inference_mode()
    def generate_diffusion(self, prompt_ids: torch.Tensor, total_len: int = 32, steps: int = 6):
        B, P = prompt_ids.shape
        gen_len = total_len - P
        device = prompt_ids.device
        canvas = torch.full((B, total_len), self.mask_token_id, dtype=torch.long, device=device)
        canvas[:, :P] = prompt_ids

        history = [canvas[0, P:].clone()]
        for step in range(steps):
            logits = self.backbone(canvas).logits[:, P:, :]
            probs = F.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)

            still_masked = (canvas[:, P:] == self.mask_token_id)
            n_rem = still_masked[0].sum().item()
            if n_rem == 0:
                break
            conf = conf.masked_fill(~still_masked, -1.0)
            k = n_rem if step == steps - 1 else max(1, int(round(gen_len / steps)))
            k = min(k, n_rem)
            _, top_idx = torch.topk(conf, k=k, dim=-1)
            for b in range(B):
                for idx in top_idx[b]:
                    if still_masked[b, idx]:
                        canvas[b, P + idx] = pred[b, idx]
            history.append(canvas[0, P:].clone())
        return canvas, history

base_sentences = [
    "因为太阳光进入大气层时，蓝光的波长比较短，容易被空气分子散射，所以我们看到的天空是蓝色的。",
    "机器学习是人工智能的核心分支，它通过算法让计算机从海量数据中自动学习规律，并用于预测未来。",
    "周末下午我特地去了一趟图书馆，在书架上挑了两本最喜欢的科幻小说安静地阅读。",
    "周末下午我特地去了一趟牙科诊所，请专业医生帮我检查牙齿并清理了牙结石。",
    "周末下午我特地去了一趟生鲜超市，买了新鲜的苹果、牛奶和面包准备做早餐。",
    "深度学习使用多层神经网络模拟人类大脑连接，在图像识别与自然语言处理领域取得了巨大突破。",
    "地球围绕着太阳公转，公转一圈的时间大约是三百六十五天，形成了四季更替的自然现象。",
    "大语言模型在预训练阶段阅读海量文本学习世界知识，在微调阶段学习遵守人类指令回答问题。"
]

url = 'https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/main/pretrain_t2t_mini.jsonl'
extra_texts = []
with requests.get(url, stream=True) as r:
    for i, line in enumerate(r.iter_lines()):
        if i >= 300:
            break
        if line:
            txt = json.loads(line.decode('utf-8'))['text'].strip().replace('\n', '')
            if 30 <= len(txt) <= 80:
                extra_texts.append(txt)

all_texts = base_sentences * 40 + extra_texts[:120]
random.seed(42)
random.shuffle(all_texts)

SEQ_LEN = 32
encoded_rows = []
for t in all_texts:
    ids = [tokenizer.bos_token_id] + tokenizer(t, add_special_tokens=False).input_ids[:SEQ_LEN - 2] + [tokenizer.eos_token_id]
    ids = ids + [tokenizer.pad_token_id] * (SEQ_LEN - len(ids))
    encoded_rows.append(ids)
data_tensor = torch.tensor(encoded_rows, dtype=torch.long, device='cuda')

print("=== [1] Training Original AR-MiniMind vs Modified Diffusion-MiniMind (400 Steps, GradClip=1.0, Cosine LR) ===", flush=True)
torch.manual_seed(42)
cfg = MiniMindConfig(hidden_size=512, num_hidden_layers=8, use_moe=False)
ar_model = MiniMindForCausalLM(cfg).cuda().bfloat16()
ar_opt = torch.optim.AdamW(ar_model.parameters(), lr=3e-4, weight_decay=0.01)

torch.manual_seed(42)
diff_model = DiffusionMiniMind(cfg, mask_token_id=MASK_ID).cuda().bfloat16()
diff_opt = torch.optim.AdamW(diff_model.parameters(), lr=3e-4, weight_decay=0.01)

STEPS = 400
BS = 32
# [audit #16] equal budgets: both models get the same steps, batch size, LR and cosine schedule.
for _name in ("AR-MiniMind", "Diffusion-MiniMind"):
    print(f"[audit #16] budget {_name}: {STEPS} steps x BS {BS}, AdamW lr 3e-4 -> 3e-5 cosine, clip 1.0", flush=True)

t0 = time.time()
ar_model.train()
for step in range(1, STEPS + 1):
    lr = 3e-5 + 0.5 * (3e-4 - 3e-5) * (1 + math.cos(math.pi * step / STEPS))
    for pg in ar_opt.param_groups: pg['lr'] = lr
    idx = torch.randint(0, len(data_tensor), (BS,), device='cuda')
    batch = data_tensor[idx]
    res = ar_model(batch, labels=batch.clone())
    ar_opt.zero_grad()
    res.loss.backward()
    torch.nn.utils.clip_grad_norm_(ar_model.parameters(), 1.0)
    ar_opt.step()
    if step in (1, 100, 200, 300, 400):
        print(f"  [AR-MiniMind]        Step {step:3d}/{STEPS} | Loss: {res.loss.item():.4f} | Supervised Tokens/Pass: 100.0%", flush=True)
ar_train_time = time.time() - t0

t0 = time.time()
diff_model.train()
for step in range(1, STEPS + 1):
    lr = 3e-5 + 0.5 * (3e-4 - 3e-5) * (1 + math.cos(math.pi * step / STEPS))
    for pg in diff_opt.param_groups: pg['lr'] = lr
    idx = torch.randint(0, len(data_tensor), (BS,), device='cuda')
    batch = data_tensor[idx]
    loss, mask_ratio = diff_model.forward_loss(batch, pad_id=tokenizer.pad_token_id)
    diff_opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(diff_model.parameters(), 1.0)
    diff_opt.step()
    if step in (1, 100, 200, 300, 400):
        print(f"  [Diffusion-MiniMind] Step {step:3d}/{STEPS} | Loss: {loss.item():.4f} | Supervised Tokens/Pass: {mask_ratio*100:.1f}%", flush=True)
diff_train_time = time.time() - t0

ar_model.eval()
diff_model.eval()

print("\n=== [Benchmark 1] Polaroid Photo Emergence: Step-by-Step Parallel Unmasking ===", flush=True)
prompt_text = "因为太阳光进入大气层时，"
prompt_ids = torch.tensor([[tokenizer.bos_token_id] + tokenizer(prompt_text, add_special_tokens=False).input_ids], device='cuda')
_, history = diff_model.generate_diffusion(prompt_ids, total_len=SEQ_LEN, steps=6)

def render_canvas(token_ids):
    parts = []
    for tid in token_ids.tolist():
        if tid == MASK_ID:
            parts.append("[▓]")
        elif tid in (tokenizer.pad_token_id, tokenizer.eos_token_id):
            continue
        else:
            parts.append(tokenizer.decode([tid]))
    return "".join(parts)

print(f"Prompt: {prompt_text}", flush=True)
for s_idx, snap in enumerate(history):
    print(f"  Step {s_idx}/6: {prompt_text}{render_canvas(snap)}", flush=True)

print("\n=== [Benchmark 2] Generation Speed & Forward Passes (Generating 64 Tokens, Batch=8) ===", flush=True)
GEN_LEN = 64
BENCH_BS = 8
bench_prompt = torch.full((BENCH_BS, 8), tokenizer.bos_token_id, dtype=torch.long, device='cuda')
_ = ar_model.generate(bench_prompt, max_new_tokens=8, use_cache=True, eos_token_id=None)
_ = diff_model.generate_diffusion(bench_prompt, total_len=16, steps=2)
torch.cuda.synchronize()

t0 = time.time()
_ = ar_model.generate(bench_prompt, max_new_tokens=GEN_LEN, use_cache=False, do_sample=False, eos_token_id=None)
torch.cuda.synchronize()
t_ar_nocache = (time.time() - t0) * 1000

t0 = time.time()
ar_bench_out = ar_model.generate(bench_prompt, max_new_tokens=GEN_LEN, use_cache=True, do_sample=False, eos_token_id=None)  # [audit #16] keep output for quality print
torch.cuda.synchronize()
t_ar_cache = (time.time() - t0) * 1000

for d_steps in (16, 8, 4):
    t0 = time.time()
    d_canvas, _ = diff_model.generate_diffusion(bench_prompt, total_len=8 + GEN_LEN, steps=d_steps)
    torch.cuda.synchronize()
    t_diff = (time.time() - t0) * 1000
    tok_per_sec = (BENCH_BS * GEN_LEN) / (t_diff / 1000.0)
    # [audit #16] no "x faster" claim: fewer steps change output quality, so print a decoded sample per step count.
    print(f"  Diffusion ({d_steps:2d} steps | {d_steps:2d} Forward Passes): {t_diff:6.1f} ms | {tok_per_sec:7.1f} tokens/s", flush=True)
    print(f"     sample[0] @ {d_steps:2d} steps: {render_canvas(d_canvas[0, 8:])}", flush=True)

print(f"  AR + KV-Cache (64 steps | 64 Forward Passes): {t_ar_cache:6.1f} ms | {(BENCH_BS*GEN_LEN)/(t_ar_cache/1000.0):7.1f} tokens/s", flush=True)
print(f"  AR No-Cache   (64 steps | 64 Forward Passes): {t_ar_nocache:6.1f} ms | {(BENCH_BS*GEN_LEN)/(t_ar_nocache/1000.0):7.1f} tokens/s", flush=True)
print(f"     AR sample[0] (greedy, KV-cache): {render_canvas(ar_bench_out[0, 8:])}", flush=True)  # [audit #16]
print("  [audit #16] speed comparison not at equal quality — see #12", flush=True)

print("\n=== [Benchmark 3] Global Bidirectional Infilling (Seeing the Future Clue!) ===", flush=True)
test_cases = [
    ("周末下午我特地去了一趟", "图书馆", "，在书架上挑了两本最喜欢的科幻小说安静地阅读。"),
    ("周末下午我特地去了一趟", "牙科诊所", "，请专业医生帮我检查牙齿并清理了牙结石。"),
    ("周末下午我特地去了一趟", "生鲜超市", "，买了新鲜的苹果、牛奶和面包准备做早餐。")
]

# [audit #16] these three sentences are in base_sentences (trained x40), so this measures memorized infilling, not generalisation.
print("[audit #16] NOTE: cloze sentences are training sentences -> this tests memorized infilling (see #16)", flush=True)

@torch.inference_mode()
def ar_seq_loglik(ids):
    """[audit #16] total log-likelihood of ids[1:] given ids[:-1] under the causal AR model."""
    t = torch.tensor([ids], device='cuda')
    lp = F.log_softmax(ar_model(t).logits[0, :-1, :].float(), dim=-1)
    return lp.gather(-1, t[0, 1:].unsqueeze(-1)).sum().item()

candidates = [m for _, m, _ in test_cases]  # [audit #16] AR picks among the same 3 candidate fills
ar_correct, diff_correct, total_masked = 0, 0, 0
ar_sent_correct, diff_sent_correct, ar_greedy_sent_correct = 0, 0, 0
for prefix_str, target_middle_str, suffix_str in test_cases:
    full_str = prefix_str + target_middle_str + suffix_str
    p_ids = [tokenizer.bos_token_id] + tokenizer(prefix_str, add_special_tokens=False).input_ids
    m_ids = tokenizer(target_middle_str, add_special_tokens=False).input_ids
    s_ids = tokenizer(suffix_str, add_special_tokens=False).input_ids
    full_ids = (p_ids + m_ids + s_ids)[:SEQ_LEN - 1] + [tokenizer.eos_token_id]
    full_ids = full_ids + [tokenizer.pad_token_id] * (SEQ_LEN - len(full_ids))

    # AR can only see p_ids to predict m_ids
    ar_inp = torch.tensor([p_ids], device='cuda')
    ar_out = ar_model.generate(ar_inp, max_new_tokens=len(m_ids), do_sample=False, use_cache=True)
    ar_pred_ids = ar_out[0, len(p_ids):len(p_ids)+len(m_ids)].tolist()
    ar_pred_str = tokenizer.decode(ar_pred_ids)

    # Diffusion sees p_ids + [MASK]*len(m_ids) + s_ids all at once!
    diff_seq = list(full_ids)
    for k_i in range(len(p_ids), len(p_ids) + len(m_ids)):
        diff_seq[k_i] = MASK_ID
    diff_inp = torch.tensor([diff_seq], device='cuda')
    with torch.inference_mode():
        diff_logits = diff_model.backbone(diff_inp).logits
    diff_pred_ids = diff_logits[0, len(p_ids):len(p_ids)+len(m_ids), :].argmax(dim=-1).tolist()
    diff_pred_str = tokenizer.decode(diff_pred_ids)

    ar_hit = sum(1 for a, b in zip(ar_pred_ids, m_ids) if a == b)
    diff_hit = sum(1 for a, b in zip(diff_pred_ids, m_ids) if a == b)
    ar_correct += ar_hit
    diff_correct += diff_hit
    total_masked += len(m_ids)

    # [audit #16] fair AR protocol: score prefix + candidate + suffix (+EOS) by total log-likelihood, pick argmax.
    cand_ll = []
    for cand in candidates:
        c_ids = ([tokenizer.bos_token_id] + tokenizer(prefix_str + cand + suffix_str, add_special_tokens=False).input_ids)[:SEQ_LEN - 1] + [tokenizer.eos_token_id]
        cand_ll.append(ar_seq_loglik(c_ids))
    ar_pick = candidates[max(range(len(candidates)), key=lambda i: cand_ll[i])]
    ar_sent_correct += int(ar_pick == target_middle_str)
    ar_greedy_sent_correct += int(ar_pred_ids == m_ids)
    diff_sent_correct += int(diff_pred_ids == m_ids)  # [audit #16] sentence-level: fill exactly equals the answer

    print(f"Clue in Suffix: '{prefix_str} [___] {suffix_str}' (Ground Truth: [{target_middle_str}])", flush=True)
    print(f"  -> AR-MiniMind greedy (Blind to Suffix) guessed: [{ar_pred_str}]", flush=True)
    print(f"  -> AR-MiniMind LL-rank (sees suffix via scoring): [{ar_pick}]  LL={', '.join(f'{c}:{v:.1f}' for c, v in zip(candidates, cand_ll))}", flush=True)
    print(f"  -> Diffusion-MiniMind (Full Sight) guessed: [{diff_pred_str}]", flush=True)

n_items = len(test_cases)
print(f"\n[Infilling Token Accuracy]: AR-MiniMind greedy = {ar_correct}/{total_masked} ({ar_correct/total_masked*100:.1f}%) vs Diffusion-MiniMind = {diff_correct}/{total_masked} ({diff_correct/total_masked*100:.1f}%)", flush=True)
# [audit #16] sentence-level accuracy with denominators
print(f"[Infilling Sentence Accuracy]: AR LL-rank (3-way) = {ar_sent_correct}/{n_items} | AR greedy exact = {ar_greedy_sent_correct}/{n_items} | Diffusion exact fill = {diff_sent_correct}/{n_items}", flush=True)
print("[audit #16] NOTE: AR LL-rank is a 3-way choice (chance 1/3); diffusion is open-vocabulary but is told the answer length via the mask count.", flush=True)

torch.save(diff_model.state_dict(), '/content/minimind/out/diffusion_minimind_512.pth')
