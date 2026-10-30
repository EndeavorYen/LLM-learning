# ==============================================================================
# ORIGINAL EXECUTED SCRIPT EXTRACTED VERBATIM FROM TRANSCRIPT (AST-UNESCAPED)
# Transcript Step Index : 339
# Timestamp             : 2026-10-30T05:38:21Z
# Original Colab Session: stage1_thought_assoc_v3
# ==============================================================================

import os, sys, time, math, random, subprocess
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
MASK_ID = 3
SLOT_PAD_ID = 4
BOS_ID = tokenizer.bos_token_id
STYLE_DETAILED_ID = 5   # Style token: render thought in full detailed language
STYLE_CONCISE_ID = 6    # Style token: render the EXACT SAME thought in concise language

class AssociativeThoughtMiniMind(nn.Module):
    def __init__(self, config: MiniMindConfig, max_clause_slots: int = 12):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.max_clause_slots = max_clause_slots
        self.backbone = MiniMindForCausalLM(config)
        for layer in self.backbone.model.layers:
            layer.self_attn.is_causal = False

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size, nhead=8, dim_feedforward=config.hidden_size*2,
            batch_first=True, norm_first=True
        )
        self.associative_brain = nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=False)
        self.next_thought_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.stop_gate = nn.Linear(config.hidden_size, 1)

        self.thought_to_prefix = nn.Linear(config.hidden_size, config.hidden_size * 2)
        self._active_prefix = None

        def _inject_thought_hook(module, inp, out):
            if self._active_prefix is not None:
                out = out.clone()
                out[:, :2, :] = self._active_prefix
            return out

        self.backbone.model.embed_tokens.register_forward_hook(_inject_thought_hook)

    def encode_clause_to_thought(self, token_ids: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        self._active_prefix = None
        h = self.backbone.model(token_ids)[0]
        mask_f = valid_mask.unsqueeze(-1).type_as(h)
        pooled = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        return F.normalize(pooled, p=2, dim=-1)

    def forward_associative_step(self, thought_seq: torch.Tensor) -> tuple:
        B, T, D = thought_seq.shape
        causal_mask = torch.triu(torch.ones(T, T, device=thought_seq.device, dtype=torch.bool), diagonal=1)
        brain_h = self.associative_brain(thought_seq, mask=causal_mask)
        pred_next_z = F.normalize(self.next_thought_proj(brain_h), p=2, dim=-1)
        stop_logits = self.stop_gate(brain_h).squeeze(-1)
        return pred_next_z, stop_logits

    def map_thought_to_logits(self, z: torch.Tensor, style_ids: torch.Tensor, canvas_ids: torch.Tensor) -> torch.Tensor:
        B, L = canvas_ids.shape
        self._active_prefix = self.thought_to_prefix(z).reshape(B, 2, self.hidden_size)
        dummy_prefix_ids = torch.full((B, 2), BOS_ID, dtype=torch.long, device=canvas_ids.device)
        full_ids = torch.cat([dummy_prefix_ids, style_ids.unsqueeze(1), canvas_ids], dim=1)
        logits = self.backbone(full_ids).logits[:, 3:, :].contiguous()
        self._active_prefix = None
        return logits

    @torch.inference_mode()
    def think_and_speak(self, initial_thought_z: torch.Tensor, intent_z: torch.Tensor = None,
                        style_id: int = STYLE_DETAILED_ID, max_hops: int = 5, render_steps: int = 2):
        device = initial_thought_z.device
        thoughts = [initial_thought_z]
        if intent_z is not None:
            curr_seq = torch.stack([F.normalize(initial_thought_z + 0.25 * intent_z, p=2, dim=-1)], dim=1)
        else:
            curr_seq = initial_thought_z.unsqueeze(1)

        for hop in range(max_hops - 1):
            pred_z_seq, stop_logits = self.forward_associative_step(curr_seq)
            next_z = pred_z_seq[:, -1, :]
            thoughts.append(next_z)
            if torch.sigmoid(stop_logits[:, -1]).item() > 0.65 and hop >= 1:
                break
            curr_seq = torch.cat([curr_seq, next_z.unsqueeze(1)], dim=1)

        all_z = torch.cat(thoughts, dim=0)
        K = all_z.size(0)
        L = self.max_clause_slots
        canvas = torch.full((K, L), MASK_ID, dtype=torch.long, device=device)
        styles = torch.full((K,), style_id, dtype=torch.long, device=device)

        for s in range(render_steps):
            logits = self.map_thought_to_logits(all_z, styles, canvas)
            conf, pred = F.softmax(logits, dim=-1).max(dim=-1)
            still_masked = (canvas == MASK_ID)
            n_rem = still_masked[0].sum().item()
            if n_rem == 0: break
            conf = conf.masked_fill(~still_masked, -1.0)
            k_unmask = n_rem if s == render_steps - 1 else max(1, int(round(L / render_steps)))
            _, top_idx = torch.topk(conf, k=min(k_unmask, n_rem), dim=-1)
            for b in range(K):
                for idx in top_idx[b]:
                    if still_masked[b, idx]:
                        canvas[b, idx] = pred[b, idx]

        return all_z, canvas

thought_chains = [
    ("物理光學", [
        ("因为太阳光进入大气层时，", "太阳光进大气层，"),
        ("蓝光的波长比较短，", "蓝光波长较短，"),
        ("容易被空气分子散射，", "易被空气散射，"),
        ("所以我们看到的天空是蓝色的。", "故天空呈蓝色。")
    ]),
    ("人工智慧", [
        ("机器学习是人工智能的核心，", "机器学习属AI核心，"),
        ("通过算法从海量数据中学习规律，", "靠算法从数据学规律，"),
        ("并自动做出准确的预测。", "自动做出精准预测。")
    ]),
    ("閱讀知識", [
        ("周末下午我特地出门，", "周末下午我出门，"),
        ("去了一趟安静的图书馆，", "去了安静的图书馆，"),
        ("挑了两本科幻小说阅读。", "挑了两本小说阅读。")
    ]),
    ("健康醫療", [
        ("周末下午我特地出门，", "周末下午我出门，"),
        ("去了一趟专业的牙科诊所，", "去了专业牙科诊所，"),
        ("请医生检查并清理了牙结石。", "请医生清理牙结石。")
    ]),
    ("美食早餐", [
        ("周末下午我特地出门，", "周末下午我出门，"),
        ("去了一趟热闹的生鲜超市，", "去了热闹生鲜超市，"),
        ("买了新鲜苹果和牛奶做早餐。", "买苹果牛奶做早餐。")
    ])
]

MAX_SLOTS = 12
def pad_clause(txt):
    ids = tokenizer(txt, add_special_tokens=False).input_ids[:MAX_SLOTS]
    valid = [1] * len(ids) + [0] * (MAX_SLOTS - len(ids))
    padded = ids + [SLOT_PAD_ID] * (MAX_SLOTS - len(ids))
    return padded, valid

cfg = MiniMindConfig(hidden_size=512, num_hidden_layers=8, use_moe=False)
torch.manual_seed(42)
model = AssociativeThoughtMiniMind(cfg, max_clause_slots=MAX_SLOTS).cuda().bfloat16()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

chain_data = []
for intent_name, nodes in thought_chains:
    intent_ids, intent_val = pad_clause(intent_name)
    intent_t = torch.tensor([intent_ids], device='cuda')
    intent_v = torch.tensor([intent_val], device='cuda')
    node_tensors = []
    for det_txt, con_txt in nodes:
        d_ids, d_val = pad_clause(det_txt)
        c_ids, c_val = pad_clause(con_txt)
        node_tensors.append((
            torch.tensor([d_ids], device='cuda'), torch.tensor([d_val], device='cuda'),
            torch.tensor([c_ids], device='cuda'), torch.tensor([c_val], device='cuda'),
            det_txt, con_txt
        ))
    chain_data.append((intent_name, intent_t, intent_v, node_tensors))

print("=== [1] Training Associative-Thought MiniMind (Latent Thought Hopping + Dual-Style Parallel Mapping) ===", flush=True)
STEPS = 250
model.train()
for step in range(1, STEPS + 1):
    lr = 3e-5 + 0.5 * (3e-4 - 3e-5) * (1 + math.cos(math.pi * step / STEPS))
    for pg in opt.param_groups: pg['lr'] = lr
    opt.zero_grad()

    total_assoc_loss = 0.0
    total_map_loss = 0.0
    total_align_loss = 0.0

    batch_z, batch_style, batch_canvas, batch_lbl = [], [], [], []

    for intent_name, intent_t, intent_v, node_tensors in chain_data:
        intent_z = model.encode_clause_to_thought(intent_t, intent_v)
        z_list = []
        for d_t, d_v, c_t, c_v, _, _ in node_tensors:
            z_d = model.encode_clause_to_thought(d_t, d_v)
            z_c = model.encode_clause_to_thought(c_t, c_v)
            total_align_loss = total_align_loss + (1.0 - (z_d * z_c).sum())
            z_mean = F.normalize(z_d + z_c, p=2, dim=-1)
            z_list.append(z_mean)

            for style_tok, tgt_t in ((STYLE_DETAILED_ID, d_t[0]), (STYLE_CONCISE_ID, c_t[0])):
                for mask_prob in (1.0, 0.5):
                    m = (torch.rand(MAX_SLOTS, device='cuda') < mask_prob)
                    if not m.any(): m[0] = True
                    canv = tgt_t.clone(); canv[m] = MASK_ID
                    lbl = tgt_t.clone(); lbl[~m] = -100
                    batch_z.append(z_mean[0])
                    batch_style.append(style_tok)
                    batch_canvas.append(canv)
                    batch_lbl.append(lbl)

        z_seq = torch.cat(z_list, dim=0).unsqueeze(0)
        z_in = z_seq.clone()
        z_in[:, 0, :] = F.normalize(z_in[:, 0, :] + 0.25 * intent_z, p=2, dim=-1)
        pred_next_z, stop_logits = model.forward_associative_step(z_in[:, :-1, :])
        target_next_z = z_seq[:, 1:, :].detach()
        cos_loss = (1.0 - (pred_next_z * target_next_z).sum(dim=-1)).mean()

        stop_tgt = torch.zeros_like(stop_logits)
        stop_tgt[:, -1] = 1.0
        gate_loss = F.binary_cross_entropy_with_logits(stop_logits, stop_tgt)
        total_assoc_loss = total_assoc_loss + cos_loss + 0.5 * gate_loss

    bz = torch.stack(batch_z, dim=0)
    bs = torch.tensor(batch_style, dtype=torch.long, device='cuda')
    bc = torch.stack(batch_canvas, dim=0)
    bl = torch.stack(batch_lbl, dim=0)
    logits = model.map_thought_to_logits(bz, bs, bc)
    total_map_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), bl.reshape(-1), ignore_index=-100)

    loss = total_assoc_loss + total_map_loss + 0.5 * total_align_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if step in (1, 80, 160, 250):
        print(f"  Step {step:3d}/{STEPS} | Thought-Hop Loss: {total_assoc_loss.item():.4f} | Language-Map Loss: {total_map_loss.item():.4f} | Concept-Align Loss: {total_align_loss.item():.4f}", flush=True)

model.eval()

def clean_clause(t_ids):
    c = [tid for tid in t_ids.tolist() if tid not in (MASK_ID, SLOT_PAD_ID, BOS_ID)]
    return tokenizer.decode(c)

print("\n=== [Benchmark 1] Pure Mental Association Hopping (Thinking in 512-D Vectors, NOT Words!) ===", flush=True)
_, intent_t0, intent_v0, nodes0 = chain_data[0]
z1_sky = model.encode_clause_to_thought(nodes0[0][0], nodes0[0][1])
intent_z0 = model.encode_clause_to_thought(intent_t0, intent_v0)
gt_sky_zs = [model.encode_clause_to_thought(n[0], n[1]) for n in nodes0]

all_z_det, canvas_det = model.think_and_speak(z1_sky, intent_z=intent_z0, style_id=STYLE_DETAILED_ID, max_hops=5, render_steps=2)
_, canvas_con = model.think_and_speak(z1_sky, intent_z=intent_z0, style_id=STYLE_CONCISE_ID, max_hops=5, render_steps=2)

print("Input Seed Thought z_1: [太阳光进入大气层] (1 single 512-D vector!)", flush=True)
for idx in range(all_z_det.size(0)):
    sim = F.cosine_similarity(all_z_det[idx:idx+1], gt_sky_zs[idx]).item()
    hop_tag = "Seed z_1  " if idx == 0 else f"Hop {idx} -> z_{idx+1}"
    print(f"  {hop_tag} (Cosine Sim to GT Concept = {sim*100:5.1f}%):", flush=True)
    print(f"     -> [自然映射 Detailed]: {clean_clause(canvas_det[idx])}", flush=True)
    print(f"     -> [換句話說 Concise ]: {clean_clause(canvas_con[idx])}", flush=True)

print("\n=== [Benchmark 2] Associative Branching: Same Opening Thought + Different Mental Intent -> Different Association Chains! ===", flush=True)
for chain_idx in (2, 3, 4):
    intent_name, intent_t, intent_v, nodes = chain_data[chain_idx]
    z1_wknd = model.encode_clause_to_thought(nodes[0][0], nodes[0][1])
    intent_z = model.encode_clause_to_thought(intent_t, intent_v)
    _, canv = model.think_and_speak(z1_wknd, intent_z=intent_z, style_id=STYLE_DETAILED_ID, max_hops=4, render_steps=2)
    clauses = [clean_clause(canv[k]) for k in range(canv.size(0))]
    print(f"  [心念意圖: {intent_name}] Seed: '{clauses[0]}'  ==>  聯想第1跳: '{clauses[1]}'  ==>  聯想第2跳: '{clauses[2]}'", flush=True)

print("\n=== [Benchmark 3] Paradigm Speed Comparison: Token-by-Token AR (36 steps) vs Thought-Hopping + Parallel Mapping (3 Hops + 2 Map Steps) ===", flush=True)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    _ = model.think_and_speak(z1_sky, intent_z=intent_z0, style_id=STYLE_DETAILED_ID, max_hops=4, render_steps=2)
torch.cuda.synchronize()
t_thought = (time.time() - t0) / 20.0 * 1000.0

ar_inp = torch.tensor([[BOS_ID] + tokenizer("因为太阳光", add_special_tokens=False).input_ids], device='cuda')
_ = model.backbone.generate(ar_inp, max_new_tokens=36, do_sample=False, use_cache=True, eos_token_id=None)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    _ = model.backbone.generate(ar_inp, max_new_tokens=36, do_sample=False, use_cache=True, eos_token_id=None)
torch.cuda.synchronize()
t_ar = (time.time() - t0) / 20.0 * 1000.0

print(f"  Traditional AR (36 Token-by-Token Steps)            : {t_ar:6.2f} ms (36 Backbone Forward Passes)", flush=True)
print(f"  Associative Thought (3 Vector Hops + 2 Map Passes)  : {t_thought:6.2f} ms (Only 2 Backbone Forward Passes!) -> {t_ar/t_thought:.2f}x Faster!", flush=True)

torch.save(model.state_dict(), '/content/minimind/out/associative_thought_minimind_512.pth')
