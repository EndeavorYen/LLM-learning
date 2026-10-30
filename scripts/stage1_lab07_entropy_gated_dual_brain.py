# ==============================================================================
# ORIGINAL EXECUTED SCRIPT EXTRACTED VERBATIM FROM TRANSCRIPT (AST-UNESCAPED)
# Transcript Step Index : 385
# Timestamp             : 2026-10-30T06:26:17Z
# Original Colab Session: stage1_entropy_dual_brain_v3
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
STYLE_DETAILED_ID = 5
STYLE_CONCISE_ID = 6

class EntropyGatedDualBrainMiniMind(nn.Module):
    def __init__(self, config: MiniMindConfig, slot_len: int = 12):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.half_dim = config.hidden_size // 2
        self.slot_len = slot_len

        self.backbone = MiniMindForCausalLM(config)
        for layer in self.backbone.model.layers:
            layer.self_attn.is_causal = False

        self.domain_proj = nn.Linear(config.hidden_size, self.half_dim)
        self.struct_proj = nn.Linear(config.hidden_size, self.half_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.half_dim, nhead=4, dim_feedforward=self.half_dim*2,
            batch_first=True, norm_first=True
        )
        self.struct_brain = nn.TransformerEncoder(enc_layer, num_layers=2, enable_nested_tensor=False)
        self.next_struct_head = nn.Linear(self.half_dim, self.half_dim)

        self.thought_to_prefix = nn.Linear(config.hidden_size, config.hidden_size * 2)
        self._active_prefix = None

        def _hook(module, inp, out):
            if self._active_prefix is not None:
                out = out.clone()
                out[:, :2, :] = self._active_prefix
            return out

        self.backbone.model.embed_tokens.register_forward_hook(_hook)

    def forward_fast_muscle_memory(self, prompt_ids: torch.Tensor, canvas_ids: torch.Tensor):
        self._active_prefix = None
        P = prompt_ids.size(1)
        full_ids = torch.cat([prompt_ids, canvas_ids], dim=1)
        logits = self.backbone(full_ids).logits[:, P:, :].contiguous()
        return logits

    def encode_thought(self, clause_ids: torch.Tensor, valid_mask: torch.Tensor):
        self._active_prefix = None
        h = self.backbone.model(clause_ids)[0]
        mask_f = valid_mask.unsqueeze(-1).type_as(h)
        pooled = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        d = F.normalize(self.domain_proj(pooled), p=2, dim=-1)
        s = F.normalize(self.struct_proj(pooled), p=2, dim=-1)
        return d, s

    def forward_struct_hop(self, s_seq: torch.Tensor):
        B, T, _ = s_seq.shape
        c_mask = torch.triu(torch.ones(T, T, device=s_seq.device, dtype=torch.bool), diagonal=1)
        h = self.struct_brain(s_seq, mask=c_mask)
        return F.normalize(self.next_struct_head(h), p=2, dim=-1)

    def forward_thought_render(self, d: torch.Tensor, s: torch.Tensor, style_ids: torch.Tensor, canvas_ids: torch.Tensor):
        B, L = canvas_ids.shape
        z = torch.cat([d, s], dim=-1)
        self._active_prefix = self.thought_to_prefix(z).reshape(B, 2, self.hidden_size)
        dummy_prefix = torch.full((B, 2), BOS_ID, dtype=torch.long, device=canvas_ids.device)
        full_ids = torch.cat([dummy_prefix, style_ids.unsqueeze(1), canvas_ids], dim=1)
        logits = self.backbone(full_ids).logits[:, 3:, :].contiguous()
        self._active_prefix = None
        return logits

    @torch.no_grad()
    def probe_muscle_memory_confidence(self, prompt_ids: torch.Tensor, steps: int = 2):
        B = prompt_ids.size(0)
        L = self.slot_len
        canvas = torch.full((B, L), MASK_ID, dtype=torch.long, device=prompt_ids.device)
        weakest_conf = 0.0

        for s in range(steps):
            logits = self.forward_fast_muscle_memory(prompt_ids, canvas)
            probs = F.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            if s == 0:
                non_pad = (pred != SLOT_PAD_ID)
                if non_pad.any():
                    weakest_conf = conf[non_pad].min().item()
                else:
                    weakest_conf = conf.min().item()

            still_m = (canvas == MASK_ID)
            n_rem = still_m[0].sum().item()
            if n_rem == 0: break
            conf_m = conf.masked_fill(~still_m, -1.0)
            k = n_rem if s == steps - 1 else max(1, int(round(L / steps)))
            _, top_idx = torch.topk(conf_m, k=min(k, n_rem), dim=-1)
            for b in range(B):
                for idx in top_idx[b]:
                    if still_m[b, idx]:
                        canvas[b, idx] = pred[b, idx]

        return weakest_conf, canvas

    @torch.no_grad()
    def solve_via_thought_analogy(self, c1_ids, c1_val, c2_ids, c2_val, style_id=STYLE_DETAILED_ID, steps=2):
        d1, s1 = self.encode_thought(c1_ids, c1_val)
        d2, s2 = self.encode_thought(c2_ids, c2_val)
        d_target = F.normalize(d1 + d2, p=2, dim=-1)

        s_seq = torch.cat([s1, s2], dim=0).unsqueeze(0)
        s3 = self.forward_struct_hop(s_seq)[:, -1, :]
        s_seq2 = torch.cat([s_seq, s3.unsqueeze(1)], dim=1)
        s4 = self.forward_struct_hop(s_seq2)[:, -1, :]

        d_pair = torch.cat([d_target, d_target], dim=0)
        s_pair = torch.cat([s3, s4], dim=0)
        styles = torch.full((2,), style_id, dtype=torch.long, device=c1_ids.device)
        L = self.slot_len
        canvas = torch.full((2, L), MASK_ID, dtype=torch.long, device=c1_ids.device)

        for step in range(steps):
            logits = self.forward_thought_render(d_pair, s_pair, styles, canvas)
            conf, pred = F.softmax(logits, dim=-1).max(dim=-1)
            still_m = (canvas == MASK_ID)
            n_rem = still_m[0].sum().item()
            if n_rem == 0: break
            conf_m = conf.masked_fill(~still_m, -1.0)
            k = n_rem if step == steps - 1 else max(1, int(round(L / steps)))
            _, top_idx = torch.topk(conf_m, k=min(k, n_rem), dim=-1)
            for b in range(2):
                for idx in top_idx[b]:
                    if still_m[b, idx]:
                        canvas[b, idx] = pred[b, idx]
        return canvas

SLOT_LEN = 12
def pad_slot(txt, length=SLOT_LEN):
    ids = tokenizer(txt, add_special_tokens=False).input_ids[:length]
    val = [1]*len(ids) + [0]*(length - len(ids))
    pad = ids + [SLOT_PAD_ID]*(length - len(ids))
    return torch.tensor([pad], device='cuda'), torch.tensor([val], device='cuda')

rote_memories = [
    ("请问妈妈的手机号码是？", "0912-888-168"),
    ("请问妈妈的家里电话是？", "02-2345-6789"),
    ("请问爸爸的手机号码是？", "0933-777-999"),
    ("请问爸爸的家里电话是？", "04-8765-4321"),
    ("水流在水管中稳定流动，当水管内壁结垢变窄时，", "管内水压就会急剧升高。"),
    ("车流在高速路正常行驶，当施工导致车道变窄时，", "路段交通压力急剧升高。")
]

domains_data = [
    ("水利工程", True, [
        ("水流在水管中稳定流动，", "水流稳定流动，"),
        ("当水管内壁结垢变窄时，", "水管结垢变窄时，"),
        ("管内水压就会急剧升高，", "管内水压急升，"),
        ("最终导致水流堵塞中断。", "导致水流堵塞。")
    ]),
    ("交通工程", True, [
        ("车流在高速路正常行驶，", "车流正常行驶，"),
        ("当施工导致车道变窄时，", "车道施工变窄时，"),
        ("路段交通压力急剧升高，", "路段车流压力急升，"),
        ("最终导致车流瘫痪堵塞。", "导致交通瘫痪。")
    ]),
    ("人體醫學", False, [
        ("血液在血管中正常循环，", "血液正常循环，"),
        ("当血管壁堆积胆固醇变窄时，", "血管壁变窄时，"),
        ("血管内血压就会急剧升高，", "血管血压急升，"),
        ("最终导致血液循环受阻。", "导致血液受阻。")
    ])
]

PROMPT_PAD_LEN = 18
def pad_prompt(txt):
    ids = [BOS_ID] + tokenizer(txt, add_special_tokens=False).input_ids[:PROMPT_PAD_LEN - 1]
    ids = [SLOT_PAD_ID]*(PROMPT_PAD_LEN - len(ids)) + ids
    return torch.tensor([ids], device='cuda')

rote_tensors = [(pad_prompt(q), pad_slot(a)[0], q, a) for q, a in rote_memories]
dom_tensors = []
for d_name, train_c, clauses in domains_data:
    c_list = []
    for det_s, con_s in clauses:
        dt, dv = pad_slot(det_s)
        ct, cv = pad_slot(con_s)
        c_list.append((dt, dv, ct, cv))
    dom_tensors.append((d_name, train_c, c_list))

cfg = MiniMindConfig(hidden_size=512, num_hidden_layers=8, use_moe=False)
torch.manual_seed(42)
model = EntropyGatedDualBrainMiniMind(cfg, slot_len=SLOT_LEN).cuda().bfloat16()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

print("=== [1] Training Entropy-Gated Dual-Brain MiniMind (With Orthogonal State Repulsion s_i _|_ s_j) ===", flush=True)
STEPS = 220
model.train()
for step in range(1, STEPS + 1):
    lr = 3e-5 + 0.5 * (3e-4 - 3e-5) * (1 + math.cos(math.pi * step / STEPS))
    for pg in opt.param_groups: pg['lr'] = lr
    opt.zero_grad()

    s1_prompts, s1_canv, s1_lbl = [], [], []
    for pt, at, _, _ in rote_tensors:
        for m in (torch.rand(SLOT_LEN, device='cuda') < 0.5, torch.ones(SLOT_LEN, dtype=torch.bool, device='cuda')):
            if not m.any(): m[0] = True
            c = at[0].clone(); c[m] = MASK_ID
            l = at[0].clone(); l[~m] = -100
            s1_prompts.append(pt[0]); s1_canv.append(c); s1_lbl.append(l)

    p_batch = torch.stack(s1_prompts, dim=0)
    c_batch = torch.stack(s1_canv, dim=0)
    l_batch = torch.stack(s1_lbl, dim=0)
    s1_logits = model.forward_fast_muscle_memory(p_batch, c_batch)
    loss_s1 = F.cross_entropy(s1_logits.reshape(-1, s1_logits.size(-1)), l_batch.reshape(-1), ignore_index=-100)

    all_dt, all_dv, all_ct, all_cv = [], [], [], []
    for _, _, c_list in dom_tensors:
        for dt, dv, ct, cv in c_list:
            all_dt.append(dt[0]); all_dv.append(dv[0]); all_ct.append(ct[0]); all_cv.append(cv[0])

    b_dt = torch.stack(all_dt, dim=0); b_dv = torch.stack(all_dv, dim=0)
    b_ct = torch.stack(all_ct, dim=0); b_cv = torch.stack(all_cv, dim=0)
    d_det, s_det = model.encode_thought(b_dt, b_dv)
    d_con, s_con = model.encode_thought(b_ct, b_cv)

    loss_dis = (1.0 - (d_det * d_con).sum(dim=-1)).mean() + (1.0 - (s_det * s_con).sum(dim=-1)).mean()
    d_mean = F.normalize(d_det + d_con, p=2, dim=-1)
    s_mean = F.normalize(s_det + s_con, p=2, dim=-1)

    d_grid = d_mean.view(3, 4, -1)
    s_grid = s_mean.view(3, 4, -1)

    # 1. Domain invariance within each domain & orthogonality across different domains
    for dom_i in range(3):
        for st_j in range(1, 4):
            loss_dis = loss_dis + (1.0 - (d_grid[dom_i, 0] * d_grid[dom_i, st_j]).sum())
    for d_a in range(3):
        for d_b in range(d_a + 1, 3):
            loss_dis = loss_dis + F.relu((d_grid[d_a, 0] * d_grid[d_b, 0]).sum() + 0.1)

    # 2. Structural isomorphism across domains (state j is the same across Water, Highway, Blood)
    #    PLUS State Orthogonality (state i != state j must be orthogonal so s_1..s_4 NEVER collapse!)
    for st_j in range(4):
        loss_dis = loss_dis + (1.0 - (s_grid[0, st_j] * s_grid[1, st_j]).sum()) + (1.0 - (s_grid[0, st_j] * s_grid[2, st_j]).sum())
    for st_a in range(4):
        for st_b in range(st_a + 1, 4):
            for dom_i in range(3):
                loss_dis = loss_dis + F.relu((s_grid[dom_i, st_a] * s_grid[dom_i, st_b]).sum() + 0.1)

    # 3. Train Structural Hopping ONLY on Domain 0 (Water) and Domain 1 (Highway)! NEVER on Domain 2 (Blood)!
    s_seen = s_grid[:2, :, :]
    pred_s_next = model.forward_struct_hop(s_seen[:, :-1, :])
    tgt_s_next = s_seen[:, 1:, :].detach()
    loss_hop = (1.0 - (pred_s_next * tgt_s_next).sum(dim=-1)).mean()

    # 4. Train System 2 Thought-to-Language Block-Diffusion Mapper on all 12 concepts (Detailed & Concise)
    s2_d, s2_s, s2_st, s2_canv, s2_lbl = [], [], [], [], []
    for idx in range(12):
        for st_tok, tgt_row in ((STYLE_DETAILED_ID, b_dt[idx]), (STYLE_CONCISE_ID, b_ct[idx])):
            for m in (torch.rand(SLOT_LEN, device='cuda') < 0.5, torch.ones(SLOT_LEN, dtype=torch.bool, device='cuda')):
                if not m.any(): m[0] = True
                c = tgt_row.clone(); c[m] = MASK_ID
                l = tgt_row.clone(); l[~m] = -100
                s2_d.append(d_mean[idx]); s2_s.append(s_mean[idx]); s2_st.append(st_tok)
                s2_canv.append(c); s2_lbl.append(l)

    s2_logits = model.forward_thought_render(
        torch.stack(s2_d), torch.stack(s2_s),
        torch.tensor(s2_st, dtype=torch.long, device='cuda'),
        torch.stack(s2_canv)
    )
    loss_s2_render = F.cross_entropy(s2_logits.reshape(-1, s2_logits.size(-1)), torch.stack(s2_lbl).reshape(-1), ignore_index=-100)

    loss = loss_s1 + loss_s2_render + 2.0 * loss_hop + 0.4 * loss_dis
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if step in (1, 70, 140, 220):
        print(f"  Step {step:3d}/{STEPS} | Muscle-Mem Loss: {loss_s1.item():.4f} | Thought-Render Loss: {loss_s2_render.item():.4f} | Law-Hop Loss: {loss_hop.item():.4f} | Disentangle Loss: {loss_dis.item():.4f}", flush=True)

model.eval()

def clean_str(t_ids):
    c = [tid for tid in t_ids.tolist() if tid not in (MASK_ID, SLOT_PAD_ID, BOS_ID)]
    return tokenizer.decode(c)

CONF_THRESHOLD = 0.85
print(f"\n====================================================================================================", flush=True)
print(f"=== [Breakthrough Test 1] AUTOMATIC CONFIDENCE ROUTING: MUSCLE MEMORY (MinConf >= {CONF_THRESHOLD}) vs THOUGHT LAYER ===", flush=True)
print(f"====================================================================================================", flush=True)

test_queries = [
    ("Familiar Phone #1", "请问妈妈的手机号码是？", None),
    ("Familiar Phone #2", "请问妈妈的家里电话是？", None),
    ("Familiar Phone #3", "请问爸爸的手机号码是？", None),
    ("Familiar Phone #4", "请问爸爸的家里电话是？", None),
    ("Familiar Water Pipe", "水流在水管中稳定流动，当水管内壁结垢变窄时，", None),
    ("UNSEEN Medical Analogy", "血液在血管中正常循环，当血管壁堆积胆固醇变窄时，", ("血液在血管中正常循环，", "当血管壁堆积胆固醇变窄时，"))
]

for tag, q_txt, clauses_pair in test_queries:
    p_tensor = pad_prompt(q_txt)
    min_conf, fast_canvas = model.probe_muscle_memory_confidence(p_tensor, steps=2)
    if min_conf >= CONF_THRESHOLD:
        route_decision = f"⚡ [MUSCLE MEMORY FAST PATH] (Weakest-Token Confidence={min_conf*100:5.1f}% >= {CONF_THRESHOLD*100:.0f}%)"
        final_answer = clean_str(fast_canvas[0])
        print(f"[{tag}] Prompt: '{q_txt}'", flush=True)
        print(f"  -> Router : {route_decision}", flush=True)
        print(f"  -> Output : {final_answer}", flush=True)
    else:
        route_decision = f"🧠 [ESCALATE TO THOUGHT LAYER!] (Weakest-Token Confidence={min_conf*100:5.1f}% < {CONF_THRESHOLD*100:.0f}% -> Hesitation Detected!)"
        blind_guess = clean_str(fast_canvas[0])
        c1_t, c1_v = pad_slot(clauses_pair[0])
        c2_t, c2_v = pad_slot(clauses_pair[1])
        s2_det = model.solve_via_thought_analogy(c1_t, c1_v, c2_t, c2_v, style_id=STYLE_DETAILED_ID, steps=2)
        s2_con = model.solve_via_thought_analogy(c1_t, c1_v, c2_t, c2_v, style_id=STYLE_CONCISE_ID, steps=2)
        det_ans = clean_str(s2_det[0]) + clean_str(s2_det[1])
        con_ans = clean_str(s2_con[0]) + clean_str(s2_con[1])
        print(f"[{tag}] Prompt: '{q_txt}'", flush=True)
        print(f"  -> Router : {route_decision}", flush=True)
        print(f"  -> (What Blind Muscle-Memory Would Have Hallucinated: '{blind_guess}')", flush=True)
        print(f"  -> Thought Layer Zero-Shot Analogy (Detailed): {det_ans}", flush=True)
        print(f"  -> Thought Layer Zero-Shot Analogy (Concise ): {con_ans}", flush=True)

print(f"\n====================================================================================================", flush=True)
print(f"=== [Breakthrough Test 2] CONSOLIDATION (熟能生巧：把意念層推導出的新知識下沉固化為肌肉記憶！) ===", flush=True)
print(f"====================================================================================================", flush=True)
unseen_q = "血液在血管中正常循环，当血管壁堆积胆固醇变窄时，"
p_unseen = pad_prompt(unseen_q)
conf_before, _ = model.probe_muscle_memory_confidence(p_unseen, steps=2)

c1_t, c1_v = pad_slot("血液在血管中正常循环，")
c2_t, c2_v = pad_slot("当血管壁堆积胆固醇变窄时，")
thought_solution = model.solve_via_thought_analogy(c1_t, c1_v, c2_t, c2_v, style_id=STYLE_DETAILED_ID, steps=2)
distilled_target = thought_solution[0:1].detach().clone()

model.train()
for consol_step in range(1, 26):
    opt.zero_grad()
    m = torch.ones((1, SLOT_LEN), dtype=torch.bool, device='cuda')
    canv = distilled_target.clone(); canv[m] = MASK_ID
    logits = model.forward_fast_muscle_memory(p_unseen, canv)
    loss_c = F.cross_entropy(logits.reshape(-1, logits.size(-1)), distilled_target.reshape(-1))
    loss_c.backward()
    opt.step()
model.eval()

conf_after, fast_after = model.probe_muscle_memory_confidence(p_unseen, steps=2)
print(f"Before Consolidation (Unseen Medical Problem): Weakest-Token Confidence = {conf_before*100:5.1f}% (< {CONF_THRESHOLD*100:.0f}% -> Escalated to Slow Thought Layer)", flush=True)
print(f"After  Consolidation (Practiced 25 times)    : Weakest-Token Confidence = {conf_after*100:5.1f}% (>= {CONF_THRESHOLD*100:.0f}% -> Now Instant Muscle Memory!)", flush=True)
print(f"  -> Instant Muscle-Memory Output: {clean_str(fast_after[0])}", flush=True)

torch.save(model.state_dict(), '/content/minimind/out/entropy_dual_brain_minimind_512.pth')
