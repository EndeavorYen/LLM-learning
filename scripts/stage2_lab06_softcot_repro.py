# ==============================================================================
# Stage 2 / Lab 06: SoftCoT small-scale reproduction (pipeline check; plan 階段 2)
# Paper: Xu et al., "SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with
#        LLMs", ACL 2025, arXiv 2502.12134.
# Backbone  = Qwen/Qwen2.5-1.5B-Instruct (frozen, bf16)
# Assistant = Qwen/Qwen2.5-0.5B-Instruct (frozen, bf16)
# Trainable = ONE nn.Linear(896 -> 1536) projection (float32), nothing else.
# Conditions on GSM8K test: (1) Zero-Shot CoT, (2) Zero-Shot CoT-Unk, (3) SoftCoT.
# This script prints numbers only; it prints NO verdict.
# Run:  AUDIT_SEED=42 SOFTCOT_STEPS=1500 python stage2_lab06_softcot_repro.py
#       (env: AUDIT_SEED [42], SOFTCOT_STEPS [1500], SOFTCOT_EVAL_N [1319],
#             SOFTCOT_EVAL_BS [64], SOFTCOT_MICRO_BS [4])
#
# ---- Exact input layout (identical in training and inference) ---------------
# Backbone sequence (Qwen chat template, fixed system prompt):
#   <|im_start|>system\n{SYSTEM}<|im_end|>\n
#   <|im_start|>user\n{COT_INSTR}\n\nQuestion: {q}\n  [S1][S2][S3][S4]  <|im_end|>\n
#   <|im_start|>assistant\n  {target / generation}
# i.e. the N=4 soft slots sit right after the question INSIDE the user turn,
# before <|im_end|> and the assistant header. The text is built by rendering
# the chat template with a sentinel at the slot position and splitting there;
# prefix and suffix are tokenised separately (the cut is "\n" | "<|im_end|>",
# a clean token boundary) and used identically by all three conditions:
#   (1) CoT     : prefix + suffix                      (no slots)
#   (2) CoT-Unk : prefix + 4 x embed(<|endoftext|>) + suffix
#   (3) SoftCoT : prefix + 4 x proj(assistant thoughts) + suffix
# Training target (loss only here) = GSM8K solution (calculator annotations
# <<...>> stripped) + "<|im_end|>", truncated so the whole sequence <= 512.
#
# Assistant sequence (0.5B, chat template, same system prompt):
#   <|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{ASSIST_INSTR}\n\n
#   Question: {q}<|im_end|>\n<|im_start|>assistant\n [P][P][P][P]
#   with P = <|endoftext|> (id 151643, the [UNK]-style placeholder). Soft
#   thoughts = last-layer hidden states (after final norm) at the 4 P positions.
# ==============================================================================
import os, sys, time, math, random, re
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

T_START = time.time()
SEED = int(os.environ.get("AUDIT_SEED", "42"))
STEPS = int(os.environ.get("SOFTCOT_STEPS", "1500"))
EVAL_N_REQ = int(os.environ.get("SOFTCOT_EVAL_N", "1319"))
EVAL_BS = int(os.environ.get("SOFTCOT_EVAL_BS", "64"))
MICRO_BS = int(os.environ.get("SOFTCOT_MICRO_BS", "4"))
BATCH = 8
assert BATCH % MICRO_BS == 0
ACCUM = BATCH // MICRO_BS
LR, WD = 1e-5, 0.01
N_SOFT = 4
MAX_LEN = 512
MAX_NEW = 512
LOG_EVERY = 100
N_EXAMPLES = 3

BACKBONE_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ASSIST_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
COT_INSTR = "Please reason step by step, and put the final answer after ####."
ASSIST_INSTR = "Generate the key reasoning thoughts for the question."
PLACEHOLDER = "<|endoftext|>"
SENTINEL = "\u0000SOFT_SLOT\u0000"

DEV = "cuda"
print(f"[softcot] SEED={SEED} STEPS={STEPS} EVAL_N(req)={EVAL_N_REQ} batch={BATCH} "
      f"(micro {MICRO_BS} x accum {ACCUM}) lr={LR} wd={WD} N_SOFT={N_SOFT}", flush=True)
print(f"[softcot] COT_INSTR = {COT_INSTR!r}", flush=True)
print(f"[softcot] ASSIST_INSTR = {ASSIST_INSTR!r}; placeholder = {PLACEHOLDER!r}", flush=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_all(SEED)

# ==============================================================================
# [0] Models
# ==============================================================================
t0 = time.time()
tok = AutoTokenizer.from_pretrained(BACKBONE_ID)
atok = AutoTokenizer.from_pretrained(ASSIST_ID)
PH_ID = tok.convert_tokens_to_ids(PLACEHOLDER)
assert PH_ID == atok.convert_tokens_to_ids(PLACEHOLDER) == 151643, PH_ID
IM_END_ID = tok.convert_tokens_to_ids("<|im_end|>")
PAD_ID = PH_ID  # padding is always masked out (attention_mask = 0)

backbone = AutoModelForCausalLM.from_pretrained(BACKBONE_ID, torch_dtype=torch.bfloat16).to(DEV)
assistant = AutoModelForCausalLM.from_pretrained(ASSIST_ID, torch_dtype=torch.bfloat16).to(DEV)
for m in (backbone, assistant):
    m.requires_grad_(False)
assistant.eval()
H_A, H_B = assistant.config.hidden_size, backbone.config.hidden_size
assert (H_A, H_B) == (896, 1536), (H_A, H_B)
EMB = backbone.get_input_embeddings()
print(f"[softcot] models loaded in {time.time()-t0:.1f}s; hidden {H_A}->{H_B}", flush=True)

torch.manual_seed(SEED)
proj = nn.Linear(H_A, H_B).to(DEV, dtype=torch.float32)
print(f"[softcot] trainable params = {sum(p.numel() for p in proj.parameters()):,}", flush=True)

# ==============================================================================
# [1] Prompt construction
# ==============================================================================
CALC_RE = re.compile(r"<<[^>]*>>")


def backbone_parts(q):
    """Return (prefix_ids, suffix_ids) around the soft slot position."""
    user = f"{COT_INSTR}\n\nQuestion: {q.strip()}\n{SENTINEL}"
    text = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)
    pre, suf = text.split(SENTINEL)
    assert suf.startswith("<|im_end|>") and suf.endswith("<|im_start|>assistant\n"), repr(suf)
    return (tok(pre, add_special_tokens=False)["input_ids"],
            tok(suf, add_special_tokens=False)["input_ids"])


def assistant_ids(q):
    text = atok.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": f"{ASSIST_INSTR}\n\nQuestion: {q.strip()}"}],
        tokenize=False, add_generation_prompt=True)
    return atok(text, add_special_tokens=False)["input_ids"] + [PH_ID] * N_SOFT


def left_pad(seqs, pad):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, L - len(s):] = torch.tensor(s)
        mask[i, L - len(s):] = 1
    return ids, mask


@torch.no_grad()
def soft_thoughts(questions):
    """Assistant last-layer hidden states at the N placeholder positions -> (B, N, 896) fp32."""
    ids, mask = left_pad([assistant_ids(q) for q in questions], PAD_ID)
    ids, mask = ids.to(DEV), mask.to(DEV)
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    h = assistant.model(input_ids=ids, attention_mask=mask, position_ids=pos).last_hidden_state
    return h[:, -N_SOFT:, :].float()  # left padding => placeholders are the last N tokens


def splice(ids, slot_mask, soft):
    """Embed ids with the backbone table; overwrite slot positions with soft (B,N,H) if given."""
    emb = EMB(ids)
    if soft is None:
        return emb
    m = slot_mask.unsqueeze(-1).expand_as(emb)
    return emb.masked_scatter(m, soft.to(emb.dtype))


# ==============================================================================
# [2] Data
# ==============================================================================
def gold_of(ans):
    return float(ans.split("####")[-1].strip().replace(",", ""))


gsm_train = load_dataset("openai/gsm8k", "main", split="train")
gsm_test = load_dataset("openai/gsm8k", "main", split="test")
print(f"[softcot] GSM8K train={len(gsm_train)} test={len(gsm_test)}", flush=True)

t0 = time.time()
TRAIN = []  # (question, full_ids, n_ctx_before_target, slot_start)
n_trunc, n_lost_answer = 0, 0
for ex in gsm_train:
    pre, suf = backbone_parts(ex["question"])
    tgt_text = CALC_RE.sub("", ex["answer"]).strip()
    tgt = tok(tgt_text, add_special_tokens=False)["input_ids"] + [IM_END_ID]
    ctx = len(pre) + N_SOFT + len(suf)
    room = MAX_LEN - ctx
    if len(tgt) > room:
        n_trunc += 1
        tgt = tgt[:max(room, 0)]
        if "####" not in tok.decode(tgt):
            n_lost_answer += 1
    full = pre + [PH_ID] * N_SOFT + suf + tgt
    TRAIN.append((ex["question"], full, ctx, len(pre)))
print(f"[softcot] train sequences truncated at {MAX_LEN}: {n_trunc}/{len(TRAIN)}; "
      f"lost their '####' line: {n_lost_answer}", flush=True)
print(f"[softcot] train seq len: mean={np.mean([len(t[1]) for t in TRAIN]):.0f} "
      f"max={max(len(t[1]) for t in TRAIN)}  (tokenised in {time.time()-t0:.1f}s)", flush=True)

# ==============================================================================
# [3] Training (projection only)
# ==============================================================================
epochs_paper = 10
steps_paper = math.ceil(len(TRAIN) / BATCH) * epochs_paper
print(f"[softcot] DEVIATION: training {STEPS} optimizer steps (~{STEPS*BATCH/len(TRAIN):.2f} epochs) "
      f"instead of the paper's {epochs_paper} epochs (~{steps_paper} steps); constant LR, no warmup.",
      flush=True)

backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
backbone.config.use_cache = False
backbone.train()  # checkpointing is active only in train mode; Qwen2 dropout = 0.0

opt = torch.optim.AdamW(proj.parameters(), lr=LR, weight_decay=WD)
rng = random.Random(SEED)
order = []


def next_indices(k):
    global order
    out = []
    while len(out) < k:
        if not order:
            order = list(range(len(TRAIN)))
            rng.shuffle(order)
        out.append(order.pop())
    return out


def train_micro(idx):
    items = [TRAIN[i] for i in idx]
    L = max(len(it[1]) for it in items)
    ids = torch.full((len(items), L), PAD_ID, dtype=torch.long)
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    slot = torch.zeros_like(ids, dtype=torch.bool)
    for r, (_, full, ctx, s0) in enumerate(items):  # right padding
        ids[r, :len(full)] = torch.tensor(full)
        mask[r, :len(full)] = 1
        labels[r, ctx:len(full)] = torch.tensor(full[ctx:])
        slot[r, s0:s0 + N_SOFT] = True
    ids, mask, labels, slot = ids.to(DEV), mask.to(DEV), labels.to(DEV), slot.to(DEV)
    soft = proj(soft_thoughts([it[0] for it in items]))
    embeds = splice(ids, slot, soft)
    out = backbone(inputs_embeds=embeds, attention_mask=mask, labels=labels, use_cache=False)
    return out.loss


@torch.no_grad()
def norm_diag(tag):
    qs = [TRAIN[i][0] for i in range(16)]
    s = proj(soft_thoughts(qs))
    tok_norm = EMB.weight[:151643].float().norm(dim=-1).mean().item()
    print(f"[softcot] {tag}: mean ||proj soft|| = {s.norm(dim=-1).mean().item():.3f}  "
          f"mean ||backbone token emb|| = {tok_norm:.3f}  "
          f"||emb(placeholder)|| = {EMB.weight[PH_ID].float().norm().item():.3f}", flush=True)


norm_diag("before training")
t_train = time.time()
run_loss, run_n = 0.0, 0
for step in range(1, STEPS + 1):
    opt.zero_grad(set_to_none=True)
    for _ in range(ACCUM):
        loss = train_micro(next_indices(MICRO_BS)) / ACCUM
        loss.backward()
        run_loss += loss.item()
    run_n += 1
    opt.step()
    if step % LOG_EVERY == 0 or step == STEPS:
        el = time.time() - t_train
        print(f"[softcot] step {step:5d}/{STEPS}  loss(mean last {run_n}) = {run_loss/run_n:.4f}  "
              f"elapsed {el/60:.1f} min  ({el/step:.2f} s/step)", flush=True)
        run_loss, run_n = 0.0, 0
print(f"[softcot] training time: {(time.time()-t_train)/60:.1f} min", flush=True)
norm_diag("after training")

os.makedirs("/content", exist_ok=True)
PROJ_PATH = f"/content/softcot_proj_seed{SEED}.pt"
torch.save({"state_dict": proj.state_dict(), "seed": SEED, "steps": STEPS,
            "backbone": BACKBONE_ID, "assistant": ASSIST_ID, "n_soft": N_SOFT}, PROJ_PATH)
print(f"[softcot] saved projection -> {PROJ_PATH}", flush=True)

# ==============================================================================
# [4] Evaluation
# ==============================================================================
backbone.gradient_checkpointing_disable()
backbone.config.use_cache = True
backbone.eval()
proj.eval()

NUM_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def to_num(s):
    try:
        return float(s.replace(",", "").replace("$", ""))
    except ValueError:
        return None


def extract(text):
    """Number right after the LAST '####'; else after the last 'The answer is' / 'answer:';
    else the last number in the text."""
    if "####" in text:
        m = NUM_RE.search(text.split("####")[-1])
        if m:
            return to_num(m.group())
    low = text.lower()
    for pat in ("the answer is", "answer:"):
        k = low.rfind(pat)
        if k >= 0:
            m = NUM_RE.search(text[k + len(pat):])
            if m:
                return to_num(m.group())
    nums = NUM_RE.findall(text)
    return to_num(nums[-1]) if nums else None


def correct(pred, gold):
    return pred is not None and abs(pred - gold) < 1e-4


EVAL_N = min(EVAL_N_REQ, len(gsm_test))
EVAL = [(ex["question"], gold_of(ex["answer"])) for ex in gsm_test.select(range(EVAL_N))]
PARTS = [backbone_parts(q) for q, _ in EVAL]
print(f"[softcot] eval on GSM8K test n={EVAL_N}; greedy, max_new_tokens={MAX_NEW}, "
      f"batch {EVAL_BS}, left padding", flush=True)


@torch.no_grad()
def run_condition(cond):
    outs = []
    t = time.time()
    for b in range(0, EVAL_N, EVAL_BS):
        idx = list(range(b, min(b + EVAL_BS, EVAL_N)))
        seqs, slot_starts = [], []
        for i in idx:
            pre, suf = PARTS[i]
            if cond == "cot":
                seqs.append(pre + suf)
                slot_starts.append(None)
            else:
                seqs.append(pre + [PH_ID] * N_SOFT + suf)
                slot_starts.append(len(pre))
        ids, mask = left_pad(seqs, PAD_ID)
        slot = torch.zeros_like(ids, dtype=torch.bool)
        L = ids.shape[1]
        for r, (s, s0) in enumerate(zip(seqs, slot_starts)):
            if s0 is not None:
                off = L - len(s)
                slot[r, off + s0: off + s0 + N_SOFT] = True
        ids, mask, slot = ids.to(DEV), mask.to(DEV), slot.to(DEV)
        soft = proj(soft_thoughts([EVAL[i][0] for i in idx])) if cond == "softcot" else None
        embeds = splice(ids, slot, soft)  # cot_unk: slots keep embed(<|endoftext|>)
        gen = backbone.generate(inputs_embeds=embeds, attention_mask=mask, max_new_tokens=MAX_NEW,
                                do_sample=False, temperature=None, top_p=None, top_k=None,
                                repetition_penalty=1.0,  # override Qwen generation_config (1.1)
                                eos_token_id=[IM_END_ID, PH_ID], pad_token_id=PAD_ID)
        outs += tok.batch_decode(gen, skip_special_tokens=True)
        print(f"[softcot]   {cond}: {len(outs)}/{EVAL_N}  ({time.time()-t:.0f}s)", flush=True)
    print(f"[softcot] {cond} eval time: {(time.time()-t)/60:.1f} min", flush=True)
    return outs


RES = {}
for cond in ("cot", "cot_unk", "softcot"):
    texts = run_condition(cond)
    preds = [extract(x) for x in texts]
    ok = [correct(p, g) for p, (_, g) in zip(preds, EVAL)]
    RES[cond] = {"texts": texts, "preds": preds, "ok": ok}
    print(f"[softcot] ACC {cond:8s} = {sum(ok)}/{EVAL_N} = {sum(ok)/EVAL_N:.4f}", flush=True)


def mcnemar_exact(a_ok, b_ok):
    """Two-sided exact binomial test on discordant pairs. Returns (b, c, p)."""
    b = sum(1 for x, y in zip(a_ok, b_ok) if x and not y)
    c = sum(1 for x, y in zip(a_ok, b_ok) if y and not x)
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return b, c, min(1.0, p)


b1, c1, p_unk = mcnemar_exact(RES["softcot"]["ok"], RES["cot_unk"]["ok"])
b2, c2, p_cot = mcnemar_exact(RES["softcot"]["ok"], RES["cot"]["ok"])
print(f"[softcot] McNemar SoftCoT vs CoT-Unk: softcot-only={b1} unk-only={c1} p={p_unk:.4g}", flush=True)
print(f"[softcot] McNemar SoftCoT vs CoT    : softcot-only={b2} cot-only={c2} p={p_cot:.4g}", flush=True)
for a, b in (("softcot", "cot_unk"), ("softcot", "cot"), ("cot_unk", "cot")):
    diff = sum(1 for x, y in zip(RES[a]["texts"], RES[b]["texts"]) if x != y)
    print(f"[softcot] outputs differ {a} vs {b}: {diff}/{EVAL_N}", flush=True)
for cond in RES:
    no_hash = sum(1 for x in RES[cond]["texts"] if "####" not in x)
    print(f"[softcot] {cond}: outputs without '####' = {no_hash}/{EVAL_N}", flush=True)

for cond in RES:
    print(f"\n[softcot] ===== examples: {cond} =====", flush=True)
    for i in range(min(N_EXAMPLES, EVAL_N)):
        t = RES[cond]["texts"][i]
        t = t if len(t) <= 700 else t[:350] + " ... " + t[-300:]
        print(f"--- [{i}] gold={EVAL[i][1]} pred={RES[cond]['preds'][i]} ok={RES[cond]['ok'][i]}\n{t}",
              flush=True)

print(f"\n[softcot] total wall time: {(time.time()-T_START)/60:.1f} min", flush=True)
print(f"SOFTCOT_RESULT seed={SEED} steps={STEPS} eval_n={EVAL_N} "
      f"cot={sum(RES['cot']['ok'])} cot_unk={sum(RES['cot_unk']['ok'])} "
      f"softcot={sum(RES['softcot']['ok'])} p_vs_unk={p_unk:.6g} p_vs_cot={p_cot:.6g}", flush=True)
