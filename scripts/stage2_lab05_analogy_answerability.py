# ==============================================================================
# Stage 2 / Lab 05: Is the analogy v2 test set answerable? (Phase 1 positive control)
# Stronger frozen models take the same 36 unseen items as Lab 04 (12 unseen domains x 3 laws):
#   (a) candidate ranking B1/B2 by mean log-prob (identical to Lab 04 eval_lm), zero-shot and few-shot;
#   (b) direct multiple choice: given the three law descriptions, which law does a sentence belong to?
#       Every item under all 3 cyclic option orders; correct = majority (>= 2 of 3 rotations).
# Models are evaluated one at a time (load, evaluate, delete). No training. Prints numbers only, NO verdict.
# Run on Colab L4:  python stage2_lab05_analogy_answerability.py
#   optional: LAB05_MODELS="Qwen/Qwen2.5-0.5B,Qwen/Qwen2.5-7B-Instruct" to override the model list
# ==============================================================================
import os, gc, time, random, json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

SEED = int(os.environ.get("AUDIT_SEED", "42"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
T_START = time.time()
print(f"[lab05] SEED={SEED} (nothing is sampled; seeds set for completeness)", flush=True)

MODELS = [m.strip() for m in os.environ.get(
    "LAB05_MODELS",
    "Qwen/Qwen2.5-0.5B,Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-7B-Instruct",
).split(",") if m.strip()]
DETAIL_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # per-item table printed for this model

# ==============================================================================
# [0] Data (same loading and item set as Lab 04)
# ==============================================================================
_DATA_CANDIDATES = ["/content/analogy_v2.json", "benchmarks/analogy_v2.json"]
try:
    _DATA_CANDIDATES.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks", "analogy_v2.json"))
except NameError:
    pass
DATA_PATH = next((p for p in _DATA_CANDIDATES if os.path.exists(p)), None)
if DATA_PATH is None:
    raise FileNotFoundError(f"analogy_v2.json not found in {_DATA_CANDIDATES}")
with open(DATA_PATH, encoding="utf-8") as f:
    DATA = json.load(f)
print(f"[lab05] data = {DATA_PATH} ({DATA.get('version')})", flush=True)

N_LAWS, N_STAGES, N_PHR = 3, 4, 2
LAWS = DATA["laws"]
assert len(LAWS) == N_LAWS and all(int(LAWS[i]["id"]) == i and len(LAWS[i]["stages"]) == N_STAGES for i in range(N_LAWS))
SEEN_DOMS = [d["name"] for d in DATA["seen_domains"]]
UNSEEN_DOMS = [d["name"] for d in DATA["unseen_domains"]]

SEEN = []  # (text, dom name, law, stage, phrasing)
for dom in DATA["seen_domains"]:
    for law in range(N_LAWS):
        cells = dom["chains"][str(law)]
        assert len(cells) == N_STAGES
        for st in range(N_STAGES):
            for ph in range(N_PHR):
                SEEN.append((cells[st][ph], dom["name"], law, st, ph))
assert len(SEEN) == 72

UNSEEN = {}  # (ui, law, stage) -> text
ITEMS = []   # (ui, law)
for ui, dom in enumerate(DATA["unseen_domains"]):
    for law in range(N_LAWS):
        chain = dom["chains"][str(law)]
        assert len(chain) == N_STAGES
        for st in range(N_STAGES):
            UNSEEN[(ui, law, st)] = chain[st]
        ITEMS.append((ui, law))
assert len(ITEMS) == 36

CUE = [(c["text"], c["domain"], int(c["law"]), int(c["stage"]), int(c["lexically_resembles"][0])) for c in DATA["cue_swap"]]
N_CUE = len(CUE)

# Few-shot prefix: identical to Lab 04 (first seen domain, phrasing 0, one chain per law)
FEWSHOT_PREFIX = "\n".join("".join(DATA["seen_domains"][0]["chains"][str(l)][st][0] for st in range(N_STAGES))
                           for l in range(N_LAWS)) + "\n"
_forbidden = set(UNSEEN.values()) | {c[0] for c in CUE}
assert not any(t in FEWSHOT_PREFIX for t in _forbidden), "few-shot prefix contains an unseen/cue-swap sentence"
print(f"[lab05] seen={len(SEEN)} | unseen items={len(ITEMS)} | cue-swap={N_CUE}\n[lab05] few-shot prefix:\n{FEWSHOT_PREFIX}", flush=True)

# ==============================================================================
# [1] MCQ prompt: three law descriptions as options, domain name + target sentence only
# ==============================================================================
LETTERS = ["A", "B", "C"]


def law_desc(l: int) -> str:
    return f"{LAWS[l]['name']}：" + " → ".join(LAWS[l]["stages"])


def mcq_question(domain: str, text: str, rot: int) -> str:
    """Option position j shows law (j + rot) % 3."""
    opts = "\n".join(f"{LETTERS[j]}. {law_desc((j + rot) % N_LAWS)}" for j in range(N_LAWS))
    return ("下面是三种因果规律，每种规律都依次经历四个阶段：\n"
            f"{opts}\n\n"
            f"领域：{domain}\n"
            f"句子：{text.rstrip('，。,.')}\n\n"
            "这个句子描述的是哪一种规律中的某个阶段？只回答一个字母（A、B 或 C）。")


print("[lab05] MCQ prompt example (rotation 0):\n" + mcq_question("<领域>", "<句子>", 0), flush=True)

# ==============================================================================
# [2] Per-model evaluation
# ==============================================================================
tokenizer = None
model = None


@torch.no_grad()
def cand_mean_logprob(prompt: str, cand: str) -> float:
    """Same as Lab 04: tokenise prompt and candidate separately and concatenate (the prompt's trailing
    '，' must not merge into the candidate); mean log-prob of the candidate tokens."""
    p_ids = tokenizer(prompt).input_ids
    c_ids = tokenizer(cand).input_ids
    f_ids = p_ids + c_ids
    n_c = len(c_ids)
    if not (1 <= n_c < len(f_ids)):
        raise RuntimeError(f"bad candidate span: {prompt!r} {cand!r} {n_c}")
    ids = torch.tensor([f_ids], device="cuda")
    lp = F.log_softmax(model(input_ids=ids).logits.float()[0, :-1, :], dim=-1)
    tok_lp = lp.gather(-1, ids[0, 1:].unsqueeze(-1))[:, 0]
    return float(tok_lp[-n_c:].mean().item())


@torch.no_grad()
def eval_lm(prefix: str, tag: str):
    """Same as Lab 04 eval_lm (raw text, no chat template). Returns B1, B2 and per-item picks."""
    b1s = b2s = 0
    picks = {}
    for (ui, law) in ITEMS:
        c = [UNSEEN[(ui, law, st)] for st in range(N_STAGES)]
        prompt = prefix + c[0] + c[1]
        cand2 = [UNSEEN[(ui, l, 2)] for l in range(N_LAWS)]
        cand3 = [UNSEEN[(ui, l, 3)] for l in range(N_LAWS)]
        lp2 = [cand_mean_logprob(prompt, t) for t in cand2]
        b1 = max(range(N_LAWS), key=lambda i: lp2[i])
        prompt3 = prompt + cand2[b1]  # B2 conditioned on the model's own B1 pick
        lp3 = [cand_mean_logprob(prompt3, t) for t in cand3]
        b2 = max(range(N_LAWS), key=lambda i: lp3[i])
        b1s += int(b1 == law); b2s += int(b2 == law)
        picks[(ui, law)] = (b1, b2)
    print(f"  [{tag}] B1={b1s}/36 B2={b2s}/36", flush=True)
    return b1s, b2s, picks


def letter_token_ids():
    """Per letter: token ids of 'A' and ' A' (each must be a single token)."""
    out = []
    for L in LETTERS:
        ids = []
        for v in (L, " " + L):
            t = tokenizer.encode(v, add_special_tokens=False)
            if len(t) != 1:
                raise RuntimeError(f"letter variant {v!r} is not a single token: {t}")
            ids.append(t[0])
        out.append(sorted(set(ids)))
    return out


@torch.no_grad()
def mcq_predict(domain: str, text: str, is_instruct: bool, letter_ids):
    """Returns predicted law for each of the 3 rotations."""
    preds = []
    for rot in range(N_LAWS):
        q = mcq_question(domain, text, rot)
        if is_instruct:
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": q}],
                                                   tokenize=False, add_generation_prompt=True)
        else:
            prompt = q + "\n答案："
        ids = torch.tensor([tokenizer(prompt).input_ids], device="cuda")
        lp = F.log_softmax(model(input_ids=ids).logits.float()[0, -1, :], dim=-1)
        scores = [max(float(lp[t].item()) for t in letter_ids[j]) for j in range(N_LAWS)]
        j = max(range(N_LAWS), key=lambda k: scores[k])
        preds.append((j + rot) % N_LAWS)
    return preds


def majority(preds, target):
    return int(sum(p == target for p in preds) >= 2)


ALL_RESULTS = []
for MODEL_ID in MODELS:
    t_model = time.time()
    is_instruct = "instruct" in MODEL_ID.lower()
    print("\n" + "=" * 100, flush=True)
    print(f"=== Model {MODEL_ID} | bf16 | eval | MCQ format: {'chat template' if is_instruct else 'plain text'} ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  loaded in {time.time()-t_model:.1f}s | layers={model.config.num_hidden_layers} | hidden={model.config.hidden_size}", flush=True)

    with torch.no_grad():
        # ---- (a) ranking ----
        t0 = time.time()
        B1z, B2z, picks_z = eval_lm("", "ranking zero-shot")
        B1f, B2f, picks_f = eval_lm(FEWSHOT_PREFIX, "ranking few-shot")
        print(f"  ranking time {time.time()-t0:.1f}s", flush=True)

        # ---- (b) MCQ ----
        t0 = time.time()
        letter_ids = letter_token_ids()
        mcq = {}  # (ui, law, stage) -> preds per rotation
        for (ui, law) in ITEMS:
            for st in (2, 3):
                mcq[(ui, law, st)] = mcq_predict(UNSEEN_DOMS[ui], UNSEEN[(ui, law, st)], is_instruct, letter_ids)
        mcq_s2 = sum(majority(mcq[(ui, law, 2)], law) for (ui, law) in ITEMS)
        mcq_s3 = sum(majority(mcq[(ui, law, 3)], law) for (ui, law) in ITEMS)
        rot_s2 = sum(p == law for (ui, law) in ITEMS for p in mcq[(ui, law, 2)])
        rot_s3 = sum(p == law for (ui, law) in ITEMS for p in mcq[(ui, law, 3)])

        cue_preds = [mcq_predict(dn, txt, is_instruct, letter_ids) for (txt, dn, law, st, lex) in CUE]
        cue_mean = sum(majority(cue_preds[i], CUE[i][2]) for i in range(N_CUE))
        cue_cue = sum(majority(cue_preds[i], CUE[i][4]) for i in range(N_CUE))

        seen_preds = [mcq_predict(dn, txt, is_instruct, letter_ids) for (txt, dn, law, st, ph) in SEEN]
        mcq_seen = sum(majority(seen_preds[i], SEEN[i][2]) for i in range(len(SEEN)))
        seen_by_stage = [sum(majority(seen_preds[i], SEEN[i][2]) for i in range(len(SEEN)) if SEEN[i][3] == st)
                         for st in range(N_STAGES)]

        # position-bias diagnostic: how often each option letter is chosen (all unseen S2/S3 queries)
        pos_count = [0, 0, 0]
        for (ui, law, st), preds in mcq.items():
            for rot, pl in enumerate(preds):
                pos_count[(pl - rot) % N_LAWS] += 1
        print(f"  MCQ time {time.time()-t0:.1f}s", flush=True)

    print(f"  MCQ unseen S2 = {mcq_s2}/36 (per-rotation {rot_s2}/108) | S3 = {mcq_s3}/36 (per-rotation {rot_s3}/108) | chance 12/36", flush=True)
    print(f"  MCQ seen = {mcq_seen}/72 | by stage S0..S3 = {seen_by_stage} (each /18)", flush=True)
    print(f"  MCQ cue-swap: meaning-correct {cue_mean}/9 | follows-cue {cue_cue}/9", flush=True)
    print(f"  MCQ letter choice counts over 216 unseen queries: A={pos_count[0]} B={pos_count[1]} C={pos_count[2]}", flush=True)
    print("  cue-swap per item (true / cue law | preds per rotation):", flush=True)
    for i, (txt, dn, law, st, lex) in enumerate(CUE):
        print(f"    [{dn} S{st}] true L{law} cue L{lex} | {['L'+str(p) for p in cue_preds[i]]} | {txt}", flush=True)

    if MODEL_ID == DETAIL_MODEL:
        print(f"\n=== {MODEL_ID} per-item detail (MCQ preds per rotation r0/r1/r2; ranking picks; * = law correct) ===", flush=True)
        print(f"{'domain':<10} {'law':>3} {'st':>2} | {'MCQ r0/r1/r2':<14} {'maj':<4} | {'rank0':<6} {'rankF':<6} | sentence", flush=True)
        for (ui, law) in ITEMS:
            for st in (2, 3):
                preds = mcq[(ui, law, st)]
                kz = picks_z[(ui, law)][0 if st == 2 else 1]
                kf = picks_f[(ui, law)][0 if st == 2 else 1]
                ok = "*" if majority(preds, law) else " "
                print(f"{UNSEEN_DOMS[ui]:<10} {'L'+str(law):>3} {'S'+str(st):>2} | "
                      f"{'/'.join('L'+str(p) for p in preds):<14} {ok:<4} | "
                      f"{'L'+str(kz)+('*' if kz == law else ' '):<6} {'L'+str(kf)+('*' if kf == law else ' '):<6} | "
                      f"{UNSEEN[(ui, law, st)]}", flush=True)
        print("  (rank0/rankF on S2 rows = B1 pick; on S3 rows = B2 pick, conditioned on the model's own B1 pick)", flush=True)
        print("  seen sentences not majority-correct:", flush=True)
        for i, (txt, dn, law, st, ph) in enumerate(SEEN):
            if not majority(seen_preds[i], law):
                print(f"    [{dn} L{law} S{st} p{ph}] {['L'+str(p) for p in seen_preds[i]]} | {txt}", flush=True)

    line = (f"ANS_RESULT model={MODEL_ID} B1_zero={B1z} B2_zero={B2z} B1_few={B1f} B2_few={B2f} "
            f"MCQ_S2={mcq_s2} MCQ_S3={mcq_s3} MCQ_seen={mcq_seen}/72 CUE_mean={cue_mean} CUE_cue={cue_cue}")
    ALL_RESULTS.append(line)
    print(line, flush=True)
    print(f"  model total time {time.time()-t_model:.1f}s", flush=True)

    del model
    model = None
    gc.collect()
    torch.cuda.empty_cache()

print("\n" + "=" * 100, flush=True)
print("=== SUMMARY (k/36 unless noted; chance 12/36; cue-swap /9; Lab 04 0.5B base reference: B1 zero 20, few 18) ===", flush=True)
for line in ALL_RESULTS:
    print(line, flush=True)
print(f"[lab05] total time {time.time()-T_START:.1f}s", flush=True)
