# ==============================================================================
# Stage 2 / Lab 10: 7B answerability gate for the association test v4 (issue #21)
# Pre-registered gate: Qwen2.5-7B-Instruct sees only (domain, stage 0, stage 1) and picks which of the two
# branch sentences is the stage-k continuation; correct only if right under BOTH option orders.
# Bars: k = 2 >= 85%, k = 3 and 4 >= 80% (48 items each).
# Extra validity check (not a bar): the same question with stage 0 removed (stage 1 only) should be near 50%,
# because paired laws share an identical stage-1 sentence.
# Prints numbers only.
# ==============================================================================
import os, time, json, random
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

random.seed(42); torch.manual_seed(42)
MODEL_ID = os.environ.get("LAB10_MODEL", "Qwen/Qwen2.5-7B-Instruct")
_cands = ["/content/analogy_v4.json", "benchmarks/analogy_v4.json"]
DATA_PATH = next(p for p in _cands if os.path.exists(p))
DATA = json.load(open(DATA_PATH, encoding="utf-8"))
PAIRS = [tuple(p) for p in DATA["pairs"]]
PAIR_OF = {l: i for i, p in enumerate(PAIRS) for l in p}
LAWS = DATA["laws"]
print(f"[lab10] data={DATA_PATH} model={MODEL_ID}", flush=True)

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda").eval()
print(f"[lab10] loaded in {time.time() - t0:.1f}s", flush=True)

LETTERS = ["A", "B"]
LETTER_IDS = []
for L in LETTERS:
    ids = set()
    for v in (L, " " + L):
        t = tok.encode(v, add_special_tokens=False)
        if len(t) != 1:
            raise RuntimeError(f"letter {v!r} is not a single token: {t}")
        ids.add(t[0])
    LETTER_IDS.append(sorted(ids))

ORD = {2: "第一步", 3: "第二步", 4: "第三步"}


def question(domain, context, k, opt_a, opt_b):
    return (f"领域：{domain}\n"
            f"已知情况：{context}\n\n"
            f"请判断：按照这个情况自然发展下去，事情接下来的{ORD[k]}，更可能是下面哪一句？\n"
            f"A. {opt_a.rstrip('，。')}\n"
            f"B. {opt_b.rstrip('，。')}\n\n"
            "只回答一个字母（A 或 B）。")


@torch.no_grad()
def choose(q):
    prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    ids = torch.tensor([tok(prompt).input_ids], device="cuda")
    lp = F.log_softmax(model(input_ids=ids).logits.float()[0, -1, :], dim=-1)
    sc = [max(float(lp[t]) for t in LETTER_IDS[j]) for j in range(2)]
    return 0 if sc[0] >= sc[1] else 1


def run(with_s0: bool):
    res = {k: 0 for k in (2, 3, 4)}
    per_order = {k: 0 for k in (2, 3, 4)}
    letter_a = 0
    wrong = []
    for dom in DATA["unseen_domains"]:
        for law in range(len(LAWS)):
            ch = dom["chains"][str(law)]
            partner = [l for l in PAIRS[PAIR_OF[law]] if l != law][0]
            ctx = (ch[0] + ch[1]) if with_s0 else ch[1]
            for k in (2, 3, 4):
                true_s, other_s = ch[k], dom["chains"][str(partner)][k]
                p1 = choose(question(dom["name"], ctx, k, true_s, other_s))   # correct = A
                p2 = choose(question(dom["name"], ctx, k, other_s, true_s))   # correct = B
                ok1, ok2 = p1 == 0, p2 == 1
                letter_a += (p1 == 0) + (p2 == 0)
                per_order[k] += ok1 + ok2
                res[k] += int(ok1 and ok2)
                if with_s0 and not (ok1 and ok2):
                    wrong.append((dom["name"], law, k, ok1, ok2, true_s))
    return res, per_order, letter_a, wrong


N = len(DATA["unseen_domains"]) * len(LAWS)
# env LAB10_MODES: "gate" (s0+s1), "s1only", or "both" (default) -- lets the two parts run as separate, shorter jobs
_modes = os.environ.get("LAB10_MODES", "both")
for with_s0 in [m for m, name in ((True, "gate"), (False, "s1only")) if _modes in (name, "both")]:
    t = time.time()
    res, per_order, letter_a, wrong = run(with_s0)
    tag = "s0+s1 (gate)" if with_s0 else "s1 only (validity check)"
    print(f"\n=== {tag} | {N} items per stage | correct under both orders (chance 25%) "
          f"| per-order correct /{2 * N} (chance 50%) | {time.time() - t:.0f}s ===", flush=True)
    for k in (2, 3, 4):
        print(f"  stage {k}: both-orders {res[k]}/{N} ({100 * res[k] / N:.1f}%) | per-order {per_order[k]}/{2 * N} "
              f"({100 * per_order[k] / (2 * N):.1f}%)", flush=True)
    print(f"  letter A chosen {letter_a}/{6 * N}", flush=True)
    if with_s0:
        print("  items not correct under both orders (domain, law, stage, order-AB ok, order-BA ok, sentence):", flush=True)
        for w in wrong:
            print(f"    {w}", flush=True)
    print(f"V4_GATE mode={'s0s1' if with_s0 else 's1only'} " + " ".join(f"k{k}={res[k]}" for k in (2, 3, 4))
          + " " + " ".join(f"k{k}_order={per_order[k]}" for k in (2, 3, 4)) + f" n={N}", flush=True)
