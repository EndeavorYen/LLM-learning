# ==============================================================================
# Stage 2 / Lab 15: cross-domain sentence-recognition diagnostic (B1). PRE-REGISTERED on issue #25 (comment of
#   2026-11-01, before any code). Numbers only, NO verdict.
# Question: which frozen encoder ("eye") + recognizer ("judgement") maps sentences of domains it has never seen to
#   their abstract class? Main metric: leave-domains-out cross-validation INSIDE the training domains only.
# Hard rule: the 8 test domains of state_tracking_v1 and the 8 unseen domains of analogy_v5 are NOT used. Test stories
#   and `unseen_domains` are dropped right after parsing, unread. Every text is built from training-domain templates
#   only (filtered by domain BEFORE any text exists) and asserted before it reaches an encoder; the test-domain template
#   strings are used for nothing but that assertion (never extracted, trained on or printed).
# Data
#   st  benchmarks/state_tracking_v1/{stories,templates}.json (sha256 pinned). The 16 domains with split "train" in
#       templates.json (asserted == the domains of the stories' train split). Per domain: 8 init sentences = STATE PART
#       only (level[l] + "，" + safeguard[s] + "。", Lab 14 amendment 2026-11-01) and every event paraphrase (7 IN,
#       7 OUT, 6 BREAK, 6 FIX, 6 NOOP) = 40 sentences, 13 classes (Lab 14: 8 init classes level x safeguard + 5 event
#       types). Every init state part and event text of the train / val_seen stories is asserted to be in this table
#       with the same class and domain.
#   v5  benchmarks/analogy_v5.json (sha256 pinned), seen_domains only (20). 27 classes per domain (Lab 12 CLASS_KEYS /
#       cls_of): per pair (a, b): (a,0) (b,0) the shared (pair,1) and (a,k) (b,k), k = 2..4. One sentence per class.
# Folds (fixed; independent of the seed): domain names sorted, shuffled with random.Random(2026), chunked.
#   st  4 folds x 4 domains; the other 12 = pool; training domains = first k of the pool (sorted, then shuffled with
#       random.Random(2026 + fold)), k in {4, 8, 12} (nested).   v5  5 folds x 4; pool 16; k in {4, 8, 16}.
#   fold = 0-based. Held-out = every sentence of the fold's 4 domains; training = every sentence of the k domains.
# Seeds: 42, 1, 2 in ONE run (loop); they vary only recognizer init and training. Features are extracted once.
# Encoders (frozen, bf16, no gradients, float32 features; one model on the GPU at a time):
#   Q05 Qwen/Qwen2.5-0.5B | Q15 Qwen/Qwen2.5-1.5B-Instruct | Q7 Qwen/Qwen2.5-7B-Instruct, each with
#     -mean    last hidden layer, mean over the sentence tokens (Lab 12 extract_qwen_896_feature, right padding +
#              attention mask), max_length 128, no truncation (asserted)
#     -prompt  text '这句话："{s}"\n用一个词概括这句话描述的事件：' (no chat template); last hidden layer state at the LAST
#              REAL token: right padding, index = attention_mask.sum - 1 (right padding asserted). Verified against
#              an unbatched run (batch of 1) of a few sentences for every encoder (same token, cos >= 0.99 asserted).
#   BGE BAAI/bge-m3 via AutoModel: [CLS] last hidden state, L2-normalised (official dense embedding), max_length 128.
# Recognizers (trained on the training sentences of the fold / subset only; evaluated on the held-out sentences):
#   all   x = normalize(f - mu), mu = mean feature of the TRAINING sentences of that fold / subset only.
#   NEAR  normalised class means of the training x; nearest by cosine.
#   LIN   Linear(D, C), CE. AdamW lr 1e-3 -> 1e-4 cosine (Lab 01 cosine_lr), 500 steps full batch, wd 0.01, clip 1.0.
#   MLP   D -> 256 -> GELU -> C, same training.
#   FUNNEL Lab 14 funnel (D -> 256 -> GELU -> 64, normalised), Lab 12/14 s pull/push loss with the closest-pair push
#         term over the C classes, 400 steps (Lab 12 schedule: same optimiser); nearest class anchor = normalised class
#         mean of the training sentences in funnel space.
#   XDOM  projection D -> 256 -> GELU -> 64, normalised; supervised contrastive loss (tau 0.1, 500 steps, same
#         optimiser). Positives of an anchor = training sentences of the same class from a DIFFERENT domain; same-domain
#         same-class pairs (and the anchor itself) are excluded from numerator AND denominator; anchors without a
#         positive are skipped. Prediction: nearest normalised class mean in projection space.
#   Every model is built after seed_all(seed) and trained after seed_all(seed) (Lab 12 placement); clip 1.0 everywhere
#   (the Lab 12 / 14 head recipe).
# Zero-shot reference (st only, Qwen2.5-7B-Instruct with chat template, while it is loaded; all 640 train-domain
#   sentences, no training): letter options, scored by the max log-prob of the letter tokens ("X" / " X") at the next
#   position (Lab 10). Each question is asked in the original and the reversed option order; correct = right in both.
#   event: 5 options (IN, OUT, BREAK, FIX, NOOP). init: (a) level 0..3 (A-D) and (b) safeguard 正常 / 失效 (A, B);
#   correct = (a) and (b) both right in both orders; orderN = (a) and (b) both right in order N. Prompt = domain name +
#   sentence + question + options.
# Machine lines: L15_FOLD, L15_FEAT, L15_CV (+ fit counts), L15_STAGE (v5 per stage), L15_ZS / L15_ZSPART / L15_ZSCLS,
#   then L15_CONFIG as the LAST line (run complete). Aggregate: python tools/aggregate_lab15.py lab15.log [...]
# Run (Colab, one GPU >= 24 GB for the 7B model):  python stage2_lab15_recognition.py > lab15.log
#   Optional LAB15_MODELS=Q05,Q15,Q7,BGE (default: all) runs a subset of the encoder models, e.g. to split the run over
#   two sessions; the aggregator merges the logs and refuses duplicates. The zero-shot runs only when Q7 is loaded.
# Copied code is marked "(copied from Lab NN)"; everything else is new for Lab 15.
# ==============================================================================
import os, sys, gc, time, math, random, json, hashlib
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log(msg: str = ""):
    print(msg, flush=True)


# ---- pre-registered configuration (issue #25) ----------------------------------------------------------------------
SEEDS = (42, 1, 2)
FOLD_SEED = 2026
DATASETS = ("st", "v5")
N_FOLDS = {"st": 4, "v5": 5}
FOLD_SIZE = 4
KS = {"st": (4, 8, 12), "v5": (4, 8, 16)}
REG_K = {"st": 12, "v5": 16}
MODELS = {  # code -> (Hugging Face id, kind)
    "Q05": ("Qwen/Qwen2.5-0.5B", "qwen"),
    "Q15": ("Qwen/Qwen2.5-1.5B-Instruct", "qwen"),
    "Q7": ("Qwen/Qwen2.5-7B-Instruct", "qwen"),
    "BGE": ("BAAI/bge-m3", "bge"),
}
EXPECTED_DIM = {"Q05": 896, "Q15": 1536, "Q7": 3584, "BGE": 1024}
QWEN_POOLINGS = ("mean", "prompt")
ENCODERS = ["Q05-mean", "Q05-prompt", "Q15-mean", "Q15-prompt", "Q7-mean", "Q7-prompt", "BGE"]
RECOGNIZERS = ["NEAR", "LIN", "MLP", "FUNNEL", "XDOM"]
ZS_MODEL = "Q7"
SENT_MAXLEN = 128
FEAT_BATCH = 64                       # Lab 12 extraction batch size
PROMPT_TMPL = '这句话："{s}"\n用一个词概括这句话描述的事件：'
REC_LR, REC_LR_FLOOR, REC_WD, REC_CLIP = 1e-3, 1e-4, 0.01, 1.0
CE_STEPS = 500                        # LIN / MLP / XDOM
FUNNEL_STEPS = 400                    # Lab 12 schedule (HEAD_STEPS)
HIDDEN, PROJ_DIM = 256, 64
XDOM_TAU = 0.1
PUSH_MARGIN = 0.1                     # Lab 12 s push: relu(cos(class means) + 0.1)
CHECK_MIN_COS = 0.99                  # batched vs unbatched pooled feature (bf16 kernels may differ slightly)

EXPECTED_ST_SHA256 = "ed6d16bd3e625198bba74d28e7b20f6bdba9c399a3eb94b9a0d1f113f11e3108"   # #23, commit d20be68
EXPECTED_TPL_SHA256 = "1bfd769efdf6dd3d61ded0e6aaff3eb70b394af2c266e2f02cd1ed9b598bca70"  # README of state_tracking_v1
EXPECTED_V5_SHA256 = "d2a9e877595829b4ff1560be90cd04739acdb11be782b14a067c4b9c4d24f196"   # Lab 12 [review S3]

# (copied from Lab 14) state classes and sentence classes
CLASSES = ["L0_ok", "L1_ok", "L2_ok", "L3_ok", "L0_broken", "L1_broken", "L2_broken", "L3_broken", "BURST"]
CI = {c: i for i, c in enumerate(CLASSES)}
EVENTS = ("IN", "OUT", "BREAK", "FIX", "NOOP")
N_INIT = 8                         # init classes = the first 8 classes (level x safeguard); BURST is never initial
N_SCLS = N_INIT + len(EVENTS)      # 13 sentence classes
SCLS_NAMES = CLASSES[:N_INIT] + [f"ev:{e}" for e in EVENTS]

# zero-shot options (order 1 = this order; order 2 = reversed)
ZS_EVENT_Q = "这句话描述的是下面哪一种事件？"
ZS_EVENT_OPTS = ["系统里的量增加一个单位", "系统里的量减少一个单位", "安全机制坏了/失效", "安全机制修好/恢复",
                 "与系统无关的事"]                         # IN, OUT, BREAK, FIX, NOOP
ZS_LEVEL_Q = "目前系统里有几个单位？"
ZS_LEVEL_OPTS = ["0", "1", "2", "3"]
ZS_SG_Q = "安全机制目前是否正常？"
ZS_SG_OPTS = ["正常", "失效"]                               # ok, broken
LETTERS = "ABCDE"


# (copied from Lab 12)
def seed_all(seed: int):
    """(copied from Lab 09/11)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# (copied from Lab 12)
def cosine_lr(step: int, total: int, peak: float, floor: float) -> float:
    """Copied from Lab 01 (via Lab 09/11): cosine schedule from peak (step 0) to floor (step == total)."""
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * step / total))


def _script_dir_candidates(*parts):
    try:
        return [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", *parts)]
    except NameError:
        return []


def _read_first(cands, what):
    path = next((p for p in cands if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(f"{what} not found in {cands}")
    with open(path, "rb") as f:  # read bytes once: parse + sha256 of the exact file used (as Lab 12)
        raw = f.read()
    return path, raw, hashlib.sha256(raw).hexdigest()


# ==============================================================================
# [0] Data
# ==============================================================================
def load_st():
    """state_tracking_v1: the sentence table of the 16 TRAIN domains, built from templates.json; the test stories are
    dropped unread; test-domain template strings go into `forbidden` only (for the leak assertion)."""
    path, raw, sha = _read_first(["/content/stories.json", "benchmarks/state_tracking_v1/stories.json"]
                                 + _script_dir_candidates("benchmarks", "state_tracking_v1", "stories.json"),
                                 "stories.json")
    assert sha == EXPECTED_ST_SHA256, f"stories.json is not the registered file: {sha}"
    data = json.loads(raw.decode("utf-8"))
    assert data.get("version") == "state_tracking_v1", data.get("version")
    assert data["classes"] == CLASSES, f"DATA ERROR: classes {data['classes']}"
    assert set(data["splits"]) == {"train", "val_seen", "test"}, set(data["splits"])
    data["splits"].pop("test")                     # test stories: never read in Lab 15
    tpl_path = os.path.join(os.path.dirname(path), "templates.json")
    _p, tpl_raw, tpl_sha = _read_first([tpl_path], "templates.json (next to stories.json)")
    assert tpl_sha == data.get("templates_sha256") == EXPECTED_TPL_SHA256, \
        f"templates.json is not the registered file: {tpl_sha} (stories.json records {data.get('templates_sha256')})"
    tpl = json.loads(tpl_raw.decode("utf-8"))
    split_of = {d["name"]: d["split"] for d in tpl["domains"]}
    assert len(split_of) == len(tpl["domains"]), "duplicate domain name in templates.json"
    assert set(split_of.values()) == {"train", "test"}, set(split_of.values())
    train_doms = sorted(n for n, s in split_of.items() if s == "train")
    n_test_doms = sum(s != "train" for s in split_of.values())
    assert len(train_doms) == 16, len(train_doms)
    story_doms = sorted({st["domain"] for st in data["splits"]["train"]})
    assert story_doms == train_doms, "DATA ERROR: templates.json train domains != domains of the stories' train split"
    assert {st["domain"] for st in data["splits"]["val_seen"]} <= set(train_doms), "DATA ERROR: val_seen domains"

    # sentence table: TRAIN domains only (filter by domain first; no other domain's text is ever built here)
    texts, cls, dom = [], [], []
    state_part = {}

    def add(t, c, d_name):
        texts.append(t)
        cls.append(c)
        dom.append(d_name)

    for d in tpl["domains"]:
        if split_of[d["name"]] != "train":
            continue
        lv_txt, sg_txt = d["init"]["level"], d["init"]["safeguard"]
        assert len(lv_txt) == 4 and set(sg_txt) == {"ok", "broken"}, d["name"]
        for lv in range(4):
            for sg in ("ok", "broken"):
                t = lv_txt[lv] + "，" + sg_txt[sg] + "。"
                state_part[(d["name"], lv, sg)] = t
                add(t, CI[f"L{lv}_{sg}"], d["name"])
        assert set(d["events"]) == set(EVENTS), d["name"]
        for e in EVENTS:
            for t in d["events"][e]:
                assert isinstance(t, str) and t, d["name"]
                add(t, N_INIT + EVENTS.index(e), d["name"])
    assert len(set(texts)) == len(texts), "DATA ERROR: a sentence occurs twice in the train-domain table"
    assert set(dom) == set(train_doms)

    # leak-assertion strings of the test domains (used by assert_allowed only: never extracted, trained on or printed)
    forbidden = set()
    for d in tpl["domains"]:
        if split_of[d["name"]] == "train":
            continue
        forbidden.add(d["init"]["rules"])
        for lt in d["init"]["level"]:
            for st_ in d["init"]["safeguard"].values():
                forbidden.add(lt + "，" + st_ + "。")
                forbidden.add(d["init"]["rules"] + "；" + lt + "，" + st_ + "。")
        for e in EVENTS:
            forbidden.update(d["events"][e])
    del tpl
    assert not forbidden & set(texts), "LEAK / DATA ERROR: a test-domain string is in the train-domain table"

    # every sentence of the train / val_seen stories is in the table (same class, same domain)
    row = {t: i for i, t in enumerate(texts)}
    used = set()

    def init_state_text(st) -> str:
        """(copied from Lab 14, [amendment 2026-11-01]) the state part = the text after the LAST '；' of init_text,
        checked against templates.json."""
        t = st["init_text"]
        if "；" not in t:
            raise RuntimeError(f"DATA ERROR: init_text has no '；': {t!r}")
        part = t.rsplit("；", 1)[1]
        want = state_part.get((st["domain"], int(st["init_level"]), st["init_safeguard"]))
        if part != want:
            raise RuntimeError(f"DATA ERROR: init state part {part!r} != templates {want!r}")
        return part

    n_story = 0
    for sp in ("train", "val_seen"):
        for st in data["splits"][sp]:
            n_story += 1
            items = [(init_state_text(st), CI[f"L{int(st['init_level'])}_{st['init_safeguard']}"])]
            items += [(t, N_INIT + EVENTS.index(e)) for e, t in zip(st["events"], st["event_texts"])]
            for t, c in items:
                i = row.get(t)
                if i is None or cls[i] != c or dom[i] != st["domain"]:
                    raise RuntimeError(f"DATA ERROR: story {st['id']}: sentence {t!r} (class {SCLS_NAMES[c]}) is not "
                                       f"in the train-domain table with that class and domain")
                used.add(t)
    cnt = Counter(cls)
    log(f"[l15] st: stories.json sha256={sha} | templates.json sha256={tpl_sha} | {len(train_doms)} train domains "
        f"(templates.json == stories' train split; {n_test_doms} test domains NOT used) | {len(texts)} sentences "
        f"({len(texts) // len(train_doms)} per domain); {len(used)} of them occur in the {n_story} train / val_seen "
        f"stories, all of which are in the table | per class: "
        + " ".join(f"{SCLS_NAMES[c]}={cnt[c]}" for c in range(N_SCLS)))
    return dict(name="st", sha=sha, tpl_sha=tpl_sha, texts=texts, cls=cls, dom=dom, doms=train_doms, n_cls=N_SCLS,
                cls_names=SCLS_NAMES, forbidden=forbidden,
                group=["init" if c < N_INIT else "ev" for c in cls])


def load_v5():
    """analogy_v5: the 27 class sentences of each of the 20 seen domains; `unseen_domains` is dropped unread."""
    path, raw, sha = _read_first(["/content/analogy_v5.json", "benchmarks/analogy_v5.json"]
                                 + _script_dir_candidates("benchmarks", "analogy_v5.json"), "analogy_v5.json")
    assert sha == EXPECTED_V5_SHA256, f"analogy_v5.json is not the registered file: {sha}"
    data = json.loads(raw.decode("utf-8"))
    data.pop("unseen_domains")                     # never read in Lab 15
    assert data.get("version") == "analogy_v5", data.get("version")
    # (copied from Lab 12) laws, pairs, 27 classes
    n_laws, n_stages, eval_stages = 6, 5, (2, 3, 4)
    assert len(data["laws"]) == n_laws
    pairs = [tuple(int(x) for x in p) for p in data["pairs"]]
    assert len(pairs) == 3 and sorted(x for p in pairs for x in p) == list(range(n_laws)), pairs
    pair_of = {}
    for _p, (_a, _b) in enumerate(pairs):
        pair_of[_a] = pair_of[_b] = _p
    class_keys = []  # ("L", law, stage) or ("P", pair, 1)
    for _p, (_a, _b) in enumerate(pairs):
        class_keys += [("L", _a, 0), ("L", _b, 0), ("P", _p, 1)]
        for _k in eval_stages:
            class_keys += [("L", _a, _k), ("L", _b, _k)]
    assert len(class_keys) == 27
    cls_index = {k: i for i, k in enumerate(class_keys)}

    def cls_of(law: int, st: int) -> int:
        return cls_index[("P", pair_of[law], 1)] if st == 1 else cls_index[("L", law, st)]

    seen = data["seen_domains"]
    names = [d["name"] for d in seen]
    assert len(names) == 20 and len(set(names)) == 20, names
    texts, cls, dom = [], [], []
    for d in seen:
        ch = d["chains"]
        for law in range(n_laws):
            assert len(ch[str(law)]) == n_stages and all(isinstance(t, str) and t for t in ch[str(law)])
        for (a, b) in pairs:  # (copied from Lab 12 check_domain)
            if ch[str(a)][1] != ch[str(b)][1]:
                raise RuntimeError(f"{d['name']}: stage-1 sentences of pair ({a},{b}) differ")
        rows = [(ch[str(a)][1], cls_of(a, 1)) for (a, b) in pairs]       # the shared stage-1 sentence, once per pair
        rows += [(ch[str(law)][st], cls_of(law, st)) for law in range(n_laws) for st in (0, 2, 3, 4)]
        for t, c in rows:
            texts.append(t)
            cls.append(c)
            dom.append(d["name"])
    assert len(set(texts)) == len(texts) == 20 * 27, "DATA ERROR: v5 seen sentences are not unique"
    cnt = Counter(cls)
    assert all(cnt[c] == 20 for c in range(27)), cnt
    for d in names:  # one sentence per class per domain
        assert sorted(c for c, dd in zip(cls, dom) if dd == d) == list(range(27)), d
    cls_names = [f"{'L' if k == 'L' else 'P'}{i}S{s}" for (k, i, s) in class_keys]
    stage = [class_keys[c][2] for c in cls]
    log(f"[l15] v5: analogy_v5.json sha256={sha} | {len(names)} seen domains (unseen_domains dropped unread) | "
        f"{len(texts)} sentences, 27 classes x 20 domains")
    return dict(name="v5", sha=sha, texts=texts, cls=cls, dom=dom, doms=sorted(names), n_cls=27, cls_names=cls_names,
                stage=stage)


def make_folds(doms, n_folds: int, size: int = FOLD_SIZE, seed: int = FOLD_SEED):
    """Fixed folds: sorted names shuffled with Random(seed), chunked; pool of fold f = the other domains, sorted and
    shuffled with Random(seed + f) (training subset of size k = its first k: nested)."""
    d = sorted(doms)
    assert len(d) == n_folds * size, (len(d), n_folds, size)
    random.Random(seed).shuffle(d)
    folds = [d[i * size:(i + 1) * size] for i in range(n_folds)]
    pools = []
    for f in range(n_folds):
        pool = sorted(x for x in d if x not in folds[f])
        random.Random(seed + f).shuffle(pool)
        pools.append(pool)
    return folds, pools


# ==============================================================================
# [1] Encoders (frozen, bf16, features in float32)
# ==============================================================================
class LeakGuard:
    """Every text that reaches a tokenizer goes through check(): its sentence must be in the train-domain tables, and
    neither the sentence nor the model text may equal or contain a test-domain string."""

    def __init__(self, allowed, forbidden):
        self.allowed, self.forbidden = set(allowed), set(forbidden)
        self.n_checked = 0

    def check(self, sentence: str, model_text: str = None):
        if sentence not in self.allowed:
            raise RuntimeError(f"LEAK GUARD: text not in the training-domain tables: {sentence!r}")
        for t in (sentence, model_text):
            if t is None:
                continue
            if t in self.forbidden or any(f in t for f in self.forbidden):
                raise RuntimeError("LEAK: a test-domain string reached an encoder")
        self.n_checked += 1


def _right_padded(mask):
    n = mask.sum(dim=1)
    ar = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
    if not torch.equal(mask.bool(), ar < n.unsqueeze(1)):
        raise RuntimeError("expected right padding (every row = real tokens, then padding only)")
    return n


@torch.no_grad()
def qwen_pool(tok, model, texts, pooling: str):
    """-> (features (B, D) float32, token id at the pooled position (prompt) / first token (mean), real lengths (B,)).
    mean: (copied from Lab 01 / Lab 12 extract_qwen_896_feature) last hidden state, masked mean, right padding.
    prompt: last hidden state at the last real token, index = attention_mask.sum - 1 under right padding (asserted);
    a causal model never lets a real token see the padding after it, and positions are 0..n-1 as in a batch of 1."""
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=SENT_MAXLEN).to(DEVICE)
    m = enc.attention_mask
    n = _right_padded(m)
    h = model.model(input_ids=enc.input_ids, attention_mask=m).last_hidden_state.float()
    rows = torch.arange(h.shape[0], device=h.device)
    if pooling == "mean":
        mask_f = m.unsqueeze(-1).float()
        out = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        tid = enc.input_ids[:, 0]
    elif pooling == "prompt":
        last = n - 1
        out = h[rows, last]
        tid = enc.input_ids[rows, last]
    else:
        raise ValueError(pooling)
    return out, tid, n


@torch.no_grad()
def bge_pool(tok, model, texts, pooling: str = "cls"):
    """BAAI/bge-m3 official dense embedding: [CLS] (position 0) of the last hidden state, L2-normalised."""
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=SENT_MAXLEN).to(DEVICE)
    m = enc.attention_mask
    n = _right_padded(m)
    h = model(input_ids=enc.input_ids, attention_mask=m).last_hidden_state.float()
    return F.normalize(h[:, 0], p=2, dim=-1), enc.input_ids[:, 0], n


def model_text(pooling: str, s: str) -> str:
    return PROMPT_TMPL.format(s=s) if pooling == "prompt" else s


def extract(tok, pool_fn, model, sentences, pooling: str, guard: LeakGuard, tag: str):
    """Features of `sentences` in batches of FEAT_BATCH. No truncation: every model text's token count (tokenised alone)
    must be <= SENT_MAXLEN and the padded, truncating batch must keep exactly that many tokens (as Lab 14)."""
    mt = [model_text(pooling, s) for s in sentences]
    for s, t in zip(sentences, mt):
        guard.check(s, t)
    n_tok = [len(tok(t).input_ids) for t in mt]
    j = int(np.argmax(n_tok))
    if n_tok[j] > SENT_MAXLEN:
        raise RuntimeError(f"DATA ERROR: {tag}: {n_tok[j]} tokens > max_length {SENT_MAXLEN}: {mt[j]!r}")
    parts, tids = [], []
    for i in range(0, len(mt), FEAT_BATCH):
        h, tid, n = pool_fn(tok, model, mt[i:i + FEAT_BATCH], pooling)
        assert [int(x) for x in n.tolist()] == n_tok[i:i + FEAT_BATCH], f"{tag}: batch tokenisation != per-text counts"
        parts.append(h.cpu())
        tids += [int(x) for x in tid.tolist()]
    H = torch.cat(parts, dim=0)
    assert H.shape[0] == len(sentences) and bool(torch.isfinite(H).all()), tag
    return H, n_tok, tids


def unbatched_check(tok, pool_fn, model, sentences, pooling: str, H, n_tok, tids, tag: str):
    """Pooled features of a few sentences recomputed one at a time (batch of 1, no padding) must match the batched ones:
    same token count and pooled token exactly (prompt: = the text's last token), cosine >= CHECK_MIN_COS.
    Returns (min cos, max |diff| / max |x|, rows checked)."""
    idx = sorted({0, 1, 2, len(sentences) - 1, int(np.argmax(n_tok)), int(np.argmin(n_tok))})
    min_cos, max_rel = 1.0, 0.0
    for i in idx:
        mt = model_text(pooling, sentences[i])
        h1, tid1, n1 = pool_fn(tok, model, [mt], pooling)
        if int(n1[0]) != n_tok[i] or int(tid1[0]) != tids[i]:
            raise RuntimeError(f"{tag}: batched / unbatched pooling disagree on row {i} (length or pooled token)")
        if pooling == "prompt" and tids[i] != int(tok(mt).input_ids[-1]):
            raise RuntimeError(f"{tag}: the pooled token of row {i} is not the text's last token")
        a, b = h1[0].cpu(), H[i]
        cos = float(F.cosine_similarity(a, b, dim=0))
        rel = float((a - b).abs().max() / b.abs().max().clamp(min=1e-12))
        min_cos, max_rel = min(min_cos, cos), max(max_rel, rel)
    if min_cos < CHECK_MIN_COS:
        raise RuntimeError(f"{tag}: batched vs unbatched pooled features differ (min cos {min_cos:.6f})")
    return min_cos, max_rel, len(idx)


def free_gpu():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def run_model(code: str, data: dict, guard: LeakGuard, zs_fn=None):
    """Load one encoder model, extract every feature it provides for every dataset, run the zero-shot if asked, free it.
    -> {encoder: {dataset: (N, D) float32 CPU tensor}}"""
    mid, kind = MODELS[code]
    log(f"\n=== [1] Loading '{mid}' ({code}, frozen, bf16) ===")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(mid)
    tok.padding_side = "right"
    if kind == "qwen":
        model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map=DEVICE)
        pool_fn, poolings = qwen_pool, QWEN_POOLINGS
        hid = model.config.hidden_size
    else:
        model = AutoModel.from_pretrained(mid, dtype=torch.bfloat16, device_map=DEVICE)
        pool_fn, poolings = bge_pool, ("cls",)
        hid = model.config.hidden_size
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    _p0 = next(model.parameters(), None)  # (Lab 12 [review N5]) the CPU test stub has no parameters and a FAKE version
    assert (_p0 is not None and _p0.dtype == torch.bfloat16) or "FAKE" in getattr(transformers, "__version__", "")
    assert hid == EXPECTED_DIM[code], (code, hid)
    log(f"[l15] {code}: loaded in {time.time() - t0:.1f}s | hidden={hid}"
        + (f" | GPU memory allocated {torch.cuda.memory_allocated() / 2 ** 30:.1f} GiB" if DEVICE == "cuda" else ""))
    feats = {}
    for pooling in poolings:
        enc_name = code if kind == "bge" else f"{code}-{pooling}"
        feats[enc_name] = {}
        for dname in DATASETS:
            t1 = time.time()
            H, n_tok, tids = extract(tok, pool_fn, model, data[dname]["texts"], pooling, guard, f"{enc_name}/{dname}")
            mc, mr, nchk = unbatched_check(tok, pool_fn, model, data[dname]["texts"], pooling, H, n_tok, tids,
                                           f"{enc_name}/{dname}")
            feats[enc_name][dname] = H
            sec = time.time() - t1
            log(f"[l15] features {enc_name}/{dname}: {tuple(H.shape)} float32 | max tokens {max(n_tok)} "
                f"(limit {SENT_MAXLEN}, no truncation) | {sec:.1f}s | unbatched check on {nchk} rows: min cos "
                f"{mc:.6f}, max |diff|/max|x| {mr:.2e} | fingerprint {float(H.double().sum()).hex()}")
            log(f"L15_FEAT enc={enc_name} data={dname} n={H.shape[0]} dim={H.shape[1]} max_tokens={max(n_tok)} "
                f"sec={sec:.1f} check_rows={nchk} check_min_cos={mc:.6f}")
    if zs_fn is not None:
        zs_fn(tok, model)
    del model, tok
    free_gpu()
    log(f"[l15] {code}: done in {time.time() - t0:.1f}s, model freed"
        + (f" (GPU memory allocated {torch.cuda.memory_allocated() / 2 ** 30:.2f} GiB)" if DEVICE == "cuda" else ""))
    return feats


# ==============================================================================
# [2] Zero-shot reference (st only; Qwen2.5-7B-Instruct, chat template)
# ==============================================================================
def letter_token_ids(tok, letters):
    """(copied from Lab 10) every letter must be a single token, with and without a leading space."""
    out = []
    for L in letters:
        ids = set()
        for v in (L, " " + L):
            t = tok.encode(v, add_special_tokens=False)
            if len(t) != 1:
                raise RuntimeError(f"letter {v!r} is not a single token: {t}")
            ids.add(t[0])
        out.append(sorted(ids))
    return out


def make_chooser(tok, model):
    """(adapted from Lab 10 choose) chat template, next-token log-probs; option score = max over the letter's tokens;
    the highest-scoring option wins, ties -> the earlier letter."""
    letter_ids = letter_token_ids(tok, LETTERS)

    @torch.no_grad()
    def choose(q: str, n_opt: int) -> int:
        prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
        ids = torch.tensor([tok(prompt).input_ids], device=DEVICE)
        lp = F.log_softmax(model(input_ids=ids).logits.float()[0, -1, :], dim=-1)
        sc = [max(float(lp[t]) for t in letter_ids[j]) for j in range(n_opt)]
        return max(range(n_opt), key=lambda j: (sc[j], -j))

    return choose


def mcq(domain: str, sentence: str, question: str, options) -> str:
    L = LETTERS[:len(options)]
    ans = (f"{L[0]} 或 {L[1]}" if len(L) == 2 else "、".join(L[:-1]) + f" 或 {L[-1]}")
    return (f"领域：{domain}\n句子：{sentence}\n\n问题：{question}\n"
            + "".join(f"{l}. {o}\n" for l, o in zip(L, options))
            + f"\n只回答一个字母（{ans}）。")


def ask_both_orders(choose, domain, sentence, question, options, true_idx, guard: LeakGuard):
    """-> (ok order 1, ok order 2, letter chosen in order 1, letter chosen in order 2). Order 2 = reversed options."""
    n = len(options)
    q1 = mcq(domain, sentence, question, options)
    q2 = mcq(domain, sentence, question, options[::-1])
    for q in (q1, q2):
        guard.check(sentence, q)
    p1, p2 = choose(q1, n), choose(q2, n)
    return p1 == true_idx, p2 == n - 1 - true_idx, LETTERS[p1], LETTERS[p2]


def run_zero_shot(st: dict, choose, guard: LeakGuard):
    """All 640 train-domain sentences of st. Prints numbers and returns the counts."""
    t0 = time.time()
    ev = defaultdict(lambda: [0, 0, 0, 0])          # class name -> [both, order1, order2, total]
    ini = defaultdict(lambda: [0, 0, 0, 0])
    part = {p: [0, 0, 0, 0] for p in ("init_level", "init_safeguard")}
    letters = {k: Counter() for k in ("event/o1", "event/o2", "level/o1", "level/o2", "sg/o1", "sg/o2")}
    for s, c, d in zip(st["texts"], st["cls"], st["dom"]):
        if c >= N_INIT:
            e = c - N_INIT
            o1, o2, l1, l2 = ask_both_orders(choose, d, s, ZS_EVENT_Q, ZS_EVENT_OPTS, e, guard)
            r = ev[SCLS_NAMES[c]]
            letters["event/o1"][l1] += 1
            letters["event/o2"][l2] += 1
        else:
            lv, sg = c % 4, (1 if c >= 4 else 0)
            a1, a2, la1, la2 = ask_both_orders(choose, d, s, ZS_LEVEL_Q, ZS_LEVEL_OPTS, lv, guard)
            b1, b2, lb1, lb2 = ask_both_orders(choose, d, s, ZS_SG_Q, ZS_SG_OPTS, sg, guard)
            for key, (x1, x2) in (("init_level", (a1, a2)), ("init_safeguard", (b1, b2))):
                part[key][0] += int(x1 and x2)
                part[key][1] += int(x1)
                part[key][2] += int(x2)
                part[key][3] += 1
            letters["level/o1"][la1] += 1
            letters["level/o2"][la2] += 1
            letters["sg/o1"][lb1] += 1
            letters["sg/o2"][lb2] += 1
            o1, o2 = (a1 and b1), (a2 and b2)
            r = ini[SCLS_NAMES[c]]
        r[0] += int(o1 and o2)
        r[1] += int(o1)
        r[2] += int(o2)
        r[3] += 1
    tot = {}
    for kind, tab in (("event", ev), ("init", ini)):
        tot[kind] = [sum(v[i] for v in tab.values()) for i in range(4)]
    log(f"\n=== [2] zero-shot reference (Qwen2.5-7B-Instruct, chat template, all {len(st['texts'])} train-domain "
        f"sentences, no training) | {time.time() - t0:.0f}s ===")
    log("  correct = right under BOTH option orders (order 2 = reversed); init = level AND safeguard both right")
    for kind, tab in (("event", ev), ("init", ini)):
        b, o1, o2, n = tot[kind]
        log(f"  {kind:<6}: both {b}/{n} ({100 * b / n:.1f}%) | order1 {o1}/{n} ({100 * o1 / n:.1f}%) | order2 {o2}/{n} "
            f"({100 * o2 / n:.1f}%)")
        for name in sorted(tab, key=SCLS_NAMES.index):
            v = tab[name]
            log(f"      {name:<10} both {v[0]:>3}/{v[3]:<3} order1 {v[1]:>3} order2 {v[2]:>3}")
    for key, v in part.items():
        log(f"  {key:<15} both {v[0]}/{v[3]} | order1 {v[1]} | order2 {v[2]}")
    n_opts = {"event": len(ZS_EVENT_OPTS), "level": len(ZS_LEVEL_OPTS), "sg": len(ZS_SG_OPTS)}
    for key, cnt in letters.items():
        log(f"  letters chosen {key:<9}: " + " ".join(f"{l}={cnt[l]}" for l in LETTERS[:n_opts[key.split('/')[0]]]))
    for kind in ("event", "init"):
        b, o1, o2, n = tot[kind]
        log(f"L15_ZS kind={kind} correct={b} total={n} order1={o1} order2={o2}")
    for key, v in part.items():
        log(f"L15_ZSPART kind={key} correct={v[0]} total={v[3]} order1={v[1]} order2={v[2]}")
    for kind, tab in (("event", ev), ("init", ini)):
        for name in sorted(tab, key=SCLS_NAMES.index):
            v = tab[name]
            log(f"L15_ZSCLS kind={kind} cls={name} correct={v[0]} total={v[3]} order1={v[1]} order2={v[2]}")
    return dict(tot=tot, part=part, ev=dict(ev), ini=dict(ini))


# ==============================================================================
# [3] Recognizers
# ==============================================================================
def preprocess(Fd, tr, te):
    """x = normalize(f - mu); mu = mean of the TRAINING rows only."""
    mu = Fd[tr].mean(dim=0)
    return F.normalize(Fd[tr] - mu, p=2, dim=-1), F.normalize(Fd[te] - mu, p=2, dim=-1)


def class_means(Z, y, n_cls: int):
    oh = F.one_hot(y, n_cls).float()
    cnt = oh.sum(0)
    assert bool((cnt > 0).all()), "a class has no training sentence"
    return F.normalize(oh.T @ Z / cnt.unsqueeze(-1), p=2, dim=-1)


class Funnel(nn.Module):
    """(copied from Lab 14) Lab 12 DualHead's struct_funnel (D -> 256 -> 64, GELU) with the s-normalisation; also the
    XDOM projection (same shape)."""

    def __init__(self, d_in: int, s_dim: int = PROJ_DIM):
        super().__init__()
        self.struct_funnel = nn.Sequential(nn.Linear(d_in, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, s_dim))

    def forward(self, x):
        return F.normalize(self.struct_funnel(x), p=2, dim=-1)


def funnel_loss(s, y, n_cls: int):
    """(adapted from Lab 14 funnel_loss / Lab 12) s pull / push with the closest-pair term over the n_cls classes."""
    oh = F.one_hot(y, n_cls).float()
    s_mean = F.normalize(oh.T @ s / oh.sum(0).unsqueeze(-1), p=2, dim=-1)
    l_pull = (1.0 - (s * s_mean[y]).sum(-1)).mean()
    off = ~torch.eye(n_cls, device=s.device, dtype=torch.bool)
    _sp = F.relu((s_mean @ s_mean.T)[off] + PUSH_MARGIN)
    l_push = _sp.mean() + _sp.max()  # closest-pair term (from v3)
    return l_pull + l_push


def xdom_masks(y, d):
    """-> (pos, excl): pos[i, j] = same class, different domain; excl[i, j] = j is not in anchor i's denominator
    (j == i or same class AND same domain)."""
    same_c = y.unsqueeze(1) == y.unsqueeze(0)
    same_d = d.unsqueeze(1) == d.unsqueeze(0)
    eye = torch.eye(len(y), device=y.device, dtype=torch.bool)
    return same_c & ~same_d, eye | (same_c & same_d)


def supcon_from_logits(S, pos, excl):
    """Supervised contrastive loss (SupCon L_out) on the logit matrix S = z z^T / tau. For anchor i: the denominator
    runs over every j not in excl[i]; the loss is -mean over positives p of log(exp S_ip / sum_j exp S_ij). Anchors
    without a positive are skipped (an anchor with a positive always has a finite denominator). Excluded pairs get
    exactly zero gradient."""
    has = pos.any(dim=1)
    if not bool(has.any()):
        raise RuntimeError("XDOM: no anchor has a cross-domain positive")
    S_a, pos_a, excl_a = S[has], pos[has], excl[has]
    assert not bool((pos_a & excl_a).any())
    log_den = torch.logsumexp(S_a.masked_fill(excl_a, float("-inf")), dim=1)
    lp = torch.where(pos_a, S_a - log_den.unsqueeze(1), torch.zeros_like(S_a))
    return -(lp.sum(1) / pos_a.sum(1).float()).mean()


def train_loop(model, loss_fn, steps: int, seed: int):
    """(adapted from Lab 12 train_mlp / Lab 14 train_funnel) full batch, AdamW lr 1e-3 -> 1e-4 cosine, wd 0.01,
    clip 1.0; seed_all before the training run."""
    seed_all(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=REC_LR, weight_decay=REC_WD)
    model.train()
    loss = None
    for step in range(1, steps + 1):
        for pg in opt.param_groups:
            pg["lr"] = cosine_lr(step, steps, REC_LR, REC_LR_FLOOR)
        opt.zero_grad()
        loss = loss_fn(model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), REC_CLIP)
        opt.step()
    model.eval()
    return float(loss.item())


def fit_predict(rec: str, X_tr, y_tr, d_tr, X_te, n_cls: int, seed: int):
    """-> (predictions on X_te, predictions on X_tr (training fit), final training loss or nan)."""
    D = X_tr.shape[1]
    if rec == "NEAR":
        means = class_means(X_tr, y_tr, n_cls)
        return (X_te @ means.T).argmax(-1), (X_tr @ means.T).argmax(-1), float("nan")
    seed_all(seed)  # before model init
    if rec == "LIN":
        model = nn.Linear(D, n_cls)
    elif rec == "MLP":
        model = nn.Sequential(nn.Linear(D, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, n_cls))
    elif rec in ("FUNNEL", "XDOM"):
        model = Funnel(D, PROJ_DIM)
    else:
        raise ValueError(rec)
    model = model.to(DEVICE).float()
    if rec in ("LIN", "MLP"):
        loss = train_loop(model, lambda m: F.cross_entropy(m(X_tr), y_tr), CE_STEPS, seed)
        with torch.no_grad():
            return model(X_te).argmax(-1), model(X_tr).argmax(-1), loss
    if rec == "FUNNEL":
        loss = train_loop(model, lambda m: funnel_loss(m(X_tr), y_tr, n_cls), FUNNEL_STEPS, seed)
    else:
        pos, excl = xdom_masks(y_tr, d_tr)

        def _xl(m):
            z = m(X_tr)
            return supcon_from_logits(z @ z.T / XDOM_TAU, pos, excl)

        loss = train_loop(model, _xl, CE_STEPS, seed)
    with torch.no_grad():
        z_tr, z_te = model(X_tr), model(X_te)
        anchors = class_means(z_tr, y_tr, n_cls)
        return (z_te @ anchors.T).argmax(-1), (z_tr @ anchors.T).argmax(-1), loss


def cv_dataset(ds: dict, enc: str, H, folds, pools, results: dict):
    """Every (k, fold, seed, recognizer) of one dataset for one encoder; prints L15_CV (+ L15_STAGE for v5)."""
    name, n_cls = ds["name"], ds["n_cls"]
    Fd = H.to(DEVICE)
    y = torch.tensor(ds["cls"], device=DEVICE)
    dom_id = {d: i for i, d in enumerate(ds["doms"])}
    dvec = torch.tensor([dom_id[d] for d in ds["dom"]], device=DEVICE)
    t0 = time.time()
    for f, held in enumerate(folds):
        te = [i for i, d in enumerate(ds["dom"]) if d in held]
        te_t = torch.tensor(te, device=DEVICE)
        for k in KS[name]:
            tr_doms = pools[f][:k]
            assert len(tr_doms) == k and not set(tr_doms) & set(held)
            tr = [i for i, d in enumerate(ds["dom"]) if d in tr_doms]
            assert not set(tr) & set(te)
            tr_t = torch.tensor(tr, device=DEVICE)
            X_tr, X_te = preprocess(Fd, tr_t, te_t)
            y_tr, y_te, d_tr = y[tr_t], y[te_t], dvec[tr_t]
            for seed in SEEDS:
                for rec in RECOGNIZERS:
                    p_te, p_tr, loss = fit_predict(rec, X_tr, y_tr, d_tr, X_te, n_cls, seed)
                    hit = (p_te == y_te).cpu().tolist()
                    fit_c = int((p_tr == y_tr).sum())
                    c, t = sum(hit), len(hit)
                    r = dict(c=c, t=t, fit_c=fit_c, fit_t=len(tr), loss=loss)
                    if name == "st":
                        ev = [h for h, i in zip(hit, te) if ds["group"][i] == "ev"]
                        ini = [h for h, i in zip(hit, te) if ds["group"][i] == "init"]
                        r.update(ec=sum(ev), et=len(ev), ic=sum(ini), it=len(ini))
                    else:
                        r.update(ec=-1, et=-1, ic=-1, it=-1)
                        r["stage"] = {s: (sum(h for h, i in zip(hit, te) if ds["stage"][i] == s),
                                          sum(1 for i in te if ds["stage"][i] == s)) for s in range(5)}
                    results[(name, enc, rec, k)][(f, seed)] = r
                    log(f"L15_CV data={name} enc={enc} rec={rec} k={k} fold={f} seed={seed} correct={c} total={t} "
                        f"ev_correct={r['ec']} ev_total={r['et']} init_correct={r['ic']} init_total={r['it']} "
                        f"fit_correct={fit_c} fit_total={len(tr)}")
                    if name == "v5":
                        for s, (sc, stt) in r["stage"].items():
                            log(f"L15_STAGE data={name} enc={enc} rec={rec} k={k} fold={f} seed={seed} stage={s} "
                                f"correct={sc} total={stt}")
    log(f"[l15] CV {name} / {enc}: {len(folds)} folds x {len(KS[name])} k x {len(SEEDS)} seeds x "
        f"{len(RECOGNIZERS)} recognizers in {time.time() - t0:.1f}s")


# ==============================================================================
# [4] Report helpers (numbers only)
# ==============================================================================
def pooled(results, key, fields=("c", "t")):
    rs = list(results.get(key, {}).values())
    return tuple(sum(r[f] for r in rs) for f in fields), len(rs)


def pct(c, t):
    return f"{100 * c / t:6.1f}" if t > 0 else "     -"


def print_tables(results, encs, data):
    for name in DATASETS:
        rk = REG_K[name]
        n_exp = N_FOLDS[name] * len(SEEDS)
        log(f"\n=== SUMMARY data={name}: mean CV accuracy (%) on held-out domains at k={rk} (registered k), exact counts "
            f"pooled over {N_FOLDS[name]} folds x {len(SEEDS)} seeds ===")
        groups = [("all", ("c", "t"))] + ([("event sentences", ("ec", "et")), ("init sentences", ("ic", "it"))]
                                          if name == "st" else [])
        for gname, flds in groups:
            log(f"--- {gname} ---")
            log(f"{'encoder':<12}" + "".join(f"{r:>8}" for r in RECOGNIZERS))
            for enc in encs:
                cells = []
                for rec in RECOGNIZERS:
                    (c, t), n = pooled(results, (name, enc, rec, rk), flds)
                    assert n == n_exp, (name, enc, rec, n)
                    cells.append(f"{pct(c, t):>8}")
                log(f"{enc:<12}" + "".join(cells))
        log(f"--- k-curves data={name} (%): " + " / ".join(f"k={k}" for k in KS[name]) + " ---")
        for enc in encs:
            for rec in RECOGNIZERS:
                vals = []
                for k in KS[name]:
                    (c, t), _n = pooled(results, (name, enc, rec, k))
                    vals.append(pct(c, t))
                log(f"  {enc + ' + ' + rec:<24}" + " ".join(vals))
        if name == "v5":
            log(f"--- per stage at k={rk} (%): stage 0 / 1 / 2 / 3 / 4 ---")
            for enc in encs:
                for rec in RECOGNIZERS:
                    rs = results[(name, enc, rec, rk)].values()
                    vals = [pct(sum(r["stage"][s][0] for r in rs), sum(r["stage"][s][1] for r in rs)) for s in range(5)]
                    log(f"  {enc + ' + ' + rec:<24}" + " ".join(vals))
        log(f"--- training fit at k={rk} (%, eval mode, pooled) ---")
        for enc in encs:
            cells = []
            for rec in RECOGNIZERS:
                (c, t), _n = pooled(results, (name, enc, rec, rk), ("fit_c", "fit_t"))
                cells.append(f"{pct(c, t):>8}")
            log(f"{enc:<12}" + "".join(cells))


# ==============================================================================
# main
# ==============================================================================
def main():
    t_start = time.time()
    log(f"[l15] device={DEVICE} | seeds={SEEDS} (one run) | python {sys.version.split()[0]} | torch {torch.__version__} "
        f"| numpy {np.__version__} | transformers {getattr(transformers, '__version__', '?')} "
        f"({getattr(transformers, '__file__', '?')})")
    try:
        with open(os.path.abspath(__file__), "rb") as _f:
            script_sha = hashlib.sha256(_f.read()).hexdigest()
    except (NameError, OSError):
        script_sha = "unavailable"  # e.g. executed through a notebook kernel; the runner records the repo file's sha
    log(f"[l15] script sha256={script_sha}")
    env_tag = f"torch={torch.__version__},transformers={getattr(transformers, '__version__', '?')},device={DEVICE}"
    if DEVICE == "cuda":
        env_tag += f",gpu={torch.cuda.get_device_name(0).replace(' ', '_')}"
    if "FAKE" in getattr(transformers, "__version__", ""):
        log("!" * 120 + "\n!!! [l15] FAKE transformers stub: TEST RUN, NOT a registered run\n" + "!" * 120)

    models_env = os.environ.get("LAB15_MODELS", "").strip()
    models = [m.strip() for m in models_env.split(",") if m.strip()] if models_env else list(MODELS)
    bad = [m for m in models if m not in MODELS]
    if bad or len(set(models)) != len(models):
        raise ValueError(f"LAB15_MODELS={models_env!r}: unknown or repeated model code(s); allowed {list(MODELS)}")
    models = [m for m in MODELS if m in models]  # registered order
    if models != list(MODELS):
        log(f"[l15] NOTE: LAB15_MODELS={models_env!r}: this log covers only {models}; merge the logs of the other "
            f"models with tools/aggregate_lab15.py")

    # ---- data ----
    data = {"st": load_st(), "v5": load_v5()}
    guard = LeakGuard(data["st"]["texts"] + data["v5"]["texts"], data["st"]["forbidden"])
    folds = {}
    for name in DATASETS:
        fo, po = make_folds(data[name]["doms"], N_FOLDS[name])
        folds[name] = (fo, po)
        log(f"[l15] folds {name} (Random({FOLD_SEED}) over the sorted names; pool of fold f shuffled with "
            f"Random({FOLD_SEED} + f); training domains = first k of the pool, k in {KS[name]}):")
        for f in range(N_FOLDS[name]):
            n_held = sum(d in fo[f] for d in data[name]["dom"])
            log(f"  fold {f}: held out {fo[f]} ({n_held} sentences) | pool order {po[f]}")
            log(f"L15_FOLD data={name} fold={f} heldout={','.join(fo[f])} pool={','.join(po[f])}")

    # ---- encoders: features once, one model at a time; CV right after each model is freed ----
    results = defaultdict(dict)
    encs_run = []
    zs_done = False
    for code in models:
        zs_fn = None
        if code == ZS_MODEL:
            def zs_fn(tok, model):
                run_zero_shot(data["st"], make_chooser(tok, model), guard)
        feats = run_model(code, data, guard, zs_fn)
        zs_done |= zs_fn is not None
        for enc in feats:
            encs_run.append(enc)
            for name in DATASETS:
                cv_dataset(data[name], enc, feats[enc][name], *folds[name], results)
        del feats
        free_gpu()
    assert encs_run == [e for e in ENCODERS if e.split("-")[0] in models], encs_run

    log(f"\n[l15] leak guard: {guard.n_checked} encoder / zero-shot inputs checked, all from the "
        f"{len(data['st']['texts'])} st + {len(data['v5']['texts'])} v5 training-domain sentences; none equals or "
        f"contains any of the {len(data['st']['forbidden'])} test-domain template strings; v5 unseen_domains and the "
        f"st test stories were dropped unread")
    print_tables(results, encs_run, data)
    if not zs_done:
        log("\n[l15] zero-shot reference not run in this log (Q7 not in LAB15_MODELS)")
    log(f"\n[l15] total time {time.time() - t_start:.1f}s")
    log(f"L15_CONFIG data_sha_st={data['st']['sha']} data_sha_v5={data['v5']['sha']} templates_sha={data['st']['tpl_sha']} "
        f"env={env_tag} script_sha={script_sha} models={','.join(models)} seeds={','.join(map(str, SEEDS))} "
        f"zs={int(zs_done)}")


if __name__ == "__main__":
    main()
