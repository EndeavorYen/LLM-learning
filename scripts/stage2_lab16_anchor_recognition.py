# ==============================================================================
# Stage 2 / Lab 16: anchor examples in new domains + domain-count curve (B1). PRE-REGISTERED on issue #25 (comment
#   "第一步之一 c / d", 2026-11-01, before any code). Numbers only, NO verdict.
# Question A (part A, "1c"): in a domain never seen in training, if each of the 5 event types gets s labelled example
#   sentences ("anchors"), how well are the domain's other event sentences recognised? s = 1 primary; 2, 3 secondary.
# Question B (part B, "1d"): with 12 new training domains (28 in all), how does cross-domain recognition grow with the
#   number of training domains k (4 .. 24)? Only to measure the curve's shape.
# Hard rule: the 8 test domains of state_tracking_v1 are NOT used (as Lab 15: test stories dropped unread; every text
#   is built from training-domain templates and asserted before it reaches an encoder).
# Data
#   st     benchmarks/state_tracking_v1/{stories,templates}.json (sha256 pinned), the 16 train domains, Lab 15's
#          sentence table: per domain 8 init sentences (level[l] + "，" + safeguard[s] + "。") and 32 event sentences
#          (7 IN, 7 OUT, 6 BREAK, 6 FIX, 6 NOOP); 13 classes.
#   hard   (part A) benchmarks/state_tracking_v1/hard_cases.json (sha256 pinned, Lab 15 phase 1b).
#   extra  (part B) benchmarks/state_tracking_v1/extra_domains.json (sha256 pinned): 12 new train domains, same
#          schema, same table construction (accepted by tools/validate_extra_domains.py and a blind check).
# Folds (Lab 15 make_folds, Random(2026)): part A = Lab 15's st folds (4 folds x 4 held-out of 16; pool 12;
#   k in {4, 8, 12}); part B = 7 folds x 4 held-out of 28; pool 24; k in {4, 6, 8, 12, 16, 20, 24} (nested).
# Encoders (frozen; Lab 15 code): part A all 7 (Q05/Q15/Q7 x mean/prompt, BGE); part B Q15-prompt, Q7-prompt, BGE.
# Recognizers trained per (fold, k, seed) on the k training domains (Lab 15 recipes, 13 classes):
#   MLP     x = normalize(f - mu), mu = mean of the training rows; D -> 256 -> GELU -> 13, CE, 500 steps.
#   FUNNEL  same x; Lab 14 funnel (D -> 256 -> GELU -> 64, normalised), pull / push loss, 400 steps; anchors =
#           normalised class means of the training rows in funnel space.
#   C-MLP / C-FUNNEL  the same models trained on domain-centred x_c = normalize(f - m_d): m_d of a training domain =
#           mean over the 5 event classes of that domain's class-mean feature (class-balanced event mean).
# Anchor examples ("support") of a held-out domain: for each event type, s sentences drawn at random from the domain's
#   sentences of that type (rng = Random(sha256(part|seed|fold|domain|s|draw)); the same support for every k and
#   method); R = 10 draws per (fold, seed, s). Query = the domain's other event sentences (32 - 5 s). The sentence group
#   is known (a story's first sentence is the initial state, the rest are events), so events are scored 5-way.
#   m_d of a held-out domain = mean over the 5 event types of the mean of its support features.
# Methods (event sentences, 5-way):
#   ZERO-MLP     MLP, argmax over the 5 event logits (no example; reference)
#   ZERO-FUNNEL  FUNNEL, nearest of the 5 training event anchors (reference)
#   RAW-NN       no training: nearest class prototype in x space (prototype = normalised mean of the class's s
#                normalised support vectors; s = 1: the example itself)
#   MLP-NN       the same, in the MLP hidden layer (256-d after GELU, normalised)
#   FUNNEL-NN    the same, in funnel space: the new domain's anchors = its examples ("re-anchoring")
#   C-MLP        C-MLP on x_c (held-out m_d from the support), argmax over the 5 event logits
#   C-MLP-NN     nearest prototype in the C-MLP hidden layer
#   C-FUNNEL-NN  nearest prototype in C-FUNNEL space
#   7B-ICL       (part A, reference) Qwen2.5-7B-Instruct, chat template: the domain's 5 s=1 examples (sentence + the
#                correct option text, in random order) then Lab 15's 5-option event question; letter scored as in
#                Lab 15; correct = right in both option orders. All 16 train domains (no training), 2 draws each;
#                ZS0 = Lab 15's zero-shot prompt (no example) on the same queries.
# Initial-state sentences (8-way, no support; s = 1 draws only): ZERO-MLP (x) vs C-MLP (x_c with the draw's m_d).
# Hard cases (part A, k = 12, s = 1): the held-out domains' NEG / OTHER / TEMP sentences, 5-way, every method (C-methods
#   centred with the draw's m_d). NEG / OTHER correct = NOOP; TEMP correct = its event; as_surface = literal event.
# ZERO-MLP 13-way on every held-out sentence (Lab 15's L15_CV quantity) for every (fold, k, seed).
# Seeds 42, 1, 2 in one run; features extracted once per encoder.
# Machine lines: L16_FOLD, L16_FEAT, L16_CV13, L16_SHOT, L16_INIT, L16_HARD, L16_CONF, L16_ICL, L16_ZS0, then
#   L16_CONFIG as the LAST line. Part B lines carry old / new counts (held-out registered vs extra domains).
#   Aggregate: python tools/aggregate_lab16.py lab16_*.log
# Run (Colab GPU; the 7B needs >= 24 GB):  LAB16_PART=A python stage2_lab16_anchor_recognition.py > a.log
#   LAB16_MODELS=Q05,Q15,BGE / Q7 splits a part over sessions (the aggregator merges and refuses duplicates).
# Determinism (review L8): torch.use_deterministic_algorithms(True, warn_only=True), CUBLAS_WORKSPACE_CONFIG=:4096:8.
# Copied code is marked "(copied from Lab 15)"; everything else is new for Lab 16.
# ==============================================================================
import os, sys, gc, time, math, random, json, hashlib
from collections import Counter, defaultdict
import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # [review L8] before CUDA is initialised
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.use_deterministic_algorithms(True, warn_only=True)  # [review L8] deterministic kernels where they exist


def log(msg: str = ""):
    print(msg, flush=True)


# ---- pre-registered configuration (issue #25, "第一步之一 c / d") -----------------------------------------------------
SEEDS = (42, 1, 2)
FOLD_SEED = 2026
FOLD_SIZE = 4
PARTS = {
    "A": dict(n_folds=4, ks=(4, 8, 12), reg_k=12, shots=(1, 2, 3), models=("Q05", "Q15", "Q7", "BGE"),
              poolings={"Q05": ("mean", "prompt"), "Q15": ("mean", "prompt"), "Q7": ("mean", "prompt"),
                        "BGE": ("cls",)}),
    "B": dict(n_folds=7, ks=(4, 6, 8, 12, 16, 20, 24), reg_k=12, shots=(1,), models=("Q15", "Q7", "BGE"),
              poolings={"Q15": ("prompt",), "Q7": ("prompt",), "BGE": ("cls",)}),
}
N_DRAWS = 10
ICL_DRAWS = 2
METHODS = ["ZERO-MLP", "ZERO-FUNNEL", "RAW-NN", "MLP-NN", "FUNNEL-NN", "C-MLP", "C-MLP-NN", "C-FUNNEL-NN"]
INIT_METHODS = ["ZERO-MLP", "C-MLP"]
MODELS = {  # (copied from Lab 15) code -> (Hugging Face id, kind)
    "Q05": ("Qwen/Qwen2.5-0.5B", "qwen"),
    "Q15": ("Qwen/Qwen2.5-1.5B-Instruct", "qwen"),
    "Q7": ("Qwen/Qwen2.5-7B-Instruct", "qwen"),
    "BGE": ("BAAI/bge-m3", "bge"),
}
EXPECTED_DIM = {"Q05": 896, "Q15": 1536, "Q7": 3584, "BGE": 1024}
ZS_MODEL = "Q7"
SENT_MAXLEN = 128
FEAT_BATCH = 64
PROMPT_TMPL = '这句话："{s}"\n用一个词概括这句话描述的事件：'
REC_LR, REC_LR_FLOOR, REC_WD, REC_CLIP = 1e-3, 1e-4, 0.01, 1.0
CE_STEPS = 500
FUNNEL_STEPS = 400
HIDDEN, PROJ_DIM = 256, 64
PUSH_MARGIN = 0.1
CHECK_MIN_COS = 0.99

EXPECTED_ST_SHA256 = "ed6d16bd3e625198bba74d28e7b20f6bdba9c399a3eb94b9a0d1f113f11e3108"
EXPECTED_TPL_SHA256 = "1bfd769efdf6dd3d61ded0e6aaff3eb70b394af2c266e2f02cd1ed9b598bca70"
EXPECTED_HARD_SHA256 = "e145539bc8b37133a226a5d390e6cfebc2492d85e2cc090501d8464fc5afe8aa"
EXPECTED_EXTRA_SHA256 = "c26febd25091b0b797067127e5a1c02773c209f829408a7551f006a814bab232"  # accepted 2026-11-01: validator PASS, blind check 84/84
N_EXTRA = 12
HARD_CATS = ("NEG", "OTHER", "TEMP", "HEDGE")

# (copied from Lab 15) classes
CLASSES = ["L0_ok", "L1_ok", "L2_ok", "L3_ok", "L0_broken", "L1_broken", "L2_broken", "L3_broken", "BURST"]
CI = {c: i for i, c in enumerate(CLASSES)}
EVENTS = ("IN", "OUT", "BREAK", "FIX", "NOOP")
N_INIT = 8
N_SCLS = N_INIT + len(EVENTS)
SCLS_NAMES = CLASSES[:N_INIT] + [f"ev:{e}" for e in EVENTS]
N_EV = len(EVENTS)
NOOP_LOCAL = EVENTS.index("NOOP")

# (copied from Lab 15) zero-shot question and options (order 1 = this order; order 2 = reversed)
ZS_EVENT_Q = "这句话描述的是下面哪一种事件？"
ZS_EVENT_OPTS = ["系统里的量增加一个单位", "系统里的量减少一个单位", "安全机制坏了/失效", "安全机制修好/恢复",
                 "与系统无关的事"]                         # IN, OUT, BREAK, FIX, NOOP
LETTERS = "ABCDE"


# ---- (copied from Lab 15) helpers -----------------------------------------------------------------------
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

# ---- (copied from Lab 15) data: st sentence table, hard cases, folds -------------------------------------------
# ==============================================================================
# [0] Data
# ==============================================================================
def load_st():
    """state_tracking_v1: the sentence table of the 16 TRAIN domains, built from templates.json; the test stories are
    dropped unread; test-domain template strings go into `forbidden` only (for the leak assertion)."""
    dirs = ["/content", "benchmarks/state_tracking_v1"] + _script_dir_candidates("benchmarks", "state_tracking_v1")
    st_dir = next((d for d in dirs if os.path.exists(os.path.join(d, "stories.json"))
                   and os.path.exists(os.path.join(d, "templates.json"))), None)
    if st_dir is None:
        raise FileNotFoundError(f"no directory with both stories.json and templates.json in {dirs}")
    path, raw, sha = _read_first([os.path.join(st_dir, "stories.json")], "stories.json")
    assert sha == EXPECTED_ST_SHA256, f"stories.json is not the registered file: {sha}"
    data = json.loads(raw.decode("utf-8"))
    assert data.get("version") == "state_tracking_v1", data.get("version")
    assert data["classes"] == CLASSES, f"DATA ERROR: classes {data['classes']}"
    assert set(data["splits"]) == {"train", "val_seen", "test"}, set(data["splits"])
    data["splits"].pop("test")                     # test stories: never read in Lab 15
    _p, tpl_raw, tpl_sha = _read_first([os.path.join(st_dir, "templates.json")], "templates.json")
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
    log(f"[l16] st: {st_dir}/stories.json sha256={sha} | templates.json sha256={tpl_sha} | {len(train_doms)} train domains "
        f"(templates.json == stories' train split; {n_test_doms} test domains NOT used) | {len(texts)} sentences "
        f"({len(texts) // len(train_doms)} per domain); {len(used)} of them occur in the {n_story} train / val_seen "
        f"stories, all of which are in the table | per class: "
        + " ".join(f"{SCLS_NAMES[c]}={cnt[c]}" for c in range(N_SCLS)))
    return dict(name="st", sha=sha, tpl_sha=tpl_sha, texts=texts, cls=cls, dom=dom, doms=train_doms, n_cls=N_SCLS,
                cls_names=SCLS_NAMES, forbidden=forbidden, st_dir=st_dir,
                group=["init" if c < N_INIT else "ev" for c in cls])

def load_hard(st: dict):
    """Phase 1b: hard_cases.json next to stories.json; train domains only (asserted == st's 16 train domains)."""
    path, raw, sha = _read_first([os.path.join(st["st_dir"], "hard_cases.json")], "hard_cases.json")
    assert sha == EXPECTED_HARD_SHA256, f"hard_cases.json is not the registered file: {sha}"
    h = json.loads(raw.decode("utf-8"))
    assert h.get("version") == "state_tracking_v1_hard", h.get("version")
    assert h.get("templates_sha256") == st["tpl_sha"], "hard_cases.json was built from another templates.json"
    names = [d["name"] for d in h["domains"]]
    assert sorted(names) == sorted(st["doms"]) and len(set(names)) == len(names), "hard-case domains != st train domains"
    lab = {e: N_INIT + EVENTS.index(e) for e in EVENTS}
    texts, cat, surf, ycls, dom = [], [], [], [], []
    for d in h["domains"]:
        for it in d["items"]:
            assert it["category"] in HARD_CATS and it["surface"] in EVENTS[:4], it
            want = {"NEG": "NOOP", "OTHER": "NOOP", "TEMP": it["surface"], "HEDGE": "UNCERTAIN"}[it["category"]]
            assert it["label"] == want, it
            texts.append(it["text"])
            cat.append(it["category"])
            surf.append(lab[it["surface"]])
            ycls.append(-1 if it["label"] == "UNCERTAIN" else lab[it["label"]])
            dom.append(d["name"])
    assert len(set(texts)) == len(texts), "DATA ERROR: duplicate hard-case sentence"
    assert not set(texts) & set(st["texts"]), "DATA ERROR: a hard-case sentence equals a template sentence"
    cnt = Counter(cat)
    log(f"[l16] hard: {path} sha256={sha} | {len(texts)} sentences in {len(names)} train domains | "
        + " ".join(f"{c}={cnt[c]}" for c in HARD_CATS))
    return dict(name="hard", sha=sha, texts=texts, cat=cat, surface=surf, cls=ycls, dom=dom)


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

# ---- (copied from Lab 15) encoders ------------------------------------------------------------------------
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

# ---- (copied from Lab 15) 7B letter chooser and multiple-choice prompt ----------------------------------------
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

# ---- (copied from Lab 15) recognizer building blocks ---------------------------------------------------------
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


# ==============================================================================
# [0b] Part B data: the 12 extra train domains
# ==============================================================================
def load_extra(st: dict):
    """extra_domains.json next to stories.json; the same sentence-table construction as load_st (Lab 15)."""
    if EXPECTED_EXTRA_SHA256 is None:
        raise RuntimeError("part B: EXPECTED_EXTRA_SHA256 is not set (extra_domains.json not accepted yet)")
    path, raw, sha = _read_first([os.path.join(st["st_dir"], "extra_domains.json")], "extra_domains.json")
    assert sha == EXPECTED_EXTRA_SHA256, f"extra_domains.json is not the registered file: {sha}"
    x = json.loads(raw.decode("utf-8"))
    assert x.get("version") == "state_tracking_v1_extra", x.get("version")
    assert x.get("templates_sha256") == st["tpl_sha"], "extra_domains.json was built against another templates.json"
    doms = x["domains"]
    names = [d["name"] for d in doms]
    assert len(doms) == N_EXTRA and len(set(names)) == N_EXTRA, names
    _tp, tpl_raw, _ts = _read_first([os.path.join(st["st_dir"], "templates.json")], "templates.json")
    reg_names = [d["name"] for d in json.loads(tpl_raw.decode("utf-8"))["domains"]]  # [review L5] all 24 (16 + 8)
    assert len(reg_names) == 24 and not any(a in b or b in a for a in names for b in reg_names), \
        "an extra domain name equals or contains a registered (train or test) domain name"
    del tpl_raw
    texts, cls, dom = [], [], []
    for d in doms:
        assert d["split"] == "train", d["name"]
        lv_txt, sg_txt = d["init"]["level"], d["init"]["safeguard"]
        assert len(lv_txt) == 4 and set(sg_txt) == {"ok", "broken"}, d["name"]
        for lv in range(4):
            for sg in ("ok", "broken"):
                texts.append(lv_txt[lv] + "，" + sg_txt[sg] + "。")
                cls.append(CI[f"L{lv}_{sg}"])
                dom.append(d["name"])
        assert set(d["events"]) == set(EVENTS), d["name"]
        for e, n in zip(EVENTS, (7, 7, 6, 6, 6)):
            assert len(d["events"][e]) == n, (d["name"], e)
            for t in d["events"][e]:
                assert isinstance(t, str) and t, d["name"]
                texts.append(t)
                cls.append(N_INIT + EVENTS.index(e))
                dom.append(d["name"])
    assert len(set(texts)) == len(texts), "DATA ERROR: a sentence occurs twice in the extra table"
    assert not set(texts) & set(st["texts"]), "DATA ERROR: an extra sentence equals a registered sentence"
    assert not st["forbidden"] & set(texts), "LEAK / DATA ERROR: a test-domain string is in the extra table"
    log(f"[l16] extra: {path} sha256={sha} | {N_EXTRA} new train domains | {len(texts)} sentences")
    return dict(sha=sha, texts=texts, cls=cls, dom=dom, doms=sorted(names))


def merge_tables(st: dict, extra: dict):
    """Part B sentence table: the 16 registered + 12 extra train domains; `new` marks the extra domains."""
    out = dict(st)
    out["texts"] = st["texts"] + extra["texts"]
    out["cls"] = st["cls"] + extra["cls"]
    out["dom"] = st["dom"] + extra["dom"]
    out["doms"] = sorted(st["doms"] + extra["doms"])
    out["new_doms"] = set(extra["doms"])
    out["group"] = ["init" if c < N_INIT else "ev" for c in out["cls"]]
    return out


# ==============================================================================
# [1b] Feature extraction for one model (adapted from Lab 15 run_model: st (+ hard) tables, the part's poolings)
# ==============================================================================
def preflight16(part: str, models, data: dict, guard: LeakGuard):
    for code in models:
        mid, kind = MODELS[code]
        tok = AutoTokenizer.from_pretrained(mid)
        worst = 0
        for pooling in PARTS[part]["poolings"][code]:
            for name in ("st", "hard"):
                for s in data.get(name, {}).get("texts", []):
                    t = model_text(pooling, s)
                    guard.check(s, t)
                    worst = max(worst, len(tok(t).input_ids))
        if worst > SENT_MAXLEN:
            raise RuntimeError(f"DATA ERROR: {code}: a model text has {worst} tokens > max_length {SENT_MAXLEN}")
        msg = ""
        if code == ZS_MODEL and part == "A":
            letter_token_ids(tok, LETTERS)
            st = data["st"]
            d0 = st["doms"][0]
            rows5 = ev_rows_by_class(st, d0)  # [review L4] one example per event type, as in run_icl
            ex = [(st["texts"][r[0]], ZS_EVENT_OPTS[e]) for e, r in enumerate(rows5)]
            q = icl_mcq(d0, ex, st["texts"][rows5[0][1]], ZS_EVENT_OPTS)
            guard.check(st["texts"][rows5[0][1]], q)
            p = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
            msg = f" | ICL: letters A-E single tokens, an example prompt has {len(tok(p).input_ids)} tokens"
        log(f"[l16] preflight {code}: tokenizer ok, longest model text {worst} tokens (limit {SENT_MAXLEN}){msg}")


def run_model16(part: str, code: str, data: dict, guard: LeakGuard, zs_fn=None):
    mid, kind = MODELS[code]
    log(f"\n=== [1] Loading '{mid}' ({code}, frozen, bf16) ===")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(mid)
    tok.padding_side = "right"
    if kind == "qwen":
        model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map=DEVICE)
        pool_fn = qwen_pool
    else:
        model = AutoModel.from_pretrained(mid, dtype=torch.bfloat16, device_map=DEVICE)
        pool_fn = bge_pool
    hid = model.config.hidden_size
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    _p0 = next(model.parameters(), None)
    assert (_p0 is not None and _p0.dtype == torch.bfloat16) or "FAKE" in getattr(transformers, "__version__", "")
    assert hid == EXPECTED_DIM[code], (code, hid)
    log(f"[l16] {code}: loaded in {time.time() - t0:.1f}s | hidden={hid}"
        + (f" | GPU memory allocated {torch.cuda.memory_allocated() / 2 ** 30:.1f} GiB" if DEVICE == "cuda" else ""))
    feats = {}
    for pooling in PARTS[part]["poolings"][code]:
        enc_name = code if kind == "bge" else f"{code}-{pooling}"
        feats[enc_name] = {}
        for dname in ("st", "hard"):
            if dname not in data:
                continue
            t1 = time.time()
            H, n_tok, tids = extract(tok, pool_fn, model, data[dname]["texts"], pooling, guard, f"{enc_name}/{dname}")
            mc, mr, nchk = unbatched_check(tok, pool_fn, model, data[dname]["texts"], pooling, H, n_tok, tids,
                                           f"{enc_name}/{dname}")
            feats[enc_name][dname] = H
            sec = time.time() - t1
            log(f"[l16] features {enc_name}/{dname}: {tuple(H.shape)} float32 | max tokens {max(n_tok)} | {sec:.1f}s "
                f"| unbatched check on {nchk} rows: min cos {mc:.6f}, max |diff|/max|x| {mr:.2e} | fingerprint "
                f"{float(H.double().sum()).hex()}")
            log(f"L16_FEAT part={part} enc={enc_name} data={dname} n={H.shape[0]} dim={H.shape[1]} "
                f"max_tokens={max(n_tok)} sec={sec:.1f} check_rows={nchk} check_min_cos={mc:.6f}")
    if zs_fn is not None:
        zs_fn(tok, model)
    del model, tok
    free_gpu()
    log(f"[l16] {code}: done in {time.time() - t0:.1f}s, model freed")
    return feats


# ==============================================================================
# [2] 7B in-context examples (part A) and the no-example zero-shot on the same queries
# ==============================================================================
def icl_mcq(domain: str, examples, sentence: str, options) -> str:
    """Lab 15 mcq with the domain's labelled examples (sentence + correct option TEXT, so the option order of the
    question does not matter) placed before the sentence."""
    L = LETTERS[:len(options)]
    ans = "、".join(L[:-1]) + f" 或 {L[-1]}"
    ex = "".join(f"例句：{s}\n事件：{o}\n\n" for s, o in examples)
    return (f"领域：{domain}\n下面是这个领域里已经标好事件类型的例句：\n\n{ex}"
            f"句子：{sentence}\n\n问题：{ZS_EVENT_Q}\n" + "".join(f"{l}. {o}\n" for l, o in zip(L, options))
            + f"\n只回答一个字母（{ans}）。")


def draw_rng(part: str, seed, fold, dom: str, s: int, r: int) -> random.Random:
    h = hashlib.sha256(f"{part}|{seed}|{fold}|{dom}|{s}|{r}".encode("utf-8")).hexdigest()
    return random.Random(int(h[:16], 16))


def ev_rows_by_class(table: dict, dom: str):
    """-> 5 lists (IN .. NOOP) of row indices of `dom`'s event sentences, in table order."""
    return [[i for i, (c, d) in enumerate(zip(table["cls"], table["dom"])) if d == dom and c == N_INIT + e]
            for e in range(N_EV)]


def draw_support(rows5, s: int, rng: random.Random):
    """-> (support rows, their local class 0..4, query rows, their local class), s per class."""
    sup, ysup, qry, yq = [], [], [], []
    for e, rows in enumerate(rows5):
        assert len(rows) > s, (len(rows), s)
        pick = set(rng.sample(rows, s))
        for i in rows:
            if i in pick:
                sup.append(i)
                ysup.append(e)
            else:
                qry.append(i)
                yq.append(e)
    return sup, ysup, qry, yq


def run_icl(st: dict, choose, guard: LeakGuard):
    """Part A reference: Qwen2.5-7B-Instruct with the domain's s = 1 examples (ICL) vs without (ZS0), same queries."""
    t0 = time.time()
    zs0 = {}
    for i, (s, c, d) in enumerate(zip(st["texts"], st["cls"], st["dom"])):
        if c >= N_INIT:
            o1, o2, _l1, _l2 = ask_both_orders(choose, d, s, ZS_EVENT_Q, ZS_EVENT_OPTS, c - N_INIT, guard)
            zs0[i] = (o1, o2)
    n = len(ZS_EVENT_OPTS)
    tot = {"ICL": [0, 0, 0, 0], "ZS0": [0, 0, 0, 0]}
    for d in st["doms"]:
        rows5 = ev_rows_by_class(st, d)
        for r in range(ICL_DRAWS):
            rng = draw_rng("ICL", 0, 0, d, 1, r)
            sup, ysup, qry, yq = draw_support(rows5, 1, rng)
            order = list(range(len(sup)))
            rng.shuffle(order)
            ex = [(st["texts"][sup[j]], ZS_EVENT_OPTS[ysup[j]]) for j in order]
            for t, _ in ex:
                guard.check(t)
            cnt = {"ICL": [0, 0, 0, 0], "ZS0": [0, 0, 0, 0]}
            for i, e in zip(qry, yq):
                q1 = icl_mcq(d, ex, st["texts"][i], ZS_EVENT_OPTS)
                q2 = icl_mcq(d, ex, st["texts"][i], ZS_EVENT_OPTS[::-1])
                for q in (q1, q2):
                    guard.check(st["texts"][i], q)
                a1 = choose(q1, n) == e
                a2 = choose(q2, n) == n - 1 - e
                for key, (x1, x2) in (("ICL", (a1, a2)), ("ZS0", zs0[i])):
                    cnt[key][0] += int(x1 and x2)
                    cnt[key][1] += int(x1)
                    cnt[key][2] += int(x2)
                    cnt[key][3] += 1
            for key in ("ICL", "ZS0"):
                b, o1, o2, nq = cnt[key]
                log(f"L16_{key} part=A dom={d} draw={r} correct={b} total={nq} order1={o1} order2={o2}")
                for j in range(4):
                    tot[key][j] += cnt[key][j]
    log(f"\n=== [2] 7B with the domain's examples (ICL) vs without (ZS0), same queries | {time.time() - t0:.0f}s ===")
    for key in ("ICL", "ZS0"):
        b, o1, o2, nq = tot[key]
        log(f"  {key}: both {b}/{nq} ({100 * b / max(nq, 1):.1f}%) | order1 {o1} | order2 {o2}")


# ==============================================================================
# [3] Recognizers (Lab 15 recipes) and the anchor methods
# ==============================================================================
def train_mlp(X, y, seed: int):
    """(Lab 15 fit_scorer, rec MLP) seed_all before init; train_loop seeds again before training."""
    seed_all(seed)
    model = nn.Sequential(nn.Linear(X.shape[1], HIDDEN), nn.GELU(), nn.Linear(HIDDEN, N_SCLS)).to(DEVICE).float()
    loss = train_loop(model, lambda m: F.cross_entropy(m(X), y), CE_STEPS, seed)
    return model, loss


def train_funnel(X, y, seed: int):
    """(Lab 15 fit_scorer, rec FUNNEL) -> (model, anchors = normalised training class means in funnel space, loss)."""
    seed_all(seed)
    model = Funnel(X.shape[1], PROJ_DIM).to(DEVICE).float()
    loss = train_loop(model, lambda m: funnel_loss(m(X), y, N_SCLS), FUNNEL_STEPS, seed)
    with torch.no_grad():
        anchors = class_means(model(X), y, N_SCLS)
    return model, anchors, loss


def mlp_hidden(model, X):
    return F.normalize(model[1](model[0](X)), p=2, dim=-1)


def class_center(Fd, rows5):
    """Class-balanced event mean: mean over the 5 event classes of the mean feature of that class's rows."""
    return torch.stack([Fd[torch.tensor(r, device=Fd.device)].mean(dim=0) for r in rows5]).mean(dim=0)


def nn_predict(Zq, Zs, ys):
    """Nearest class prototype (prototype = normalised mean of the class's normalised support vectors)."""
    Zq, Zs = F.normalize(Zq, p=2, dim=-1), F.normalize(Zs, p=2, dim=-1)
    ys_t = torch.tensor(ys, device=Zs.device)
    P = F.normalize(torch.stack([Zs[ys_t == e].mean(dim=0) for e in range(N_EV)]), p=2, dim=-1)
    return (Zq @ P.T).argmax(-1)


@torch.no_grad()
def method_preds(M, Fd, X, rows_q, sup, ysup, m_d, Fextra=None):
    """Event predictions (local 0..4) of every method for query rows `rows_q` of one held-out domain.
    M = trained models; X = mu-normalised features of ALL table rows; m_d = the domain centre from its support.
    Fextra (optional): raw features of extra query rows (hard cases), replacing Fd[rows_q] / X[rows_q]."""
    q = torch.tensor(rows_q, device=DEVICE) if Fextra is None else None
    s_t = torch.tensor(sup, device=DEVICE)
    Fq = Fd[q] if Fextra is None else Fextra
    Xq = X[q] if Fextra is None else F.normalize(Fextra - M["mu"], p=2, dim=-1)
    Xs = X[s_t]
    Xcq = F.normalize(Fq - m_d, p=2, dim=-1)
    Xcs = F.normalize(Fd[s_t] - m_d, p=2, dim=-1)
    ev = slice(N_INIT, N_SCLS)
    out = {}
    out["ZERO-MLP"] = M["mlp"](Xq)[:, ev].argmax(-1)
    zq = M["fun"](Xq)
    out["ZERO-FUNNEL"] = (zq @ M["fun_anc"][ev].T).argmax(-1)
    out["RAW-NN"] = nn_predict(Xq, Xs, ysup)
    out["MLP-NN"] = nn_predict(mlp_hidden(M["mlp"], Xq), mlp_hidden(M["mlp"], Xs), ysup)
    out["FUNNEL-NN"] = nn_predict(zq, M["fun"](Xs), ysup)
    out["C-MLP"] = M["cmlp"](Xcq)[:, ev].argmax(-1)
    out["C-MLP-NN"] = nn_predict(mlp_hidden(M["cmlp"], Xcq), mlp_hidden(M["cmlp"], Xcs), ysup)
    out["C-FUNNEL-NN"] = nn_predict(M["cfun"](Xcq), M["cfun"](Xcs), ysup)
    assert list(out) == METHODS
    return {k: v.tolist() for k, v in out.items()}


def run_cv(part: str, table: dict, enc: str, H, folds, pools, hard=None, Hh=None):
    """Every (fold, k, seed) of one part for one encoder: train MLP / FUNNEL / C-MLP / C-FUNNEL on the k training
    domains, then score the held-out domains (ZERO-MLP 13-way; every anchor method for every s and draw)."""
    cfg = PARTS[part]
    Fd = H.to(DEVICE)
    y = torch.tensor(table["cls"], device=DEVICE)
    new_doms = table.get("new_doms", set())
    rows5_of = {d: ev_rows_by_class(table, d) for d in table["doms"]}
    init_rows_of = {d: [i for i, (c, dd) in enumerate(zip(table["cls"], table["dom"])) if dd == d and c < N_INIT]
                    for d in table["doms"]}
    if hard is not None:
        Fh = Hh.to(DEVICE)
        hard_rows_of = {d: [i for i, (dd, cat) in enumerate(zip(hard["dom"], hard["cat"])) if dd == d and cat != "HEDGE"]
                        for d in table["doms"]}
    t0 = time.time()
    for f, held in enumerate(folds):
        te = [i for i, d in enumerate(table["dom"]) if d in held]
        te_t = torch.tensor(te, device=DEVICE)
        for k in cfg["ks"]:
            tr_doms = pools[f][:k]
            assert len(tr_doms) == k and not set(tr_doms) & set(held)
            tr = [i for i, d in enumerate(table["dom"]) if d in tr_doms]
            assert not set(tr) & set(te)
            tr_t = torch.tensor(tr, device=DEVICE)
            mu = Fd[tr_t].mean(dim=0)
            X = F.normalize(Fd - mu, p=2, dim=-1)       # every row; training rows define mu only
            centers = {d: class_center(Fd, rows5_of[d]) for d in tr_doms}
            Xc_tr = F.normalize(Fd[tr_t] - torch.stack([centers[table["dom"][i]] for i in tr]), p=2, dim=-1)
            y_tr = y[tr_t]
            for seed in SEEDS:
                M = {"mu": mu}
                M["mlp"], l1 = train_mlp(X[tr_t], y_tr, seed)
                M["fun"], M["fun_anc"], l2 = train_funnel(X[tr_t], y_tr, seed)
                M["cmlp"], l3 = train_mlp(Xc_tr, y_tr, seed)
                M["cfun"], M["cfun_anc"], l4 = train_funnel(Xc_tr, y_tr, seed)
                for m in ("mlp", "fun", "cmlp", "cfun"):
                    M[m].eval()
                with torch.no_grad():
                    p13 = M["mlp"](X[te_t]).argmax(-1)
                    fit = int((M["mlp"](X[tr_t]).argmax(-1) == y_tr).sum())
                hit = (p13 == y[te_t]).tolist()
                grp = [table["group"][i] for i in te]
                old = [table["dom"][i] not in new_doms for i in te]
                log(f"L16_CV13 part={part} enc={enc} k={k} fold={f} seed={seed} correct={sum(hit)} total={len(hit)} "
                    f"ev_correct={sum(h for h, g in zip(hit, grp) if g == 'ev')} ev_total={grp.count('ev')} "
                    f"init_correct={sum(h for h, g in zip(hit, grp) if g == 'init')} init_total={grp.count('init')} "
                    f"c_old={sum(h for h, o in zip(hit, old) if o)} t_old={sum(old)} "
                    f"c_new={sum(h for h, o in zip(hit, old) if not o)} t_new={len(old) - sum(old)} "
                    f"fit_correct={fit} fit_total={len(tr)} losses={l1:.4f},{l2:.4f},{l3:.4f},{l4:.4f}")
                for s in cfg["shots"]:
                    cnt = {m: [0, 0, 0, 0, 0, 0] for m in METHODS}       # c, t, c_old, t_old, c_new, t_new
                    conf = {m: [[0] * N_EV for _ in range(N_EV)] for m in METHODS}
                    init_cnt = {m: [0, 0] for m in INIT_METHODS}
                    hard_cnt = {(m, cat): [0, 0, 0] for m in METHODS for cat in HARD_CATS[:3]}
                    for r in range(N_DRAWS):
                        for d in held:
                            rng = draw_rng(part, seed, f, d, s, r)
                            sup, ysup, qry, yq = draw_support(rows5_of[d], s, rng)
                            local5 = [[sup[j] for j in range(len(sup)) if ysup[j] == e] for e in range(N_EV)]
                            m_d = class_center(Fd, local5)
                            P = method_preds(M, Fd, X, qry, sup, ysup, m_d)
                            is_old = d not in new_doms
                            for m in METHODS:
                                hits = [int(p == t) for p, t in zip(P[m], yq)]
                                c = cnt[m]
                                c[0] += sum(hits)
                                c[1] += len(hits)
                                c[2 if is_old else 4] += sum(hits)
                                c[3 if is_old else 5] += len(hits)
                                if k == cfg["reg_k"] and s == 1:
                                    for p, t in zip(P[m], yq):
                                        conf[m][t][p] += 1
                            if s == 1:  # initial-state sentences, 8-way
                                ir = torch.tensor(init_rows_of[d], device=DEVICE)
                                yi = y[ir]
                                with torch.no_grad():
                                    pz = M["mlp"](X[ir])[:, :N_INIT].argmax(-1)
                                    pc = M["cmlp"](F.normalize(Fd[ir] - m_d, p=2, dim=-1))[:, :N_INIT].argmax(-1)
                                for m, p in (("ZERO-MLP", pz), ("C-MLP", pc)):
                                    init_cnt[m][0] += int((p == yi).sum())
                                    init_cnt[m][1] += len(init_rows_of[d])
                            if hard is not None and k == cfg["reg_k"] and s == 1:
                                hr = hard_rows_of[d]
                                assert hr, d
                                Ph = method_preds(M, Fd, X, None, sup, ysup, m_d,
                                                  Fextra=Fh[torch.tensor(hr, device=DEVICE)])
                                for m in METHODS:
                                    for j, i in enumerate(hr):
                                        hc = hard_cnt[(m, hard["cat"][i])]
                                        hc[0] += int(Ph[m][j] == hard["cls"][i] - N_INIT)
                                        hc[1] += 1
                                        hc[2] += int(Ph[m][j] == hard["surface"][i] - N_INIT)
                    for m in METHODS:
                        c = cnt[m]
                        log(f"L16_SHOT part={part} enc={enc} method={m} k={k} s={s} fold={f} seed={seed} correct={c[0]} "
                            f"total={c[1]} c_old={c[2]} t_old={c[3]} c_new={c[4]} t_new={c[5]}")
                    if s == 1:
                        for m in INIT_METHODS:
                            log(f"L16_INIT part={part} enc={enc} method={m} k={k} fold={f} seed={seed} "
                                f"correct={init_cnt[m][0]} total={init_cnt[m][1]}")
                    if k == cfg["reg_k"] and s == 1:
                        for m in METHODS:
                            flat = ",".join(str(conf[m][t][p]) for t in range(N_EV) for p in range(N_EV))
                            log(f"L16_CONF part={part} enc={enc} method={m} k={k} fold={f} seed={seed} m={flat}")
                    if hard is not None and k == cfg["reg_k"] and s == 1:
                        for m in METHODS:
                            for cat in HARD_CATS[:3]:
                                hc = hard_cnt[(m, cat)]
                                log(f"L16_HARD part={part} enc={enc} method={m} fold={f} seed={seed} cat={cat} "
                                    f"correct={hc[0]} total={hc[1]} as_surface={hc[2]}")
        log(f"[l16] {enc}: fold {f} done ({time.time() - t0:.0f}s)")


# ==============================================================================
# main
# ==============================================================================
def main():
    t_start = time.time()
    part = os.environ.get("LAB16_PART", "").strip()
    if part not in PARTS:
        raise ValueError(f"LAB16_PART must be one of {list(PARTS)}, got {part!r}")
    cfg = PARTS[part]
    log(f"[l16] part={part} device={DEVICE} | seeds={SEEDS} | python {sys.version.split()[0]} | torch "
        f"{torch.__version__} | numpy {np.__version__} | transformers {getattr(transformers, '__version__', '?')}")
    try:
        with open(os.path.abspath(__file__), "rb") as _f:
            script_sha = hashlib.sha256(_f.read()).hexdigest()
    except (NameError, OSError):
        script_sha = "unavailable"  # executed through a notebook kernel; the runner records the repo file's sha
    log(f"[l16] script sha256={script_sha}")
    env_tag = f"torch={torch.__version__},transformers={getattr(transformers, '__version__', '?')},device={DEVICE}"
    if DEVICE == "cuda":
        env_tag += f",gpu={torch.cuda.get_device_name(0).replace(' ', '_')}"
    if "FAKE" in getattr(transformers, "__version__", ""):
        log("!" * 120 + "\n!!! [l16] FAKE transformers stub: TEST RUN, NOT a registered run\n" + "!" * 120)

    models_env = os.environ.get("LAB16_MODELS", "").strip()
    models = [m.strip() for m in models_env.split(",") if m.strip()] if models_env else list(cfg["models"])
    bad = [m for m in models if m not in cfg["models"]]
    if bad or len(set(models)) != len(models):
        raise ValueError(f"LAB16_MODELS={models_env!r}: unknown or repeated code(s); part {part} allows {cfg['models']}")
    models = [m for m in cfg["models"] if m in models]
    if models != list(cfg["models"]):
        log(f"[l16] NOTE: this log covers only {models}; merge the other logs with tools/aggregate_lab16.py")

    st = load_st()
    data = {}
    extra_sha = "-"
    if part == "A":
        data["st"] = st
        data["hard"] = load_hard(st)
        allowed = st["texts"] + data["hard"]["texts"]
    else:
        extra = load_extra(st)
        extra_sha = extra["sha"]
        data["st"] = merge_tables(st, extra)
        allowed = data["st"]["texts"]
    table = data["st"]
    guard = LeakGuard(allowed, st["forbidden"])
    folds, pools = make_folds(table["doms"], cfg["n_folds"])
    log(f"[l16] part {part}: {len(table['doms'])} train domains, {cfg['n_folds']} folds x {FOLD_SIZE} held out, "
        f"k in {cfg['ks']}, shots {cfg['shots']}, {N_DRAWS} draws, methods {METHODS}")
    for f in range(cfg["n_folds"]):
        log(f"L16_FOLD part={part} fold={f} heldout={','.join(folds[f])} pool={','.join(pools[f])}")
    preflight16(part, models, data, guard)

    icl_done = False
    encs_run = []
    for code in models:
        zs_fn = None
        if code == ZS_MODEL and part == "A":
            def zs_fn(tok, model):
                run_icl(table, make_chooser(tok, model), guard)
        feats = run_model16(part, code, data, guard, zs_fn)
        icl_done |= zs_fn is not None
        for enc, fe in feats.items():
            encs_run.append(enc)
            run_cv(part, table, enc, fe["st"], folds, pools,
                   hard=data.get("hard"), Hh=fe.get("hard"))
        del feats
        free_gpu()
    log(f"\n[l16] leak guard: {guard.n_checked} encoder / 7B inputs checked, all from the {len(allowed)} training-domain "
        f"sentences; none equals or contains any of the {len(st['forbidden'])} test-domain template strings")
    log(f"[l16] total time {time.time() - t_start:.1f}s")
    log(f"L16_CONFIG part={part} data_sha_st={st['sha']} templates_sha={st['tpl_sha']} "
        f"data_sha_hard={data['hard']['sha'] if 'hard' in data else '-'} data_sha_extra={extra_sha} env={env_tag} "
        f"script_sha={script_sha} models={','.join(models)} encs={','.join(encs_run)} seeds={','.join(map(str, SEEDS))} "
        f"shots={','.join(map(str, cfg['shots']))} draws={N_DRAWS} icl={int(icl_done)}")


if __name__ == "__main__":
    main()
