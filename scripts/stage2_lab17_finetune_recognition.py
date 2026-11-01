# ==============================================================================
# Stage 2 / Lab 17: strengthen cross-domain recognition (B1). PRE-REGISTERED on issue #25 (comment "第一步之一 e",
#   2026-11-01, before any code). Numbers only, NO verdict.
# Question: with (1) a LoRA-fine-tuned eye, (2) a "what changes x direction" factorised head and (3) hard sentences
#   in training, do event sentences of domains never trained on reach 95% (one example per event type), and is
#   negation recognised? 2 x 2 x 2 factorial per fine-tunable eye; Q7 frozen reference (4 cells).
# Hard rule: the 8 test domains of state_tracking_v1 are NOT used (Lab 15/16 loaders and leak guard).
# Data (sha256 pinned)
#   st     stories.json / templates.json: the 16 registered train domains (Lab 15 table: 8 init + 32 event sentences).
#   extra  extra_domains.json v1.1: the 12 Lab 16 extra train domains; the only change from v1 renames 渡轮's
#          safeguard 超载报警器 -> 限载感应器 (v1 contained the test safeguard name 超载报警; validator X4 now checks).
#   hard   hard_cases.json (16 registered) + hard_cases_extra.json (12 extra; validator PASS, blind check 144/144).
#          Only NEG / OTHER (label NOOP) and TEMP (label = its event) are used; HEDGE is not.
# Folds: Lab 15 make_folds over the 28 domains (Random(2026)), 4 folds x 7 held out; the other 21 train. Seeds 42,1,2.
# Eyes
#   Q15  Qwen/Qwen2.5-1.5B-Instruct, Lab 15 prompt '这句话："{s}"\n用一个词概括这句话描述的事件：', last real token.
#   Q3E  Qwen/Qwen3-Embedding-0.6B, official query format 'Instruct: {task}\nQuery:{s}', last real token.
#   Q7   Qwen/Qwen2.5-7B-Instruct, as Q15; frozen only (reference).
#   Right padding, index = attention_mask.sum - 1 (asserted), max_length 128, no truncation (asserted).
# Factor 1, LoRA (off / on; Q15 and Q3E only): rank 16, alpha 32, dropout 0.05 on every q/k/v/o projection; base
#   weights frozen (bf16), LoRA weights float32 (A Kaiming-uniform, B zero). Trained per (eye, fold, seed, hard) on
#   the fold's 21 training domains only (+ their NEG/OTHER/TEMP sentences when hard = on). Each step: 8 training
#   domains x 12 of their sentences (rng from (seed, fold, hard)); loss = cross-domain supervised contrastive loss
#   (Lab 15 XDOM masks: positives = same class, other domain; tau 0.1) on the normalised pooled features + 13-class
#   cross-entropy of a linear head on the pooled features (head discarded). 150 steps, AdamW lr 2e-4 (wd 0), 10
#   linear warm-up steps then cosine to 0, clip 1.0. Then all features are re-extracted with the adapted eye.
# Factor 2, head (FLAT / FACT): FLAT = Lab 16 MLP D -> 256 -> GELU -> 13. FACT = the same trunk with three heads:
#   init 8 | kind 3 (quantity / safeguard / nothing) | direction 2 (+ / -), shared by quantity and safeguard
#   (IN, FIX = +; OUT, BREAK = -). Loss: init rows CE(init); event rows CE(kind) + CE(direction) (NOOP: kind only).
#   Event score = log P(kind) + log P(direction) (NOOP: log P(nothing)). Both: Lab 15 recipe (AdamW 1e-3 -> 1e-4
#   cosine, wd 0.01, clip 1.0, 500 full-batch steps, seed_all before init and before training).
# Factor 3, hard (off / on): the training domains' NEG / OTHER / TEMP sentences join the LoRA and head training.
# Readout (Lab 16)
#   ZERO (s = 0)  x = normalize(f - mu), mu = mean of the fold's normal training rows; head on x.
#   C    (s = 1)  domain-centred: training domain centre = class-balanced mean of its event features; held-out domain
#                 centre = mean of its support (one example per event type); x_c = normalize(f - centre); head on x_c.
#                 Hard training rows are centred with their own domain's centre.
#   Support: per held-out domain, one random sentence per event type, R = 10 draws per (fold, seed), rng =
#   sha256("17|seed|fold|domain|1|draw") (same supports for every eye and cell); queries = its other 27 event sentences.
#   ZERO is scored on the same queries. Events 5-way (sentence group known); init 8-way; hard NEG / OTHER / TEMP of
#   the held-out domains 5-way (correct = NOOP / NOOP / its event; as_surface = the literal event).
# Primary (registered): eye with the better main cell (LoRA on + FACT + hard on, method C), event accuracy on the
#   held-out REGISTERED-style domains (the 16 of templates.json) >= 95%. Negation: same eye, main cell, registered
#   domains NEG -> NOOP >= 90%, and normal events at most 1 point below (hard off, all else equal).
# Machine lines: L17_FOLD, L17_FEAT, L17_LORA, L17_EV (+ conf), L17_INIT, L17_HARD, then L17_CONFIG as the LAST line.
#   Aggregate: python tools/aggregate_lab17.py lab17_*.log
# Run: LAB17_EYES=Q15 python stage2_lab17_finetune_recognition.py > a.log ; LAB17_EYES=Q3E,Q7 ... > b.log
# Code marked "(copied from Lab 15 / 16)" is reused verbatim; everything else is new for Lab 17.
# ==============================================================================
import os, sys, gc, time, math, random, json, hashlib
from collections import Counter, defaultdict
import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.use_deterministic_algorithms(True, warn_only=True)


def log(msg: str = ""):
    print(msg, flush=True)


# ---- pre-registered configuration (issue #25, "第一步之一 e") ----------------------------------------------------------
SEEDS = (42, 1, 2)
FOLD_SEED = 2026
FOLD_SIZE = 7
N_FOLDS = 4
N_DRAWS = 10
EYES = {  # code -> (Hugging Face id, kind, template, hidden size, fine-tuned?)
    "Q15": ("Qwen/Qwen2.5-1.5B-Instruct", "causal", "q25", 1536, True),
    "Q3E": ("Qwen/Qwen3-Embedding-0.6B", "embed", "q3e", 1024, True),
    "Q7": ("Qwen/Qwen2.5-7B-Instruct", "causal", "q25", 3584, False),
}
EYE_ORDER = ("Q15", "Q3E", "Q7")
PROMPT_TMPL = '这句话："{s}"\n用一个词概括这句话描述的事件：'
Q3E_TASK = "Identify which abstract event or state of the system this sentence describes"
Q3E_TMPL = "Instruct: " + Q3E_TASK + "\nQuery:{s}"
SENT_MAXLEN = 128
FEAT_BATCH = 64
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
LORA_STEPS, LORA_LR, LORA_WARMUP, LORA_CLIP = 150, 2e-4, 10, 1.0
LORA_DOMS_PER_STEP, LORA_PER_DOM = 8, 12
SUPCON_TAU = 0.1
REC_LR, REC_LR_FLOOR, REC_WD, REC_CLIP = 1e-3, 1e-4, 0.01, 1.0
CE_STEPS = 500
HIDDEN = 256
HEADS = ("FLAT", "FACT")
METHODS = ("ZERO", "C")

EXPECTED_ST_SHA256 = "ed6d16bd3e625198bba74d28e7b20f6bdba9c399a3eb94b9a0d1f113f11e3108"
EXPECTED_TPL_SHA256 = "1bfd769efdf6dd3d61ded0e6aaff3eb70b394af2c266e2f02cd1ed9b598bca70"
EXPECTED_HARD_SHA256 = "e145539bc8b37133a226a5d390e6cfebc2492d85e2cc090501d8464fc5afe8aa"
EXPECTED_EXTRA_SHA256 = "f44dc28a2a2df593981c919949b47e13304a814cec4e960123564206190ca7cf"   # extra_domains.json v1.1
EXPECTED_HARD_EXTRA_SHA256 = "e31ad5839bbc1133c6d3e6e5caec6a5125a4ffd9bcffbf330fb793b58416fc53"
N_EXTRA = 12
HARD_CATS = ("NEG", "OTHER", "TEMP", "HEDGE")
USED_HARD = ("NEG", "OTHER", "TEMP")

# (copied from Lab 15) classes
CLASSES = ["L0_ok", "L1_ok", "L2_ok", "L3_ok", "L0_broken", "L1_broken", "L2_broken", "L3_broken", "BURST"]
CI = {c: i for i, c in enumerate(CLASSES)}
EVENTS = ("IN", "OUT", "BREAK", "FIX", "NOOP")
N_INIT = 8
N_SCLS = N_INIT + len(EVENTS)
SCLS_NAMES = CLASSES[:N_INIT] + [f"ev:{e}" for e in EVENTS]
N_EV = len(EVENTS)
# factorised event code: kind 0 quantity / 1 safeguard / 2 nothing; direction 0 + / 1 -
EV_KIND = (0, 0, 1, 1, 2)
EV_DIR = (0, 1, 1, 0, -1)


# ---- (copied from Lab 16 / 15) helpers --------------------------------------------------------------
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


# ---- (copied from Lab 16 / 15) data ---------------------------------------------------------------------
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
    log(f"[l17] st: {st_dir}/stories.json sha256={sha} | templates.json sha256={tpl_sha} | {len(train_doms)} train domains "
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
    log(f"[l17] hard: {path} sha256={sha} | {len(texts)} sentences in {len(names)} train domains | "
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


# ---- (copied from Lab 16 / 15) leak guard, padding check, gpu ---------------------------------------------
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



def free_gpu():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# ---- (copied from Lab 15) training loop -------------------------------------------------------------
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



# ---- (copied from Lab 16) extra domains, supports, domain centre -------------------------------------
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
    log(f"[l17] extra: {path} sha256={sha} | {N_EXTRA} new train domains | {len(texts)} sentences")
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



def class_center(Fd, rows5):
    """Class-balanced event mean: mean over the 5 event classes of the mean feature of that class's rows."""
    return torch.stack([Fd[torch.tensor(r, device=Fd.device)].mean(dim=0) for r in rows5]).mean(dim=0)




# ==============================================================================
# [0b] extra hard cases + merged hard table
# ==============================================================================
def load_hard_extra(table: dict):
    """hard_cases_extra.json: the 12 extra domains, same item format as hard_cases.json."""
    path, raw, sha = _read_first([os.path.join(table["st_dir"], "hard_cases_extra.json")], "hard_cases_extra.json")
    assert sha == EXPECTED_HARD_EXTRA_SHA256, f"hard_cases_extra.json is not the registered file: {sha}"
    h = json.loads(raw.decode("utf-8"))
    assert h.get("version") == "state_tracking_v1_hard_extra", h.get("version")
    assert h.get("extra_domains_sha256") == EXPECTED_EXTRA_SHA256, "hard_cases_extra.json built on another extra file"
    names = [d["name"] for d in h["domains"]]
    assert sorted(names) == sorted(table["new_doms"]) and len(set(names)) == len(names), names
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
    log(f"[l17] hard extra: {path} sha256={sha} | {len(texts)} sentences in {len(names)} extra domains")
    return dict(sha=sha, texts=texts, cat=cat, surface=surf, cls=ycls, dom=dom)


def merge_hard(a: dict, b: dict, table: dict):
    out = {k: a[k] + b[k] for k in ("texts", "cat", "surface", "cls", "dom")}
    assert len(set(out["texts"])) == len(out["texts"]), "DATA ERROR: duplicate hard sentence across the two files"
    assert not set(out["texts"]) & set(table["texts"]), "DATA ERROR: a hard sentence equals a table sentence"
    assert set(out["dom"]) == set(table["doms"])
    out["sha"], out["sha_extra"] = a["sha"], b["sha"]
    return out


# ==============================================================================
# [1] Eyes: tokenisation, last-token pooling (with or without gradient), LoRA
# ==============================================================================
def model_text(eye: str, s: str) -> str:
    return (Q3E_TMPL if EYES[eye][2] == "q3e" else PROMPT_TMPL).format(s=s)


def backbone_of(eye: str, model):
    return model.model if EYES[eye][1] == "causal" else model


def pool_last(tok, backbone, mtexts):
    """Last real token of the last hidden layer, right padding (asserted); float32 (B, D) + real lengths."""
    enc = tok(mtexts, return_tensors="pt", padding=True, truncation=True, max_length=SENT_MAXLEN).to(DEVICE)
    m = enc.attention_mask
    n = _right_padded(m)
    h = backbone(input_ids=enc.input_ids, attention_mask=m).last_hidden_state
    rows = torch.arange(h.shape[0], device=h.device)
    return h[rows, n - 1].float(), n


@torch.no_grad()
def extract17(tok, backbone, sentences, eye: str, guard: LeakGuard, tag: str, check: bool):
    mt = [model_text(eye, s) for s in sentences]
    for s, t in zip(sentences, mt):
        guard.check(s, t)
    n_tok = [len(tok(t).input_ids) for t in mt]
    if max(n_tok) > SENT_MAXLEN:
        raise RuntimeError(f"DATA ERROR: {tag}: {max(n_tok)} tokens > {SENT_MAXLEN}")
    parts = []
    for i in range(0, len(mt), FEAT_BATCH):
        h, n = pool_last(tok, backbone, mt[i:i + FEAT_BATCH])
        assert [int(x) for x in n.tolist()] == n_tok[i:i + FEAT_BATCH], f"{tag}: batch lengths != per-text counts"
        parts.append(h.cpu())
    H = torch.cat(parts)
    assert H.shape[0] == len(sentences) and bool(torch.isfinite(H).all()), tag
    min_cos = 1.0
    if check:  # (as Lab 15) a few rows recomputed one at a time must match the batched features
        for i in sorted({0, 1, len(mt) - 1, int(np.argmax(n_tok))}):
            h1, _n = pool_last(tok, backbone, [mt[i]])
            min_cos = min(min_cos, float(F.cosine_similarity(h1[0].cpu(), H[i], dim=0)))
        if min_cos < 0.99:
            raise RuntimeError(f"{tag}: batched vs unbatched features differ (min cos {min_cos:.6f})")
    return H, max(n_tok), min_cos


class LoRALinear(nn.Module):
    """y = base(x) + (dropout(x) A^T B^T) * alpha / r; base frozen; A, B float32; off until `enabled`."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        dev = base.weight.device
        self.A = nn.Parameter(torch.zeros(LORA_R, base.in_features, device=dev, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, LORA_R, device=dev, dtype=torch.float32))
        self.scale = LORA_ALPHA / LORA_R
        self.enabled = False
        self.drop_on = False

    def reset(self):
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        y = self.base(x)
        if not self.enabled:
            return y
        xd = F.dropout(x.float(), p=LORA_DROPOUT, training=self.drop_on)
        return y + ((xd @ self.A.T) @ self.B.T * self.scale).to(y.dtype)


def inject_lora(backbone):
    mods = []
    for _name, module in list(backbone.named_modules()):
        for t in LORA_TARGETS:
            sub = getattr(module, t, None)
            if isinstance(sub, nn.Linear):
                lin = LoRALinear(sub)
                setattr(module, t, lin)
                mods.append(lin)
    if not mods:
        raise RuntimeError("no q/k/v/o projection found for LoRA")
    for p in backbone.parameters():
        p.requires_grad = False
    for m in mods:
        m.A.requires_grad = True
        m.B.requires_grad = True
    return mods


def xdom_masks(y, d):
    """(copied from Lab 15)"""
    same_c = y.unsqueeze(1) == y.unsqueeze(0)
    same_d = d.unsqueeze(1) == d.unsqueeze(0)
    eye = torch.eye(len(y), device=y.device, dtype=torch.bool)
    return same_c & ~same_d, eye | (same_c & same_d)


def supcon_from_logits(S, pos, excl):
    """(copied from Lab 15)"""
    has = pos.any(dim=1)
    if not bool(has.any()):
        raise RuntimeError("XDOM: no anchor has a cross-domain positive")
    S_a, pos_a, excl_a = S[has], pos[has], excl[has]
    assert not bool((pos_a & excl_a).any())
    log_den = torch.logsumexp(S_a.masked_fill(excl_a, float("-inf")), dim=1)
    lp = torch.where(pos_a, S_a - log_den.unsqueeze(1), torch.zeros_like(S_a))
    return -(lp.sum(1) / pos_a.sum(1).float()).mean()


def train_lora(eye, tok, backbone, mods, pool_by_dom, guard, seed, fold, hard_on):
    """pool_by_dom: training domain -> list of (sentence, class). Returns the loss trace."""
    tag_seed = int(hashlib.sha256(f"17-lora|{eye}|{seed}|{fold}|{int(hard_on)}".encode()).hexdigest()[:8], 16)
    seed_all(tag_seed)
    for m in mods:
        m.reset()
        m.enabled, m.drop_on = True, True
    D = EYES[eye][3]
    head = nn.Linear(D, N_SCLS).to(DEVICE).float()
    params = [p for m in mods for p in (m.A, m.B)] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=LORA_LR, weight_decay=0.0)
    rng = random.Random(tag_seed)
    doms = sorted(pool_by_dom)
    dom_id = {d: i for i, d in enumerate(doms)}
    trace = []
    for step in range(1, LORA_STEPS + 1):
        lr = LORA_LR * step / LORA_WARMUP if step <= LORA_WARMUP else \
            0.5 * LORA_LR * (1 + math.cos(math.pi * (step - LORA_WARMUP) / (LORA_STEPS - LORA_WARMUP)))
        for pg in opt.param_groups:
            pg["lr"] = lr
        batch = []
        for d in rng.sample(doms, LORA_DOMS_PER_STEP):
            batch += [(s, c, dom_id[d]) for s, c in rng.sample(pool_by_dom[d], LORA_PER_DOM)]
        mt = [model_text(eye, s) for s, _c, _d in batch]
        for (s, _c, _d), t in zip(batch, mt):
            guard.check(s, t)
        h, _n = pool_last(tok, backbone, mt)
        y = torch.tensor([c for _s, c, _d in batch], device=DEVICE)
        dvec = torch.tensor([d for _s, _c, d in batch], device=DEVICE)
        z = F.normalize(h, dim=-1)
        pos, excl = xdom_masks(y, dvec)
        loss = supcon_from_logits(z @ z.T / SUPCON_TAU, pos, excl) + F.cross_entropy(head(h), y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, LORA_CLIP)
        opt.step()
        if step in (1, 50, 100, LORA_STEPS):
            trace.append(f"{step}:{float(loss.item()):.4f}")
    for m in mods:
        m.drop_on = False
    return trace


def lora_off(mods):
    for m in mods:
        m.enabled, m.drop_on = False, False


# ==============================================================================
# [2] Heads (FLAT / FACT) and the readout
# ==============================================================================
class FactHead(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(D, HIDDEN), nn.GELU())
        self.init = nn.Linear(HIDDEN, N_INIT)
        self.kind = nn.Linear(HIDDEN, 3)
        self.dir = nn.Linear(HIDDEN, 2)

    def forward(self, x):
        h = self.trunk(x)
        return self.init(h), self.kind(h), self.dir(h)


def fact_loss(model, X, y):
    li, lk, ld = model(X)
    is_init = y < N_INIT
    e = (y - N_INIT).clamp(min=0)
    kind = torch.tensor(EV_KIND, device=y.device)[e]
    dr = torch.tensor(EV_DIR, device=y.device)[e]
    loss = 0.0
    if bool(is_init.any()):
        loss = loss + F.cross_entropy(li[is_init], y[is_init])
    ev = ~is_init
    if bool(ev.any()):
        loss = loss + F.cross_entropy(lk[ev], kind[ev])
        hd = ev & (dr >= 0)
        if bool(hd.any()):
            loss = loss + F.cross_entropy(ld[hd], dr[hd])
    return loss


@torch.no_grad()
def ev_scores(head_kind: str, model, X):
    """(N, 5) event scores."""
    if head_kind == "FLAT":
        return model(X)[:, N_INIT:]
    _li, lk, ld = model(X)
    pk, pd = F.log_softmax(lk, -1), F.log_softmax(ld, -1)
    return torch.stack([pk[:, 0] + pd[:, 0], pk[:, 0] + pd[:, 1], pk[:, 1] + pd[:, 1], pk[:, 1] + pd[:, 0], pk[:, 2]], 1)


@torch.no_grad()
def init_scores(head_kind: str, model, X):
    return model(X)[:, :N_INIT] if head_kind == "FLAT" else model(X)[0]


def train_head(head_kind: str, X, y, seed: int):
    """(Lab 15 recipe) seed_all before init; train_loop seeds again before training."""
    seed_all(seed)
    D = X.shape[1]
    if head_kind == "FLAT":
        model = nn.Sequential(nn.Linear(D, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, N_SCLS)).to(DEVICE).float()
        loss = train_loop(model, lambda m: F.cross_entropy(m(X), y), CE_STEPS, seed)
    else:
        model = FactHead(D).to(DEVICE).float()
        loss = train_loop(model, lambda m: fact_loss(m, X, y), CE_STEPS, seed)
    model.eval()
    return model, loss


def evaluate(eye, lora_on, hard_on, table, hard, Fd, Fh, folds, pools, f, seed):
    """One (eye, LoRA, hard, fold, seed): train the 4 heads (ZERO / C x FLAT / FACT) and score the held-out domains."""
    held = folds[f]
    tr_doms = pools[f]
    new_doms = table["new_doms"]
    y = torch.tensor(table["cls"], device=DEVICE)
    tr = [i for i, d in enumerate(table["dom"]) if d in tr_doms]
    assert not {table["dom"][i] for i in tr} & set(held) and len(tr_doms) == 21
    tr_t = torch.tensor(tr, device=DEVICE)
    htr = [i for i, (d, c) in enumerate(zip(hard["dom"], hard["cls"]))
           if d in tr_doms and c >= 0 and hard["cat"][i] in USED_HARD] if hard_on else []
    htr_t = torch.tensor(htr, device=DEVICE, dtype=torch.long)
    yh = torch.tensor([max(c, 0) for c in hard["cls"]], device=DEVICE)
    rows5_of = {d: ev_rows_by_class(table, d) for d in table["doms"]}
    centers = {d: class_center(Fd, rows5_of[d]) for d in tr_doms}
    mu = Fd[tr_t].mean(dim=0)
    X, Xh = F.normalize(Fd - mu, dim=-1), F.normalize(Fh - mu, dim=-1)
    Xc_tr = F.normalize(Fd[tr_t] - torch.stack([centers[table["dom"][i]] for i in tr]), dim=-1)
    if htr:
        Xhc_tr = F.normalize(Fh[htr_t] - torch.stack([centers[hard["dom"][i]] for i in htr]), dim=-1)
    models = {}
    for hk in HEADS:
        Xz = torch.cat([X[tr_t], Xh[htr_t]]) if htr else X[tr_t]
        Xc = torch.cat([Xc_tr, Xhc_tr]) if htr else Xc_tr
        yy = torch.cat([y[tr_t], yh[htr_t]]) if htr else y[tr_t]
        models[("ZERO", hk)], _l1 = train_head(hk, Xz, yy, seed)
        models[("C", hk)], _l2 = train_head(hk, Xc, yy, seed)
    keys = [(m, hk) for hk in HEADS for m in METHODS]
    ev = {k: [0] * 6 for k in keys}                                  # c, t, c_old, t_old, c_new, t_new
    conf = {k: [[0] * N_EV for _ in range(N_EV)] for k in keys}
    ini = {k: [0] * 6 for k in keys}
    hd = {(k, c): [0] * 7 for k in keys for c in USED_HARD}          # + as_surface
    for r in range(N_DRAWS):
        for d in held:
            rng = draw_rng("17", seed, f, d, 1, r)
            sup, ysup, qry, yq = draw_support(rows5_of[d], 1, rng)
            m_d = class_center(Fd, [[sup[j] for j in range(len(sup)) if ysup[j] == e] for e in range(N_EV)])
            q_t = torch.tensor(qry, device=DEVICE)
            ir = [i for i, (c, dd) in enumerate(zip(table["cls"], table["dom"])) if dd == d and c < N_INIT]
            ir_t = torch.tensor(ir, device=DEVICE)
            hr = [i for i, (dd, c) in enumerate(zip(hard["dom"], hard["cat"])) if dd == d and c in USED_HARD]
            hr_t = torch.tensor(hr, device=DEVICE)
            old = d not in new_doms
            feats = {"ZERO": (X[q_t], X[ir_t], Xh[hr_t]),
                     "C": (F.normalize(Fd[q_t] - m_d, dim=-1), F.normalize(Fd[ir_t] - m_d, dim=-1),
                           F.normalize(Fh[hr_t] - m_d, dim=-1))}
            for (m, hk) in keys:
                xq, xi, xh = feats[m]
                mdl = models[(m, hk)]
                pq = ev_scores(hk, mdl, xq).argmax(-1).tolist()
                hits = [int(p == t) for p, t in zip(pq, yq)]
                a = ev[(m, hk)]
                a[0] += sum(hits); a[1] += len(hits)
                a[2 if old else 4] += sum(hits); a[3 if old else 5] += len(hits)
                for p, t in zip(pq, yq):
                    conf[(m, hk)][t][p] += 1
                pi = init_scores(hk, mdl, xi).argmax(-1)
                ci = int((pi == y[ir_t]).sum())
                b = ini[(m, hk)]
                b[0] += ci; b[1] += len(ir)
                b[2 if old else 4] += ci; b[3 if old else 5] += len(ir)
                ph = ev_scores(hk, mdl, xh).argmax(-1).tolist()
                for j, i in enumerate(hr):
                    c = hd[((m, hk), hard["cat"][i])]
                    ok = int(ph[j] == hard["cls"][i] - N_INIT)
                    c[0] += ok; c[1] += 1
                    c[2 if old else 4] += ok; c[3 if old else 5] += 1
                    c[6] += int(ph[j] == hard["surface"][i] - N_INIT)
    tag = f"eye={eye} lora={int(lora_on)} hard={int(hard_on)}"
    for (m, hk) in keys:
        a, b = ev[(m, hk)], ini[(m, hk)]
        flat = ",".join(str(conf[(m, hk)][t][p]) for t in range(N_EV) for p in range(N_EV))
        log(f"L17_EV {tag} head={hk} method={m} fold={f} seed={seed} correct={a[0]} total={a[1]} c_old={a[2]} "
            f"t_old={a[3]} c_new={a[4]} t_new={a[5]} conf={flat}")
        log(f"L17_INIT {tag} head={hk} method={m} fold={f} seed={seed} correct={b[0]} total={b[1]} c_old={b[2]} "
            f"t_old={b[3]} c_new={b[4]} t_new={b[5]}")
        for cat in USED_HARD:
            c = hd[((m, hk), cat)]
            log(f"L17_HARD {tag} head={hk} method={m} fold={f} seed={seed} cat={cat} correct={c[0]} total={c[1]} "
                f"c_old={c[2]} t_old={c[3]} c_new={c[4]} t_new={c[5]} as_surface={c[6]}")


# ==============================================================================
# main
# ==============================================================================
def main():
    t_start = time.time()
    log(f"[l17] device={DEVICE} | seeds={SEEDS} | python {sys.version.split()[0]} | torch {torch.__version__} | "
        f"transformers {getattr(transformers, '__version__', '?')}")
    try:
        with open(os.path.abspath(__file__), "rb") as _f:
            script_sha = hashlib.sha256(_f.read()).hexdigest()
    except (NameError, OSError):
        script_sha = "unavailable"
    env_tag = f"torch={torch.__version__},transformers={getattr(transformers, '__version__', '?')},device={DEVICE}"
    if DEVICE == "cuda":
        env_tag += f",gpu={torch.cuda.get_device_name(0).replace(' ', '_')}"
    if "FAKE" in getattr(transformers, "__version__", ""):
        log("!" * 100 + "\n!!! [l17] FAKE transformers stub: TEST RUN, NOT a registered run\n" + "!" * 100)
    eyes_env = os.environ.get("LAB17_EYES", "").strip()
    eyes = [e.strip() for e in eyes_env.split(",") if e.strip()] if eyes_env else list(EYE_ORDER)
    if any(e not in EYES for e in eyes) or len(set(eyes)) != len(eyes):
        raise ValueError(f"LAB17_EYES={eyes_env!r}; allowed {list(EYES)}")
    eyes = [e for e in EYE_ORDER if e in eyes]

    st = load_st()
    extra = load_extra(st)
    table = merge_tables(st, extra)
    hard = merge_hard(load_hard(st), load_hard_extra(table), table)
    guard = LeakGuard(table["texts"] + hard["texts"], st["forbidden"])
    folds, pools = make_folds(table["doms"], N_FOLDS, size=FOLD_SIZE)
    for f in range(N_FOLDS):
        log(f"L17_FOLD fold={f} heldout={','.join(folds[f])} pool={','.join(pools[f])}")
    for eye in eyes:  # preflight: tokenizer + lengths before any download of weights
        tok = AutoTokenizer.from_pretrained(EYES[eye][0])
        worst = max(len(tok(model_text(eye, s)).input_ids) for s in table["texts"] + hard["texts"])
        assert worst <= SENT_MAXLEN, (eye, worst)
        log(f"[l17] preflight {eye}: longest model text {worst} tokens")

    for eye in eyes:
        mid, kind, _tmpl, dim, tunable = EYES[eye]
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(mid)
        tok.padding_side = "right"
        model = (AutoModelForCausalLM if kind == "causal" else AutoModel).from_pretrained(
            mid, dtype=torch.bfloat16, device_map=DEVICE)
        model.eval()
        backbone = backbone_of(eye, model)
        hid = model.config.hidden_size
        assert hid == dim, (eye, hid)
        mods = inject_lora(backbone) if tunable else []
        for p in model.parameters():
            assert not p.requires_grad or tunable
        log(f"[l17] {eye}: {mid} loaded in {time.time() - t0:.1f}s | hidden {hid} | LoRA modules {len(mods)}")

        def feats(tag, check):
            H, mx, mc = extract17(tok, backbone, table["texts"], eye, guard, f"{eye}/{tag}/st", check)
            Hh, mxh, mch = extract17(tok, backbone, hard["texts"], eye, guard, f"{eye}/{tag}/hard", check)
            log(f"L17_FEAT eye={eye} tag={tag} n={H.shape[0]} n_hard={Hh.shape[0]} dim={H.shape[1]} "
                f"max_tokens={max(mx, mxh)} check_min_cos={min(mc, mch):.6f} fp={float(H.double().sum()).hex()}")
            return H.to(DEVICE), Hh.to(DEVICE)

        Ff, Ffh = feats("frozen", True)
        for f in range(N_FOLDS):
            for seed in SEEDS:
                for hard_on in (False, True):
                    evaluate(eye, False, hard_on, table, hard, Ff, Ffh, folds, pools, f, seed)
                    if not tunable:
                        continue
                    pool_by_dom = defaultdict(list)
                    for s, c, d in zip(table["texts"], table["cls"], table["dom"]):
                        if d in pools[f]:
                            pool_by_dom[d].append((s, c))
                    if hard_on:
                        for s, c, d, cat in zip(hard["texts"], hard["cls"], hard["dom"], hard["cat"]):
                            if d in pools[f] and c >= 0 and cat in USED_HARD:
                                pool_by_dom[d].append((s, c))
                    assert not set(pool_by_dom) & set(folds[f]) and len(pool_by_dom) == 21
                    t1 = time.time()
                    trace = train_lora(eye, tok, backbone, mods, pool_by_dom, guard, seed, f, hard_on)
                    model.eval()
                    Fl, Flh = feats(f"lora-f{f}-s{seed}-h{int(hard_on)}", False)
                    lora_off(mods)
                    log(f"L17_LORA eye={eye} fold={f} seed={seed} hard={int(hard_on)} loss={','.join(trace)} "
                        f"sec={time.time() - t1:.1f}")
                    evaluate(eye, True, hard_on, table, hard, Fl, Flh, folds, pools, f, seed)
                    del Fl, Flh
            log(f"[l17] {eye}: fold {f} done ({time.time() - t0:.0f}s)")
        del model, backbone, mods, tok, Ff, Ffh
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    log(f"[l17] leak guard: {guard.n_checked} inputs checked; none contains a test-domain string")
    log(f"[l17] total time {time.time() - t_start:.1f}s")
    log(f"L17_CONFIG data_sha_st={st['sha']} templates_sha={st['tpl_sha']} data_sha_extra={extra['sha']} "
        f"data_sha_hard={hard['sha']} data_sha_hard_extra={hard['sha_extra']} env={env_tag} script_sha={script_sha} "
        f"eyes={','.join(eyes)} seeds={','.join(map(str, SEEDS))} draws={N_DRAWS}")


if __name__ == "__main__":
    main()
