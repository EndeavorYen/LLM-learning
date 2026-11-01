# ==============================================================================
# Stage 2 / Lab 17b: comprehension probe (B1). PRE-REGISTERED on issue #25 (comment "理解檢測（Lab 17b）", 2026-11-01,
#   before any data or code). Numbers only, NO verdict.
# Question: does the Lab 17 recogniser READ the event sentence ("看懂") or match surface cues ("猜字")? Measured on 6
#   brand-new probe domains (comprehension_probe_v1/probe.json) with minimal pairs, cue-free and restructured sentences.
# Hard rule: the 8 test domains of state_tracking_v1 are NOT used (Lab 15/16/17 loaders and leak guard).
# Data (sha256 pinned)
#   st / extra / hard / hard extra: as Lab 17 (the 28 training domains and their hard sentences).
#   probe  comprehension_probe_v1/probe.json (validator tools/validate_probe.py PASS; independent blind check).
#          Per probe domain: a normal part (templates.json format: 8 init + 32 event sentences) and probe items
#          DIR 4 pairs, REAL 2 groups x 4 (actual / negated / intended / hypothetical), SYS 2 pairs, MENTION 2 pairs,
#          CUE 10, STR 8.
# Training (exactly Lab 17, but on ALL 28 training domains, no folds): LoRA (Q15, Q3E) rank 16 / alpha 32 / dropout
#   0.05 on q/k/v/o, 150 steps, 8 domains x 12 sentences per step, cross-domain SupCon (tau 0.1) + 13-class CE, lr 2e-4,
#   10 warm-up steps then cosine; head FLAT only (D -> 256 -> GELU -> 13, Lab 15 recipe). hard off / on = the training
#   domains' NEG / OTHER / TEMP sentences join LoRA and head training (Lab 17 factor 3).
# Cells: Q15, Q3E: frozen / LoRA x hard off / on; Q7 (Qwen2.5-7B-Instruct): frozen x hard off / on. Seeds 42, 1, 2.
# Readout (Lab 17): ZERO  x = normalize(f - mu), mu = mean of the 28 domains' normal rows (no example).
#                   C     x = normalize(f - m_d), m_d = mean of the probe domain's support: one NORMAL event sentence per
#                         event type, R = 10 draws per (seed, domain), rng = sha256("17b|seed|0|domain|1|draw").
#   Scored per draw (ZERO once per seed): the domain's other 27 normal event sentences (5-way), its 8 init sentences
#   (8-way), and all its probe items (5-way). A pair is right when both of its sentences are right in the same draw;
#   REAL = 3 pairs per group (actual with negated / intended / hypothetical).
# References (Q7 model, after its features): 7B-ZS0 = Lab 15 zero-shot multiple-choice question (no example);
#   7B-ICL = Lab 16 icl_mcq with the probe domain's 5 normal examples (one per event type), ICL_DRAWS = 2 draws,
#   rng = sha256("17b-ICL|0|0|domain|1|draw"). Correct = right in both option orders.
# Metrics: pair accuracy DIR / REAL / SYS / MENTION; CUE, STR, normal-event accuracy; CUE - normal.
# Judgement (registered; method C; best cell = small-eye cell (Q15 / Q3E: frozen or LoRA, hard off or on) with the
#   highest DIR + REAL pair accuracy; Q7 eye and 7B references are auxiliary [pre-run clarification, review 1]; DIR + REAL
#   = pooled 10 pairs per domain (4 DIR + 6 REAL) [review 2]):
#   看懂 if DIR+REAL pairs >= 90% and CUE at most 5 points below normal; 猜字 if pairs < 70% or CUE more than 10 points
#   below normal; otherwise 部分看懂. SYS, MENTION, STR are auxiliary.
# Machine lines: L17B_FEAT, L17B_LORA, L17B_RES, L17B_ITEM, then L17B_CONFIG as the LAST line.
#   Aggregate: python tools/aggregate_lab17b.py lab17b_*.log
# Run: LAB17B_EYES=Q15 python stage2_lab17b_comprehension_probe.py > a.log ; LAB17B_EYES=Q3E,Q7 ... > b.log
# Code marked "(copied from Lab 15 / 16 / 17)" is reused verbatim except log tags; everything else is new.
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


# ---- pre-registered configuration (issue #25, "理解檢測（Lab 17b）") ------------------------------------------------
SEEDS = (42, 1, 2)
N_DRAWS = 10
ICL_DRAWS = 2
N_PROBE = 6
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
METHODS = ("ZERO", "C")

EXPECTED_ST_SHA256 = "ed6d16bd3e625198bba74d28e7b20f6bdba9c399a3eb94b9a0d1f113f11e3108"
EXPECTED_TPL_SHA256 = "1bfd769efdf6dd3d61ded0e6aaff3eb70b394af2c266e2f02cd1ed9b598bca70"
EXPECTED_HARD_SHA256 = "e145539bc8b37133a226a5d390e6cfebc2492d85e2cc090501d8464fc5afe8aa"
EXPECTED_EXTRA_SHA256 = "f44dc28a2a2df593981c919949b47e13304a814cec4e960123564206190ca7cf"   # extra_domains.json v1.1
EXPECTED_HARD_EXTRA_SHA256 = "e31ad5839bbc1133c6d3e6e5caec6a5125a4ffd9bcffbf330fb793b58416fc53"
EXPECTED_PROBE_SHA256 = "3beede5245d048af2a5a6652079a974e19a1305887710eca6c7d1ffbc83258b8"  # accepted 2026-11-01: validator PASS, blind check 306/312 then 9 revised items 30/30
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
    log(f"[l17b] st: {st_dir}/stories.json sha256={sha} | templates.json sha256={tpl_sha} | {len(train_doms)} train domains "
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
    log(f"[l17b] hard: {path} sha256={sha} | {len(texts)} sentences in {len(names)} train domains | "
        + " ".join(f"{c}={cnt[c]}" for c in HARD_CATS))
    return dict(name="hard", sha=sha, texts=texts, cat=cat, surface=surf, cls=ycls, dom=dom)



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
    log(f"[l17b] extra: {path} sha256={sha} | {N_EXTRA} new train domains | {len(texts)} sentences")
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
    log(f"[l17b] hard extra: {path} sha256={sha} | {len(texts)} sentences in {len(names)} extra domains")
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
    h = backbone(input_ids=enc.input_ids, attention_mask=m, use_cache=False).last_hidden_state
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


def train_lora(eye, tok, backbone, mods, pool_by_dom, guard, seed, hard_on):
    """pool_by_dom: training domain -> list of (sentence, class). Returns the loss trace."""
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tag_seed = int(hashlib.sha256(f"17b-lora|{eye}|{seed}|{int(hard_on)}".encode()).hexdigest()[:8], 16)
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
    if DEVICE == "cuda":
        trace.append(f"maxmem_GiB:{torch.cuda.max_memory_allocated() / 2 ** 30:.1f}")
    for m in mods:
        m.drop_on = False
    return trace


def lora_off(mods):
    for m in mods:
        m.enabled, m.drop_on = False, False



# ==============================================================================
# [2] Head (FLAT) and the readout (copied from Lab 17, FLAT branch only)
# ==============================================================================
@torch.no_grad()
def ev_scores(head_kind: str, model, X):
    """(N, 5) event scores."""
    assert head_kind == "FLAT"
    return model(X)[:, N_INIT:]


@torch.no_grad()
def init_scores(head_kind: str, model, X):
    assert head_kind == "FLAT"
    return model(X)[:, :N_INIT]


def train_head(head_kind: str, X, y, seed: int):
    """(Lab 15 recipe) seed_all before init; train_loop seeds again before training."""
    assert head_kind == "FLAT"
    seed_all(seed)
    D = X.shape[1]
    model = nn.Sequential(nn.Linear(D, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, N_SCLS)).to(DEVICE).float()
    loss = train_loop(model, lambda m: F.cross_entropy(m(X), y), CE_STEPS, seed)
    model.eval()
    return model, loss


# ==============================================================================
# [0c] the comprehension probe (new for Lab 17b)
# ==============================================================================
PAIR_KINDS = ("DIR", "REAL", "SYS", "MENTION")
REAL_FORMS = ("negated", "intended", "hypothetical")


def load_probe(table: dict, hard: dict, forbidden):
    """probe.json -> normal table of the probe domains (init + events, Lab 15 construction) and the probe items/pairs."""
    cands = ["/content/probe.json", os.path.join(table["st_dir"], "..", "comprehension_probe_v1", "probe.json")] \
        + _script_dir_candidates("benchmarks", "comprehension_probe_v1", "probe.json")
    path, raw, sha = _read_first(cands, "probe.json")
    assert EXPECTED_PROBE_SHA256 is not None, "probe.json not accepted yet (EXPECTED_PROBE_SHA256 unset)"
    assert sha == EXPECTED_PROBE_SHA256, f"probe.json is not the registered file: {sha}"
    p = json.loads(raw.decode("utf-8"))
    assert p.get("version") == "comprehension_probe_v1", p.get("version")
    assert p.get("templates_sha256") == table["tpl_sha"] and p.get("extra_domains_sha256") == EXPECTED_EXTRA_SHA256
    doms = p["domains"]
    names = [d["name"] for d in doms]
    assert len(doms) == N_PROBE and len(set(names)) == N_PROBE, names
    assert not any(a in b or b in a for a in names for b in table["doms"]), "a probe domain name overlaps a train domain"
    lab = {e: EVENTS.index(e) for e in EVENTS}
    texts, cls, dom = [], [], []
    items = []           # dict(dom, kind, id, text, y)  (y = local event index 0..4)
    pairs = []           # (kind, sub, item index a, item index b)

    def item(d, kind, iid, x):
        items.append(dict(dom=d, kind=kind, id=iid, text=x["text"], y=lab[x["label"]]))
        return len(items) - 1

    for d in doms:
        nm = d["name"]
        assert d["split"] == "probe", nm
        lv_txt, sg_txt = d["init"]["level"], d["init"]["safeguard"]
        assert len(lv_txt) == 4 and set(sg_txt) == {"ok", "broken"}, nm
        for lv in range(4):
            for sg in ("ok", "broken"):
                texts.append(lv_txt[lv] + "，" + sg_txt[sg] + "。")
                cls.append(CI[f"L{lv}_{sg}"])
                dom.append(nm)
        for e, n in zip(EVENTS, (7, 7, 6, 6, 6)):
            assert len(d["events"][e]) == n, (nm, e)
            for t in d["events"][e]:
                texts.append(t)
                cls.append(N_INIT + EVENTS.index(e))
                dom.append(nm)
        pr = d["probe"]
        for kind in ("DIR", "SYS", "MENTION"):
            for q in pr[kind]:
                pairs.append((kind, "", item(nm, kind, q["id"] + "a", q["a"]), item(nm, kind, q["id"] + "b", q["b"])))
        for g in pr["REAL"]:
            forms = [it["form"] for it in g["items"]]
            assert forms == ["actual", *REAL_FORMS], forms
            ia = item(nm, "REAL", g["id"] + "-actual", g["items"][0])
            for it in g["items"][1:]:
                assert it["label"] == "NOOP", it
                pairs.append(("REAL", it["form"], ia, item(nm, "REAL", g["id"] + "-" + it["form"], it)))
        for kind in ("CUE", "STR"):
            for x in pr[kind]:
                item(nm, kind, x["id"], x)
    itexts = [it["text"] for it in items]
    assert len(set(itexts)) == len(itexts), "DATA ERROR: duplicate probe item"
    assert len(set(texts)) == len(texts), "DATA ERROR: duplicate probe normal sentence"
    old = set(table["texts"]) | set(hard["texts"])
    assert not (set(texts) | set(itexts)) & old, "DATA ERROR: a probe sentence equals a training sentence"
    assert not any(f in t for t in texts + itexts for f in forbidden), "LEAK / DATA ERROR: test string in probe.json"
    cnt = Counter(it["kind"] for it in items)
    log(f"[l17b] probe: {path} sha256={sha} | {N_PROBE} probe domains ({','.join(names)}) | {len(texts)} normal "
        f"sentences | items " + " ".join(f"{k}={cnt[k]}" for k in ("DIR", "REAL", "SYS", "MENTION", "CUE", "STR"))
        + f" | pairs {len(pairs)}")
    return dict(sha=sha, texts=texts, cls=cls, dom=dom, doms=names, items=items, pairs=pairs)


class Tally:
    """Counts for one (source, method, seed): normal events, init, single items by kind, pairs by kind / REAL form,
    and per-item prediction counts."""

    def __init__(self, probe):
        self.c = defaultdict(lambda: [0, 0])
        self.pred = defaultdict(lambda: [0] * (N_EV + 1))   # last slot: 7B option orders disagree (pred -1)
        self.probe = probe

    def add(self, key, ok):
        self.c[key][0] += int(ok)
        self.c[key][1] += 1

    def score_items(self, dom, preds):
        """preds: item index -> predicted local event (all items of `dom`)."""
        P = self.probe
        for i, it in enumerate(P["items"]):
            if it["dom"] != dom:
                continue
            ok = preds[i] == it["y"]
            self.add("item_" + it["kind"], ok)
            self.add(f"lab_{it['kind']}_{EVENTS[it['y']]}", ok)
            self.pred[i][preds[i]] += 1
        for kind, sub, a, b in P["pairs"]:
            if P["items"][a]["dom"] != dom:
                continue
            ok = preds[a] == P["items"][a]["y"] and preds[b] == P["items"][b]["y"]
            self.add("pair_" + kind, ok)
            if sub:
                self.add("pair_REAL_" + sub, ok)
            self.add("pair_DIRREAL" if kind in ("DIR", "REAL") else "pair_AUX", ok)

    def log(self, tag, items_too: bool):
        keys = sorted(self.c)
        log(f"L17B_RES {tag} " + " ".join(f"{k}={self.c[k][0]}/{self.c[k][1]}" for k in keys))
        if items_too:
            for i, it in enumerate(self.probe["items"]):
                log(f"L17B_ITEM {tag} item={it['dom']}/{it['id']} kind={it['kind']} y={EVENTS[it['y']]} "
                    f"preds={','.join(map(str, self.pred[i]))}")


def evaluate17b(eye, lora_on, hard_on, table, hard, probe, Fd, Fh, Fp, Fi, seed, tally_by_method):
    """Train the FLAT head (ZERO / C) on all 28 domains and score the probe domains."""
    y = torch.tensor(table["cls"], device=DEVICE)
    tr = list(range(len(table["cls"])))
    tr_t = torch.tensor(tr, device=DEVICE)
    htr = [i for i, c in enumerate(hard["cls"]) if c >= 0 and hard["cat"][i] in USED_HARD] if hard_on else []
    htr_t = torch.tensor(htr, device=DEVICE, dtype=torch.long)
    yh = torch.tensor([max(c, 0) for c in hard["cls"]], device=DEVICE)
    centers = {d: class_center(Fd, ev_rows_by_class(table, d)) for d in table["doms"]}
    mu = Fd.mean(dim=0)
    X, Xh = F.normalize(Fd - mu, dim=-1), F.normalize(Fh - mu, dim=-1)
    Xc_tr = F.normalize(Fd - torch.stack([centers[d] for d in table["dom"]]), dim=-1)
    Xz, Xc, yy = X, Xc_tr, y
    if htr:
        Xhc = F.normalize(Fh[htr_t] - torch.stack([centers[hard["dom"][i]] for i in htr]), dim=-1)
        Xz, Xc, yy = torch.cat([X, Xh[htr_t]]), torch.cat([Xc_tr, Xhc]), torch.cat([y, yh[htr_t]])
    models = {"ZERO": train_head("FLAT", Xz, yy, seed)[0], "C": train_head("FLAT", Xc, yy, seed)[0]}
    yp = torch.tensor(probe["cls"], device=DEVICE)
    for d in probe["doms"]:
        rows5 = ev_rows_by_class(probe, d)
        ir = [i for i, (c, dd) in enumerate(zip(probe["cls"], probe["dom"])) if dd == d and c < N_INIT]
        ii = [i for i, it in enumerate(probe["items"]) if it["dom"] == d]
        ir_t, ii_t = torch.tensor(ir, device=DEVICE), torch.tensor(ii, device=DEVICE)
        for m in METHODS:
            tl = tally_by_method[m]
            for r in (range(N_DRAWS) if m == "C" else (0,)):
                if m == "C":
                    sup, ysup, qry, yq = draw_support(rows5, 1, draw_rng("17b", seed, 0, d, 1, r))
                    ctr = class_center(Fp, [[sup[j] for j in range(len(sup)) if ysup[j] == e] for e in range(N_EV)])
                else:
                    qry = [i for rows in rows5 for i in rows]
                    yq = [e for e, rows in enumerate(rows5) for _i in rows]
                    ctr = mu
                q_t = torch.tensor(qry, device=DEVICE)
                xq, xi, xit = (F.normalize(A - ctr, dim=-1) for A in (Fp[q_t], Fp[ir_t], Fi[ii_t]))
                mdl = models[m]
                pq = ev_scores("FLAT", mdl, xq).argmax(-1).tolist()
                for p, t in zip(pq, yq):
                    tl.add("normal", p == t)
                    tl.add(f"normal_{EVENTS[t]}", p == t)
                pi = init_scores("FLAT", mdl, xi).argmax(-1)
                for p, t in zip(pi.tolist(), yp[ir_t].tolist()):
                    tl.add("init", p == t)
                pit = ev_scores("FLAT", mdl, xit).argmax(-1).tolist()
                tl.score_items(d, dict(zip(ii, pit)))


# ==============================================================================
# [3] 7B references: zero-shot (ZS0) and in-context examples (ICL), Lab 15 / 16 prompts
# ==============================================================================
ZS_EVENT_Q = "这句话描述的是下面哪一种事件？"
ZS_EVENT_OPTS = ["系统里的量增加一个单位", "系统里的量减少一个单位", "安全机制坏了/失效", "安全机制修好/恢复",
                 "与系统无关的事"]                         # (copied from Lab 15) IN, OUT, BREAK, FIX, NOOP
LETTERS = "ABCDE"


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
    """(copied from Lab 15) chat template, next-token log-probs; option score = max over the letter's tokens;
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
    """(copied from Lab 15)"""
    L = LETTERS[:len(options)]
    ans = (f"{L[0]} 或 {L[1]}" if len(L) == 2 else "、".join(L[:-1]) + f" 或 {L[-1]}")
    return (f"领域：{domain}\n句子：{sentence}\n\n问题：{question}\n"
            + "".join(f"{l}. {o}\n" for l, o in zip(L, options))
            + f"\n只回答一个字母（{ans}）。")


def icl_mcq(domain: str, examples, sentence: str, options) -> str:
    """(copied from Lab 16)"""
    L = LETTERS[:len(options)]
    ans = "、".join(L[:-1]) + f" 或 {L[-1]}"
    ex = "".join(f"例句：{s}\n事件：{o}\n\n" for s, o in examples)
    return (f"领域：{domain}\n下面是这个领域里已经标好事件类型的例句：\n\n{ex}"
            f"句子：{sentence}\n\n问题：{ZS_EVENT_Q}\n" + "".join(f"{l}. {o}\n" for l, o in zip(L, options))
            + f"\n只回答一个字母（{ans}）。")


def run_7b_refs(tok, model, probe, guard, tag_env):
    """7B-ZS0 (once) and 7B-ICL (ICL_DRAWS draws) on every probe domain: normal events + all probe items.
    A sentence is right when it is right in both option orders; its 'prediction' = the order-1 choice if both orders
    agree, else -1 (counted as wrong). ZS0 also needs both orders (as Lab 15)."""
    t0 = time.time()
    choose = make_chooser(tok, model)
    n = len(ZS_EVENT_OPTS)

    def ask(make_q, sentence):
        q1, q2 = make_q(ZS_EVENT_OPTS), make_q(ZS_EVENT_OPTS[::-1])
        for q in (q1, q2):
            guard.check(sentence, q)
        p1, p2 = choose(q1, n), n - 1 - choose(q2, n)
        return p1 if p1 == p2 else -1

    for key in ("ZS0", "ICL"):
        tl = Tally(probe)
        for d in probe["doms"]:
            rows5 = ev_rows_by_class(probe, d)
            ii = [i for i, it in enumerate(probe["items"]) if it["dom"] == d]
            for r in (range(ICL_DRAWS) if key == "ICL" else (0,)):
                if key == "ICL":
                    rng = draw_rng("17b-ICL", 0, 0, d, 1, r)
                    sup, ysup, qry, yq = draw_support(rows5, 1, rng)
                    order = list(range(len(sup)))
                    rng.shuffle(order)
                    ex = [(probe["texts"][sup[j]], ZS_EVENT_OPTS[ysup[j]]) for j in order]
                    for t, _o in ex:
                        guard.check(t)
                    mk = lambda s: (lambda opts: icl_mcq(d, ex, s, opts))
                else:
                    qry = [i for rows in rows5 for i in rows]
                    yq = [e for e, rows in enumerate(rows5) for _i in rows]
                    mk = lambda s: (lambda opts: mcq(d, s, ZS_EVENT_Q, opts))
                for i, e in zip(qry, yq):
                    p = ask(mk(probe["texts"][i]), probe["texts"][i])
                    tl.add("normal", p == e)
                    tl.add(f"normal_{EVENTS[e]}", p == e)
                preds = {}
                for i in ii:
                    s = probe["items"][i]["text"]
                    p = ask(mk(s), s)
                    preds[i] = p          # -1 = the two option orders disagree: wrong, counted in the last slot
                tl.score_items(d, preds)
        tl.log(f"src=7B-{key} lora=0 hard=0 method={key} seed=all {tag_env}", True)
    log(f"[l17b] 7B references done ({time.time() - t0:.0f}s)")


# ==============================================================================
# main
# ==============================================================================
def main():
    t_start = time.time()
    log(f"[l17b] device={DEVICE} | seeds={SEEDS} | python {sys.version.split()[0]} | torch {torch.__version__} | "
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
        log("!" * 100 + "\n!!! [l17b] FAKE transformers stub: TEST RUN, NOT a registered run\n" + "!" * 100)
    eyes_env = os.environ.get("LAB17B_EYES", "").strip()
    eyes = [e.strip() for e in eyes_env.split(",") if e.strip()] if eyes_env else list(EYE_ORDER)
    if any(e not in EYES for e in eyes) or len(set(eyes)) != len(eyes):
        raise ValueError(f"LAB17B_EYES={eyes_env!r}; allowed {list(EYES)}")
    eyes = [e for e in EYE_ORDER if e in eyes]

    st = load_st()
    extra = load_extra(st)
    table = merge_tables(st, extra)
    hard = merge_hard(load_hard(st), load_hard_extra(table), table)
    probe = load_probe(table, hard, st["forbidden"])
    guard = LeakGuard(table["texts"] + hard["texts"] + probe["texts"] + [it["text"] for it in probe["items"]],
                      st["forbidden"])
    all_texts = table["texts"] + hard["texts"] + probe["texts"] + [it["text"] for it in probe["items"]]
    for eye in eyes:  # preflight: tokenizer + lengths before any download of weights
        tok = AutoTokenizer.from_pretrained(EYES[eye][0])
        worst = max(len(tok(model_text(eye, s)).input_ids) for s in all_texts)
        assert worst <= SENT_MAXLEN, (eye, worst)
        log(f"[l17b] preflight {eye}: longest model text {worst} tokens")

    for eye in eyes:
        mid, kind, _tmpl, dim, tunable = EYES[eye]
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(mid)
        tok.padding_side = "right"
        model = (AutoModelForCausalLM if kind == "causal" else AutoModel).from_pretrained(
            mid, dtype=torch.bfloat16, device_map=DEVICE)
        model.eval()
        model.requires_grad_(False)  # (Lab 17 review D1) HF loads weights with requires_grad=True
        backbone = backbone_of(eye, model)
        hid = model.config.hidden_size
        assert hid == dim, (eye, hid)
        mods = inject_lora(backbone) if tunable else []
        n_layers = getattr(model.config, "num_hidden_layers", None)
        if tunable and "FAKE" not in getattr(transformers, "__version__", ""):
            assert len(mods) == len(LORA_TARGETS) * n_layers, (len(mods), n_layers)
        lora_ids = {id(m.A) for m in mods} | {id(m.B) for m in mods}
        assert all(id(p) in lora_ids for p in model.parameters() if p.requires_grad)
        last_ids = {tok(model_text(eye, s)).input_ids[-1] for s in table["texts"][:5]}
        log(f"[l17b] {eye}: {mid} loaded in {time.time() - t0:.1f}s | hidden {hid} | layers {n_layers} | LoRA modules "
            f"{len(mods)} | revision {getattr(model.config, '_commit_hash', '?')} | pooled token ids {sorted(last_ids)}")

        def feats(tag, check):
            out = []
            for nm, txts in (("st", table["texts"]), ("hard", hard["texts"]), ("pnorm", probe["texts"]),
                             ("pitem", [it["text"] for it in probe["items"]])):
                H, mx, mc = extract17(tok, backbone, txts, eye, guard, f"{eye}/{tag}/{nm}", check)
                out.append((H.to(DEVICE), mx, mc))
            log(f"L17B_FEAT eye={eye} tag={tag} n={out[0][0].shape[0]} n_hard={out[1][0].shape[0]} "
                f"n_pnorm={out[2][0].shape[0]} n_pitem={out[3][0].shape[0]} dim={out[0][0].shape[1]} "
                f"max_tokens={max(o[1] for o in out)} check_min_cos={min(o[2] for o in out):.6f} "
                f"fp={float(out[0][0].double().sum()).hex()}")
            return [o[0] for o in out]

        Ff = feats("frozen", True)
        for hard_on in (False, True):
            tallies = {m: Tally(probe) for m in METHODS}
            for seed in SEEDS:
                per_seed = {m: Tally(probe) for m in METHODS}
                evaluate17b(eye, False, hard_on, table, hard, probe, *Ff, seed, per_seed)
                for m in METHODS:
                    per_seed[m].log(f"src={eye} lora=0 hard={int(hard_on)} method={m} seed={seed}", False)
                    for k, v in per_seed[m].c.items():
                        tallies[m].c[k][0] += v[0]; tallies[m].c[k][1] += v[1]
                    for k, v in per_seed[m].pred.items():
                        tallies[m].pred[k] = [a + b for a, b in zip(tallies[m].pred[k], v)]
            for m in METHODS:
                tallies[m].log(f"src={eye} lora=0 hard={int(hard_on)} method={m} seed=all", True)
            if not tunable:
                continue
            tallies = {m: Tally(probe) for m in METHODS}
            for seed in SEEDS:
                pool_by_dom = defaultdict(list)
                for s, c, d in zip(table["texts"], table["cls"], table["dom"]):
                    pool_by_dom[d].append((s, c))
                if hard_on:
                    for s, c, d, cat in zip(hard["texts"], hard["cls"], hard["dom"], hard["cat"]):
                        if c >= 0 and cat in USED_HARD:
                            pool_by_dom[d].append((s, c))
                assert len(pool_by_dom) == 28 and not set(pool_by_dom) & set(probe["doms"])
                t1 = time.time()
                trace = train_lora(eye, tok, backbone, mods, pool_by_dom, guard, seed, hard_on)
                model.eval()
                Fl = feats(f"lora-s{seed}-h{int(hard_on)}", False)
                lora_off(mods)
                log(f"L17B_LORA eye={eye} seed={seed} hard={int(hard_on)} loss={','.join(trace)} "
                    f"sec={time.time() - t1:.1f}")
                per_seed = {m: Tally(probe) for m in METHODS}
                evaluate17b(eye, True, hard_on, table, hard, probe, *Fl, seed, per_seed)
                for m in METHODS:
                    per_seed[m].log(f"src={eye} lora=1 hard={int(hard_on)} method={m} seed={seed}", False)
                    for k, v in per_seed[m].c.items():
                        tallies[m].c[k][0] += v[0]; tallies[m].c[k][1] += v[1]
                    for k, v in per_seed[m].pred.items():
                        tallies[m].pred[k] = [a + b for a, b in zip(tallies[m].pred[k], v)]
                del Fl
                free_gpu()
            for m in METHODS:
                tallies[m].log(f"src={eye} lora=1 hard={int(hard_on)} method={m} seed=all", True)
        if eye == "Q7":
            run_7b_refs(tok, model, probe, guard, "")
        log(f"[l17b] {eye}: done ({time.time() - t0:.0f}s)")
        del model, backbone, mods, tok, Ff
        free_gpu()
    log(f"[l17b] leak guard: {guard.n_checked} inputs checked; none contains a test-domain string")
    log(f"[l17b] total time {time.time() - t_start:.1f}s")
    log(f"L17B_CONFIG data_sha_st={st['sha']} templates_sha={st['tpl_sha']} data_sha_extra={extra['sha']} "
        f"data_sha_hard={hard['sha']} data_sha_hard_extra={hard['sha_extra']} data_sha_probe={probe['sha']} "
        f"env={env_tag} script_sha={script_sha} eyes={','.join(eyes)} seeds={','.join(map(str, SEEDS))} "
        f"draws={N_DRAWS} icl_draws={ICL_DRAWS}")


if __name__ == "__main__":
    main()
