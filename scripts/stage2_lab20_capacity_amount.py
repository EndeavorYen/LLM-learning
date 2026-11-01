# ==============================================================================
# Stage 2 / Lab 20 (roadmap step 2b-1): rules that change: capacity 2..5 and IN / OUT amounts 1..3. PRE-REGISTERED on
#   issue #25 (comment "Lab 20（第二步 2b-1）「規則變活：容量＋數量」", 2026-11-02, before any new data or code).
#   Numbers only, NO verdict.
# Data: benchmarks/story_v2/stories.json (tools/make_stories_v2.py, fixed rng; 22 never-trained domains; amount-1
#   sentences = the story_v1 pools, amount 2 / 3 = benchmarks/amount_v1/test.json); eye training adds
#   benchmarks/amount_v1/train.json (28 training domains). The 8 state_tracking_v1 test domains are NOT used.
# Job (LAB20_JOB=eye, LAB20_SEED 42 / 1 / 2): the Lab 19 eye (Q15 LoRA, system-aware prompt, no noise practice) with
#   the amount_v1 training sentences added and an amount cross-entropy (3-way, IN / OUT rows, weight 1) in the LoRA
#   loss; a FLAT amount head on the readout-C features; readout C with each domain's FIXED first support sentence.
#   Systems: ORACLE+EXEC, EYE+EXEC (argmax event + amount -> rule), ORACLE+HOP, EYE+HOP (GRU 64 over event (5) + amount
#   (3) inputs, initial input = level one-hot + safeguard + capacity thermometer; trained on abstract stories of
#   length 1-4 with capacity 2 / 3 / 5 only, 6000 steps).
# Judgement (EYE+EXEC, normal LONG 8-12; amount accuracy on amount_v1 test IN / OUT sentences): LONG >= 80 and amount
#   >= 90 -> rules can vary; amount < 80 -> amounts not read; otherwise partly usable. Learned reasoner: on capacity-4
#   normal stories |EYE+HOP - EYE+EXEC| <= 3 and ORACLE+HOP >= 95 -> learned capacity.
# Machine lines: L20_LORA, L20_SENT, L20_AMT, L20_HOP, L20_RES, L20_ERRS, then L20_CONFIG as the LAST line.
# Code from Lab 17-18b is reused verbatim (log tags changed); the [5] section is new.
# ==============================================================================
import os, sys, gc, time, math, random, json, hashlib
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # [review 18b-1]
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


# ---- pre-registered configuration (issue #25, "Lab 18b「讀的補強」") -----------------------------------------------
SEEDS = (42, 1, 2, 3, 4)
N_DRAWS = 10
N_PROBE = 6
N_RTEST = 8
CELLS = ((0, 0), (0, 1), (1, 0), (1, 1))   # (A system-aware prompt, B noise practice)
GROUP_PART = 6
NOISE_PER_STEP = 24
EYES = {  # code -> (Hugging Face id, kind, template, hidden size, fine-tuned?)
    "Q15": ("Qwen/Qwen2.5-1.5B-Instruct", "causal", "q25", 1536, True),
    "Q3E": ("Qwen/Qwen3-Embedding-0.6B", "embed", "q3e", 1024, True),
}
EYE_ORDER = ("Q15", "Q3E")
PROMPT_TMPL = '这句话："{s}"\n用一个词概括这句话描述的事件：'
Q3E_TASK = "Identify which abstract event or state of the system this sentence describes"
Q3E_TMPL = "Instruct: " + Q3E_TASK + "\nQuery:{s}"
SENT_MAXLEN = 256  # [amendment] 128 in Lab 17, 160 in 18a; system-aware prompts and noised rows are longer (no truncation, asserted)
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
EXPECTED_RTRAIN_SHA256 = "ffede6eca9e2777a213c926c21b9b75981aad4fef50711b391c49d9149efeaf8"  # accepted: validator PASS, blind 416/420, 4 OTHER items clarified and re-checked
EXPECTED_RTRAIN2_SHA256 = "1d7f1c6fac0b39b7bae6ad7c81472986f97853c9dd33120441a5cd022b92942a"  # accepted: validator PASS, blind 419/420 (1 unclear OTHER_B clarified, re-checked)
EXPECTED_RTEST2_SHA256 = "904a97b811942ef95efede95d367603364e3542a2ff2f6ac6737f4e3121f2de2"   # accepted: validator PASS, blind 1256/1256
EXPECTED_RTEST_SHA256 = "49616f38581105d0c69c6e9b1509d8d763027fdb3d3b7ec199ea8a07d03c4c4a"
EXPECTED_STORY_SHA256 = "69c5866bf9e4f4715bde1dfff4c9660d7fbc409d0498d73f7258cc0ab9650be6"
SEEDS19 = (42, 1, 2)
THINK_MAX_NEW = 2048   # registered thinking budget
THINK_SAMPLING = dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20)  # [review 2] Qwen3 card: no greedy in thinking mode

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
    log(f"[l20] st: {st_dir}/stories.json sha256={sha} | templates.json sha256={tpl_sha} | {len(train_doms)} train domains "
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
    log(f"[l20] hard: {path} sha256={sha} | {len(texts)} sentences in {len(names)} train domains | "
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
    log(f"[l20] extra: {path} sha256={sha} | {N_EXTRA} new train domains | {len(texts)} sentences")
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
    log(f"[l20] hard extra: {path} sha256={sha} | {len(texts)} sentences in {len(names)} extra domains")
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
    log(f"[l20] probe: {path} sha256={sha} | {N_PROBE} probe domains ({','.join(names)}) | {len(texts)} normal "
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






# ==============================================================================
# [1b] Lab 18b: system-aware prompt (factor A) and noise (factor B)
# ==============================================================================
PROMPT_SYS = '这句话："{s}"\n这句话对「{n}」造成了什么变化？用一个词概括：'
Q3E_SYS = "Instruct: Identify what change this sentence describes for the system 「{n}」\nQuery:{s}"
SYS_ON = False          # factor A of the current cell (set in main)
NAME_OF = {}            # sentence -> its domain (system) name; every encoded sentence is registered here


def model_text(eye: str, s: str) -> str:
    """Factor A off: the Lab 17 templates. On: the system-aware templates with the sentence's own domain name."""
    if not SYS_ON:
        return (Q3E_TMPL if EYES[eye][2] == "q3e" else PROMPT_TMPL).format(s=s)
    return (Q3E_SYS if EYES[eye][2] == "q3e" else PROMPT_SYS).format(s=s, n=NAME_OF[s])


def make_noised(texts, doms, distractors, seed):
    """One noised copy per row: an unrelated template NOOP sentence of ANOTHER training domain joined with '，',
    before (half) or after (half) the row; a QUESTION-style row ending in '？' always gets it before.
    rng = Random(sha256('18b-noise|seed'))."""
    rng = random.Random(int(hashlib.sha256(f"18b-noise|{seed}".encode()).hexdigest()[:8], 16))
    out = []
    for t, d in zip(texts, doms):
        dist = rng.choice([x for x, dd in distractors if dd != d])
        before = t.endswith("？") or rng.random() < 0.5
        out.append(dist[:-1] + "，" + t if before else t[:-1] + "，" + dist)
    return out


def train_lora(eye, tok, backbone, mods, rows, guard, seed, cell, noised=None, train_set=None):
    """(Lab 17, extended for 18b) rows: dict(texts, cls, dom, gid) of the training rows (gid None outside contrast
    groups). Each step: 8 domains x 12 sentences; per domain 6 from one random contrast group (all of it if it has
    fewer than 6) and the rest from the domain's other rows (group batching, all cells). noised: None (factor B off)
    or one noised text per row: then 24 of the batch rows also go through the eye in their noised form, and the loss
    adds mean(1 - cos(h, h_noised)) + symmetric KL of the auxiliary head's distributions."""
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tag_seed = int(hashlib.sha256(f"18b-lora|{eye}|{seed}".encode()).hexdigest()[:8], 16)  # same for every cell
    seed_all(tag_seed)
    for m in mods:
        m.reset()
        m.enabled, m.drop_on = True, True
    D = EYES[eye][3]
    head = nn.Linear(D, N_SCLS).to(DEVICE).float()
    amt_head = nn.Linear(D, N_AMT).to(DEVICE).float()   # [Lab 20] amount loss on IN / OUT rows (weight 1)
    params = [p for m in mods for p in (m.A, m.B)] + list(head.parameters()) + list(amt_head.parameters())
    opt = torch.optim.AdamW(params, lr=LORA_LR, weight_decay=0.0)
    rng = random.Random(tag_seed)
    # [review 18b-2] separate generator for the noised-row pick, so B-on cells keep the B-off batch order
    nrng = random.Random(int(hashlib.sha256(f"18b-noisepick|{eye}|{seed}".encode()).hexdigest()[:8], 16))
    backbone.train()  # [review 18b-1] enables gradient checkpointing (attention dropout is 0 in these models); LoRA
    #                   dropout is controlled by its own flag; main() calls model.eval() after training
    by_dom = defaultdict(list)
    groups = defaultdict(lambda: defaultdict(list))
    for r, (d, g) in enumerate(zip(rows["dom"], rows["gid"])):
        by_dom[d].append(r)
        if g is not None:
            groups[d][g].append(r)
    doms = sorted(by_dom)
    assert len(doms) == 28 and all(groups[d] for d in doms)
    dom_id = {d: i for i, d in enumerate(doms)}
    trace = []
    for step in range(1, LORA_STEPS + 1):
        lr = LORA_LR * step / LORA_WARMUP if step <= LORA_WARMUP else \
            0.5 * LORA_LR * (1 + math.cos(math.pi * (step - LORA_WARMUP) / (LORA_STEPS - LORA_WARMUP)))
        for pg in opt.param_groups:
            pg["lr"] = lr
        batch = []
        for d in rng.sample(doms, LORA_DOMS_PER_STEP):
            g = rng.choice(sorted(groups[d]))
            mem = groups[d][g]
            part = rng.sample(mem, min(GROUP_PART, len(mem)))
            rest = [r for r in by_dom[d] if r not in set(part)]
            batch += part + rng.sample(rest, LORA_PER_DOM - len(part))
        texts = [rows["texts"][r] for r in batch]
        nz = nrng.sample(range(len(batch)), NOISE_PER_STEP) if noised is not None else []
        ntexts = [noised[batch[i]] for i in nz]
        mt = [model_text(eye, s) for s in texts + ntexts]
        for s, t in zip(texts + ntexts, mt):
            guard.check(s, t)
        for s in texts:
            assert train_set is None or s in train_set, "LEAK: a non-training sentence in a LoRA batch"
        # [pre-run amendment] two passes with separate padding, so the longer noised rows do not pad the whole batch:
        # (1) clean rows: SupCon + CE, backward (frees the clean graph); (2) noised rows: consistency to the DETACHED
        # clean features / distribution (stop-gradient on the clean side), backward. Gradients accumulate; one step.
        opt.zero_grad()
        h, _n = pool_last(tok, backbone, mt[:len(batch)])
        y = torch.tensor([rows["cls"][r] for r in batch], device=DEVICE)
        dvec = torch.tensor([dom_id[rows["dom"][r]] for r in batch], device=DEVICE)
        z = F.normalize(h, dim=-1)
        pos, excl = xdom_masks(y, dvec)
        loss = supcon_from_logits(z @ z.T / SUPCON_TAU, pos, excl) + F.cross_entropy(head(h), y)
        a = torch.tensor([rows["amt"][r] for r in batch], device=DEVICE)
        if bool((a > 0).any()):
            loss = loss + F.cross_entropy(amt_head(h[a > 0]), a[a > 0] - 1)
        if nz:
            hc = h[torch.tensor(nz, device=DEVICE)].detach()
            lp = F.log_softmax(head(hc), -1).detach()
        loss.backward()
        loss_v = float(loss.item())
        if nz:
            del h, z
            hn, _n2 = pool_last(tok, backbone, mt[len(batch):])
            cons = (1 - F.cosine_similarity(hc, hn, dim=-1)).mean()
            lq = F.log_softmax(head(hn), -1)
            kl = 0.5 * (F.kl_div(lq, lp, log_target=True, reduction="batchmean")
                        + F.kl_div(lp, lq, log_target=True, reduction="batchmean"))
            (cons + kl).backward()
            loss_v += float((cons + kl).item())
        torch.nn.utils.clip_grad_norm_(params, LORA_CLIP)
        opt.step()
        if step in (1, 50, 100, LORA_STEPS):
            trace.append(f"{step}:{loss_v:.4f}")
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
# [0d] reading data (Lab 18a format; v1 + v2)
# ==============================================================================
EVENTS4 = ("IN", "OUT", "BREAK", "FIX")
HYPO_CUES = ("如果", "要是", "假如", "假设", "万一", "倘若", "假使", "若", "的话")
OPP = {"IN": "OUT", "OUT": "IN", "BREAK": "FIX", "FIX": "BREAK"}


def label_of(event, variant):
    """(same rule as tools/validate_reading.py)"""
    variant = variant[:-2] if variant.endswith("_B") else variant
    if variant in ("BASE", "POSNEG", "PRONOUN", "CHATTER", "TRANSFER"):
        return event
    if variant == "DIRSWAP":
        return OPP[event]
    return "NOOP"


def _reading_path(folder, name):
    return ["/content/" + name, os.path.join(os.getcwd(), "benchmarks", folder, name)] \
        + _script_dir_candidates("benchmarks", folder, name)


def load_reading_train(table: dict, forbidden, folder, name, version, sha_expected, tag):
    path, raw, sha = _read_first(_reading_path(folder, name), name)
    assert sha_expected is not None and sha == sha_expected, f"{name} not registered: {sha}"
    p = json.loads(raw.decode("utf-8"))
    assert p.get("version") == version
    assert sorted(d["name"] for d in p["domains"]) == sorted(table["doms"]), "reading train domains != the 28"
    texts, cls, dom, var, gid = [], [], [], [], []
    for d in p["domains"]:
        for g in d["groups"]:
            e = g["event"]
            for v, t in [("BASE", g["base"])] + sorted(g["variants"].items()):
                assert v != "HYPO" and not any(c in t for c in HYPO_CUES), (d["name"], g["id"], v)
                texts.append(t)
                cls.append(N_INIT + EVENTS.index(label_of(e, v)))
                dom.append(d["name"])
                var.append(v)
                gid.append(f"{tag}/{g['id']}")
    assert len(set(texts)) == len(texts), f"DATA ERROR: duplicate sentence in {name}"
    assert not set(texts) & set(table["texts"]), "DATA ERROR: a reading sentence equals a table sentence"
    assert not any(f in t for t in texts for f in forbidden), f"LEAK / DATA ERROR: test string in {name}"
    log(f"[l20] reading train {tag}: {path} sha256={sha} | {len(texts)} sentences | "
        + " ".join(f"{k}={v}" for k, v in sorted(Counter(var).items())))
    return dict(sha=sha, texts=texts, cls=cls, dom=dom, var=var, gid=gid)


def load_reading_test(table: dict, forbidden, other_texts, other_doms, folder, name, version, sha_expected, tag):
    path, raw, sha = _read_first(_reading_path(folder, name), name)
    assert sha_expected is not None and sha == sha_expected, f"{name} not registered: {sha}"
    p = json.loads(raw.decode("utf-8"))
    assert p.get("version") == version
    doms = [d["name"] for d in p["domains"]]
    assert len(doms) == N_RTEST and len(set(doms)) == N_RTEST, doms
    assert not any(a in b or b in a for a in doms for b in list(table["doms"]) + list(other_doms)), \
        f"a {tag} test domain overlaps another domain"
    sup, items, texts = {}, [], []
    for d in p["domains"]:
        nm = d["name"]
        for e in EVENTS:
            assert len(d["support"][e]) == 3, (nm, e)
            sup[(nm, EVENTS.index(e))] = []
            for t in d["support"][e]:
                sup[(nm, EVENTS.index(e))].append(len(texts))
                texts.append(t)
        for g in d["groups"]:
            e = g["event"]
            assert "HYPO" in g["variants"], (nm, g["id"])
            for v, t in [("BASE", g["base"])] + sorted(g["variants"].items()):
                items.append(dict(dom=nm, gid=g["id"], var=v, text=t, y=EVENTS.index(label_of(e, v))))
        for i, t in enumerate(d["noop"]):
            items.append(dict(dom=nm, gid=f"noop{i}", var="NOOPSENT", text=t, y=EVENTS.index("NOOP")))
    allt = texts + [it["text"] for it in items]
    assert len(set(allt)) == len(allt), f"DATA ERROR: duplicate sentence in {name}"
    assert not set(allt) & set(other_texts), f"DATA ERROR: a {tag} test sentence equals another sentence"
    assert not any(f in t for t in allt for f in forbidden), f"LEAK / DATA ERROR: test-split string in {name}"
    log(f"[l20] reading test {tag}: {path} sha256={sha} | {N_RTEST} domains ({','.join(doms)}) | {len(texts)} support "
        f"| {len(items)} scored")
    return dict(sha=sha, doms=doms, texts=texts, sup=sup, items=items, tag=tag)


# ---- scoring (copied from Lab 18a) -----------------------------------------------------------------------
class ReadTally:
    def __init__(self, rt):
        self.rt = rt
        self.c = defaultdict(lambda: [0, 0])
        self.pred = defaultdict(lambda: [0] * N_EV)

    def add(self, key, ok):
        self.c[key][0] += int(ok)
        self.c[key][1] += 1

    def score(self, dom, preds):
        groups = defaultdict(list)
        for i, it in enumerate(self.rt["items"]):
            if it["dom"] != dom:
                continue
            ok = preds[i] == it["y"]
            self.pred[i][preds[i]] += 1
            self.add("var_" + it["var"], ok)
            if it["var"] == "BASE":
                self.add("normal", ok)
                self.add("normal_" + EVENTS[it["y"]], ok)
            if it["var"] not in ("HYPO", "NOOPSENT"):
                groups[it["gid"]].append(ok)
                self.add("taught_items", ok)
        for gid, oks in groups.items():
            self.add("group_all", all(oks))

    def line(self):
        return " ".join(f"{k}={self.c[k][0]}/{self.c[k][1]}" for k in sorted(self.c))

    def log_items(self, tag):
        for i, it in enumerate(self.rt["items"]):
            log(f"L18B_ITEM {tag} item={it['dom']}/{it['gid']}/{it['var']} y={EVENTS[it['y']]} "
                f"preds={','.join(map(str, self.pred[i]))}")


def merge_tally(dst, src):
    for k, v in src.c.items():
        dst.c[k][0] += v[0]
        dst.c[k][1] += v[1]
    for k, v in src.pred.items():
        dst.pred[k] = [a + b for a, b in zip(dst.pred[k], v)]


def score_reading(rt, models, mu, F_sup, F_it, seed, tl):
    for d in rt["doms"]:
        ii = [i for i, it in enumerate(rt["items"]) if it["dom"] == d]
        ii_t = torch.tensor(ii, device=DEVICE)
        for m in METHODS:
            for r in (range(N_DRAWS) if m == "C" else (0,)):
                if m == "C":
                    rng = draw_rng("18b-" + rt["tag"], seed, 0, d, 1, r)
                    pick = [rng.choice(rt["sup"][(d, e)]) for e in range(N_EV)]
                    ctr = F_sup[torch.tensor(pick, device=DEVICE)].mean(dim=0)
                else:
                    ctr = mu
                p = ev_scores("FLAT", models[m], F.normalize(F_it[ii_t] - ctr, dim=-1)).argmax(-1).tolist()
                tl[m].score(d, dict(zip(ii, p)))


def score_probe(probe, models, mu, Fp, Fi, seed, ptl):
    """(copied from Lab 18a) the Lab 17b readout."""
    for d in probe["doms"]:
        rows5 = ev_rows_by_class(probe, d)
        ii = [i for i, it in enumerate(probe["items"]) if it["dom"] == d]
        ii_t = torch.tensor(ii, device=DEVICE)
        ir = [i for i, (c, dd) in enumerate(zip(probe["cls"], probe["dom"])) if dd == d and c < N_INIT]
        for m in METHODS:
            for r in (range(N_DRAWS) if m == "C" else (0,)):
                if m == "C":
                    sup, ysup, qry, yq = draw_support(rows5, 1, draw_rng("17b", seed, 0, d, 1, r))
                    ctr = class_center(Fp, [[sup[j] for j in range(len(sup)) if ysup[j] == e] for e in range(N_EV)])
                else:
                    qry = [i for rows in rows5 for i in rows]
                    yq = [e for e, rows in enumerate(rows5) for _i in rows]
                    ctr = mu
                pq = ev_scores("FLAT", models[m], F.normalize(Fp[torch.tensor(qry, device=DEVICE)] - ctr, dim=-1)).argmax(-1).tolist()
                for pp, t in zip(pq, yq):
                    ptl[m].add("normal", pp == t)
                    ptl[m].add(f"normal_{EVENTS[t]}", pp == t)
                pi = init_scores("FLAT", models[m], F.normalize(Fp[torch.tensor(ir, device=DEVICE)] - ctr, dim=-1)).argmax(-1)
                for pp, t in zip(pi.tolist(), [probe["cls"][i] for i in ir]):
                    ptl[m].add("init", pp == t)
                pit = ev_scores("FLAT", models[m], F.normalize(Fi[ii_t] - ctr, dim=-1)).argmax(-1).tolist()
                ptl[m].score_items(d, dict(zip(ii, pit)))


def train_heads(tr, F_tr, seed, F_noise=None):
    """FLAT heads (ZERO / C) on the training rows (+ their noised copies with the same labels when factor B is on;
    a noised row is centred like its own row)."""
    y = torch.tensor(tr["cls"], device=DEVICE)
    norm_rows = [i for i, k in enumerate(tr["kind"]) if k == "normal"]
    centers = {}
    for d in sorted(set(tr["dom"])):
        rows5 = [[i for i in norm_rows if tr["dom"][i] == d and tr["cls"][i] == N_INIT + e] for e in range(N_EV)]
        centers[d] = class_center(F_tr, rows5)
    mu = F_tr[torch.tensor(norm_rows, device=DEVICE)].mean(dim=0)
    C = torch.stack([centers[d] for d in tr["dom"]])
    X = {"ZERO": F.normalize(F_tr - mu, dim=-1), "C": F.normalize(F_tr - C, dim=-1)}
    yy = y
    if F_noise is not None:
        X = {"ZERO": torch.cat([X["ZERO"], F.normalize(F_noise - mu, dim=-1)]),
             "C": torch.cat([X["C"], F.normalize(F_noise - C, dim=-1)])}
        yy = torch.cat([y, y])
    return {m: train_head("FLAT", X[m], yy, seed)[0] for m in METHODS}, mu




# ==============================================================================
# [5] Lab 20: capacity 2..5 and IN / OUT amounts 1..3 (new)
# ==============================================================================
STATES = [f"L{lv}_{sg}" for sg in ("ok", "broken") for lv in range(6)] + ["BURST"]   # 13 final classes
SI = {s: i for i, s in enumerate(STATES)}
CAPS = (2, 3, 4, 5)
HOP_TRAIN_CAPS = (2, 3, 5)            # registered: capacity 4 is never seen by the learned reasoner
HOP_STEPS, HOP_HIDDEN, HOP_BATCH = 6000, 64, 256
N_AMT = 3
SEEDS20 = (42, 1, 2)
AMT_FORMS = ("BASE", "NUMFORM", "DIRSWAP", "NEG", "CHATTER_NUM")
OPP_IO = {"IN": "OUT", "OUT": "IN"}
EXPECTED_AMT_TRAIN_SHA256 = None
EXPECTED_AMT_TEST_SHA256 = None
EXPECTED_STORY2_SHA256 = None


def amt_label(event, k, form):
    """(label, amount) of a form of an amount_v1 group; amount 0 = not an IN / OUT."""
    if form == "DIRSWAP":
        return OPP_IO[event], k
    if form == "NEG":
        return "NOOP", 0
    return event, k


def simulate(cap, lv, sg, events):
    """(Lab 20 rule) events = [(type, amount)] -> final class name.
    IN k: level + k <= cap -> + k; else safeguard ok -> level = cap; broken -> BURST (absorbing). OUT k: max(0, lv - k)."""
    burst = False
    for e, k in events:
        if burst:
            continue
        if e == "IN":
            if lv + k <= cap:
                lv += k
            elif sg == "ok":
                lv = cap
            else:
                burst = True
        elif e == "OUT":
            lv = max(0, lv - k)
        elif e == "BREAK":
            sg = "broken"
        elif e == "FIX":
            sg = "ok"
    return "BURST" if burst else f"L{lv}_{sg}"


def load_amount(name, version, sha_expected, doms_expected, per, forbidden, tag):
    path, raw, sha = _read_first(_reading_path("amount_v1", name), name)
    assert sha_expected is not None and sha == sha_expected, f"{name} not registered: {sha}"
    p = json.loads(raw.decode("utf-8"))
    assert p.get("version") == version
    assert sorted(d["name"] for d in p["domains"]) == sorted(doms_expected), f"{name}: domains"
    texts, cls, dom, var, gid, amt = [], [], [], [], [], []
    for d in p["domains"]:
        want = sorted(f"{e}{k}-{i}" for e in ("IN", "OUT") for k in (2, 3) for i in range(1, per + 1))
        assert sorted(g["id"] for g in d["groups"]) == want, (name, d["name"])
        for g in d["groups"]:
            assert sorted(g["forms"]) == sorted(AMT_FORMS), (d["name"], g["id"])
            for f in AMT_FORMS:
                t = g["forms"][f]
                assert not any(c in t for c in HYPO_CUES), (d["name"], g["id"], f)
                lab, k = amt_label(g["event"], g["amount"], f)
                texts.append(t)
                cls.append(N_INIT + EVENTS.index(lab))
                dom.append(d["name"])
                var.append("AMT-" + f)
                gid.append(f"{tag}/{g['id']}")
                amt.append(k)
    assert len(set(texts)) == len(texts), f"DATA ERROR: duplicate sentence in {name}"
    assert not any(f in t for t in texts for f in forbidden), f"LEAK / DATA ERROR: test string in {name}"
    log(f"[l20] amount {tag}: {path} sha256={sha} | {len(texts)} sentences in {len(p['domains'])} domains")
    return dict(sha=sha, texts=texts, cls=cls, dom=dom, var=var, gid=gid, amt=amt)


def load_stories():
    # on Colab the story file is uploaded as /content/story_v2_stories.json (state_tracking_v1 also has a stories.json)
    path, raw, sha = _read_first(["/content/story_v2_stories.json"] + _reading_path("story_v2", "stories.json")[1:],
                                 "story_v2 stories.json")
    assert EXPECTED_STORY2_SHA256 is not None and sha == EXPECTED_STORY2_SHA256, f"stories.json not registered: {sha}"
    d = json.loads(raw.decode("utf-8"))
    assert d.get("version") == "story_v2"
    for s in d["stories"]:
        assert s["capacity"] in CAPS
        assert simulate(s["capacity"], s["init_level"], s["init_safeguard"],
                        [(x["label"], x["amount"]) for x in s["sentences"]]) == s["final"]
    log(f"[l20] stories: {path} sha256={sha} | {len(d['stories'])} stories in {len(d['domains'])} domains | "
        + " ".join(f"{k}={v}" for k, v in sorted(Counter(s['kind'] for s in d['stories']).items())))
    return d


def length_band(n):
    return "SHORT" if n <= 4 else ("MID" if n <= 7 else "LONG")


class StoryTally:
    def __init__(self):
        self.c = defaultdict(lambda: [0, 0])

    def add(self, s, ok):
        band = length_band(s["length"])
        keys = [f"len{s['length']}", s["kind"], band + ("" if s["kind"] == "normal" else "_trap"), "all"]
        if s["kind"] == "normal":
            keys += [f"nlen{s['length']}", f"ncap{s['capacity']}", f"n{band}_cap{s['capacity']}"]
        for k in keys:
            self.c[k][0] += int(ok)
            self.c[k][1] += 1

    def line(self):
        return " ".join(f"{k}={self.c[k][0]}/{self.c[k][1]}" for k in sorted(self.c))


def hop_init(cap, lv, sg):
    """initial-state input: level one-hot (0..5), safeguard one-hot (ok, broken), capacity thermometer (>=1 .. >=5)."""
    v = [0.0] * 13
    v[lv] = 1.0
    v[6 + (sg == "broken")] = 1.0
    for j in range(cap):
        v[8 + j] = 1.0
    return v


class Hop(nn.Module):
    """Learned state tracker: GRU over [event (5) + amount (3)] per step, h0 from hop_init (13)."""

    def __init__(self):
        super().__init__()
        self.init = nn.Linear(13, HOP_HIDDEN)
        self.gru = nn.GRU(N_EV + N_AMT, HOP_HIDDEN, batch_first=True)
        self.out = nn.Linear(HOP_HIDDEN, len(STATES))

    def forward(self, init_vec, ev, lengths):
        h0 = torch.tanh(self.init(init_vec)).unsqueeze(0)
        o, _ = self.gru(ev, h0)
        last = o[torch.arange(o.shape[0], device=o.device), lengths - 1]
        return self.out(last)


def onehot_step(e, k):
    v = [0.0] * (N_EV + N_AMT)
    v[EVENTS.index(e)] = 1.0
    v[N_EV + max(k, 1) - 1] = 1.0
    return v


def train_hop(seed):
    """Random abstract stories of length 1..4, capacity in HOP_TRAIN_CAPS, uniform events, uniform amounts 1..3 (for a
    non-IN/OUT step the amount input is a random one-hot that the rule ignores), uniform initial level 0..cap;
    AdamW 3e-3, 6000 steps, batch 256. Returns the model and its one-hot accuracy on fresh length-1..12 stories per
    capacity 2..5."""
    seed_all(seed)
    rng = random.Random(seed)
    hop = Hop().to(DEVICE)
    opt = torch.optim.AdamW(hop.parameters(), lr=3e-3)

    def batch(nmin, nmax, B, caps):
        inits, seqs, ys, lens = [], [], [], []
        for _ in range(B):
            n, cap = rng.randint(nmin, nmax), rng.choice(caps)
            lv, sg = rng.randint(0, cap), rng.choice(["ok", "broken"])
            ev = [(rng.choice(EVENTS), rng.randint(1, 3)) for _ in range(n)]
            inits.append(hop_init(cap, lv, sg))
            seqs.append(torch.tensor([onehot_step(e, k) for e, k in ev]))
            ys.append(SI[simulate(cap, lv, sg, [(e, k if e in ("IN", "OUT") else 0) for e, k in ev])])
            lens.append(n)
        X = nn.utils.rnn.pad_sequence(seqs, batch_first=True).to(DEVICE)
        return (torch.tensor(inits, device=DEVICE), X, torch.tensor(ys, device=DEVICE), torch.tensor(lens, device=DEVICE))

    for step in range(HOP_STEPS):
        i, X, y, L = batch(1, 4, HOP_BATCH, HOP_TRAIN_CAPS)
        loss = F.cross_entropy(hop(i, X, L), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    hop.eval()
    acc = {}
    with torch.no_grad():
        for cap in CAPS:
            i, X, y, L = batch(1, 12, 2000, (cap,))
            acc[cap] = float((hop(i, X, L).argmax(-1) == y).float().mean())
    return hop, acc


def story_pool(stories, amt_test):
    """Unique sentences of the stories + every amount_v1 test sentence + the per-domain support (one-example readout).
    gold: text -> (event index, amount, variant)."""
    texts, dom, idx, gold = [], [], {}, {}

    def add(t, d, g):
        if t not in idx:
            idx[t] = len(texts)
            texts.append(t)
            dom.append(d)
            gold[t] = g
        else:
            assert gold[t][:2] == g[:2], f"DATA ERROR: {t} has two labels"

    for s in stories["stories"]:
        for x in s["sentences"]:
            add(x["text"], s["domain"], (EVENTS.index(x["label"]), x["amount"], x["var"]))
    for t, c, d, v, k in zip(amt_test["texts"], amt_test["cls"], amt_test["dom"], amt_test["var"], amt_test["amt"]):
        add(t, d, (c - N_INIT, k, v))
    sup = {}
    for d in stories["domains"]:
        for e in EVENTS:
            t = d["support"][e][0]           # fixed: the first support sentence of each event type
            assert t not in idx
            sup[(d["name"], EVENTS.index(e))] = len(texts)
            idx[t] = len(texts)
            texts.append(t)
            dom.append(d["name"])
    return texts, dom, idx, sup, gold


def train_amt_head(X, y, seed):
    """FLAT amount head (3-way) on readout-C features of the training IN / OUT rows (same recipe as train_head)."""
    seed_all(seed)
    model = nn.Sequential(nn.Linear(X.shape[1], HIDDEN), nn.GELU(), nn.Linear(HIDDEN, N_AMT)).to(DEVICE).float()
    loss = train_loop(model, lambda m: F.cross_entropy(m(X), y), CE_STEPS, seed)
    model.eval()
    return model, loss


def run_eye(stories, seed):
    """Train the Lab 19 eye (+ amount_v1 training sentences, + an amount loss) for one seed, read every pool sentence
    (readout C, centre = mean of the domain's fixed support), then score the four systems."""
    global SYS_ON
    st = load_st()
    extra = load_extra(st)
    table = merge_tables(st, extra)
    hard = merge_hard(load_hard(st), load_hard_extra(table), table)
    r1 = load_reading_train(table, st["forbidden"], "reading_v1", "train_groups.json", "reading_v1_train",
                            EXPECTED_RTRAIN_SHA256, "v1")
    r2 = load_reading_train(table, st["forbidden"], "reading_v2", "train_groups_v2.json", "reading_v2_train",
                            EXPECTED_RTRAIN2_SHA256, "v2")
    a_tr = load_amount("train.json", "amount_v1_train", EXPECTED_AMT_TRAIN_SHA256, table["doms"], 4, st["forbidden"], "amt")
    a_te = load_amount("test.json", "amount_v1_test", EXPECTED_AMT_TEST_SHA256, [d["name"] for d in stories["domains"]],
                       2, st["forbidden"], "amt_test")
    io = (N_INIT + EVENTS.index("IN"), N_INIT + EVENTS.index("OUT"))
    hrows = [i for i, (c, cat) in enumerate(zip(hard["cls"], hard["cat"])) if c >= 0 and cat in USED_HARD]
    tr = dict(texts=list(table["texts"]), cls=list(table["cls"]), dom=list(table["dom"]),
              kind=["normal"] * len(table["texts"]), gid=[None] * len(table["texts"]),
              amt=[1 if c in io else 0 for c in table["cls"]])   # registered: every existing IN / OUT sentence = 1
    for i in hrows:
        for k, v in (("texts", hard["texts"][i]), ("cls", hard["cls"][i]), ("dom", hard["dom"][i]), ("kind", "hard"),
                     ("gid", None), ("amt", 1 if hard["cls"][i] in io else 0)):
            tr[k].append(v)
    for r in (r1, r2):
        for t, c, d, g in zip(r["texts"], r["cls"], r["dom"], r["gid"]):
            for k, v in (("texts", t), ("cls", c), ("dom", d), ("kind", "pair"), ("gid", g), ("amt", 1 if c in io else 0)):
                tr[k].append(v)
    assert not set(a_tr["texts"]) & set(tr["texts"]), "DATA ERROR: an amount train sentence equals another sentence"
    for t, c, d, g, a in zip(a_tr["texts"], a_tr["cls"], a_tr["dom"], a_tr["gid"], a_tr["amt"]):
        for k, v in (("texts", t), ("cls", c), ("dom", d), ("kind", "amount"), ("gid", g), ("amt", a)):
            tr[k].append(v)
    texts, dom, idx, sup, gold = story_pool(stories, a_te)
    assert not set(texts) & set(tr["texts"]), "LEAK: a story / amount test sentence is a training sentence"
    assert not set(d["name"] for d in stories["domains"]) & set(table["doms"])
    for t, d in zip(tr["texts"], tr["dom"]):
        NAME_OF[t] = d
    for t, d in zip(texts, dom):
        NAME_OF[t] = d
    guard = LeakGuard(list(NAME_OF), st["forbidden"])
    train_set = set(tr["texts"])
    eye = "Q15"
    mid, kind, _tmpl, dim, _tun = EYES[eye]
    tok = AutoTokenizer.from_pretrained(mid)
    tok.padding_side = "right"
    SYS_ON = True
    worst = max(len(tok(model_text(eye, s)).input_ids) for s in list(NAME_OF))
    assert worst <= SENT_MAXLEN, worst
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map=DEVICE)
    model.eval()
    model.requires_grad_(False)
    backbone = backbone_of(eye, model)
    mods = inject_lora(backbone)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if "FAKE" not in getattr(transformers, "__version__", ""):
        assert len(mods) == len(LORA_TARGETS) * model.config.num_hidden_layers
    t1 = time.time()
    trace = train_lora(eye, tok, backbone, mods, tr, guard, seed, "A1B0", None, train_set)
    model.eval()
    F_tr, _a, _b = extract17(tok, backbone, tr["texts"], eye, guard, f"{eye}/s{seed}/train", False)
    F_st, _c, _d = extract17(tok, backbone, texts, eye, guard, f"{eye}/s{seed}/pool", False)
    F_tr, F_st = F_tr.to(DEVICE), F_st.to(DEVICE)
    lora_off(mods)
    log(f"L20_LORA seed={seed} loss={','.join(trace)} sec={time.time() - t1:.1f} longest_tokens={worst} "
        f"fp={float(F_tr.double().sum()).hex()}")
    # event heads (as Lab 19) + the amount head on the same readout-C features
    models, mu = train_heads(tr, F_tr, seed, None)
    norm_rows = [i for i, k in enumerate(tr["kind"]) if k == "normal"]
    centers = {}
    for d in sorted(set(tr["dom"])):
        rows5 = [[i for i in norm_rows if tr["dom"][i] == d and tr["cls"][i] == N_INIT + e] for e in range(N_EV)]
        centers[d] = class_center(F_tr, rows5)
    arows = [i for i, a in enumerate(tr["amt"]) if a > 0]
    XA = F.normalize(F_tr[torch.tensor(arows, device=DEVICE)]
                     - torch.stack([centers[tr["dom"][i]] for i in arows]), dim=-1)
    amt_head, _l = train_amt_head(XA, torch.tensor([tr["amt"][i] - 1 for i in arows], device=DEVICE), seed)
    log(f"[l20] amount head: {len(arows)} training IN/OUT rows | per amount "
        + " ".join(f"{k}={v}" for k, v in sorted(Counter(tr['amt'][i] for i in arows).items())))
    probs = torch.zeros(len(texts), N_EV, device=DEVICE)
    aprobs = torch.zeros(len(texts), N_AMT, device=DEVICE)
    for d in stories["domains"]:
        rows = [i for i, dd in enumerate(dom) if dd == d["name"]]
        ctr = F_st[torch.tensor([sup[(d["name"], e)] for e in range(N_EV)], device=DEVICE)].mean(dim=0)
        with torch.no_grad():
            Xc = F.normalize(F_st[rows] - ctr, dim=-1)
            probs[rows] = F.softmax(ev_scores("FLAT", models["C"], Xc), -1)
            aprobs[rows] = F.softmax(amt_head(Xc), -1)
    pred = probs.argmax(-1).tolist()
    apred = (aprobs.argmax(-1) + 1).tolist()

    def read(t):
        e = EVENTS[pred[idx[t]]]
        return e, (apred[idx[t]] if e in ("IN", "OUT") else 0)

    # sentence-level accuracy: events (all pool sentences) and amounts (gold IN / OUT sentences)
    sc, ac = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for t, (g, k, v) in gold.items():
        ok = pred[idx[t]] == g
        for key in ("all", v):
            sc[key][0] += int(ok)
            sc[key][1] += 1
        if k > 0:
            aok = apred[idx[t]] == k
            keys = [f"amt{k}", v, "registered" if v.startswith("AMT-") else "k1_existing"]
            keys += ["registered_event_right"] if v.startswith("AMT-") and ok else []
            for key in keys:
                ac[key][0] += int(aok)
                ac[key][1] += 1
    log(f"L20_SENT seed={seed} " + " ".join(f"{k}={a}/{b}" for k, (a, b) in sorted(sc.items())))
    log(f"L20_AMT seed={seed} " + " ".join(f"{k}={a}/{b}" for k, (a, b) in sorted(ac.items())))
    hop, hop_acc = train_hop(seed)
    log(f"L20_HOP seed={seed} " + " ".join(f"onehot_len1-12_cap{c}={a:.4f}" for c, a in sorted(hop_acc.items())))
    ex, hp, orc, ohp = StoryTally(), StoryTally(), StoryTally(), StoryTally()
    dec = defaultdict(lambda: [0, 0])      # (kind group, error type) -> [final right, stories]
    harm = defaultdict(lambda: [0, 0])     # variant -> [stories made wrong when this variant was misread, misreads]
    for s in stories["stories"]:
        cap, lv, sg = s["capacity"], s["init_level"], s["init_safeguard"]
        gl = [(x["label"], x["amount"]) for x in s["sentences"]]
        p = [read(x["text"]) for x in s["sentences"]]
        orc.add(s, simulate(cap, lv, sg, gl) == s["final"])
        ok = simulate(cap, lv, sg, p) == s["final"]
        ex.add(s, ok)
        n_type = sum(a[0] != b[0] for a, b in zip(gl, p))
        n_amt = sum(a[0] == b[0] and a[1] != b[1] for a, b in zip(gl, p))
        et = "none" if n_type + n_amt == 0 else ("type" if n_amt == 0 else ("amount" if n_type == 0 else "both"))
        grp = "normal" if s["kind"] == "normal" else "trap"
        dec[(grp, et)][0] += int(ok)
        dec[(grp, et)][1] += 1
        for x, a, b in zip(s["sentences"], gl, p):
            if a != b:
                harm[x["var"]][1] += 1
                harm[x["var"]][0] += int(not ok)
        iv = torch.tensor([hop_init(cap, lv, sg)], device=DEVICE)
        L = torch.tensor([s["length"]], device=DEVICE)
        with torch.no_grad():
            P = torch.cat([probs, aprobs], -1)[torch.tensor([idx[x["text"]] for x in s["sentences"]], device=DEVICE)]
            hy = hop(iv, P.unsqueeze(0), L).argmax(-1).item()
            G = torch.tensor([onehot_step(e, k) for e, k in gl], device=DEVICE).unsqueeze(0)
            gy = hop(iv, G, L).argmax(-1).item()
        hp.add(s, STATES[hy] == s["final"])
        ohp.add(s, STATES[gy] == s["final"])
    log(f"L20_RES seed={seed} system=ORACLE+EXEC {orc.line()}")
    log(f"L20_RES seed={seed} system=EYE+EXEC {ex.line()}")
    log(f"L20_RES seed={seed} system=ORACLE+HOP {ohp.line()}")
    log(f"L20_RES seed={seed} system=EYE+HOP {hp.line()}")
    log(f"L20_ERRS seed={seed} by_error_type " + " ".join(f"{g}_{e}={a}/{b}" for (g, e), (a, b) in sorted(dec.items())))
    log(f"L20_ERRS seed={seed} harm_by_variant " + " ".join(f"{k}={a}/{b}" for k, (a, b) in sorted(harm.items())))
    SYS_ON = False
    return guard


# ==============================================================================
# main
# ==============================================================================
def main():
    t_start = time.time()
    job = os.environ.get("LAB20_JOB", "").strip()
    assert job == "eye", f"LAB20_JOB={job!r}"
    seed = int(os.environ.get("LAB20_SEED", "0"))
    assert seed in SEEDS20, seed
    log(f"[l20] job={job} seed={seed} device={DEVICE} | torch {torch.__version__} | transformers "
        f"{getattr(transformers, '__version__', '?')}")
    try:
        with open(os.path.abspath(__file__), "rb") as _f:
            script_sha = hashlib.sha256(_f.read()).hexdigest()
    except (NameError, OSError):
        script_sha = "unavailable"
    env_tag = f"torch={torch.__version__},transformers={getattr(transformers, '__version__', '?')},device={DEVICE}"
    if DEVICE == "cuda":
        env_tag += f",gpu={torch.cuda.get_device_name(0).replace(' ', '_')}"
    if "FAKE" in getattr(transformers, "__version__", ""):
        log("!" * 100 + "\n!!! [l20] FAKE transformers stub: TEST RUN, NOT a registered run\n" + "!" * 100)
    stories = load_stories()
    tpl_forbidden = load_st()["forbidden"]
    raw = json.dumps(stories["stories"], ensure_ascii=False)
    assert not any(f in raw for f in tpl_forbidden if len(f) > 3), "LEAK: test-domain string in stories"
    run_eye(stories, seed)
    log(f"[l20] total time {time.time() - t_start:.1f}s")
    log(f"L20_CONFIG job={job} seed={seed} data_sha_story={EXPECTED_STORY2_SHA256} "
        f"data_sha_amt_train={EXPECTED_AMT_TRAIN_SHA256} data_sha_amt_test={EXPECTED_AMT_TEST_SHA256} env={env_tag} "
        f"script_sha={script_sha} hop_train_caps={','.join(map(str, HOP_TRAIN_CAPS))} hop_steps={HOP_STEPS}")


if __name__ == "__main__":
    main()
