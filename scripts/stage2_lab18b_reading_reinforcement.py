# ==============================================================================
# Stage 2 / Lab 18b: reading reinforcement (B1). PRE-REGISTERED on issue #25 (comment "Lab 18b「讀的補強」",
#   2026-11-01, before any data or code). Numbers only, NO verdict. Last round of roadmap step 1.
# Question: do two human-like ways of filtering noise help the recogniser read naturally written, noisy sentences of
#   NEW domains as whole contrast groups?  A = read with the question in mind (system-aware prompt naming the system);
#   B = noise practice with an invariance loss. 2 x 2, eyes Q15 / Q3E, seeds 42 / 1 / 2 / 3 / 4.
# Hard rule: the 8 test domains of state_tracking_v1 are NOT used (Lab 15-18a loaders and leak guard).
# Data (sha256 pinned): st / extra / hard / hard extra / probe as Lab 17b; reading_v1/train_groups.json and
#   reading_v1/test.json as Lab 18a; reading_v2/train_groups_v2.json (28 domains x 4 groups in 4 natural styles, extra
#   CHATTER_B / POSNEG_B / OTHER_B / TRANSFER_B); reading_v2/test_v2.json (8 new sealed domains, 4 new styles; primary).
# All cells: training rows = normal table + Lab 17 hard (NEG / OTHER / TEMP) + reading groups v1 + v2. LoRA recipe of
#   Lab 17 (150 steps, rank 16, SupCon tau 0.1 + 13-class CE, lr 2e-4) with GROUP BATCHING: per step 8 domains x 12
#   sentences, 6 of them from one random contrast group of the domain, 6 from its other rows. LoRA seed depends on
#   (eye, seed) only, so the 4 cells share init and batch order. FLAT head (Lab 15 recipe).
# Factor A (system-aware prompt): off = Lab 17 templates; on = Q15 '这句话："{s}"\n这句话对「{name}」造成了什么变化？用一个词
#   概括：' / Q3E 'Instruct: Identify what change this sentence describes for the system 「{name}」\nQuery:{s}', name =
#   the sentence's domain name (training, support and test alike).
# Factor B (noise practice): one noised copy per training row (an unrelated template NOOP sentence of ANOTHER training
#   domain joined with '，' before / after, rng per seed). LoRA: 24 batch rows also encoded noised; loss += mean(1 - cos)
#   + symmetric KL of the auxiliary head. Head: noised rows added with the same labels (centred like their own row).
# Readout: ZERO / C (C primary; support = one per event type from the test domain's support, 10 draws).
# Metrics (per test): group_all (base + taught variants, same draw), normal (base), var_*, HYPO, NOOPSENT; Lab 17b probe.
# Judgement (registered; test v2, readout C; best cell over 2 x 2 x eyes by group_all):
#   讀得廣 group_all >= 80 and normal >= 95 and HYPO >= 70; 教了才會 the same with HYPO < 70; 還沒讀懂 otherwise.
#   A factor counts as effective if its main effect >= +3 points AND it points the same way in >= 4 of 5 seeds.
# Machine lines: L18B_LORA, L18B_RES (test=v2 / v1), L18B_PROBE, L18B_ITEM (v2), then L18B_CONFIG as the LAST line.
#   Aggregate: python tools/aggregate_lab18b.py lab18b_*.log
# ==============================================================================
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
SENT_MAXLEN = 160  # [amendment] 128 in Lab 17; the natural-style test sentences reach 132 tokens with the Q3E prompt (no truncation, asserted)
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
EXPECTED_RTRAIN2_SHA256 = None  # set when reading_v2/train_groups_v2.json is accepted
EXPECTED_RTEST2_SHA256 = None   # set when reading_v2/test_v2.json is accepted
EXPECTED_RTEST_SHA256 = "49616f38581105d0c69c6e9b1509d8d763027fdb3d3b7ec199ea8a07d03c4c4a"   # accepted: validator PASS, blind 1256/1256 (one annotator row-shift re-checked 20/20)

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
    log(f"[l18b] st: {st_dir}/stories.json sha256={sha} | templates.json sha256={tpl_sha} | {len(train_doms)} train domains "
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
    log(f"[l18b] hard: {path} sha256={sha} | {len(texts)} sentences in {len(names)} train domains | "
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
    log(f"[l18b] extra: {path} sha256={sha} | {N_EXTRA} new train domains | {len(texts)} sentences")
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
    log(f"[l18b] hard extra: {path} sha256={sha} | {len(texts)} sentences in {len(names)} extra domains")
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
    log(f"[l18b] probe: {path} sha256={sha} | {N_PROBE} probe domains ({','.join(names)}) | {len(texts)} normal "
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
    params = [p for m in mods for p in (m.A, m.B)] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=LORA_LR, weight_decay=0.0)
    rng = random.Random(tag_seed)
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
        nz = rng.sample(range(len(batch)), NOISE_PER_STEP) if noised is not None else []
        ntexts = [noised[batch[i]] for i in nz]
        mt = [model_text(eye, s) for s in texts + ntexts]
        for s, t in zip(texts + ntexts, mt):
            guard.check(s, t)
        for s in texts:
            assert train_set is None or s in train_set, "LEAK: a non-training sentence in a LoRA batch"
        h_all, _n = pool_last(tok, backbone, mt)
        h, hn = h_all[:len(batch)], h_all[len(batch):]
        y = torch.tensor([rows["cls"][r] for r in batch], device=DEVICE)
        dvec = torch.tensor([dom_id[rows["dom"][r]] for r in batch], device=DEVICE)
        z = F.normalize(h, dim=-1)
        pos, excl = xdom_masks(y, dvec)
        loss = supcon_from_logits(z @ z.T / SUPCON_TAU, pos, excl) + F.cross_entropy(head(h), y)
        if nz:
            hc = h[torch.tensor(nz, device=DEVICE)]
            cons = (1 - F.cosine_similarity(hc, hn, dim=-1)).mean()
            lp, lq = F.log_softmax(head(hc), -1), F.log_softmax(head(hn), -1)
            kl = 0.5 * (F.kl_div(lq, lp, log_target=True, reduction="batchmean")
                        + F.kl_div(lp, lq, log_target=True, reduction="batchmean"))
            loss = loss + cons + kl
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
    log(f"[l18b] reading train {tag}: {path} sha256={sha} | {len(texts)} sentences | "
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
    log(f"[l18b] reading test {tag}: {path} sha256={sha} | {N_RTEST} domains ({','.join(doms)}) | {len(texts)} support "
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
# main
# ==============================================================================
def main():
    global SYS_ON
    t_start = time.time()
    log(f"[l18b] device={DEVICE} | seeds={SEEDS} | python {sys.version.split()[0]} | torch {torch.__version__} | "
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
        log("!" * 100 + "\n!!! [l18b] FAKE transformers stub: TEST RUN, NOT a registered run\n" + "!" * 100)
    eyes_env = os.environ.get("LAB18B_EYES", "").strip()
    eyes = [e.strip() for e in eyes_env.split(",") if e.strip()] if eyes_env else list(EYE_ORDER)
    if any(e not in EYES for e in eyes) or len(set(eyes)) != len(eyes):
        raise ValueError(f"LAB18B_EYES={eyes_env!r}; allowed {list(EYES)}")
    eyes = [e for e in EYE_ORDER if e in eyes]

    st = load_st()
    extra = load_extra(st)
    table = merge_tables(st, extra)
    hard = merge_hard(load_hard(st), load_hard_extra(table), table)
    probe = load_probe(table, hard, st["forbidden"])
    r1 = load_reading_train(table, st["forbidden"], "reading_v1", "train_groups.json", "reading_v1_train",
                            EXPECTED_RTRAIN_SHA256, "v1")
    r2 = load_reading_train(table, st["forbidden"], "reading_v2", "train_groups_v2.json", "reading_v2_train",
                            EXPECTED_RTRAIN2_SHA256, "v2")
    assert not set(r1["texts"]) & set(r2["texts"]) and not (set(r1["texts"]) | set(r2["texts"])) & set(hard["texts"])
    trained = table["texts"] + hard["texts"] + r1["texts"] + r2["texts"]
    probe_items = [it["text"] for it in probe["items"]]
    t2 = load_reading_test(table, st["forbidden"], trained + probe["texts"] + probe_items, probe["doms"],
                           "reading_v2", "test_v2.json", "reading_v2_test", EXPECTED_RTEST2_SHA256, "v2")
    t1 = load_reading_test(table, st["forbidden"], trained + probe["texts"] + probe_items, list(probe["doms"]) + t2["doms"],
                           "reading_v1", "test.json", "reading_v1_test", EXPECTED_RTEST_SHA256, "v1")
    assert not (set(t1["texts"]) | {it["text"] for it in t1["items"]}) & (set(t2["texts"]) | {it["text"] for it in t2["items"]})

    # training rows (all cells): normal table + Lab 17 hard (NEG / OTHER / TEMP) + reading groups v1 + v2
    hrows = [i for i, (c, cat) in enumerate(zip(hard["cls"], hard["cat"])) if c >= 0 and cat in USED_HARD]
    tr = dict(texts=list(table["texts"]), cls=list(table["cls"]), dom=list(table["dom"]),
              kind=["normal"] * len(table["texts"]), gid=[None] * len(table["texts"]))
    for i in hrows:
        for k, v in (("texts", hard["texts"][i]), ("cls", hard["cls"][i]), ("dom", hard["dom"][i]), ("kind", "hard"),
                     ("gid", None)):
            tr[k].append(v)
    for r in (r1, r2):
        for t, c, d, g in zip(r["texts"], r["cls"], r["dom"], r["gid"]):
            for k, v in (("texts", t), ("cls", c), ("dom", d), ("kind", "pair"), ("gid", g)):
                tr[k].append(v)
    distractors = [(t, d) for t, c, d in zip(table["texts"], table["cls"], table["dom"]) if c == N_INIT + EVENTS.index("NOOP")]
    assert len(distractors) == 28 * 6
    # names (factor A) for every sentence that can reach an encoder
    for t, d in zip(tr["texts"], tr["dom"]):
        NAME_OF[t] = d
    for t, d in zip(probe["texts"], probe["dom"]):
        NAME_OF[t] = d
    for it in probe["items"]:
        NAME_OF[it["text"]] = it["dom"]
    for rt in (t2, t1):
        for (d, _e), rows in rt["sup"].items():
            for i in rows:
                NAME_OF[rt["texts"][i]] = d
        for it in rt["items"]:
            NAME_OF[it["text"]] = it["dom"]
    noised = {s: make_noised(tr["texts"], tr["dom"], distractors, s) for s in SEEDS}
    for s in SEEDS:
        for t, d in zip(noised[s], tr["dom"]):
            NAME_OF[t] = d
    allowed = list(NAME_OF)
    guard = LeakGuard(allowed, st["forbidden"])
    train_set = set(tr["texts"])
    cue_rows = [t for t in tr["texts"] if any(c in t for c in HYPO_CUES)]
    log(f"[l18b] training rows {len(tr['texts'])} (normal {tr['kind'].count('normal')}, hard {tr['kind'].count('hard')}, "
        f"pair {tr['kind'].count('pair')}; groups {len({g for g in tr['gid'] if g})}) | rows with a hypothetical cue "
        f"character: {len(cue_rows)} " + " | ".join(cue_rows[:5]))
    for eye in eyes:
        tok = AutoTokenizer.from_pretrained(EYES[eye][0])
        worst = 0
        for SYS_ON in (False, True):
            worst = max(worst, max(len(tok(model_text(eye, s)).input_ids) for s in allowed))
        assert worst <= SENT_MAXLEN, (eye, worst)
        log(f"[l18b] preflight {eye}: longest model text {worst} tokens (both prompts, incl. noised rows)")

    for eye in eyes:
        mid, kind, _tmpl, dim, tunable = EYES[eye]
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(mid)
        tok.padding_side = "right"
        model = (AutoModelForCausalLM if kind == "causal" else AutoModel).from_pretrained(
            mid, dtype=torch.bfloat16, device_map=DEVICE)
        model.eval()
        model.requires_grad_(False)
        backbone = backbone_of(eye, model)
        assert model.config.hidden_size == dim, (eye, model.config.hidden_size)
        mods = inject_lora(backbone)
        n_layers = getattr(model.config, "num_hidden_layers", None)
        if "FAKE" not in getattr(transformers, "__version__", ""):
            assert len(mods) == len(LORA_TARGETS) * n_layers, (len(mods), n_layers)
        lora_ids = {id(m.A) for m in mods} | {id(m.B) for m in mods}
        assert all(id(p) in lora_ids for p in model.parameters() if p.requires_grad)
        log(f"[l18b] {eye}: {mid} loaded in {time.time() - t0:.1f}s | layers {n_layers} | LoRA modules {len(mods)} | "
            f"revision {getattr(model.config, '_commit_hash', '?')}")

        def enc(tag, txts):
            H, _mx, _mc = extract17(tok, backbone, txts, eye, guard, f"{eye}/{tag}", False)
            return H.to(DEVICE)

        tallies = {(c, m): (ReadTally(t2), ReadTally(t1), Tally(probe)) for c in CELLS for m in METHODS}
        for seed in SEEDS:
            for cell in CELLS:
                a_on, b_on = cell
                SYS_ON = bool(a_on)
                ctag = f"A{a_on}B{b_on}"
                t1s = time.time()
                trace = train_lora(eye, tok, backbone, mods, tr, guard, seed, ctag,
                                   noised[seed] if b_on else None, train_set)
                model.eval()
                F_tr = enc(f"{ctag}-s{seed}/train", tr["texts"])
                F_nz = enc(f"{ctag}-s{seed}/noised", noised[seed]) if b_on else None
                feats = {rt["tag"]: (enc(f"{ctag}-s{seed}/{rt['tag']}sup", rt["texts"]),
                                     enc(f"{ctag}-s{seed}/{rt['tag']}item", [it["text"] for it in rt["items"]]))
                         for rt in (t2, t1)}
                Fp, Fi = enc(f"{ctag}-s{seed}/pnorm", probe["texts"]), enc(f"{ctag}-s{seed}/pitem", probe_items)
                lora_off(mods)
                log(f"L18B_LORA eye={eye} cell={ctag} seed={seed} loss={','.join(trace)} sec={time.time() - t1s:.1f} "
                    f"fp={float(F_tr.double().sum()).hex()}")
                models, mu = train_heads(tr, F_tr, seed, F_nz)
                tl2 = {m: ReadTally(t2) for m in METHODS}
                tl1 = {m: ReadTally(t1) for m in METHODS}
                ptl = {m: Tally(probe) for m in METHODS}
                score_reading(t2, models, mu, *feats["v2"], seed, tl2)
                score_reading(t1, models, mu, *feats["v1"], seed, tl1)
                score_probe(probe, models, mu, Fp, Fi, seed, ptl)
                for m in METHODS:
                    tag = f"eye={eye} cell={ctag} method={m} seed={seed}"
                    log(f"L18B_RES test=v2 {tag} {tl2[m].line()}")
                    log(f"L18B_RES test=v1 {tag} {tl1[m].line()}")
                    log(f"L18B_PROBE {tag} " + " ".join(f"{k}={ptl[m].c[k][0]}/{ptl[m].c[k][1]}" for k in sorted(ptl[m].c)))
                    merge_tally(tallies[(cell, m)][0], tl2[m])
                    merge_tally(tallies[(cell, m)][1], tl1[m])
                    merge_tally(tallies[(cell, m)][2], ptl[m])
                del F_tr, F_nz, feats, Fp, Fi, models
                free_gpu()
            log(f"[l18b] {eye}: seed {seed} done ({time.time() - t0:.0f}s)")
        SYS_ON = False
        for cell in CELLS:
            for m in METHODS:
                tag = f"eye={eye} cell=A{cell[0]}B{cell[1]} method={m} seed=all"
                a, b, p = tallies[(cell, m)]
                log(f"L18B_RES test=v2 {tag} {a.line()}")
                log(f"L18B_RES test=v1 {tag} {b.line()}")
                log(f"L18B_PROBE {tag} " + " ".join(f"{k}={p.c[k][0]}/{p.c[k][1]}" for k in sorted(p.c)))
                a.log_items(f"test=v2 {tag}")
        del model, backbone, mods, tok
        free_gpu()
    log(f"[l18b] leak guard: {guard.n_checked} inputs checked; none contains a test-domain string")
    log(f"[l18b] total time {time.time() - t_start:.1f}s")
    log(f"L18B_CONFIG data_sha_st={st['sha']} templates_sha={st['tpl_sha']} data_sha_extra={extra['sha']} "
        f"data_sha_hard={hard['sha']} data_sha_hard_extra={hard['sha_extra']} data_sha_probe={probe['sha']} "
        f"data_sha_rtrain={r1['sha']} data_sha_rtrain2={r2['sha']} data_sha_rtest={t1['sha']} data_sha_rtest2={t2['sha']} "
        f"env={env_tag} script_sha={script_sha} eyes={','.join(eyes)} seeds={','.join(map(str, SEEDS))} "
        f"draws={N_DRAWS} cells={','.join(f'A{a}B{b}' for a, b in CELLS)}")


if __name__ == "__main__":
    main()
