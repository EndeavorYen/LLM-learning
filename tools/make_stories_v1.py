"""Build benchmarks/story_v1/stories.json for Lab 19 (issue #25, registration "Lab 19（第二步 2a）「讀＋推接起來」").

    python tools/make_stories_v1.py [--check]

Stories are composed (fixed rng) from already accepted, labelled sentences of 22 domains that were never trained on:
reading_v2/test_v2.json (8), reading_v1/test.json (8), comprehension_probe_v1/probe.json (6). The 8 test domains of
state_tracking_v1 are not touched. Support sentences (the one-example readout pool) are kept out of the stories:
reading tests -> their `support`; probe -> the FIRST normal sentence of each event type.
Rule = Lab 14 `simulate` (capacity 3; IN at 3: ok -> diverted, broken -> BURST, absorbing; OUT at 0: no change).
  normal  per domain x length 1..12: 10 stories; event label drawn with P(IN .3, OUT .2, BREAK .1, FIX .1, NOOP .3),
          then an unused sentence with that label; random initial state (level 0..3, safeguard ok / broken).
  trap    per domain 20 stories, length 8..12, 5 of each:
          T1 early break, late overflow (final BURST); T2 fix, then fill past capacity (final L3_ok);
          T3 mostly not-happened sentences (>= 60% NOOP-labelled variants of real events); T4 burst mid-story, then
          more events (final BURST).
--check: rebuild and compare with the file on disk (sha), print counts.
"""
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from validate_reading import label_of  # noqa: E402

OUT = ROOT / "benchmarks" / "story_v1" / "stories.json"
EVENTS = ("IN", "OUT", "BREAK", "FIX", "NOOP")
CAP = 3
SEED = 1902
LEVEL_WORD = ["零", "一", "两", "三"]
SRC = {"rv2": ROOT / "benchmarks" / "reading_v2" / "test_v2.json",
       "rv1": ROOT / "benchmarks" / "reading_v1" / "test.json",
       "probe": ROOT / "benchmarks" / "comprehension_probe_v1" / "probe.json"}


def simulate(lv, sg, events):
    """(Lab 14 rule) -> final class name."""
    burst = False
    for e in events:
        if burst:
            continue
        if e == "IN":
            if lv < CAP:
                lv += 1
            elif sg == "broken":
                burst = True
        elif e == "OUT":
            lv = max(0, lv - 1)
        elif e == "BREAK":
            sg = "broken"
        elif e == "FIX":
            sg = "ok"
    return "BURST" if burst else f"L{lv}_{sg}"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_domains():
    doms = []
    for src in ("rv2", "rv1"):
        t = json.loads(SRC[src].read_bytes())
        for d in t["domains"]:
            pool = []
            for g in d["groups"]:
                for v, s in [("BASE", g["base"])] + sorted(g["variants"].items()):
                    pool.append(dict(text=s, label=label_of(g["event"], v), var=v))
            pool += [dict(text=s, label="NOOP", var="NOOPSENT") for s in d["noop"]]
            doms.append(dict(name=d["name"], src=src, measure=d["measure"], rules=d["rules"],
                             ok=d["safeguard"]["ok"], broken=d["safeguard"]["broken"], pool=pool,
                             support={e: d["support"][e] for e in EVENTS}))
    p = json.loads(SRC["probe"].read_bytes())
    for d in p["domains"]:
        pool, support = [], {}
        for e in EVENTS:
            sents = d["events"][e]
            support[e] = sents[:1]
            pool += [dict(text=s, label=e, var="NORMAL") for s in sents[1:]]
        pr = d["probe"]
        for k in ("DIR", "SYS", "MENTION"):
            for q in pr[k]:
                pool += [dict(text=q[x]["text"], label=q[x]["label"], var=k) for x in "ab"]
        for g in pr["REAL"]:
            pool += [dict(text=it["text"], label=it["label"], var="REAL-" + it["form"]) for it in g["items"]]
        for k in ("CUE", "STR"):
            pool += [dict(text=x["text"], label=x["label"], var=k) for x in pr[k]]
        doms.append(dict(name=d["name"], src="probe", measure=d["measure"], rules=d["init"]["rules"],
                         ok=d["init"]["safeguard"]["ok"], broken=d["init"]["safeguard"]["broken"], pool=pool,
                         support=support))
    names = [d["name"] for d in doms]
    assert len(doms) == 22 and len(set(names)) == 22, names
    return doms


def init_text(d, lv, sg):
    amount = f"是空的（零{d['measure']}）" if lv == 0 else f"有{LEVEL_WORD[lv]}{d['measure']}"
    return f"目前{d['name']}里{amount}，{d['ok'] if sg == 'ok' else d['broken']}"


def pick(rng, d, label, used):
    cand = [x for x in d["pool"] if x["label"] == label and x["text"] not in used]
    if not cand:
        cand = [x for x in d["pool"] if x["label"] == label]
    x = rng.choice(cand)
    used.add(x["text"])
    return x


def story(rng, d, kind, labels, lv, sg, sid):
    used = set()
    sents = [pick(rng, d, lab, used) for lab in labels]
    final = simulate(lv, sg, [s["label"] for s in sents])
    return dict(id=sid, domain=d["name"], src=d["src"], kind=kind, length=len(labels), init_level=lv,
                init_safeguard=sg, init_text=init_text(d, lv, sg),
                sentences=[dict(text=s["text"], label=s["label"], var=s["var"]) for s in sents], final=final)


def trap_labels(rng, kind, n):
    """Label sequences for the trap kinds; returns (labels, init level, init safeguard)."""
    while True:
        if kind == "T1":   # early BREAK, then enough IN to overflow at the end
            lv, sg = rng.choice([0, 1]), "ok"
            labs = ["NOOP"] * n
            labs[rng.choice([0, 1])] = "BREAK"
            need = CAP - lv + 1
            pos = sorted(rng.sample(range(2, n), need))
            for p in pos:
                labs[p] = "IN"
            labs[n - 1] = "IN"
            for i in range(2, n - 1):
                if labs[i] == "NOOP" and rng.random() < 0.3:
                    labs[i] = rng.choice(["NOOP", "OUT"])
            want = "BURST"
        elif kind == "T2":  # broken, FIX early, then fill past capacity
            lv, sg = rng.choice([1, 2]), "broken"
            labs = ["NOOP"] * n
            labs[rng.choice([0, 1])] = "FIX"
            for p in sorted(rng.sample(range(2, n), CAP - lv + 2)):
                labs[p] = "IN"
            want = "L3_ok"
        elif kind == "T3":  # mostly not-happened sentences
            lv, sg = rng.randint(0, 3), rng.choice(["ok", "broken"])
            labs = [rng.choice(["IN", "OUT", "BREAK", "FIX"]) if rng.random() < 0.35 else "NOOP" for _ in range(n)]
            want = None
        else:               # T4: burst mid-story, then more events
            lv, sg = CAP, "broken"
            k = rng.randint(2, n - 4)
            # level 3 and broken stay until position k (only NOOP / BREAK before it), IN at k bursts, then anything
            labs = ["NOOP"] * k + ["IN"] + [rng.choice(EVENTS) for _ in range(n - k - 1)]
            labs[rng.randint(0, k - 1)] = rng.choice(["NOOP", "BREAK"])
            want = "BURST"
        fin = simulate(lv, sg, labs)
        if want is None or fin == want:
            if kind == "T3" and labs.count("NOOP") < 0.6 * n:
                continue
            return labs, lv, sg


def build():
    rng = random.Random(SEED)
    doms = load_domains()
    stories = []
    for d in doms:
        for n in range(1, 13):
            for k in range(10):
                labs = rng.choices(EVENTS, weights=[.3, .2, .1, .1, .3], k=n)
                stories.append(story(rng, d, "normal", labs, rng.randint(0, 3), rng.choice(["ok", "broken"]),
                                     f"{d['name']}/N{n}/{k}"))
        for kind in ("T1", "T2", "T3", "T4"):
            for k in range(5):
                n = rng.randint(8, 12)
                labs, lv, sg = trap_labels(rng, kind, n)
                stories.append(story(rng, d, kind, labs, lv, sg, f"{d['name']}/{kind}/{k}"))
    meta = dict(version="story_v1", seed=SEED, rule="Lab 14 simulate (capacity 3)",
                sources={k: sha(v) for k, v in SRC.items()},
                domains=[dict(name=d["name"], src=d["src"], measure=d["measure"], rules=d["rules"], ok=d["ok"],
                              broken=d["broken"], support=d["support"]) for d in doms])
    return dict(meta, stories=stories)


def check(data):
    tpl = json.loads((ROOT / "benchmarks" / "state_tracking_v1" / "templates.json").read_bytes())
    test_words = [w for d in tpl["domains"] if d["split"] == "test" for w in (d["name"], d["safeguard_name"])]
    raw = json.dumps(data, ensure_ascii=False)
    bad = [w for w in test_words if w in raw]
    assert not bad, f"test-split strings in stories: {bad}"
    sup = {d["name"]: {t for lst in d["support"].values() for t in lst} for d in data["domains"]}
    for s in data["stories"]:
        assert simulate(s["init_level"], s["init_safeguard"], [x["label"] for x in s["sentences"]]) == s["final"]
        assert len(s["sentences"]) == s["length"]
        assert not {x["text"] for x in s["sentences"]} & sup[s["domain"]], s["id"]
    c = Counter(s["kind"] for s in data["stories"])
    assert c["normal"] == 22 * 12 * 10 and all(c[k] == 22 * 5 for k in ("T1", "T2", "T3", "T4")), c
    print("stories:", dict(c), "| finals:", dict(Counter(s["final"] for s in data["stories"])))


if __name__ == "__main__":
    data = build()
    check(data)
    raw = json.dumps(data, ensure_ascii=False, indent=1)
    if "--check" in sys.argv:
        ok = OUT.exists() and OUT.read_text(encoding="utf-8") == raw
        print("file on disk matches a rebuild:", ok, hashlib.sha256(OUT.read_bytes()).hexdigest() if OUT.exists() else "")
        sys.exit(0 if ok else 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(raw, encoding="utf-8")
    print("wrote", OUT, hashlib.sha256(OUT.read_bytes()).hexdigest())
