"""Build benchmarks/story_v2/stories.json for Lab 20 (issue #25, registration "Lab 20（第二步 2b-1）「規則變活：容量＋數量」").

    python tools/make_stories_v2.py [--check]

Same 22 never-trained domains and amount-1 sentence pools as story_v1 (tools/make_stories_v1.load_domains; support
sentences excluded), plus benchmarks/amount_v1/test.json: BASE / NUMFORM / CHATTER_NUM -> (event, k), DIRSWAP ->
(opposite event, k), NEG -> NOOP. All story_v1 IN / OUT sentences count as amount 1.
Rule (capacity cap 2..5, events (type, k)):
  IN k:  level + k <= cap -> level + k; else safeguard ok -> level = cap; broken -> BURST (absorbing).
  OUT k: level = max(0, level - k).   BREAK / FIX / NOOP as Lab 14.
  normal  per domain x cap 2..5 x length 1..12: 3 stories; event P(IN .3, OUT .2, BREAK .1, FIX .1, NOOP .3),
          IN / OUT amount P(1 .5, 2 .3, 3 .2); initial level 0..cap, safeguard ok / broken.
  trap    per domain 20 stories, length 8..12, cap 2..5 random, 5 of each (rejection sampling on the rule trace):
          A1 a multi-unit IN that would overflow while the safeguard is ok (blocked, level = cap), final not BURST;
          A2 final BURST caused by a multi-unit IN that a single unit would not have burst;
          A3 a multi-unit OUT larger than a non-zero level (-> 0), final not BURST;
          A4 BREAK in the first two sentences, BURST caused by a multi-unit IN in the last three sentences.
--check: rebuild and compare with the file on disk.
"""
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from make_stories_v1 import load_domains  # noqa: E402

OUT = ROOT / "benchmarks" / "story_v2" / "stories.json"
AMOUNT = ROOT / "benchmarks" / "amount_v1" / "test.json"
EVENTS = ("IN", "OUT", "BREAK", "FIX", "NOOP")
OPP = {"IN": "OUT", "OUT": "IN"}
CAPS = (2, 3, 4, 5)
SEED = 2002
NUM = ["零", "一", "两", "三", "四", "五"]


def trace(cap, lv, sg, events):
    """events = [(type, k)] -> list of per-step info dicts and the final class name."""
    burst, info = False, []
    for e, k in events:
        st = dict(lv=lv, sg=sg, e=e, k=k, burst_now=False, blocked=False, over_out=False)
        if not burst:
            if e == "IN":
                if lv + k <= cap:
                    lv += k
                elif sg == "ok":
                    st["blocked"] = True
                    lv = cap
                else:
                    burst = st["burst_now"] = True
            elif e == "OUT":
                st["over_out"] = k > lv > 0
                lv = max(0, lv - k)
            elif e == "BREAK":
                sg = "broken"
            elif e == "FIX":
                sg = "ok"
        info.append(st)
    return info, ("BURST" if burst else f"L{lv}_{sg}")


def simulate(cap, lv, sg, events):
    return trace(cap, lv, sg, events)[1]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_all():
    doms = load_domains()
    am = json.loads(AMOUNT.read_bytes())
    assert am["version"] == "amount_v1_test"
    by = {d["name"]: d for d in am["domains"]}
    assert sorted(by) == sorted(d["name"] for d in doms)
    for d in doms:
        for x in d["pool"]:
            x["amount"] = 1 if x["label"] in ("IN", "OUT") else 0
        for g in by[d["name"]]["groups"]:
            e, k = g["event"], g["amount"]
            for f, t in sorted(g["forms"].items()):
                lab, amt = (OPP[e], k) if f == "DIRSWAP" else (("NOOP", 0) if f == "NEG" else (e, k))
                d["pool"].append(dict(text=t, label=lab, var="AMT-" + f, amount=amt))
    return doms, am


def init_text(d, cap, lv, sg):
    amount = f"是空的（零{d['measure']}）" if lv == 0 else f"有{NUM[lv]}{d['measure']}"
    return f"这个{d['name']}最多能装{NUM[cap]}{d['measure']}。目前{d['name']}里{amount}，{d['ok'] if sg == 'ok' else d['broken']}"


def pick(rng, d, label, amount, used):
    cand = [x for x in d["pool"] if x["label"] == label and x["amount"] == amount and x["text"] not in used]
    if not cand:
        cand = [x for x in d["pool"] if x["label"] == label and x["amount"] == amount]
    x = rng.choice(cand)
    used.add(x["text"])
    return x


def rand_events(rng, n):
    out = []
    for e in rng.choices(EVENTS, weights=[.3, .2, .1, .1, .3], k=n):
        out.append((e, rng.choices((1, 2, 3), weights=[.5, .3, .2])[0] if e in ("IN", "OUT") else 0))
    return out


def trap_ok(kind, info, final, n):
    if kind == "A1":
        return final != "BURST" and any(s["blocked"] and s["k"] >= 2 for s in info)
    if kind == "A2":
        return final == "BURST" and any(s["burst_now"] and s["k"] >= 2 and s["lv"] + 1 <= s.get("cap", 99) for s in info)
    if kind == "A3":
        return final != "BURST" and any(s["over_out"] and s["k"] >= 2 for s in info)
    return (final == "BURST" and any(s["e"] == "BREAK" for s in info[:2])
            and any(s["burst_now"] and s["k"] >= 2 for s in info[n - 3:]))


def make_story(rng, d, kind, cap, lv, sg, evs, sid):
    used = set()
    sents = [pick(rng, d, e, k, used) for e, k in evs]
    final = simulate(cap, lv, sg, [(s["label"], s["amount"]) for s in sents])
    return dict(id=sid, domain=d["name"], src=d["src"], kind=kind, length=len(evs), capacity=cap, init_level=lv,
                init_safeguard=sg, init_text=init_text(d, cap, lv, sg),
                sentences=[dict(text=s["text"], label=s["label"], amount=s["amount"], var=s["var"]) for s in sents],
                final=final)


def build():
    rng = random.Random(SEED)
    doms, _am = load_all()
    stories = []
    for d in doms:
        for cap in CAPS:
            for n in range(1, 13):
                for k in range(3):
                    stories.append(make_story(rng, d, "normal", cap, rng.randint(0, cap), rng.choice(["ok", "broken"]),
                                              rand_events(rng, n), f"{d['name']}/C{cap}/N{n}/{k}"))
        for kind in ("A1", "A2", "A3", "A4"):
            for k in range(5):
                n, cap = rng.randint(8, 12), rng.choice(CAPS)
                while True:
                    lv, sg = rng.randint(0, cap), rng.choice(["ok", "broken"])
                    evs = rand_events(rng, n)
                    if kind == "A4":
                        sg = "ok"
                        evs[rng.choice([0, 1])] = ("BREAK", 0)
                        evs = [x if x[0] != "FIX" else ("NOOP", 0) for x in evs]
                    info, final = trace(cap, lv, sg, evs)
                    for s in info:
                        s["cap"] = cap
                    if trap_ok(kind, info, final, n):
                        break
                stories.append(make_story(rng, d, kind, cap, lv, sg, evs, f"{d['name']}/{kind}/{k}"))
    meta = dict(version="story_v2", seed=SEED, rule="capacity 2..5, IN/OUT amounts 1..3 (Lab 20)",
                sources=dict(amount_v1_test=sha(AMOUNT), story_v1_sources="tools/make_stories_v1.SRC"),
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
        assert simulate(s["capacity"], s["init_level"], s["init_safeguard"],
                        [(x["label"], x["amount"]) for x in s["sentences"]]) == s["final"]
        assert len(s["sentences"]) == s["length"] and 0 <= s["init_level"] <= s["capacity"]
        assert not {x["text"] for x in s["sentences"]} & sup[s["domain"]], s["id"]
    c = Counter(s["kind"] for s in data["stories"])
    assert c["normal"] == 22 * 4 * 12 * 3 and all(c[k] == 22 * 5 for k in ("A1", "A2", "A3", "A4")), c
    amt = Counter(x["amount"] for s in data["stories"] for x in s["sentences"] if x["label"] in ("IN", "OUT"))
    print("stories:", dict(c), "| IN/OUT amounts:", dict(amt), "| finals:",
          dict(sorted(Counter(s["final"] for s in data["stories"]).items())))


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
