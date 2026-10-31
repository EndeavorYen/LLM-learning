"""Acceptance check for benchmarks/state_tracking_v1 (Lab 14 pilot data, issue #23).

Plain Python, no torch, independent of the generator (own transition function):
    python tools/validate_state_tracking.py [templates.json] [stories.json]

Errors (exit code 1):
  T1  template schema: version, capacity 3, the 9 classes and 5 events in order; 24 domains with unique
      names, 16 train + 8 test; init.rules, 4 level phrases, safeguard ok/broken; >= 5 distinct
      paraphrases per event; the 9 state_desc strings, distinct.
  T2  template form: init parts carry no sentence punctuation, so rules + "；" + level + "，" + safeguard
      + "。" is one sentence; rules state the capacity ("三" + measure); level l states its number
      (零/一/两/三, with the measure for l >= 1) and no other number; every event text is one sentence
      ending in "。" with no digit, no number word other than 一, and no 满/空/半, so no event states
      a resulting level; IN/OUT contain "一" + measure exactly once, BREAK/FIX/NOOP never; BREAK/FIX
      name the safeguard, IN/OUT/NOOP never do.
  T3  no template string (rules, level, safeguard, event, state_desc) occurs in two domains or twice in
      one domain.
  S1  stories header: version, seed, classes, templates_sha256 == sha256(templates.json).
  S2  every story: domain of the right split, n in range, init_text composed from the templates, each
      event text a paraphrase of that event in that domain, states re-simulated from init + events,
      final == states[-1], first BURST at position >= max(1, n - 3); ids unique.
  S3  balance: exactly K stories per (split, n, final class, domain); split sizes 3456 / 1440 / 1440.
  S4  uniqueness: no repeated (domain, init, events, texts) within a split, none shared by train and
      val_seen; no test-domain string inside any train/val_seen story and vice versa.
Reported (warnings only):
  init-state balance per split (warn if max/min > 1.15 in train);
  B1  order-blind lookup fitted on train (as pre-registered): (init, counts of all 5 event types) ->
      majority final; fallback (init, #IN, #OUT); fallback global majority. Accuracy per n and SHORT
      (1-4) / LONG (6-10) on val_seen and test; WARNING if test LONG > 60%.
  B2  exact Bayes-optimal order-blind classifier: P(final | init, event counts) computed by dynamic
      programming under the generating process, weighted by the split's realised (class, init) mix.
      Its expected and realised LONG accuracy are the ceiling for any BAG model (V2 bar: <= 70%).
  B3  last-event lookup fitted on train (last event type -> majority final).
"""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks" / "state_tracking_v1"
VERSION = "state_tracking_v1"
SEED = 20261031
CLASSES = ["L0_ok", "L1_ok", "L2_ok", "L3_ok", "L0_broken", "L1_broken", "L2_broken", "L3_broken", "BURST"]
EVENTS = ["IN", "OUT", "BREAK", "FIX", "NOOP"]
P_EVENT = dict(zip(EVENTS, [0.30, 0.25, 0.15, 0.15, 0.15]))
SPLITS = {"train": ("train", range(1, 5), 6), "val_seen": ("train", range(1, 11), 1),
          "test": ("test", range(1, 11), 2)}
N_TRAIN_DOM, N_TEST_DOM, MIN_PARA = 16, 8, 5
NUM = ["零", "一", "两", "三"]
NUMERALS = set("零一二两三四五六七八九十百千")
EVENT_FORBIDDEN = set("零二两三四五六七八九十百千半满空0123456789０１２３４５６７８９")
PART_PUNCT = set("。！？；，：、")
SHORT, LONG = range(1, 5), range(6, 11)
B1_WARN, V2_BAR = 0.60, 0.70


# ---------------------------------------------------------------- task semantics (independent copy)
def step(state, event):
    if state == "BURST":
        return "BURST"
    level, sg = int(state[1]), state[3:]
    if event == "IN":
        if level == 3:
            return "BURST" if sg == "broken" else state
        level += 1
    elif event == "OUT":
        level = level - 1 if level > 0 else 0
    elif event == "BREAK":
        sg = "broken"
    elif event == "FIX":
        sg = "ok"
    elif event != "NOOP":
        raise ValueError(event)
    return f"L{level}_{sg}"


def burst_floor(n):
    return max(1, n - 3)


# ---------------------------------------------------------------- templates
def check_templates(t):
    errs = []
    if t.get("version") != VERSION:
        errs.append(f"T1 version {t.get('version')!r}")
    if t.get("capacity") != 3:
        errs.append("T1 capacity must be 3")
    if t.get("classes") != CLASSES:
        errs.append("T1 classes differ from the pre-registered 9")
    if t.get("events") != EVENTS:
        errs.append("T1 events differ from IN/OUT/BREAK/FIX/NOOP")
    doms = t.get("domains", [])
    names = [d.get("name") for d in doms]
    if len(doms) != N_TRAIN_DOM + N_TEST_DOM or len(set(names)) != len(names):
        errs.append(f"T1 need {N_TRAIN_DOM + N_TEST_DOM} uniquely named domains, got {len(doms)}")
    nsplit = Counter(d.get("split") for d in doms)
    if nsplit != Counter({"train": N_TRAIN_DOM, "test": N_TEST_DOM}):
        errs.append(f"T1 split counts {dict(nsplit)}")
    owner = defaultdict(list)  # string -> [(domain, field)]
    for d in doms:
        nm, ms, sgn = d.get("name"), d.get("measure", ""), d.get("safeguard_name", "")
        if not ms or not sgn:
            errs.append(f"T1 {nm}: measure / safeguard_name missing")
            continue
        ini = d.get("init", {})
        rules, level, sg = ini.get("rules"), ini.get("level"), ini.get("safeguard", {})
        if not isinstance(rules, str) or not isinstance(level, list) or len(level) != 4 \
                or sorted(sg) != ["broken", "ok"]:
            errs.append(f"T1 {nm}: init needs rules, 4 level phrases, safeguard ok/broken")
            continue
        parts = [("rules", rules)] + [(f"level[{i}]", x) for i, x in enumerate(level)] \
            + [(f"safeguard.{k}", sg[k]) for k in ("ok", "broken")]
        for fld, txt in parts:
            owner[txt].append((nm, fld))
            if not txt or "。" in txt or txt[-1] in PART_PUNCT or txt[0] in PART_PUNCT:
                errs.append(f"T2 {nm} {fld}: init parts must carry no sentence punctuation ({txt})")
        if "三" + ms not in rules:
            errs.append(f"T2 {nm} rules: capacity '三{ms}' not stated")
        for lv, txt in enumerate(level):
            want = NUM[lv] + (ms if lv else "")
            others = (NUMERALS - {NUM[lv]}) & set(txt)
            if want not in txt or others:
                errs.append(f"T2 {nm} level[{lv}]: must state '{want}' and no other number ({txt})")
        for k in ("ok", "broken"):
            if NUMERALS & set(sg[k]):
                errs.append(f"T2 {nm} safeguard.{k}: contains a number ({sg[k]})")
        ev = d.get("events", {})
        if sorted(ev) != sorted(EVENTS):
            errs.append(f"T1 {nm}: events keys {sorted(ev)}")
            continue
        unit = "一" + ms
        for e in EVENTS:
            lst = ev[e]
            if len(lst) < MIN_PARA or len(set(lst)) != len(lst):
                errs.append(f"T1 {nm} {e}: need >= {MIN_PARA} distinct paraphrases, got {len(lst)}")
            for txt in lst:
                owner[txt].append((nm, e))
                if not txt.endswith("。") or txt.count("。") != 1 or set("！？；") & set(txt):
                    errs.append(f"T2 {nm} {e}: one sentence ending in '。' ({txt})")
                bad = EVENT_FORBIDDEN & set(txt)
                if bad:
                    errs.append(f"T2 {nm} {e}: forbidden characters {''.join(sorted(bad))} ({txt})")
                if e in ("IN", "OUT"):
                    if txt.count(unit) != 1:
                        errs.append(f"T2 {nm} {e}: must contain '{unit}' exactly once ({txt})")
                elif unit in txt:
                    errs.append(f"T2 {nm} {e}: must not contain '{unit}' ({txt})")
                if (e in ("BREAK", "FIX")) != (sgn in txt):
                    errs.append(f"T2 {nm} {e}: safeguard '{sgn}' must appear in BREAK/FIX only ({txt})")
        sd = d.get("state_desc", {})
        if list(sd) != CLASSES or len(set(sd.values())) != len(CLASSES):
            errs.append(f"T1 {nm}: state_desc needs 9 distinct strings keyed by the classes in order")
        else:
            for c in CLASSES:
                owner[sd[c]].append((nm, f"state_desc.{c}"))
                if c != "BURST" and NUM[int(c[1])] not in sd[c]:
                    errs.append(f"T2 {nm} state_desc.{c}: level number missing ({sd[c]})")
    for txt, where in owner.items():
        if len(where) > 1:
            errs.append(f"T3 string used {len(where)} times {where}: {txt}")
    return errs


def compose_init(dom, level, sg):
    ini = dom["init"]
    return f"{ini['rules']}；{ini['level'][level]}，{ini['safeguard'][sg]}。"


def domain_strings(dom):
    ini = dom["init"]
    out = [ini["rules"], *ini["level"], ini["safeguard"]["ok"], ini["safeguard"]["broken"]]
    for e in EVENTS:
        out += dom["events"][e]
    return out


# ---------------------------------------------------------------- stories
def story_key(s):
    return (s["domain"], s["init_level"], s["init_safeguard"], tuple(s["events"]), tuple(s["event_texts"]))


def check_stories(t, st, tpl_sha):
    errs = []
    if st.get("version") != VERSION or st.get("seed") != SEED or st.get("classes") != CLASSES:
        errs.append("S1 header version / seed / classes")
    if st.get("templates_sha256") != tpl_sha:
        errs.append(f"S1 templates_sha256 {st.get('templates_sha256')} != {tpl_sha}")
    splits = st.get("splits", {})
    if sorted(splits) != sorted(SPLITS):
        errs.append(f"S1 splits {sorted(splits)}")
        return errs
    doms = {d["name"]: d for d in t["domains"]}
    ids = set()
    keys = {}
    for name, (dsplit, lengths, k) in SPLITS.items():
        rows = splits[name]
        dnames = [d["name"] for d in t["domains"] if d["split"] == dsplit]
        cells = Counter()
        keys[name] = Counter(story_key(s) for s in rows)
        for s in rows:
            sid = s.get("id")
            if sid in ids:
                errs.append(f"S2 duplicate id {sid}")
            ids.add(sid)
            dom = doms.get(s.get("domain"))
            if dom is None or dom["split"] != dsplit:
                errs.append(f"S2 {sid}: domain {s.get('domain')} not a {dsplit} domain")
                continue
            n, ev, tx, states = s["n"], s["events"], s["event_texts"], s["states"]
            if n not in lengths or len(ev) != n or len(tx) != n or len(states) != n + 1:
                errs.append(f"S2 {sid}: lengths n={n} events={len(ev)} texts={len(tx)} states={len(states)}")
                continue
            lv, sg = s["init_level"], s["init_safeguard"]
            if lv not in range(4) or sg not in ("ok", "broken"):
                errs.append(f"S2 {sid}: init {lv} {sg}")
                continue
            if s["init_text"] != compose_init(dom, lv, sg):
                errs.append(f"S2 {sid}: init_text is not the template composition")
            sim = [f"L{lv}_{sg}"]
            for e, x in zip(ev, tx):
                if e not in EVENTS or x not in dom["events"][e]:
                    errs.append(f"S2 {sid}: text '{x}' is not a {e} paraphrase of {dom['name']}")
                sim.append(step(sim[-1], e))
            if sim != states or s["final"] != states[-1]:
                errs.append(f"S2 {sid}: states/final do not match the rule ({states} vs {sim})")
            if states[-1] == "BURST" and states.index("BURST") < burst_floor(n):
                errs.append(f"S2 {sid}: BURST at position {states.index('BURST')} < {burst_floor(n)}")
            cells[(n, s["final"], s["domain"])] += 1
        want = {(n, c, dn): k for n in lengths for c in CLASSES for dn in dnames}
        if cells != Counter(want):
            bad = [(key, cells.get(key, 0)) for key in sorted(set(want) | set(cells)) if cells.get(key, 0) != want.get(key)]
            errs.append(f"S3 {name}: {len(bad)} (n, class, domain) cells off quota, e.g. {bad[:3]}")
        if len(rows) != k * len(dnames) * len(CLASSES) * len(lengths):
            errs.append(f"S3 {name}: {len(rows)} stories")
        dup = sum(v - 1 for v in keys[name].values() if v > 1)
        if dup:
            errs.append(f"S4 {name}: {dup} repeated stories")
    shared = set(keys["train"]) & set(keys["val_seen"])
    if shared:
        errs.append(f"S4 {len(shared)} stories shared by train and val_seen")
    # leak scan: no string of the other domain split inside any story text
    test_str = [x for d in t["domains"] if d["split"] == "test" for x in domain_strings(d)]
    train_str = [x for d in t["domains"] if d["split"] == "train" for x in domain_strings(d)]
    for name, foreign in (("train", test_str), ("val_seen", test_str), ("test", train_str)):
        hits = 0
        for s in splits[name]:
            blob = s["init_text"] + "\n" + "\n".join(s["event_texts"])
            hits += sum(1 for x in foreign if x in blob)
        if hits:
            errs.append(f"S4 {name}: {hits} occurrences of strings from the other domain split")
    return errs


# ---------------------------------------------------------------- baselines
def init_of(s):
    return f"L{s['init_level']}_{s['init_safeguard']}"


def counts(s):
    c = Counter(s["events"])
    return tuple(c[e] for e in EVENTS)


def majority(counter):
    best = max(counter.values())
    return next(c for c in CLASSES if counter.get(c, 0) == best)


def fit_order_blind(train):
    k1, k2, glob = defaultdict(Counter), defaultdict(Counter), Counter()
    for s in train:
        c = counts(s)
        k1[(init_of(s), c)][s["final"]] += 1
        k2[(init_of(s), c[0], c[1])][s["final"]] += 1
        glob[s["final"]] += 1
    t1 = {k: majority(v) for k, v in k1.items()}
    t2 = {k: majority(v) for k, v in k2.items()}
    g = majority(glob)

    def predict(s):
        c = counts(s)
        if (init_of(s), c) in t1:
            return t1[(init_of(s), c)], 1
        if (init_of(s), c[0], c[1]) in t2:
            return t2[(init_of(s), c[0], c[1])], 2
        return g, 3
    return predict


def fit_last(train):
    tab, glob = defaultdict(Counter), Counter()
    for s in train:
        tab[s["events"][-1]][s["final"]] += 1
        glob[s["final"]] += 1
    t = {k: majority(v) for k, v in tab.items()}
    g = majority(glob)
    return lambda s: (t.get(s["events"][-1], g), 1)


def count_dp(init, n):
    """P(final, event counts, admissible | init, n) under the generating process."""
    dist = {(init, (0,) * 5): 1.0}
    for tpos in range(1, n + 1):
        nxt = defaultdict(float)
        for (st, cnt), p in dist.items():
            for i, e in enumerate(EVENTS):
                s2 = step(st, e)
                if st != "BURST" and s2 == "BURST" and tpos < burst_floor(n):
                    continue
                c2 = cnt[:i] + (cnt[i] + 1,) + cnt[i + 1:]
                nxt[(s2, c2)] += p * P_EVENT[e]
        dist = nxt
    return dist


def bayes_order_blind(rows):
    """Expected and realised accuracy per n of argmax_c P(c | init, counts), with the (class, init)
    weights of `rows` and P(counts | class, init, n) from the generating process."""
    by_n = defaultdict(list)
    for s in rows:
        by_n[s["n"]].append(s)
    exp, real = {}, {}
    for n, ss in sorted(by_n.items()):
        mix = Counter((s["final"], init_of(s)) for s in ss)
        post = defaultdict(lambda: defaultdict(float))  # (init, counts) -> class -> weight
        for init in CLASSES[:8]:
            dist = count_dp(init, n)
            z = Counter()
            for (st, _), p in dist.items():
                z[st] += p
            for (st, cnt), p in dist.items():
                w = mix.get((st, init), 0)
                if w:
                    post[(init, cnt)][st] += w / len(ss) * p / z[st]
        exp[n] = sum(max(v.values()) for v in post.values())
        rule = {k: max(CLASSES, key=lambda c: (v.get(c, 0.0), -CLASSES.index(c))) for k, v in post.items()}
        real[n] = sum(rule[(init_of(s), counts(s))] == s["final"] for s in ss) / len(ss)
    return exp, real


def acc_by_n(predict, rows):
    hit, tot, src = Counter(), Counter(), Counter()
    for s in rows:
        p, level = predict(s)
        tot[s["n"]] += 1
        hit[s["n"]] += p == s["final"]
        src[(s["n"] >= 6, level)] += 1
    return {n: hit[n] / tot[n] for n in sorted(tot)}, src


def mean_over(d, ns):
    vals = [d[n] for n in ns if n in d]
    return sum(vals) / len(vals) if vals else float("nan")


def fmt_row(label, d):
    cells = " ".join(f"{d[n] * 100:5.1f}" for n in sorted(d))
    return f"  {label:28s} {cells} | SHORT {mean_over(d, SHORT) * 100:5.1f}  LONG {mean_over(d, LONG) * 100:5.1f}"


def report(t, st):
    warns = []
    splits = st["splits"]
    print("\nInit-state balance (stories per initial state):")
    for name, rows in splits.items():
        c = Counter(init_of(s) for s in rows)
        vals = [c[i] for i in CLASSES[:8]]
        ratio = max(vals) / min(vals)
        print(f"  {name:9s} " + " ".join(f"{i}={c[i]}" for i in CLASSES[:8]) + f"   max/min {ratio:.3f}")
        if name == "train" and ratio > 1.15:
            warns.append(f"train init states unbalanced (max/min {ratio:.2f})")
    print("\nEvent mix and BURST timing:")
    for name, rows in splits.items():
        ev = Counter(e for s in rows for e in s["events"])
        tot = sum(ev.values())
        post = sum(len(s["events"]) - s["states"].index("BURST") for s in rows if s["final"] == "BURST")
        print(f"  {name:9s} " + " ".join(f"{e}={ev[e] / tot:.3f}" for e in EVENTS)
              + f"   post-BURST events {post / tot:.3f} of all")
    ns = sorted({s["n"] for s in splits["test"]})
    head = " ".join(f"{'n' + str(n):>5s}" for n in ns)
    print(f"\nBaselines, accuracy % per n (chance 11.1%)\n  {'':28s} {head}")
    ob = fit_order_blind(splits["train"])
    last = fit_last(splits["train"])
    out = {}
    for name in ("val_seen", "test"):
        a, src = acc_by_n(ob, splits[name])
        out[("B1", name)] = a
        print(fmt_row(f"B1 order-blind lookup {name}", a))
        long_n = sum(v for (lg, _), v in src.items() if lg)
        print(f"  {'':28s} LONG items answered by (init, 5 counts) {src[(True, 1)] / long_n:.1%}, "
              f"(init, #IN, #OUT) {src[(True, 2)] / long_n:.1%}, global majority {src[(True, 3)] / long_n:.1%}")
    for name in ("val_seen", "test"):
        exp, real = bayes_order_blind(splits[name])
        out[("B2", name)] = (exp, real)
        print(fmt_row(f"B2 Bayes order-blind {name}", exp) + "  (expected)")
        print(fmt_row(f"B2 Bayes order-blind {name}", real) + "  (realised)")
    for name in ("val_seen", "test"):
        a, _ = acc_by_n(last, splits[name])
        print(fmt_row(f"B3 last-event lookup {name}", a))
    b1 = mean_over(out[("B1", "test")], LONG)
    if b1 > B1_WARN:
        warns.append(f"B1 order-blind lookup on test LONG = {b1:.1%} > {B1_WARN:.0%}")
    exp, real = out[("B2", "test")]
    if max(mean_over(exp, LONG), mean_over(real, LONG)) > V2_BAR:
        warns.append(f"B2 Bayes order-blind ceiling on test LONG exceeds the V2 bar {V2_BAR:.0%}")
    return warns


def main(argv):
    tpl_path = Path(argv[1]) if len(argv) > 1 else BENCH / "templates.json"
    st_path = Path(argv[2]) if len(argv) > 2 else BENCH / "stories.json"
    raw = tpl_path.read_bytes()
    t = json.loads(raw)
    errs = check_templates(t)
    doms = t.get("domains", [])
    print(f"templates: {tpl_path}  sha256={hashlib.sha256(raw).hexdigest()}")
    for sp in ("train", "test"):
        print(f"  {sp:5s} domains: " + "、".join(d["name"] for d in doms if d.get("split") == sp))
    npara = Counter()
    for d in doms:
        for e in EVENTS:
            npara[e] = min(npara.get(e, 99), len(d.get("events", {}).get(e, [])))
    lens = [len(x) for d in doms for e in EVENTS for x in d.get("events", {}).get(e, [])]
    if lens:
        print("  min paraphrases per domain: " + " ".join(f"{e}={npara[e]}" for e in EVENTS)
              + f"; event text length {min(lens)}-{max(lens)} chars")
    if not errs:
        st_raw = st_path.read_bytes()
        st = json.loads(st_raw)
        print(f"stories:   {st_path}  sha256={hashlib.sha256(st_raw).hexdigest()}")
        print("  " + "  ".join(f"{k}={len(v)}" for k, v in st.get("splits", {}).items()))
        errs = check_stories(t, st, hashlib.sha256(raw).hexdigest())
        warns = [] if errs else report(t, st)
    else:
        warns = []
    for w in warns:
        print(f"WARNING: {w}")
    if errs:
        print(f"\nFAIL: {len(errs)} error(s)")
        for e in errs[:60]:
            print("  " + e)
        return 1
    print("\nPASS: T1-T3 templates, S1-S4 stories")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
