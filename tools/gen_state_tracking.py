"""Generate benchmarks/state_tracking_v1/stories.json (Lab 14 pilot data, issue #23).

Plain Python (no torch), fully deterministic (seed 20261031):
    python tools/gen_state_tracking.py

Task (pre-registered in #23). State: level 0..3 (capacity 3 units) x safeguard {ok, broken}, plus an
absorbing failure state BURST (9 classes). Events:
    IN     level+1; at level 3 the excess is diverted if the safeguard is ok (stays 3), BURST if broken
    OUT    level-1; nothing happens at 0
    BREAK  safeguard := broken          FIX  safeguard := ok          NOOP  no change
    any event after BURST: no change.
A story is one initial sentence (rules + initial level + safeguard status) and n event sentences; the
label is the final state.

Splits (domains from templates.json; K stories per (domain, final class, n)):
    train     16 train domains  n = 1..4   K = 6   -> 96 per (class, n), 3,456 stories
    val_seen  16 train domains  n = 1..10  K = 1   -> 16 per (class, n), 1,440 stories
    test       8 test domains   n = 1..10  K = 2   -> 16 per (class, n), 1,440 stories

Sampling, for every split and n:
  1. Final classes are exactly balanced (K * D per class) and spread evenly over the domains (K each).
  2. Initial states are balanced as well. P(final | init, n) under the event process below is computed
     exactly; Sinkhorn scaling turns it into a (class x init) table whose rows are the class quota and
     whose columns are equal (9 K D / 8 per init state), i.e. the allocation closest in KL to the
     natural one. It is rounded to integers with both margins kept, and each class's list of initial
     states is dealt round-robin over the domains.
  3. For each (domain, class, initial state): events i.i.d. with P(IN, OUT, BREAK, FIX, NOOP) =
     (.30, .25, .15, .15, .15), rejected until the final state is the class and, for BURST-final
     stories, BURST first occurs at event position >= max(1, n - 3) (so long stories are not mostly
     post-BURST filler). This is rejection sampling from the natural process, conditioned per cell.
  4. Each event text is a uniformly random paraphrase of that event from that domain.
  5. Stories are unique within a split on (domain, initial state, events, texts); val_seen also
     excludes every train story (same domains), so it is a held-out set at every length.
"""
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks" / "state_tracking_v1"
TEMPLATES = BENCH / "templates.json"
OUT = BENCH / "stories.json"

VERSION = "state_tracking_v1"
SEED = 20261031
CLASSES = ["L0_ok", "L1_ok", "L2_ok", "L3_ok", "L0_broken", "L1_broken", "L2_broken", "L3_broken", "BURST"]
INITS = CLASSES[:8]
EVENTS = ["IN", "OUT", "BREAK", "FIX", "NOOP"]
P_EVENT = [0.30, 0.25, 0.15, 0.15, 0.15]
# (split name, domain split, lengths, stories per (domain, class, n))
SPLITS = [("train", "train", range(1, 5), 6),
          ("val_seen", "train", range(1, 11), 1),
          ("test", "test", range(1, 11), 2)]
MAX_TRIES = 500_000


def step(state, event):
    if state == "BURST":
        return state
    level, sg = int(state[1]), state.split("_")[1]
    if event == "IN":
        if level < 3:
            level += 1
        elif sg == "broken":
            return "BURST"
    elif event == "OUT":
        level = max(0, level - 1)
    elif event == "BREAK":
        sg = "broken"
    elif event == "FIX":
        sg = "ok"
    return f"L{level}_{sg}"


def run(init, events):
    states = [init]
    for e in events:
        states.append(step(states[-1], e))
    return states


def burst_floor(n):
    """Earliest allowed event position (1-based) of the first BURST in a story of length n."""
    return max(1, n - 3)


def admissible(states, n):
    return states[-1] != "BURST" or states.index("BURST") >= burst_floor(n)


def final_dist(init, n):
    """Exact P(final state, story admissible | init, n) under the i.i.d. event process."""
    dist = {init: 1.0}
    for t in range(1, n + 1):
        nxt = defaultdict(float)
        for s, p in dist.items():
            for e, pe in zip(EVENTS, P_EVENT):
                s2 = step(s, e)
                if s != "BURST" and s2 == "BURST" and t < burst_floor(n):
                    continue
                nxt[s2] += p * pe
        dist = nxt
    return dist


def init_quota(n, per_class):
    """Integer (class x init) table: rows sum to per_class, columns to 9 * per_class / 8."""
    col_total = per_class * len(CLASSES) / len(INITS)
    assert col_total == int(col_total), "column target must be an integer"
    col_total = int(col_total)
    dists = [final_dist(i, n) for i in INITS]
    w = [[dists[j].get(c, 0.0) for j in range(len(INITS))] for c in CLASSES]
    for _ in range(20000):  # Sinkhorn
        for r in w:
            s = sum(r)
            r[:] = [x * per_class / s for x in r]
        worst = 0.0
        for j in range(len(INITS)):
            s = sum(w[c][j] for c in range(len(CLASSES)))
            worst = max(worst, abs(s - col_total))
            for c in range(len(CLASSES)):
                w[c][j] *= col_total / s
        if worst < 1e-10:
            break
    q = [[int(x) for x in r] for r in w]
    row_need = [per_class - sum(r) for r in q]
    col_need = [col_total - sum(q[c][j] for c in range(len(CLASSES))) for j in range(len(INITS))]
    while any(row_need):
        best = None
        for relax in (False, True):
            for c in range(len(CLASSES)):
                if row_need[c] <= 0:
                    continue
                for j in range(len(INITS)):
                    if w[c][j] <= 0 or (not relax and col_need[j] <= 0):
                        continue
                    gap = w[c][j] - q[c][j]
                    if best is None or gap > best[0]:
                        best = (gap, c, j)
            if best is not None:
                break
        _, c, j = best
        q[c][j] += 1
        row_need[c] -= 1
        col_need[j] -= 1
    return q


def init_text(dom, level, sg):
    ini = dom["init"]
    return f"{ini['rules']}；{ini['level'][level]}，{ini['safeguard'][sg]}。"


def sample_story(rng, dom, cls, init, n, taken):
    paras = dom["events"]
    for _ in range(MAX_TRIES):
        events = rng.choices(EVENTS, weights=P_EVENT, k=n)
        states = run(init, events)
        if states[-1] != cls or not admissible(states, n):
            continue
        texts = [rng.choice(paras[e]) for e in events]
        key = (dom["name"], init, tuple(events), tuple(texts))
        if key in taken:
            continue
        taken.add(key)
        level, sg = int(init[1]), init.split("_")[1]
        return {"domain": dom["name"], "n": n, "init_level": level, "init_safeguard": sg,
                "init_text": init_text(dom, level, sg), "events": events, "event_texts": texts,
                "states": states, "final": states[-1]}
    raise RuntimeError(f"no unique story for {dom['name']} {cls} init={init} n={n}")


def story_key(s):
    return (s["domain"], f"L{s['init_level']}_{s['init_safeguard']}", tuple(s["events"]), tuple(s["event_texts"]))


def generate(templates):
    rng = random.Random(SEED)
    doms = {sp: [d for d in templates["domains"] if d["split"] == sp] for sp in ("train", "test")}
    assert len(doms["train"]) == 16 and len(doms["test"]) == 8, "need 16 train + 8 test domains"
    splits, taken_by_domain_split = {}, {"train": set(), "test": set()}
    for name, dsplit, lengths, k in SPLITS:
        dl = doms[dsplit]
        taken = taken_by_domain_split[dsplit]  # val_seen inherits train's keys: no story repeats
        stories = []
        for n in lengths:
            q = init_quota(n, k * len(dl))
            for ci, cls in enumerate(CLASSES):
                inits = [INITS[j] for j in range(len(INITS)) for _ in range(q[ci][j])]
                off = (n * len(CLASSES) + ci) % len(dl)
                per_dom = defaultdict(list)
                for idx, init in enumerate(inits):
                    per_dom[(idx + off) % len(dl)].append(init)
                for di in range(len(dl)):
                    assert len(per_dom[di]) == k
                    for init in per_dom[di]:
                        s = sample_story(rng, dl[di], cls, init, n, taken)
                        stories.append({"id": f"{name}-{len(stories):05d}", **s})
        splits[name] = stories
    return splits


def dump(path, header, splits):
    lines = ["{" + ", ".join(f"{json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}" for k, v in header.items())
             + ",", ' "splits": {']
    names = list(splits)
    for si, name in enumerate(names):
        lines.append(f"  {json.dumps(name)}: [")
        rows = splits[name]
        for i, s in enumerate(rows):
            lines.append("   " + json.dumps(s, ensure_ascii=False) + ("," if i < len(rows) - 1 else ""))
        lines.append("  ]" + ("," if si < len(names) - 1 else ""))
    lines.append(" }")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    raw = TEMPLATES.read_bytes()
    templates = json.loads(raw)
    splits = generate(templates)
    header = {"version": VERSION, "templates_sha256": hashlib.sha256(raw).hexdigest(), "seed": SEED,
              "classes": CLASSES}
    dump(OUT, header, splits)
    json.loads(OUT.read_text(encoding="utf-8"))  # sanity: valid JSON
    for name, rows in splits.items():
        inits = defaultdict(int)
        for s in rows:
            inits[f"L{s['init_level']}_{s['init_safeguard']}"] += 1
        print(f"{name:9s} {len(rows):5d} stories; init states: "
              + " ".join(f"{i}={inits[i]}" for i in INITS))
    print(f"wrote {OUT.relative_to(ROOT)}  sha256={hashlib.sha256(OUT.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    sys.exit(main())
