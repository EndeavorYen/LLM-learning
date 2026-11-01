"""Aggregate Lab 17b (comprehension probe, issue #25) logs.

    python tools/aggregate_lab17b.py lab17b_A.log lab17b_B.log [--partial]

Reads the L17B_RES lines with seed=all (sums over seeds, and over draws for C / ICL), L17B_ITEM lines and L17B_CONFIG.
Prints, per cell: normal-event accuracy, CUE, CUE - normal, STR, pair accuracy DIR / REAL (+ forms) / SYS / MENTION and
DIR + REAL; then the registered judgement for the best small-eye cell (method C, highest DIR + REAL pair accuracy; 7B
references and the Q7 eye excluded; DIR + REAL = pooled 10 pairs per domain), and the probe items that cell gets wrong most often.
Registered judgement: 看懂 if DIR+REAL pairs >= 90% and CUE at most 5 points below normal; 猜字 if pairs < 70% or CUE more
than 10 points below normal; otherwise 部分看懂.
"""
import re
import sys
from collections import defaultdict

EXPECTED_PROBE_SHA = "3beede5245d048af2a5a6652079a974e19a1305887710eca6c7d1ffbc83258b8"
EVENTS = ("IN", "OUT", "BREAK", "FIX", "NOOP")
SMALL_CELLS = [(e, l, h) for e in ("Q15", "Q3E") for l in (0, 1) for h in (0, 1)]   # registered "small eyes"
Q7_CELLS = [("Q7", 0, h) for h in (0, 1)]                                             # auxiliary
REFS = [("7B-ZS0", "ZS0"), ("7B-ICL", "ICL")]


def kv(line):
    return dict(re.findall(r"(\S+?)=(\S+)", line))


def pct(cnt):
    c, t = cnt
    return 100.0 * c / t if t else float("nan")


def main(argv):
    partial = "--partial" in argv
    files = [a for a in argv[1:] if a != "--partial"]
    res, items, configs = {}, defaultdict(dict), []
    for fn in files:
        for line in open(fn, encoding="utf-8"):
            line = line.rstrip("\n")
            if line.startswith("L17B_RES ") and " seed=all" in line:
                d = kv(line)
                key = (d["src"], int(d["lora"]), int(d["hard"]), d["method"])
                assert key not in res, f"duplicate result line {key}"
                res[key] = {k: tuple(map(int, v.split("/"))) for k, v in d.items() if "/" in v}
            elif line.startswith("L17B_ITEM "):
                d = kv(line)
                key = (d["src"], int(d["lora"]), int(d["hard"]), d["method"])
                items[key][d["item"]] = (d["kind"], d["y"], list(map(int, d["preds"].split(","))))
            elif line.startswith("L17B_CONFIG "):
                configs.append(kv(line))
    if len(configs) != len(files) and not partial:
        sys.exit("every log needs exactly one L17B_CONFIG line: incomplete run (use --partial to look anyway)")
    for c in configs:
        if c.get("data_sha_probe") != EXPECTED_PROBE_SHA:
            sys.exit(f"probe sha {c.get('data_sha_probe')} != registered {EXPECTED_PROBE_SHA}")
        bad = ("FAKE" in c.get("env", "") or c.get("seeds") != "42,1,2" or c.get("draws") != "10"
               or c.get("icl_draws") != "2")
        if bad and not partial:
            sys.exit(f"not a registered run (env / seeds / draws): {c}")
    if not partial:
        if len({c.get("script_sha") for c in configs}) != 1:
            sys.exit("logs come from different script versions")
        eyes = sorted(e for c in configs for e in c.get("eyes", "").split(","))
        if eyes != ["Q15", "Q3E", "Q7"]:
            sys.exit(f"eyes across logs {eyes} != Q15, Q3E, Q7 once each")
    want = [(e, l, h, m) for e, l, h in SMALL_CELLS + Q7_CELLS for m in ("ZERO", "C")] + [(s, 0, 0, m) for s, m in REFS]
    missing = [k for k in want if k not in res]
    if missing and not partial:
        sys.exit(f"missing cells: {missing}")
    if partial:
        print("!!! PARTIAL: not all registered cells present; numbers are not the registered result\n")

    cols = ["normal", "item_CUE", "CUE-n", "item_STR", "pair_DIR", "pair_REAL", "pair_SYS", "pair_MENTION",
            "pair_DIRREAL"]
    print("cell (eye lora hard method) | " + " | ".join(cols))
    print("-" * 120)
    for k in want:
        if k not in res:
            continue
        r = res[k]
        vals = []
        for c in cols:
            if c == "CUE-n":
                vals.append(f"{pct(r['item_CUE']) - pct(r['normal']):+6.2f}")
            else:
                vals.append(f"{pct(r.get(c, (0, 0))):6.2f}")
        print(f"{k[0]:6s} L{k[1]} H{k[2]} {k[3]:4s} | " + " | ".join(vals))
    print("\nREAL pairs by form (actual vs ...), and per-label item accuracy")
    for k in want:
        if k not in res:
            continue
        r = res[k]
        forms = " ".join(f"{f}={pct(r.get('pair_REAL_' + f, (0, 0))):.1f}" for f in ("negated", "intended", "hypothetical"))
        norm = " ".join(f"{e}={pct(r.get('normal_' + e, (0, 0))):.0f}" for e in EVENTS)
        cue = " ".join(f"{e}={pct(r.get('lab_CUE_' + e, (0, 0))):.0f}" for e in EVENTS)
        print(f"{k[0]:6s} L{k[1]} H{k[2]} {k[3]:4s} | REAL {forms} | normal {norm} | CUE {cue} | "
              f"init {pct(r.get('init', (0, 0))):.1f}")

    def judge(cells, title):
        cand = [(pct(res[c + ("C",)]["pair_DIRREAL"]), c) for c in cells if c + ("C",) in res]
        if not cand:
            return None
        top = max(v for v, _c in cand)
        best = None
        for v, c in cand:
            if v != top:
                continue
            r = res[c + ("C",)]
            gap = pct(r["normal"]) - pct(r["item_CUE"])
            verdict = "看懂" if v >= 90 and gap <= 5 else ("猜字" if v < 70 or gap > 10 else "部分看懂")
            print(f"{title}: {c[0]} LoRA={c[1]} hard={c[2]}: DIR+REAL pairs {v:.2f}%, CUE {pct(r['item_CUE']):.2f}% "
                  f"vs normal {pct(r['normal']):.2f}% (CUE lower by {gap:.2f}) -> {verdict}")
            if best is None:
                best = c
        if sum(v == top for v, _c in cand) > 1:
            print(f"  (tie: {sum(v == top for v, _c in cand)} cells; all listed above)")
        return best

    print()
    best = judge(SMALL_CELLS, "REGISTERED best small-eye cell (method C)" + (" [PARTIAL RUN: not final]" if partial else ""))
    judge(SMALL_CELLS + Q7_CELLS, "auxiliary: best cell including the Q7 eye")
    if best is None:
        return

    it = items.get(best + ("C",), {})
    if it:
        print("\nItems of that cell right in < 50% of (seed, draw) scorings: item kind gold -> predictions")
        for name, (kind, y, pr) in sorted(it.items()):
            tot = sum(pr)
            if tot and pr[EVENTS.index(y)] / tot < 0.5:
                print(f"  {name:28s} {kind:7s} {y:5s} -> "
                      + " ".join(f"{e}:{n}" for e, n in zip(EVENTS + ("disagree",), pr) if n))


if __name__ == "__main__":
    main(sys.argv)
