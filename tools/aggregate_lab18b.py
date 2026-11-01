"""Aggregate Lab 18b (reading reinforcement, issue #25) logs.

    python tools/aggregate_lab18b.py lab18b_A.log lab18b_B.log [--partial]

Registered judgement (test v2, readout C): best cell = highest group_all over cells A0B0 / A0B1 / A1B0 / A1B1 x eyes;
  讀得廣 group_all >= 80 and normal >= 95 and HYPO >= 70; 教了才會 the same with HYPO < 70; 還沒讀懂 otherwise.
Factor effect (per eye): main effect = mean over the other factor of (on - off) on group_all; "effective" if >= +3
points and the per-seed effect has the same sign in >= 4 of 5 seeds.
"""
import re
import sys
from collections import defaultdict

REG = {"data_sha_st": "ed6d16bd3e625198bba74d28e7b20f6bdba9c399a3eb94b9a0d1f113f11e3108",
       "data_sha_extra": "f44dc28a2a2df593981c919949b47e13304a814cec4e960123564206190ca7cf",
       "data_sha_hard": "e145539bc8b37133a226a5d390e6cfebc2492d85e2cc090501d8464fc5afe8aa",
       "data_sha_hard_extra": "e31ad5839bbc1133c6d3e6e5caec6a5125a4ffd9bcffbf330fb793b58416fc53",
       "data_sha_probe": "3beede5245d048af2a5a6652079a974e19a1305887710eca6c7d1ffbc83258b8",
       "data_sha_rtrain": "ffede6eca9e2777a213c926c21b9b75981aad4fef50711b391c49d9149efeaf8",
       "data_sha_rtest": "49616f38581105d0c69c6e9b1509d8d763027fdb3d3b7ec199ea8a07d03c4c4a",
       "data_sha_rtrain2": "1d7f1c6fac0b39b7bae6ad7c81472986f97853c9dd33120441a5cd022b92942a", "data_sha_rtest2": "904a97b811942ef95efede95d367603364e3542a2ff2f6ac6737f4e3121f2de2"}
EYES = ("Q15", "Q3E")
CELLS = ("A0B0", "A0B1", "A1B0", "A1B1")
SEEDS = ("42", "1", "2", "3", "4")
VARS = ("BASE", "NEG", "INTENT", "QUESTION", "FUTURE", "POSNEG", "OTHER", "TRANSFER", "DIRSWAP", "PRONOUN", "CHATTER",
        "HYPO", "NOOPSENT")


def kv(line):
    return dict(re.findall(r"(\S+?)=(\S+)", line))


def pct(c):
    return 100.0 * c[0] / c[1] if c and c[1] else float("nan")


def main(argv):
    partial = "--partial" in argv
    files = [a for a in argv[1:] if a != "--partial"]
    res, prb, items, configs = {}, {}, defaultdict(dict), []
    for fn in files:
        for line in open(fn, encoding="utf-8"):
            line = line.rstrip("\n")
            if line.startswith("L18B_RES "):
                d = kv(line)
                key = (d["test"], d["eye"], d["cell"], d["method"], d["seed"])
                assert key not in res, f"duplicate {key}"
                res[key] = {k: tuple(map(int, v.split("/"))) for k, v in d.items() if "/" in v}
            elif line.startswith("L18B_PROBE "):
                d = kv(line)
                prb[(d["eye"], d["cell"], d["method"], d["seed"])] = {k: tuple(map(int, v.split("/"))) for k, v in d.items() if "/" in v}
            elif line.startswith("L18B_ITEM "):
                d = kv(line)
                items[(d["eye"], d["cell"], d["method"])][d["item"]] = (d["y"], list(map(int, d["preds"].split(","))))
            elif line.startswith("L18B_CONFIG "):
                configs.append(kv(line))
    if len(configs) != len(files) and not partial:
        sys.exit("every log needs exactly one L18B_CONFIG line (use --partial to look anyway)")
    for c in configs:
        bad = ("FAKE" in c.get("env", "") or c.get("seeds") != ",".join(SEEDS) or c.get("draws") != "10"
               or c.get("cells") != ",".join(CELLS))
        bad = bad or any(v is not None and c.get(k) != v for k, v in REG.items())
        if bad and not partial:
            sys.exit(f"not a registered run: {c}")
    if not partial:
        if None in REG.values():
            sys.exit("v2 data shas not pinned in this aggregator")
        if len({c.get("script_sha") for c in configs}) != 1:
            sys.exit("logs come from different script versions")
        if sorted(e for c in configs for e in c.get("eyes", "").split(",")) != sorted(EYES):
            sys.exit("eyes across logs != Q15, Q3E once each")
        miss = [(t, e, c, m) for t in ("v2", "v1") for e in EYES for c in CELLS for m in ("ZERO", "C")
                if (t, e, c, m, "all") not in res]
        if miss:
            sys.exit(f"missing cells: {miss}")
    if partial:
        print("!!! PARTIAL: numbers are not the registered result\n")

    for t, title in (("v2", "NEW sealed test v2 (primary)"), ("v1", "Lab 18a test v1 (auxiliary)")):
        print(f"== {title}: cell | group_all | normal | taught items | HYPO | unrelated NOOP")
        for e in EYES:
            for c in CELLS:
                for m in ("ZERO", "C"):
                    r = res.get((t, e, c, m, "all"))
                    if r:
                        print(f"  {e} {c} {m:4s} | {pct(r.get('group_all')):6.2f} | {pct(r.get('normal')):6.2f} | "
                              f"{pct(r.get('taught_items')):6.2f} | {pct(r.get('var_HYPO')):6.2f} | {pct(r.get('var_NOOPSENT')):6.2f}")
        print()
    print("per variant accuracy, test v2, readout C")
    print("cell | " + " ".join(f"{v[:7]:>7s}" for v in VARS))
    for e in EYES:
        for c in CELLS:
            r = res.get(("v2", e, c, "C", "all"))
            if r:
                print(f"{e} {c} | " + " ".join(f"{pct(r.get('var_' + v)):7.1f}" for v in VARS))
    print("\nLab 17b probe re-test (readout C): normal | DIR+REAL | DIR | REAL | REAL-hyp | SYS | MENTION | CUE | STR")
    for e in EYES:
        for c in CELLS:
            p = prb.get((e, c, "C", "all"))
            if p:
                print(f"{e} {c} | " + " | ".join(f"{pct(p.get(k)):6.2f}" for k in
                      ("normal", "pair_DIRREAL", "pair_DIR", "pair_REAL", "pair_REAL_hypothetical", "pair_SYS",
                       "pair_MENTION", "item_CUE", "item_STR")))

    print("\nFactor main effects on group_all (test v2, readout C); per-seed sign count")
    for e in EYES:
        for fac, (on, off) in (("A system-aware prompt", (("A1B0", "A1B1"), ("A0B0", "A0B1"))),
                               ("B noise practice", (("A0B1", "A1B1"), ("A0B0", "A1B0")))):
            def eff(seed):
                v = [pct(res[("v2", e, x, "C", seed)]["group_all"]) for x in on + off
                     if ("v2", e, x, "C", seed) in res]
                if len(v) < 4:
                    return None
                return (v[0] + v[1]) / 2 - (v[2] + v[3]) / 2
            tot = eff("all")
            per = [eff(s) for s in SEEDS]
            per = [x for x in per if x is not None]
            if tot is None:
                continue
            same = sum((x > 0) == (tot > 0) for x in per)
            ok = tot >= 3 and same >= 4
            nrm_on = sum(pct(res[("v2", e, x, "C", "all")]["normal"]) for x in on) / 2
            nrm_off = sum(pct(res[("v2", e, x, "C", "all")]["normal"]) for x in off) / 2
            print(f"  {e} {fac}: {tot:+.2f} (same sign in {same}/{len(per)} seeds; normal {nrm_on - nrm_off:+.2f})"
                  + ("  -> EFFECTIVE" if ok else "  -> not effective"))

    cand = [(pct(res[("v2", e, c, "C", "all")]["group_all"]), (e, c)) for e in EYES for c in CELLS
            if ("v2", e, c, "C", "all") in res]
    if not cand:
        return
    top = max(v for v, _ in cand)
    print()
    first = None
    for v, (e, c) in cand:
        if v != top:
            continue
        r = res[("v2", e, c, "C", "all")]
        if "var_HYPO" not in r:
            sys.exit("var_HYPO missing")
        nrm, hyp = pct(r["normal"]), pct(r["var_HYPO"])
        verdict = "讀得廣" if v >= 80 and nrm >= 95 and hyp >= 70 else ("教了才會" if v >= 80 and nrm >= 95 else "還沒讀懂")
        print(f"REGISTERED best cell (test v2, readout C){' [PARTIAL]' if partial else ''}: {e} {c}: group_all {v:.2f}%, "
              f"normal {nrm:.2f}%, HYPO {hyp:.2f}% -> {verdict}")
        first = first or (e, c)
    it = items.get(first + ("C",), {})
    if it:
        ev = ("IN", "OUT", "BREAK", "FIX", "NOOP")
        bad = [(n, y, pr) for n, (y, pr) in sorted(it.items()) if sum(pr) and pr[ev.index(y)] / sum(pr) < 0.5]
        by_var = defaultdict(int)
        for n, _y, _pr in bad:
            by_var[n.rsplit("/", 1)[1]] += 1
        print("\nBest cell, test v2: items right in < 50% of scorings, count by variant: "
              + " ".join(f"{k}={v}" for k, v in sorted(by_var.items())))
        for n, y, pr in bad:
            print(f"  {n:34s} {y:5s} -> " + " ".join(f"{e}:{x}" for e, x in zip(ev, pr) if x))


if __name__ == "__main__":
    main(sys.argv)
