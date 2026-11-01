"""Aggregate Lab 18a (reading breadth, issue #25) logs.

    python tools/aggregate_lab18a.py lab18a_A.log lab18a_B.log [--partial]

Reads the seed=all L18A_RES / L18A_PROBE lines (summed over seeds 42, 1, 2 and, for C, 10 draws), L18A_ITEM and
L18A_CONFIG. Prints per (eye, condition, readout): group all-correct (taught variants), normal (base) sentences, each
variant, HYPO (untaught), unrelated NOOP sentences, and the Lab 17b probe re-test; then the registered judgement:
best cell = highest group_all among PAIRS / PAIRSCR x eyes (readout C);
  讀得廣 group_all >= 80 and normal >= 95 and HYPO >= 70; 教了才會 group_all >= 80 and normal >= 95 and HYPO < 70;
  還沒讀懂 otherwise. Side effect: normal more than 1 point below HARD17 (same eye, readout C).
"""
import re
import sys
from collections import defaultdict

EXPECTED = {"data_sha_probe": "3beede5245d048af2a5a6652079a974e19a1305887710eca6c7d1ffbc83258b8",
            "data_sha_rtrain": "ffede6eca9e2777a213c926c21b9b75981aad4fef50711b391c49d9149efeaf8",
            "data_sha_rtest": "49616f38581105d0c69c6e9b1509d8d763027fdb3d3b7ec199ea8a07d03c4c4a"}
REG_DATA = {"data_sha_st": "ed6d16bd3e625198bba74d28e7b20f6bdba9c399a3eb94b9a0d1f113f11e3108",
            "data_sha_extra": "f44dc28a2a2df593981c919949b47e13304a814cec4e960123564206190ca7cf",
            "data_sha_hard": "e145539bc8b37133a226a5d390e6cfebc2492d85e2cc090501d8464fc5afe8aa",
            "data_sha_hard_extra": "e31ad5839bbc1133c6d3e6e5caec6a5125a4ffd9bcffbf330fb793b58416fc53"}
EYES = ("Q15", "Q3E")
CONDS = ("HARD17", "PAIRS", "PAIRSCR")
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
            if line.startswith(("L18A_RES ", "L18A_PROBE ")) and " seed=all" in line:
                d = kv(line)
                key = (d["eye"], d["cond"], d["method"])
                dst = res if line.startswith("L18A_RES") else prb
                assert key not in dst, f"duplicate {key}"
                dst[key] = {k: tuple(map(int, v.split("/"))) for k, v in d.items() if "/" in v}
            elif line.startswith("L18A_ITEM "):
                d = kv(line)
                items[(d["eye"], d["cond"], d["method"])][d["item"]] = (d["y"], list(map(int, d["preds"].split(","))))
            elif line.startswith("L18A_CONFIG "):
                configs.append(kv(line))
    if len(configs) != len(files) and not partial:
        sys.exit("every log needs exactly one L18A_CONFIG line (use --partial to look anyway)")
    for c in configs:
        for k, v in EXPECTED.items():
            if v is not None and c.get(k) != v:
                sys.exit(f"{k} {c.get(k)} != registered {v}")
        bad = ("FAKE" in c.get("env", "") or c.get("seeds") != "42,1,2" or c.get("draws") != "10"
               or c.get("conds") != "HARD17,PAIRS,PAIRSCR" or any(c.get(k) != v for k, v in REG_DATA.items()))
        if bad and not partial:
            sys.exit(f"not a registered run: {c}")
    if not partial:
        if None in EXPECTED.values():
            sys.exit("reading data shas not pinned in this aggregator")
        if len({c.get("script_sha") for c in configs}) != 1 or len({c.get("data_sha_rtest") for c in configs}) != 1:
            sys.exit("logs come from different script / data versions")
        if sorted(e for c in configs for e in c.get("eyes", "").split(",")) != sorted(EYES):
            sys.exit("eyes across logs != Q15, Q3E once each")
    want = [(e, c, m) for e in EYES for c in CONDS for m in ("ZERO", "C")]
    missing = [k for k in want if k not in res]
    if missing and not partial:
        sys.exit(f"missing cells: {missing}")
    if partial:
        print("!!! PARTIAL: numbers are not the registered result\n")

    print("cell (eye cond readout) | group_all | normal | taught items | HYPO | unrelated NOOP")
    print("-" * 90)
    for k in want:
        if k in res:
            r = res[k]
            print(f"{k[0]:4s} {k[1]:8s} {k[2]:4s} | {pct(r.get('group_all')):6.2f} | {pct(r.get('normal')):6.2f} | "
                  f"{pct(r.get('taught_items')):6.2f} | {pct(r.get('var_HYPO')):6.2f} | {pct(r.get('var_NOOPSENT')):6.2f}")
    print("\nper variant accuracy (readout C)")
    print("cell | " + " ".join(f"{v[:7]:>7s}" for v in VARS))
    for e in EYES:
        for c in CONDS:
            r = res.get((e, c, "C"))
            if r:
                print(f"{e} {c:8s} | " + " ".join(f"{pct(r.get('var_' + v)):7.1f}" for v in VARS))
    print("\nLab 17b probe re-test (readout C): normal | DIR+REAL pairs | DIR | REAL | REAL-hypothetical | SYS | MENTION | "
          "CUE | STR")
    for e in EYES:
        for c in CONDS:
            p = prb.get((e, c, "C"))
            if p:
                print(f"{e} {c:8s} | " + " | ".join(f"{pct(p.get(k)):6.2f}" for k in
                      ("normal", "pair_DIRREAL", "pair_DIR", "pair_REAL", "pair_REAL_hypothetical", "pair_SYS",
                       "pair_MENTION", "item_CUE", "item_STR")))

    cand = [(pct(res[(e, c, "C")]["group_all"]), (e, c)) for e in EYES for c in ("PAIRS", "PAIRSCR")
            if (e, c, "C") in res]
    if not cand:
        return
    top = max(v for v, _ in cand)
    print()
    first = None
    for v, (e, c) in cand:
        if v != top:
            continue
        r = res[(e, c, "C")]
        if "var_HYPO" not in r:
            sys.exit("var_HYPO missing")
        nrm, hyp = pct(r["normal"]), pct(r.get("var_HYPO"))
        if v >= 80 and nrm >= 95 and hyp >= 70:
            verdict = "讀得廣"
        elif v >= 80 and nrm >= 95:
            verdict = "教了才會"
        else:
            verdict = "還沒讀懂"
        base = res.get((e, "HARD17", "C"))
        side = (pct(base["normal"]) - nrm) if base else float("nan")
        print(f"REGISTERED best cell (readout C){' [PARTIAL]' if partial else ''}: {e} {c}: group_all {v:.2f}%, "
              f"normal {nrm:.2f}%, HYPO {hyp:.2f}% -> {verdict}; normal vs HARD17 {-side:+.2f}"
              + ("  [SIDE EFFECT]" if side > 1 else ""))
        first = first or (e, c)
    if sum(v == top for v, _ in cand) > 1:
        print("  (tie: all tied cells listed)")
    print("\nSide effects vs HARD17 (same eye, readout C): reading-test normal (base) | Lab 17b template-style normal"
          " (registered auxiliary side-effect measure); > 1 point drop flagged")
    for e in EYES:
        h, hp = res.get((e, "HARD17", "C")), prb.get((e, "HARD17", "C"))
        for c in ("PAIRS", "PAIRSCR"):
            r, rp = res.get((e, c, "C")), prb.get((e, c, "C"))
            if h and r and hp and rp:
                d1, d2 = pct(r["normal"]) - pct(h["normal"]), pct(rp["normal"]) - pct(hp["normal"])
                print(f"  {e} {c:8s}: {d1:+.2f} | {d2:+.2f}" + ("  [SIDE EFFECT]" if min(d1, d2) < -1 else ""))
    for e in EYES:
        a, b = res.get((e, "PAIRS", "C")), res.get((e, "PAIRSCR", "C"))
        if a and b:
            print(f"CR vs PAIRS ({e}, descriptive): group_all {pct(b['group_all']) - pct(a['group_all']):+.2f}, "
                  f"normal {pct(b['normal']) - pct(a['normal']):+.2f}, HYPO {pct(b.get('var_HYPO')) - pct(a.get('var_HYPO')):+.2f}")
    it = items.get(first + ("C",), {})
    if it:
        print("\nBest cell: items right in < 50% of scorings (item gold -> predictions IN/OUT/BREAK/FIX/NOOP)")
        ev = ("IN", "OUT", "BREAK", "FIX", "NOOP")
        bad = [(n, y, pr) for n, (y, pr) in sorted(it.items()) if sum(pr) and pr[ev.index(y)] / sum(pr) < 0.5]
        by_var = defaultdict(int)
        for n, y, pr in bad:
            by_var[n.rsplit("/", 1)[1]] += 1
        print("  count by variant: " + " ".join(f"{k}={v}" for k, v in sorted(by_var.items())))
        for n, y, pr in bad:
            print(f"  {n:34s} {y:5s} -> " + " ".join(f"{e}:{x}" for e, x in zip(ev, pr) if x))


if __name__ == "__main__":
    main(sys.argv)
