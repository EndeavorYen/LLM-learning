"""Aggregate the machine lines of Lab 17 (scripts/stage2_lab17_finetune_recognition.py; pre-registered on issue #25,
"第一步之一 e"). Numbers and the registered quantities only; the decision is applied by the reader.

Usage: python tools/aggregate_lab17.py [--partial] [--test] lab17_a.log [lab17_b.log ...]

Refuses (exit 1): a log without its final L17_CONFIG line; unpinned data; seeds other than 42,1,2 or draws other than
10; an eye in two logs; duplicate lines; differing L17_FOLD lines; a missing cell line; totals other than expected
(events 10 x 7 x 27 = 1890, init 560, NEG 280 / OTHER 140 / TEMP 280 per fold x seed) or old + new != total;
an eye not covered (unless --partial); FAKE stub / differing script sha (unless --test).
"Old" = the 16 registered domains (templates.json), "new" = the 12 extra domains. The registered primary metric is
event accuracy on held-out OLD domains.
"""
import sys
from collections import defaultdict

PINNED = {"data_sha_st": "ed6d16bd3e625198bba74d28e7b20f6bdba9c399a3eb94b9a0d1f113f11e3108",
          "templates_sha": "1bfd769efdf6dd3d61ded0e6aaff3eb70b394af2c266e2f02cd1ed9b598bca70",
          "data_sha_extra": "f44dc28a2a2df593981c919949b47e13304a814cec4e960123564206190ca7cf",
          "data_sha_hard": "e145539bc8b37133a226a5d390e6cfebc2492d85e2cc090501d8464fc5afe8aa",
          "data_sha_hard_extra": "e31ad5839bbc1133c6d3e6e5caec6a5125a4ffd9bcffbf330fb793b58416fc53"}
SEEDS = ("42", "1", "2")
FOLDS = ("0", "1", "2", "3")
EYES = {"Q15": True, "Q3E": True, "Q7": False}   # fine-tuned?
HEADS = ("FLAT", "FACT")
METHODS = ("ZERO", "C")
CATS = ("NEG", "OTHER", "TEMP")
EXP_EV, EXP_INIT, EXP_HARD = 1890, 560, {"NEG": 280, "OTHER": 140, "TEMP": 280}
EVN = ("IN", "OUT", "BREAK", "FIX", "NOOP")


def die(m):
    print("ERROR:", m)
    sys.exit("aggregate_lab17: refused")


def kv(line):
    return dict(x.split("=", 1) for x in line.split()[1:])


def pct(c, t):
    return 100.0 * c / t if t else float("nan")


argv = sys.argv[1:]
PARTIAL, TEST = "--partial" in argv, "--test" in argv
paths = [a for a in argv if not a.startswith("--")]
if not paths:
    sys.exit(__doc__)
EV, INI, HD, LORA, folds, cfgs, eye_log = {}, {}, {}, {}, {}, [], {}
for p in paths:
    mach = [l.rstrip("\n") for l in open(p, encoding="utf-8", errors="ignore") if l.startswith("L17_")]
    if not mach or not mach[-1].startswith("L17_CONFIG"):
        die(f"{p}: last machine line is not L17_CONFIG")
    c = kv(mach[-1])
    cfgs.append((p, c))
    for e in c["eyes"].split(","):
        if e in eye_log:
            die(f"eye {e} in {eye_log[e]} and {p}")
        eye_log[e] = p
    for l in mach[:-1]:
        tag, d = l.split()[0], kv(l)
        if tag == "L17_FOLD":
            if d["fold"] in folds and folds[d["fold"]] != d:
                die("L17_FOLD differs across logs")
            folds[d["fold"]] = d
            continue
        if tag == "L17_FEAT":
            continue
        if tag == "L17_LORA":
            key, tab = (d["eye"], d["fold"], d["seed"], d["hard"]), LORA
        else:
            base = (d["eye"], d["lora"], d["hard"], d["head"], d["method"], d["fold"], d["seed"])
            if tag == "L17_EV":
                key, tab = base, EV
            elif tag == "L17_INIT":
                key, tab = base, INI
            elif tag == "L17_HARD":
                key, tab = base + (d["cat"],), HD
            else:
                die(f"unknown line {tag}")
        if key in tab:
            die(f"duplicate {tag} {key}")
        tab[key] = d
for p, c in cfgs:
    for k, v in PINNED.items():
        if c.get(k) != v:
            die(f"{p}: {k}={c.get(k)} is not pinned ({v})")
    if c["seeds"].split(",") != list(SEEDS) or c["draws"] != "10":
        die(f"{p}: seeds / draws not registered")
shas = {c["script_sha"] for _p, c in cfgs}
envs = {c["env"] for _p, c in cfgs}
if not TEST and (len(shas) > 1 or any("FAKE" in e for e in envs)):
    die(f"script sha differs or FAKE stub: {shas} {envs} (--test to override)")
eyes = [e for e in EYES if e in eye_log]
missing = [e for e in EYES if e not in eye_log]
if missing and not PARTIAL:
    die(f"eyes not covered: {missing} (--partial to override)")
if sorted(folds) != list(FOLDS):
    die(f"folds {sorted(folds)}")


def cells(e):
    return [(lo, h) for lo in (("0", "1") if EYES[e] else ("0",)) for h in ("0", "1")]


for e in eyes:
    for lo, h in cells(e):
        for hk in HEADS:
            for m in METHODS:
                for f in FOLDS:
                    for s in SEEDS:
                        k = (e, lo, h, hk, m, f, s)
                        for tab, nm, exp in ((EV, "EV", EXP_EV), (INI, "INIT", EXP_INIT)):
                            if k not in tab:
                                die(f"missing L17_{nm} {k}")
                            d = tab[k]
                            if int(d["total"]) != exp or int(d["t_old"]) + int(d["t_new"]) != exp \
                                    or int(d["c_old"]) + int(d["c_new"]) != int(d["correct"]):
                                die(f"L17_{nm} {k}: totals")
                        for cat in CATS:
                            hd_ = HD.get(k + (cat,))
                            if hd_ is None or int(hd_["total"]) != EXP_HARD[cat] \
                                    or int(hd_["t_old"]) + int(hd_["t_new"]) != int(hd_["total"]) \
                                    or int(hd_["c_old"]) + int(hd_["c_new"]) != int(hd_["correct"]):
                                die(f"L17_HARD {k} {cat}: missing or totals")
        if EYES[e] and lo == "1":
            for f in FOLDS:
                for s in SEEDS:
                    if (e, f, s, h) not in LORA:
                        die(f"missing L17_LORA {e} {f} {s} {h}")


def pool(tab, k6, fields, cat=None):
    e, lo, h, hk, m = k6
    out = [0] * len(fields)
    for f in FOLDS:
        for s in SEEDS:
            d = tab[(e, lo, h, hk, m, f, s) + ((cat,) if cat else ())]
            for i, x in enumerate(fields):
                out[i] += int(d[x])
    return out


def acc(k6, which="old", tab=None, cat=None):
    tab = EV if tab is None else tab
    f = {"old": ("c_old", "t_old"), "new": ("c_new", "t_new"), "all": ("correct", "total")}[which]
    c, t = pool(tab, k6, f, cat)
    return pct(c, t), c, t


def conf(k6):
    mat = [[0] * 5 for _ in range(5)]
    e, lo, h, hk, m = k6
    for f in FOLDS:
        for s in SEEDS:
            v = [int(x) for x in EV[(e, lo, h, hk, m, f, s)]["conf"].split(",")]
            for i in range(25):
                mat[i // 5][i % 5] += v[i]
    return mat


print(f"Lab 17 aggregate | logs: {', '.join(paths)}")
for p, c in cfgs:
    print(f"  {p}: eyes={c['eyes']} script_sha={c['script_sha']} env={c['env']}")
if missing:
    print(f"  PARTIAL: eyes not covered {missing}")
print("Accuracy pooled over 4 folds x 3 seeds x 10 draws. OLD = held-out registered domains (primary). NO decision.\n")

print("=== event sentences, 5-way: OLD domains (NEW domains) per cell; ZERO = no example, C = one example per type ===")
print(f"{'eye':<5}{'LoRA':>5}{'hard':>5}{'head':>6}" + "".join(f"{m:>18}" for m in METHODS))
for e in eyes:
    for lo, h in cells(e):
        for hk in HEADS:
            row = ""
            for m in METHODS:
                a, _c, _t = acc((e, lo, h, hk, m))
                b, _c2, _t2 = acc((e, lo, h, hk, m), "new")
                row += f"{a:9.2f} ({b:5.2f})"
            print(f"{e:<5}{lo:>5}{h:>5}{hk:>6}{row}")

print("\n=== REGISTERED quantities ===")
best = None
for e in eyes:
    if not EYES[e]:
        continue
    main = (e, "1", "1", "FACT", "C")
    a, c, t = acc(main)
    neg, nc, nt = acc(main, tab=HD, cat="NEG")
    a0, c0, t0 = acc((e, "1", "0", "FACT", "C"))
    fr, _x, _y = acc((e, "0", "1", "FACT", "C"))
    print(f"  {e} main cell (LoRA + FACT + hard, C): OLD events {a:.2f}% ({c}/{t}) | NEG->NOOP OLD {neg:.2f}% "
          f"({nc}/{nt}) | events with hard off {a0:.2f}% (drop {a0 - a:+.2f}) | frozen same cell {fr:.2f}% "
          f"(LoRA gain = main minus frozen, same cell: {a - fr:+.2f})")
    if best is None or a > best[0]:
        best = (a, e, neg, a0 - a, a - fr)
if best:
    a, e, neg, drop, gain = best
    print(f"  best eye{' (PARTIAL)' if missing else ''}: {e} = {a:.2f}% | >= 95%: {a >= 95} | negation gate (NEG >= 90 and drop <= 1): "
          f"{neg >= 90 and drop <= 1} | 93-95%: {93 <= a < 95} | LoRA gain >= 2 points: {gain >= 2}")

print("\n=== main effects on OLD events, method C (mean of the other factors' cells; points) ===")
for e in eyes:
    if not EYES[e]:
        continue
    g = lambda lo, h, hk: acc((e, lo, h, hk, "C"))[0]
    lora = sum(g("1", h, hk) - g("0", h, hk) for h in "01" for hk in HEADS) / 4
    head = sum(g(lo, h, "FACT") - g(lo, h, "FLAT") for lo in "01" for h in "01") / 4
    hard_ = sum(g(lo, "1", hk) - g(lo, "0", hk) for lo in "01" for hk in HEADS) / 4
    print(f"  {e}: LoRA {lora:+.2f} | FACT vs FLAT {head:+.2f} | hard on {hard_:+.2f}")

print("\n=== IN<->OUT confusions (OLD+NEW queries), method C ===")
for e in eyes:
    for lo, h in cells(e):
        for hk in HEADS:
            mat = conf((e, lo, h, hk, "C"))
            err = sum(mat[i][j] for i in range(5) for j in range(5) if i != j)
            print(f"  {e} LoRA={lo} hard={h} {hk}: IN->OUT {mat[0][1]} OUT->IN {mat[1][0]} of {err} errors")

print("\n=== hard sentences on held-out OLD domains: NEG->NOOP / OTHER->NOOP / TEMP correct [% literal event] ===")
for e in eyes:
    for lo, h in cells(e):
        for hk in HEADS:
            for m in METHODS:
                parts = []
                for cat in CATS:
                    a, _c, _t = acc((e, lo, h, hk, m), tab=HD, cat=cat)
                    sc, st = pool(HD, (e, lo, h, hk, m), ("as_surface", "total"), cat)
                    an, _c2, _t2 = acc((e, lo, h, hk, m), "new", tab=HD, cat=cat)
                    parts.append(f"{cat} {a:6.2f} (new {an:6.2f}) [{pct(sc, st):5.1f}]")
                print(f"  {e} LoRA={lo} hard={h} {hk:<4} {m:<4} " + " | ".join(parts))

print("\n=== initial-state sentences, 8-way, OLD domains ===")
for e in eyes:
    for lo, h in cells(e):
        for hk in HEADS:
            print(f"  {e} LoRA={lo} hard={h} {hk:<4} " + " | ".join(
                f"{m} {acc((e, lo, h, hk, m), tab=INI)[0]:6.2f} (new {acc((e, lo, h, hk, m), 'new', tab=INI)[0]:6.2f})"
                for m in METHODS))

if LORA:
    print("\n=== LoRA loss traces (step:loss), first fold/seed per eye ===")
    for (e, f, s, h), d in sorted(LORA.items()):
        if f == "0" and s == "42":
            print(f"  {e} hard={h}: {d['loss']} ({d['sec']} s)")

print("\n=== confusion of the best main cell, method C (rows = true) ===")
if best:
    mat = conf((best[1], "1", "1", "FACT", "C"))
    print(f"{'':>8}" + "".join(f"{x:>8}" for x in EVN))
    for i, x in enumerate(EVN):
        print(f"{x:>8}" + "".join(f"{v:>8}" for v in mat[i]))
