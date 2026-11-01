"""Aggregate the machine lines of Lab 16 (scripts/stage2_lab16_anchor_recognition.py; pre-registered on issue #25,
"第一步之一 c / d"). Numbers and the registered quantities only; the decision is applied by the reader.

Usage: python tools/aggregate_lab16.py [--partial] [--test] [--a-best METHOD] lab16_1.log [lab16_2.log ...]
  (all logs of ONE part). Part B requires --a-best = the best anchor method of the part A aggregate (registered:
  the shape rule applies to ZERO-MLP 13-way and to that method).
  --partial  accept logs that do not cover every encoder of the part (every table is then labelled PARTIAL).
  --test     accept FAKE-stub logs and differing script shas (CPU stub tests only).

Refuses (exit status 1):
  - a log without its final L16_CONFIG line, logs of different parts, a data sha that is not pinned, seeds other
    than 42, 1, 2, draws other than 10, or shots other than the part's;
  - an encoder covered by two logs, a duplicate machine line (e.g. a log passed twice), L16_FOLD lines that differ;
  - a missing line for any covered encoder (every method x k x s x fold x seed; init, hard, confusion, 7B lines);
  - query totals that differ across methods / encoders for the same (k, s, fold, seed) (all must score the same
    sentences), totals other than the expected ones (review L2), or old + new counts that do not add up;
  - [review D2] an encoder of the part not covered (unless --partial), a Q7 log of part A without the 7B lines;
  - [review L1] different script shas or a FAKE stub (unless --test).
Best combination of part A (registered, s=1, k=12): the 6 ANCHOR methods only (review D3, #25 note of 2026-11-01);
  ZERO-MLP / ZERO-FUNNEL are reference rows and 7B-ICL is a reference.
Accuracy = exact counts pooled over folds x seeds x draws.
"""
import math
import sys
from collections import defaultdict

PINNED = {"data_sha_st": "ed6d16bd3e625198bba74d28e7b20f6bdba9c399a3eb94b9a0d1f113f11e3108",
          "templates_sha": "1bfd769efdf6dd3d61ded0e6aaff3eb70b394af2c266e2f02cd1ed9b598bca70",
          "data_sha_hard": {"A": "e145539bc8b37133a226a5d390e6cfebc2492d85e2cc090501d8464fc5afe8aa", "B": "-"},
          "data_sha_extra": {"A": "-", "B": "c26febd25091b0b797067127e5a1c02773c209f829408a7551f006a814bab232"}}  # B: accepted extra_domains.json
SEEDS = ("42", "1", "2")
DRAWS = "10"
PARTS = {"A": dict(folds=4, ks=(4, 8, 12), reg_k=12, shots=(1, 2, 3),
                   encs=("Q05-mean", "Q05-prompt", "Q15-mean", "Q15-prompt", "Q7-mean", "Q7-prompt", "BGE")),
         "B": dict(folds=7, ks=(4, 6, 8, 12, 16, 20, 24), reg_k=12, shots=(1,), encs=("Q15-prompt", "Q7-prompt", "BGE"))}
METHODS = ["ZERO-MLP", "ZERO-FUNNEL", "RAW-NN", "MLP-NN", "FUNNEL-NN", "C-MLP", "C-MLP-NN", "C-FUNNEL-NN"]
ANCHOR_METHODS = METHODS[2:]   # [review D3] methods that use the new domain's examples
N_HELD = 4                     # held-out domains per fold
N_EV_DOM = 32                  # event sentences per domain
N_INIT_DOM = 8
HARD_PER_DOM = {"NEG": 4, "OTHER": 2, "TEMP": 4}
INIT_METHODS = ["ZERO-MLP", "C-MLP"]
HARD_CATS = ("NEG", "OTHER", "TEMP")
EVENTS = ("IN", "OUT", "BREAK", "FIX", "NOOP")
N_EV = 5
ICL_DRAWS = 2


def die(msg):
    print(f"ERROR: {msg}")
    sys.exit("aggregate_lab16: refused (see ERROR above)")


def kv_of(line):
    return dict(x.split("=", 1) for x in line.split()[1:])


def pct(c, t):
    return None if not t else 100.0 * c / t


def fp(x, w=8):
    return f"{'-':>{w}}" if x is None else f"{x:{w}.2f}"


argv = sys.argv[1:]
PARTIAL = "--partial" in argv
TEST = "--test" in argv
A_BEST = None
if "--a-best" in argv:
    i = argv.index("--a-best")
    A_BEST = argv[i + 1]
    del argv[i:i + 2]
paths = [a for a in argv if a not in ("--partial", "--test")]
if not paths:
    sys.exit(__doc__)
if A_BEST is not None and A_BEST not in ANCHOR_METHODS:
    sys.exit(f"--a-best must be one of {ANCHOR_METHODS}")
cfgs, folds = [], {}
CV13, SHOT, INIT, HARD, CONF, ICL = {}, {}, {}, {}, {}, {}
enc_of_log = {}
for path in paths:
    lines = [l.rstrip("\n") for l in open(path, encoding="utf-8", errors="ignore")]
    mach = [l for l in lines if l.startswith("L16_")]
    if not mach or not mach[-1].startswith("L16_CONFIG"):
        die(f"{path}: the last machine line is not L16_CONFIG (run incomplete)")
    cfg = kv_of(mach[-1])
    cfgs.append((path, cfg))
    encs = cfg["encs"].split(",")
    for e in encs:
        if e in enc_of_log:
            die(f"encoder {e} is in {enc_of_log[e]} and {path}")
        enc_of_log[e] = path
    for l in mach[:-1]:
        tag, kv = l.split()[0], kv_of(l)
        if tag == "L16_FOLD":
            key = kv["fold"]
            if key in folds and folds[key] != kv:
                die(f"L16_FOLD fold {key} differs across logs")
            folds[key] = kv
            continue
        if tag == "L16_FEAT" or tag == "L16_CONFIG":
            continue
        if "enc" in kv and kv["enc"] not in encs:
            die(f"{path}: line for encoder {kv['enc']} not listed in its L16_CONFIG encs")
        if tag == "L16_CV13":
            tab, key = CV13, (kv["enc"], int(kv["k"]), kv["fold"], kv["seed"])
        elif tag == "L16_SHOT":
            tab, key = SHOT, (kv["enc"], kv["method"], int(kv["k"]), int(kv["s"]), kv["fold"], kv["seed"])
        elif tag == "L16_INIT":
            tab, key = INIT, (kv["enc"], kv["method"], int(kv["k"]), kv["fold"], kv["seed"])
        elif tag == "L16_HARD":
            tab, key = HARD, (kv["enc"], kv["method"], kv["fold"], kv["seed"], kv["cat"])
        elif tag == "L16_CONF":
            tab, key = CONF, (kv["enc"], kv["method"], kv["fold"], kv["seed"])
        elif tag in ("L16_ICL", "L16_ZS0"):
            tab, key = ICL, (tag[4:], kv["dom"], kv["draw"])
        else:
            die(f"unknown machine line {tag}")
        if key in tab:
            die(f"duplicate {tag} {key} ({path}); pass each log once")
        tab[key] = kv

parts = {c["part"] for _p, c in cfgs}
if len(parts) != 1:
    die(f"logs of several parts: {parts}")
part = parts.pop()
P = PARTS[part]
for path, c in cfgs:
    for k in ("data_sha_st", "templates_sha"):
        if c[k] != PINNED[k]:
            die(f"{path}: {k}={c[k]} is not the pinned file")
    for k in ("data_sha_hard", "data_sha_extra"):
        want = PINNED[k][part]
        if want is None:
            die(f"{k} is not pinned in this aggregator yet (part {part})")
        if c[k] != want:
            die(f"{path}: {k}={c[k]} is not the pinned file ({want})")
    if c["seeds"].split(",") != list(SEEDS) or c["draws"] != DRAWS:
        die(f"{path}: seeds {c['seeds']} / draws {c['draws']} are not the registered ones")
    if tuple(int(x) for x in c["shots"].split(",")) != P["shots"]:
        die(f"{path}: shots {c['shots']} != part {part}'s {P['shots']}")
if len(folds) != P["folds"]:
    die(f"{len(folds)} L16_FOLD lines, part {part} has {P['folds']} folds")
fold_doms = {d for kv in folds.values() for d in kv["heldout"].split(",")}
encs = [e for e in P["encs"] if e in enc_of_log]
missing_encs = [e for e in P["encs"] if e not in enc_of_log]
FOLDS = [str(f) for f in range(P["folds"])]
envs = {c["env"] for _p, c in cfgs}
shas = {c["script_sha"] for _p, c in cfgs}
icl_on = any(c["icl"] == "1" for _p, c in cfgs)
if missing_encs and not PARTIAL:
    die(f"encoders of part {part} not covered: {missing_encs} (pass every log, or --partial)")
for path, c in cfgs:  # [review D2] the 7B reference must come with the Q7 session of part A
    if part == "A" and "Q7" in c["models"].split(",") and c["icl"] != "1":
        die(f"{path}: part A log with Q7 but without the 7B lines (icl=0)")
if (len(shas) > 1 or len(envs) > 1 and len({e.split(',gpu=')[0] for e in envs}) > 1) and not TEST:
    die(f"script sha / environment differ across logs: {shas} {envs} (--test to override)")
if any("FAKE" in e for e in envs) and not TEST:
    die("FAKE transformers stub in a log: a test run, not a registered run (--test to override)")
PART_TAG = " [PARTIAL: not every encoder covered]" if missing_encs else ""

# ---- completeness and consistency ---------------------------------------------------------------------------------
for e in encs:
    for k in P["ks"]:
        for f in FOLDS:
            for sd in SEEDS:
                if (e, k, f, sd) not in CV13:
                    die(f"missing L16_CV13 {e} k={k} fold={f} seed={sd}")
                for m in INIT_METHODS:
                    if (e, m, k, f, sd) not in INIT:
                        die(f"missing L16_INIT {e} {m} k={k} fold={f} seed={sd}")
                for s in P["shots"]:
                    for m in METHODS:
                        if (e, m, k, s, f, sd) not in SHOT:
                            die(f"missing L16_SHOT {e} {m} k={k} s={s} fold={f} seed={sd}")
    for f in FOLDS:
        for sd in SEEDS:
            for m in METHODS:
                if (e, m, f, sd) not in CONF:
                    die(f"missing L16_CONF {e} {m} fold={f} seed={sd}")
                if part == "A":
                    for cat in HARD_CATS:
                        if (e, m, f, sd, cat) not in HARD:
                            die(f"missing L16_HARD {e} {m} fold={f} seed={sd} {cat}")
ref_tot = {}
for (e, m, k, s, f, sd), kv in SHOT.items():
    t = int(kv["total"])
    if int(kv["t_old"]) + int(kv["t_new"]) != t or int(kv["c_old"]) + int(kv["c_new"]) != int(kv["correct"]):
        die(f"L16_SHOT {e} {m} k={k} s={s} fold={f} seed={sd}: old + new != total")
    key = (k, s, f, sd)
    if ref_tot.setdefault(key, t) != t:
        die(f"L16_SHOT query totals differ for k={k} s={s} fold={f} seed={sd}")
for key, t in ref_tot.items():  # [review L2] expected totals
    if t != N_HELD * int(DRAWS) * (N_EV_DOM - N_EV * key[1]):
        die(f"L16_SHOT total {t} for k={key[0]} s={key[1]} fold={key[2]} seed={key[3]} != expected")
for key, kv in CV13.items():
    if int(kv["total"]) != N_HELD * (N_EV_DOM + N_INIT_DOM):
        die(f"L16_CV13 {key}: total {kv['total']} != {N_HELD * (N_EV_DOM + N_INIT_DOM)}")
for key, kv in INIT.items():
    if int(kv["total"]) != N_HELD * N_INIT_DOM * int(DRAWS):
        die(f"L16_INIT {key}: total {kv['total']} != {N_HELD * N_INIT_DOM * int(DRAWS)}")
for key, kv in HARD.items():
    if int(kv["total"]) != N_HELD * HARD_PER_DOM[key[4]] * int(DRAWS):
        die(f"L16_HARD {key}: total {kv['total']} != expected")
if icl_on:
    if part != "A":
        die("7B lines in a part B log")
    want = {(t, d, str(r)) for t in ("ICL", "ZS0") for d in fold_doms for r in range(ICL_DRAWS)}
    if set(ICL) != want:
        die(f"7B lines: got {len(ICL)} keys, expected ICL and ZS0 for the {len(fold_doms)} train domains x "
            f"{ICL_DRAWS} draws (missing {sorted(want - set(ICL))[:3]}, unexpected {sorted(set(ICL) - want)[:3]})")
    for (_t, d, r), kv in ICL.items():
        if int(kv["total"]) != N_EV_DOM - N_EV:
            die(f"7B {_t} {d} draw {r}: total {kv['total']} != {N_EV_DOM - N_EV}")


# ---- helpers ------------------------------------------------------------------------------------------------------
def shot(e, m, k, s, fields=("correct", "total"), ff=None):
    ff = FOLDS if ff is None else ff
    return tuple(sum(int(SHOT[(e, m, k, s, f, sd)][x]) for f in ff for sd in SEEDS) for x in fields)


def cv13(e, k, fields=("correct", "total")):
    return tuple(sum(int(CV13[(e, k, f, sd)][x]) for f in FOLDS for sd in SEEDS) for x in fields)


def acc_shot(e, m, k, s):
    return pct(*shot(e, m, k, s))


def shape(acc):
    """Registered rule: g1 = acc(12) - acc(6), g2 = acc(24) - acc(12)."""
    g1, g2 = acc[12] - acc[6], acc[24] - acc[12]
    if g1 <= 0:
        lab = "已飽和 (g1 <= 0)"
    elif g2 >= 0.75 * g1:
        lab = "持續上升"
    elif g2 >= 0.25 * g1:
        lab = "趨緩"
    else:
        lab = "飽和"
    return g1, g2, lab


def k_for(acc, target=95.0, ks=None):
    """Descriptive only: least-squares line in log2(k) through the given k; k where it reaches `target`."""
    ks = [k for k in acc if k >= 6] if ks is None else ks
    xs = [math.log2(k) for k in ks]
    ys = [acc[k] for k in ks]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    if b <= 0:
        return b, float("inf")
    e = (target - a) / b          # [review D1] no OverflowError on a nearly flat curve
    return b, (2 ** e if e < 60 else float("inf"))


# ---- report -------------------------------------------------------------------------------------------------------
print(f"Lab 16 part {part} aggregate | logs: {', '.join(paths)}")
for path, c in cfgs:
    print(f"  {path}: encs={c['encs']} icl={c['icl']} script_sha={c['script_sha']} env={c['env']}")
if len(envs) > 1 or len(shas) > 1:
    print(f"  WARNING: env / script_sha differ across logs: {envs} {shas}")
if any("FAKE" in e for e in envs):
    print("  WARNING: FAKE transformers stub in at least one log (TEST RUN)")
if missing_encs:
    print(f"  NOTE: encoders not covered: {missing_encs}")
print("Accuracy = exact counts pooled over folds x seeds x draws; held-out = domains never used to train. NO decision."
      + PART_TAG)
RK = P["reg_k"]
print("\n" + "=" * 110)

if part == "A":
    for s in P["shots"]:
        n_q = shot(encs[0], METHODS[0], RK, s)[1]
        print(f"=== event sentences, 5-way, k={RK}, s={s} example(s) per event type | queries pooled: {n_q} ===")
        print(f"{'encoder':<12}" + "".join(f"{m:>13}" for m in METHODS))
        for e in encs:
            print(f"{e:<12}" + "".join(fp(acc_shot(e, m, RK, s), 13) for m in METHODS))
    cand = [(acc_shot(e, m, RK, 1), e, m) for e in encs for m in ANCHOR_METHODS]   # [review D3]
    cand.sort(key=lambda x: (-x[0], P["encs"].index(x[1]), METHODS.index(x[2])))
    best_a, best_e, best_m = cand[0]
    bc, bt = shot(best_e, best_m, RK, 1)
    zc, zt = shot(best_e, "ZERO-MLP", RK, 1)
    print(f"\n--- registered quantities (s=1, k={RK}, event sentences; anchor methods only, ZERO-* and 7B-ICL are "
          f"references){PART_TAG} ---")
    print(f"  best combination: {best_e} + {best_m} = {best_a:.2f}% ({bc}/{bt})")
    exact_ties = [f"{e} + {m}" for a, e, m in cand[1:] if a == best_a]
    if exact_ties:
        print(f"  EXACT TIE with: {', '.join(exact_ties)}")
    zb = max((acc_shot(e, m, RK, 1), e, m) for e in encs for m in ("ZERO-MLP", "ZERO-FUNNEL"))
    print(f"  best reference without examples: {zb[1]} + {zb[2]} = {zb[0]:.2f}%")
    ties = [f"{e} + {m} {a:.2f}%" for a, e, m in cand[1:6]]
    print(f"  next five: {'; '.join(ties)}")
    print(f"  ZERO-MLP with the same encoder ({best_e}) on the same queries: {pct(zc, zt):.2f}% ({zc}/{zt})")
    print(f"  gain = {best_a - pct(zc, zt):+.2f} points")
    print(f"  per fold (pooled over seeds x draws): "
          + " ".join(f"f{f}={pct(*shot(best_e, best_m, RK, 1, ff=[f])):.2f}%" for f in FOLDS))
    print(f"  per seed: " + " ".join(
        f"{sd}={pct(sum(int(SHOT[(best_e, best_m, RK, 1, f, sd)]['correct']) for f in FOLDS), sum(int(SHOT[(best_e, best_m, RK, 1, f, sd)]['total']) for f in FOLDS)):.2f}%"
        for sd in SEEDS))
    print(f"\n--- gain over ZERO-MLP of the same encoder (points), s=1, k={RK} ---")
    print(f"{'encoder':<12}" + "".join(f"{m:>13}" for m in METHODS[1:]))
    for e in encs:
        z = acc_shot(e, "ZERO-MLP", RK, 1)
        print(f"{e:<12}" + "".join(f"{acc_shot(e, m, RK, 1) - z:13.2f}" for m in METHODS[1:]))
    print(f"\n--- k-curves, s=1 (%): " + " / ".join(f"k={k}" for k in P["ks"]) + " ---")
    for e in encs:
        for m in METHODS:
            print(f"  {e + ' + ' + m:<26}" + "".join(fp(acc_shot(e, m, k, 1)) for k in P["ks"]))
    print(f"\n--- s-curves at k={RK} (%): s=1 / 2 / 3 ---")
    for e in encs:
        for m in METHODS:
            print(f"  {e + ' + ' + m:<26}" + "".join(fp(acc_shot(e, m, RK, s)) for s in P["shots"]))
    print(f"\n--- ZERO-MLP 13-way on all held-out sentences (Lab 15's quantity), k={RK} ---")
    for e in encs:
        c, t, ec, et, ic, it = cv13(e, RK, ("correct", "total", "ev_correct", "ev_total", "init_correct", "init_total"))
        print(f"  {e:<12} all {pct(c, t):6.2f}% ({c}/{t}) | events {pct(ec, et):6.2f}% | init {pct(ic, it):6.2f}%")
    print(f"\n--- initial-state sentences, 8-way, s=1 draws, k={RK} (counts include the 10 draws: ZERO-MLP repeats "
          f"the same predictions 10 times) ---")
    for e in encs:
        cells = []
        for m in INIT_METHODS:
            c = sum(int(INIT[(e, m, RK, f, sd)]["correct"]) for f in FOLDS for sd in SEEDS)
            t = sum(int(INIT[(e, m, RK, f, sd)]["total"]) for f in FOLDS for sd in SEEDS)
            cells.append(f"{m} {pct(c, t):6.2f}% ({c}/{t})")
        print(f"  {e:<12} " + " | ".join(cells))
    print(f"\n--- hard cases on held-out domains, s=1, k={RK}: % correct [% literal event] (pooled over the 10 draws) ---")
    print(f"{'encoder + method':<28}{'NEG -> NOOP':>18}{'OTHER -> NOOP':>18}{'TEMP':>10}")
    for e in encs:
        for m in METHODS:
            cells = []
            for cat in HARD_CATS:
                c = sum(int(HARD[(e, m, f, sd, cat)]["correct"]) for f in FOLDS for sd in SEEDS)
                t = sum(int(HARD[(e, m, f, sd, cat)]["total"]) for f in FOLDS for sd in SEEDS)
                a = sum(int(HARD[(e, m, f, sd, cat)]["as_surface"]) for f in FOLDS for sd in SEEDS)
                cells.append(f"{pct(c, t):6.1f} [{pct(a, t):5.1f}]" if cat != "TEMP" else f"{pct(c, t):6.1f}")
            print(f"  {e + ' + ' + m:<26}{cells[0]:>18}{cells[1]:>18}{cells[2]:>10}")
    for e, m in ((best_e, best_m), (best_e, "ZERO-MLP")):
        mat = [[0] * 5 for _ in range(5)]
        for f in FOLDS:
            for sd in SEEDS:
                v = [int(x) for x in CONF[(e, m, f, sd)]["m"].split(",")]
                for i in range(25):
                    mat[i // 5][i % 5] += v[i]
        print(f"\n--- confusion, {e} + {m}, s=1, k={RK} (rows = true, columns = predicted) ---")
        print(f"{'':>8}" + "".join(f"{x:>8}" for x in EVENTS) + f"{'recall':>9}")
        for i, ev in enumerate(EVENTS):
            print(f"{ev:>8}" + "".join(f"{x:>8}" for x in mat[i]) + f"{pct(mat[i][i], sum(mat[i])):8.1f}%")
    if icl_on:
        print("\n--- Qwen2.5-7B-Instruct, 16 train domains x 2 draws, same queries; correct = right in both orders ---")
        for key in ("ICL", "ZS0"):
            rows = [kv for (t, _d, _r), kv in ICL.items() if t == key]
            b, o1, o2, n = (sum(int(r[x]) for r in rows) for x in ("correct", "order1", "order2", "total"))
            lab = "with the domain's 5 examples" if key == "ICL" else "no example (Lab 15 prompt)"
            print(f"  {key} ({lab}): {pct(b, n):.2f}% ({b}/{n}) | order1 {pct(o1, n):.2f}% | order2 {pct(o2, n):.2f}%")
    else:
        print("\n  7B reference not in these logs")

else:  # part B
    ks = P["ks"]
    if A_BEST is None:
        die("part B needs --a-best METHOD (the best anchor method of the part A aggregate; review D4)")
    print(f"=== REGISTERED shape rule: ZERO-MLP 13-way and the part A best method ({A_BEST}, s=1, events 5-way) ==="
          + PART_TAG)
    print(f"{'row':<34}" + "".join(f"{'k=' + str(k):>8}" for k in ks) + f"{'g1':>8}{'g2':>8}   shape")
    for e in encs:
        for lab_row, acc in ((f"{e} ZERO-MLP-13", {k: pct(*cv13(e, k)) for k in ks}),
                             (f"{e} {A_BEST}", {k: acc_shot(e, A_BEST, k, 1) for k in ks})):
            g1, g2, lab = shape(acc)
            print(f"{lab_row:<34}" + "".join(fp(acc[k]) for k in ks) + f"{g1:8.2f}{g2:8.2f}   {lab}")
    print("\n--- descriptive below (not registered) ---\n")
    print(f"=== ZERO-MLP 13-way on all held-out sentences (Lab 15's quantity) by number of training domains k ===")
    print(f"{'encoder':<12}" + "".join(f"{'k=' + str(k):>8}" for k in ks) + f"{'g1':>8}{'g2':>8}   shape")
    for e in encs:
        acc = {k: pct(*cv13(e, k)) for k in ks}
        g1, g2, lab = shape(acc)
        print(f"{e:<12}" + "".join(fp(acc[k]) for k in ks) + f"{g1:8.2f}{g2:8.2f}   {lab}")
    print("\n=== event sentences 5-way with s=1 example per event type, by k (each method) ===")
    for m in METHODS:
        print(f"--- {m} ---")
        for e in encs:
            acc = {k: acc_shot(e, m, k, 1) for k in ks}
            g1, g2, lab = shape(acc)
            print(f"  {e:<12}" + "".join(fp(acc[k]) for k in ks) + f"{g1:8.2f}{g2:8.2f}   {lab}")
    print("\n=== held-out registered (old) vs extra (new) domains, k=12 and k=24 ===")
    for e in encs:
        for k in (12, 24):
            co, to, cn, tn = cv13(e, k, ("c_old", "t_old", "c_new", "t_new"))
            line = f"  {e:<12} k={k:<3} ZERO-MLP-13 old {fp(pct(co, to), 6)} new {fp(pct(cn, tn), 6)}"
            for m in METHODS:
                co, to, cn, tn = shot(e, m, k, 1, ("c_old", "t_old", "c_new", "t_new"))
                line += f" | {m} {fp(pct(co, to), 6)}/{fp(pct(cn, tn), 6)}"
            print(line)
    print("\n=== descriptive: least-squares line in log2(k) over k >= 6; k where it reaches 95% ===")
    for e in encs:
        b, kk = k_for({k: pct(*cv13(e, k)) for k in ks})
        line = f"  {e:<12} ZERO-MLP-13 slope {b:5.2f}/doubling k95={kk:7.1f}"
        for m in METHODS:
            b, kk = k_for({k: acc_shot(e, m, k, 1) for k in ks})
            line += f" | {m} {kk:7.1f}"
        print(line)
