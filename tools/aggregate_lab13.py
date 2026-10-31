"""Aggregate the L13_RESULT / L13_NOISE / L13_PERT / L13_LONG / L13_CONFIG lines of Lab 13
(scripts/stage2_lab13_space_steps_factorial.py, exploratory) over the seeds given. Numbers only, no verdict.

Usage: python tools/aggregate_lab13.py seed42.log seed1.log seed2.log [--seeds 42,1,2] [--allow-missing]

Seeds = every seed with an L13_CONFIG line in the logs (or the --seeds list, in that order). Every method / grid point
must be present for every seed (error otherwise; --allow-missing turns that into a warning and averages over the seeds
that have the value). A configuration that differs across seeds (data sha, script sha, environment) is a warning.
DEC / BR / BRa / A / seenDEC: exact counts summed over seeds and stages, printed as % with the count / denominator
(3 seeds x 3 stages x 48 items = 432). Stress tests: each log line is already a mean over K draws; the aggregator
averages those means over seeds.
"""
import sys
from collections import defaultdict

N, N_SEEN = 48, 120   # unseen items / seen chains per seed (the Lab 13 script asserts both)
STAGES = (2, 3, 4)
SPACES = ("F64", "P64", "R896")
ORDER = [("F64", "HOP"), ("F64", "ONE-STEP"), ("F64", "X1-ONLY"),
         ("P64", "HOP"), ("P64", "ONE-STEP"), ("P64", "X1-ONLY"),
         ("R896", "HOP"), ("R896", "HOP-wide"), ("R896", "ONE-STEP"), ("R896", "X1-ONLY"),
         ("JOINT896", "PREMISE-MLP")]

argv = sys.argv[1:]
ALLOW_MISSING = "--allow-missing" in argv
want = None
paths = []
i = 0
while i < len(argv):
    a = argv[i]
    if a == "--seeds":
        want = [s.strip() for s in argv[i + 1].split(",") if s.strip()]
        i += 2
        continue
    if a.startswith("--seeds="):
        want = [s.strip() for s in a.split("=", 1)[1].split(",") if s.strip()]
    elif a != "--allow-missing":
        paths.append(a)
    i += 1
if not paths:
    sys.exit(__doc__)

cfg = {}
res = defaultdict(dict)                                      # (space, method) -> seed -> kv
noise = defaultdict(lambda: defaultdict(dict))               # (space, method) -> alpha -> seed -> [DEC2, DEC3, DEC4]
pert = defaultdict(lambda: defaultdict(dict))                # (space, method, kind) -> beta -> seed -> [DEC3, DEC4]
long_ = defaultdict(lambda: defaultdict(dict))               # (space, method) -> step -> seed -> (cos_near, law_keep)


def put(table, key, seed, val, line, path):
    if seed in table[key]:
        sys.exit(f"ERROR: duplicate line for {key} seed {seed} ({path}): {line.strip()}\n"
                 f"       pass each seed's log once")
    table[key][seed] = val


for path in paths:
    for line in open(path, encoding="utf-8", errors="ignore"):
        tag = line.split(" ", 1)[0]
        if tag not in ("L13_CONFIG", "L13_RESULT", "L13_NOISE", "L13_PERT", "L13_LONG"):
            continue
        kv = dict(x.split("=", 1) for x in line.split()[1:])
        seed = kv["seed"]
        if tag == "L13_CONFIG":
            if seed in cfg and cfg[seed] != kv:
                sys.exit(f"ERROR: seed {seed} has two different L13_CONFIG lines: {cfg[seed]} vs {kv}")
            cfg[seed] = kv
        elif tag == "L13_RESULT":
            put(res, (kv["space"], kv["method"]), seed, kv, line, path)
        elif tag == "L13_NOISE":
            put(noise[(kv["space"], kv["method"])], float(kv["alpha"]), seed,
                [float(kv[f"DEC{k}"]) for k in STAGES], line, path)
        elif tag == "L13_PERT":
            put(pert[(kv["space"], kv["method"], kv["kind"])], float(kv["beta"]), seed,
                [float(kv["DEC3"]), float(kv["DEC4"])], line, path)
        else:
            put(long_[(kv["space"], kv["method"])], int(kv["step"]), seed,
                (float(kv["cos_near"]), float(kv["law_keep"])), line, path)

problems = []
seeds = list(want) if want is not None else list(cfg)
for s in seeds:
    if s not in cfg:
        problems.append(f"seed {s}: no L13_CONFIG line in the logs given")
for key in ORDER:
    if key not in res:
        problems.append(f"L13_RESULT {key[0]}/{key[1]} absent from every log")
for name, table in ([(f"L13_RESULT {k}", v) for k, v in res.items()]
                    + [(f"L13_NOISE {k} alpha={a:g}", v) for k, g in noise.items() for a, v in g.items()]
                    + [(f"L13_PERT {k} beta={b:g}", v) for k, g in pert.items() for b, v in g.items()]
                    + [(f"L13_LONG {k} step={t}", v) for k, g in long_.items() for t, v in g.items()]):
    miss = [s for s in seeds if s not in table]
    if miss:
        problems.append(f"{name}: missing seed(s) {miss}")
for k in ("data_sha", "script_sha", "env"):
    vs = {cfg[s].get(k, "?") for s in seeds if s in cfg}
    if len(vs) > 1:
        print(f"WARNING: L13_CONFIG {k} differs across seeds: {sorted(vs)}")

print("config:")
for s in seeds:
    c = cfg.get(s, {})
    print(f"  seed={s} " + " ".join(f"{k}={c.get(k, '?')}" for k in ("data_sha", "script_sha", "env")))
if problems:
    for p in problems[:20]:
        print(("WARNING: " if ALLOW_MISSING else "ERROR: ") + p)
    if len(problems) > 20:
        print(f"... and {len(problems) - 20} more")
    if not ALLOW_MISSING:
        sys.exit("aggregate_lab13: incomplete logs (pass --allow-missing to average over the seeds present)")
print(f"seeds: {', '.join(seeds)} ({len(seeds)})")


def total(key, metric, stages=STAGES, ss=None):
    """Exact count summed over stages and seeds (default: all seeds present); (count, denominator) or None if n/a."""
    d = res.get(key, {})
    ss = [s for s in (seeds if ss is None else ss) if s in d]
    if not ss:
        return None
    v = [int(d[s][f"{metric}{k}"]) for s in ss for k in stages]
    if min(v) < 0:
        return None
    return sum(v), len(ss) * len(stages) * (N_SEEN if metric == "seenDEC" else N)


def pc(t, w=20):
    """'pct% (cnt/den)' or '-'."""
    if t is None:
        return f"{'-':>{w}}"
    return f"{f'{100.0 * t[0] / t[1]:.2f}% ({t[0]}/{t[1]})':>{w}}"


def diff_pts(t1, t2, w=10):
    if t1 is None or t2 is None:
        return f"{'-':>{w}}"
    return f"{100.0 * t1[0] / t1[1] - 100.0 * t2[0] / t2[1]:>+{w}.2f}"


# ---- factorial table ----
den = len(seeds) * len(STAGES) * N
print(f"\n=== FACTORIAL (means over {len(seeds)} seed(s), 48 unseen items, stages 2-4; exact counts / {den}) ===")
print("DEC = picker-free pairwise decision (primary); BR = DEC law -> common sentence-MLP picker; "
      "delta = HOP - ONE-STEP (points)")
hdr = (f"{'row':<28}{'HOP DEC':>20}{'ONE-STEP DEC':>20}{'delta DEC':>10}{'HOP BR':>20}{'ONE-STEP BR':>20}"
       f"{'delta BR':>10}")
print(hdr + "\n" + "-" * len(hdr))
ROWS = [(f"{sp}", (sp, "HOP"), (sp, "ONE-STEP")) for sp in SPACES]
ROWS += [("R896 (HOP-wide)", ("R896", "HOP-wide"), ("R896", "ONE-STEP")),
         ("JOINT896 (PREMISE-MLP ref)", None, ("JOINT896", "PREMISE-MLP"))]
for label, hk, ok in ROWS:
    hd, od = (total(hk, "DEC") if hk else None), total(ok, "DEC")
    hb, ob = (total(hk, "BR") if hk else None), total(ok, "BR")
    print(f"{label:<28}{pc(hd)}{pc(od)}{diff_pts(hd, od)}{pc(hb)}{pc(ob)}{diff_pts(hb, ob)}")
print("(the PREMISE-MLP reference row is shown in the ONE-STEP columns: it is a one-decision method on the joint text)")

# ---- all methods, all metrics ----
print(f"\n=== all methods (means over seeds; % with exact counts) ===")
cols = ("DEC", "BR", "BRa", "A", "seenDEC")
hdr = f"{'space':<9}{'method':<12}{'params':>12}" + "".join(f"{c:>22}" for c in cols)
print(hdr + "\n" + "-" * len(hdr))
for key in ORDER:
    if key not in res:
        continue
    ps = sorted({res[key][s]["params"] for s in seeds if s in res[key]})
    print(f"{key[0]:<9}{key[1]:<12}{'/'.join(ps):>12}" + "".join(pc(total(key, c), 22) for c in cols))
print("per stage (DEC-2 / DEC-3 / DEC-4, % of seeds x 48):")
for key in ORDER:
    if key in res:
        print(f"  {key[0]:<9}{key[1]:<12}" + "".join(pc(total(key, "DEC", (k,)), 20) for k in STAGES))
x1 = [(key, s, k, int(res[key][s][f"DEC{k}"])) for key in ORDER if key[1] == "X1-ONLY" and key in res
      for s in seeds if s in res[key] for k in STAGES]
bad = [x for x in x1 if x[3] * 2 != N]
print(f"sanity: X1-ONLY DEC == {N // 2}/{N} for every space / seed / stage: {'OK' if not bad else f'FAIL {bad}'}")


def mean(vals):
    return sum(vals) / len(vals) if vals else None


# ---- T1 noise curves ----
if noise:
    alphas = sorted({a for g in noise.values() for a in g})
    print(f"\n=== T1 noise curves: DEC mean (% of 48; mean over stages 2-4, K draws and seeds) vs alpha ===")
    hdr = f"{'space':<9}{'method':<12}" + "".join(f"{f'alpha={a:g}':>12}" for a in alphas)
    print(hdr + "\n" + "-" * len(hdr))
    keys = [k for k in ORDER if k in noise] + [k for k in noise if k not in ORDER]
    for key in keys:
        row = []
        for a in alphas:
            v = [x for s in seeds if s in noise[key].get(a, {}) for x in noise[key][a][s]]
            m = mean(v)
            row.append(f"{'-':>12}" if m is None else f"{100.0 * m / N:>11.2f}%")
        print(f"{key[0]:<9}{key[1]:<12}" + "".join(row))
    print("per stage (mean DEC-2/DEC-3/DEC-4 counts out of 48):")
    for key in keys:
        row = []
        for a in alphas:
            per = [mean([noise[key][a][s][j] for s in seeds if s in noise[key].get(a, {})]) for j in range(3)]
            row.append(f"{'-':>21}" if None in per else f"{'/'.join(f'{x:.2f}' for x in per):>21}")
        print(f"  {key[0]:<9}{key[1]:<12}" + "".join(row))

# ---- T2 perturbation curves ----
if pert:
    betas = sorted({b for g in pert.values() for b in g})
    print(f"\n=== T2 perturbation curves: mean DEC-3 / DEC-4 counts (out of 48) and [mean %] vs beta ===")
    hdr = f"{'space':<9}{'method':<12}{'u':<6}" + "".join(f"{f'beta={b:g}':>24}" for b in betas)
    print(hdr + "\n" + "-" * len(hdr))
    for key in sorted(pert, key=lambda k: (ORDER.index(k[:2]) if k[:2] in ORDER else 99, k[2])):
        row = []
        for b in betas:
            per = [mean([pert[key][b][s][j] for s in seeds if s in pert[key].get(b, {})]) for j in range(2)]
            row.append(f"{'-':>24}" if None in per else
                       f"{f'{per[0]:.2f}/{per[1]:.2f} [{100.0 * (per[0] + per[1]) / (2 * N):.2f}%]':>24}")
        print(f"{key[0]:<9}{key[1]:<12}{key[2]:<6}" + "".join(row))

# ---- T3 long-rollout curves ----
if long_:
    steps = sorted({t for g in long_.values() for t in g})
    print(f"\n=== T3 long-rollout curves (means over seeds): cos_near = mean cosine to the nearest anchor; law_keep = "
          f"fraction of items whose nearest anchor belongs to the DEC-2 law ===")
    hdr = f"{'space':<9}{'method':<12}{'':<10}" + "".join(f"{f't={t}':>8}" for t in steps)
    print(hdr + "\n" + "-" * len(hdr))
    for key in [k for k in ORDER if k in long_] + [k for k in long_ if k not in ORDER]:
        for j, name in enumerate(("cos_near", "law_keep")):
            row = []
            for t in steps:
                m = mean([long_[key][t][s][j] for s in seeds if s in long_[key].get(t, {})])
                row.append(f"{'-':>8}" if m is None else f"{m:>8.4f}")
            print(f"{(key[0] if j == 0 else ''):<9}{(key[1] if j == 0 else ''):<12}{name:<10}" + "".join(row))

# ---- per-seed values ----
print(f"\n=== per-seed factorial values (counts out of {3 * N} = 3 stages x {N}) ===")
hdr = f"{'row':<28}{'seed':>6}{'HOP DEC':>20}{'ONE-STEP DEC':>20}{'HOP BR':>20}{'ONE-STEP BR':>20}"
print(hdr + "\n" + "-" * len(hdr))
for label, hk, ok in ROWS:
    for s in seeds:
        print(f"{label:<28}{s:>6}"
              + pc(total(hk, "DEC", ss=[s]) if hk else None) + pc(total(ok, "DEC", ss=[s]))
              + pc(total(hk, "BR", ss=[s]) if hk else None) + pc(total(ok, "BR", ss=[s])))
print(f"\n=== per-seed L13_RESULT values (counts out of {N}; seenDEC out of {N_SEEN}; -1 = n/a) ===")
keys = [f"{m}{k}" for m in ("DEC", "BR", "BRa", "A", "seenDEC") for k in STAGES]
print(f"{'space':<9}{'method':<12}{'seed':>5}" + "".join(f"{k:>9}" for k in keys) + f"{'params':>12}")
for key in ORDER:
    for s in seeds:
        r = res.get(key, {}).get(s)
        if r is None:
            continue
        print(f"{key[0]:<9}{key[1]:<12}{s:>5}" + "".join(f"{int(r[k]):>9}" for k in keys) + f"{r['params']:>12}")
