"""Aggregate the V5_RESULT / V5_DIAG / V5_CONFIG lines of the three seed logs of the v5 confirmatory test
(Lab 12, issue #21). Numbers only, no verdict: the pre-registered criteria are applied by the reader.

Usage: python tools/aggregate_v5.py seed42.log seed1.log seed2.log [--exploratory]

Strict by default (review S5): the seeds must be exactly {42, 1, 2}; every method and diagnostic must be present
for every seed; all seeds must share one configuration (variant, mode, data sha, script sha, environment) and none
may use an environment override. --exploratory turns these errors into warnings.
Derived quantities (review S1) are computed from exact counts and printed with 2 decimals plus the counts, so a
threshold can never be misread from a rounded mean.
"""
import sys
from collections import defaultdict

N = 48            # unseen items per seed (Lab 12 asserts N_ITEMS == 48); a V5_DIAG line may carry its own n=
STAGES = (2, 3, 4)
REGISTERED_SEEDS = ["42", "1", "2"]
METHOD_ORDER = ["hop", "random_hop", "v0_joint", "last", "s0_rule", "premise_mlp", "qwen_zeroshot", "qwen_fewshot"]

args = [a for a in sys.argv[1:] if not a.startswith("--")]
EXPLORATORY = "--exploratory" in sys.argv[1:]
problems = []

res, diag, cfg = defaultdict(dict), defaultdict(dict), {}
for path in args:
    for line in open(path, encoding="utf-8", errors="ignore"):
        if not line.startswith(("V5_RESULT ", "V5_DIAG ", "V5_CONFIG ")):
            continue
        kv = dict(x.split("=", 1) for x in line.split()[1:])
        seed = kv["seed"]
        if line.startswith("V5_CONFIG"):
            if seed in cfg and cfg[seed] != kv:
                sys.exit(f"ERROR: seed {seed} has two different V5_CONFIG lines: {cfg[seed]} vs {kv}")
            cfg[seed] = kv
            continue
        table, key = (res, kv["method"]) if line.startswith("V5_RESULT") else (diag, kv["name"])
        if seed in table[key]:
            sys.exit(f"ERROR: duplicate {line.split()[0]} for {key} seed {seed} ({path}); pass each seed's log once")
        table[key][seed] = kv

seeds = [s for s in REGISTERED_SEEDS if s in cfg] + sorted(s for s in cfg if s not in REGISTERED_SEEDS)
if sorted(cfg) != sorted(REGISTERED_SEEDS):
    problems.append(f"seeds with a V5_CONFIG line: {sorted(cfg)}; registered: {REGISTERED_SEEDS}")
for name, table in list(res.items()) + list(diag.items()):
    missing = [s for s in seeds if s not in table]
    if missing:
        problems.append(f"{name}: missing seed(s) {missing}")
for m in METHOD_ORDER:
    if m not in res:
        problems.append(f"method {m} absent from every log")
CFG_KEYS = ("variant", "mode", "data_sha", "script_sha", "env")
for k in CFG_KEYS:
    vs = {c.get(k, "?") for c in cfg.values()}
    if len(vs) > 1:
        problems.append(f"V5_CONFIG {k} differs across seeds: {sorted(vs)}")
if any(c.get("override") != "0" for c in cfg.values()):
    problems.append("at least one log ran with an environment override (not the registered configuration)")

print("config:")
for s in seeds:
    c = cfg[s]
    print(f"  seed={s} " + " ".join(f"{k}={c.get(k, '?')}" for k in ("variant", "mode", "override") + CFG_KEYS[2:]))
if problems:
    for p in problems:
        print(("WARNING: " if EXPLORATORY else "ERROR: ") + p)
    if not EXPLORATORY:
        sys.exit("aggregate_v5: refusing to aggregate a non-registered run (pass --exploratory to override)")

BR_K = [f"BR{k}" for k in STAGES]
DEC_K = [f"DEC{k}" for k in STAGES]
FL_K = [f"FLIP{k}" for k in STAGES]
A_K = [f"A{k}" for k in STAGES]


def total(d, keys, ss=None):
    """Exact count summed over `keys` and seeds `ss` (default all); None if any value is n/a (-1) or missing."""
    ss = seeds if ss is None else ss
    v = [int(d[s][k]) for s in ss for k in keys if s in d and k in d[s]]
    return None if len(v) != len(ss) * len(keys) or min(v) < 0 else sum(v)


def pct(cnt, denom):
    return None if cnt is None else 100.0 * cnt / denom


def fp(x, w=9, dec=2):
    return f"{'-':>{w}}" if x is None else f"{x:{w - 1}.{dec}f}%"


methods = [m for m in METHOD_ORDER if m in res] + [m for m in res if m not in METHOD_ORDER]
S3 = len(seeds) * len(STAGES)

print(f"\n=== per-method means over {len(seeds)} seeds (% of {N} unseen items; exact counts / {S3 * N} in brackets) ===")
cols = ["BR mean", "DEC mean", "BR-2", "BR-3", "BR-4", "DEC-2", "DEC-3", "DEC-4", "FLIP", "PREM"]
print(f"{'method':<15}" + "".join(f"{c:>10}" for c in cols) + "   BR count")
for m in methods:
    d = res[m]
    br = total(d, BR_K)
    row = [pct(br, S3 * N), pct(total(d, DEC_K), S3 * N)]
    row += [pct(total(d, [k]), len(seeds) * N) for k in BR_K + DEC_K]
    row += [pct(total(d, FL_K), S3 * N), pct(total(d, ["PREM"]), len(seeds) * N)]
    print(f"{m:<15}" + "".join(fp(x, 10) for x in row) + f"   {'-' if br is None else f'{br}/{S3 * N}'}")

print(f"\n=== per-seed values (counts out of {N}; -1 = n/a; BR/DEC = mean of stages 2-4, count out of {3 * N}) ===")
keys = BR_K + DEC_K + A_K + FL_K + ["PREM"]
print(f"{'method':<15}{'seed':>5}" + "".join(f"{k:>7}" for k in keys) + f"{'BR':>10}{'(cnt)':>9}{'DEC':>10}")
for m in methods:
    for s in seeds:
        r = res[m].get(s)
        if r is None:
            print(f"{m:<15}{s:>5}  (missing)")
            continue
        br, dec = total(res[m], BR_K, [s]), total(res[m], DEC_K, [s])
        print(f"{m:<15}{s:>5}" + "".join(f"{int(r.get(k, -1)):>7}" for k in keys)
              + fp(pct(br, 3 * N), 10) + f"{'' if br is None else f'{br}/{3 * N}':>9}" + fp(pct(dec, 3 * N), 10))

print("\n=== diagnostics (secondary), per seed ===")
print(f"{'name':<28}{'seed':>5}{'n':>5}" + "".join(f"{k:>7}" for k in BR_K) + f"{'mean':>10}{'(cnt)':>10}")
for name, d in diag.items():
    for s in seeds:
        r = d.get(s)
        if r is None:
            print(f"{name:<28}{s:>5}  (missing)")
            continue
        n = int(r.get("n", N))
        t = total(d, BR_K, [s])
        print(f"{name:<28}{s:>5}{n:>5}" + "".join(f"{int(r[k]):>7}" for k in BR_K)
              + fp(pct(t, 3 * n), 10) + f"{'' if t is None else f'{t}/{3 * n}':>10}")
    ns = {int(r.get("n", N)) for r in d.values()}
    if len(ns) == 1:
        n = ns.pop()
        t = total(d, BR_K)
        print(f"{name:<28}{'mean':>5}{n:>5}{'':>21}" + fp(pct(t, S3 * n), 10) + f"{'' if t is None else f'{t}/{S3 * n}':>10}")

# ---- derived quantities for the pre-registered criteria (exact counts; NO verdict) ----
print("\n=== quantities named in the pre-registration (exact; the criteria are applied by the reader) ===")


def br_count(m, ss=None):
    return total(res[m], BR_K, ss) if m in res else None


def line(label, cnt, denom):
    print(f"  {label:<58} {fp(pct(cnt, denom), 9).strip():>9}   ({'-' if cnt is None else f'{cnt}/{denom}'})")


for s in seeds:
    line(f"HOP BR, seed {s}", br_count("hop", [s]), 3 * N)
line("HOP BR, mean over seeds", br_count("hop"), S3 * N)
line("RANDOM HOP BR, mean over seeds", br_count("random_hop"), S3 * N)
if "seen_hop" in diag:
    for s in seeds:
        r = diag["seen_hop"].get(s)
        if r is not None:
            n = int(r.get("n", N))
            line(f"HOP seen-domain BR, seed {s} (stages {'/'.join(r[k] for k in BR_K)} of {n})",
                 total(diag["seen_hop"], BR_K, [s]), 3 * n)
for m in ("last",):
    for s in seeds:
        b, dct = total(res[m], BR_K, [s]), total(res[m], DEC_K, [s])
        per = " ".join(f"{k}={res[m][s][k]}" for k in BR_K + DEC_K) if s in res[m] else "-"
        print(f"  LAST seed {s}: {per}   (exactly 50% requires every value == {N // 2})")
h, p, v0 = br_count("hop"), br_count("premise_mlp"), br_count("v0_joint")
if None not in (h, p):
    print(f"  D = HOP BR - PREMISE-MLP BR (points)                       {100.0 * (h - p) / (S3 * N):+9.2f}   "
          f"(counts {h} - {p} = {h - p:+d} of {S3 * N})")
if None not in (h, v0):
    print(f"  HOP BR - V0 BR (points)                                    {100.0 * (h - v0) / (S3 * N):+9.2f}   "
          f"(counts {h} - {v0} = {h - v0:+d} of {S3 * N})")
for s in seeds:
    hs, ps = br_count("hop", [s]), br_count("premise_mlp", [s])
    if None not in (hs, ps):
        print(f"  D seed {s} (points)                                         {100.0 * (hs - ps) / (3 * N):+9.2f}   "
              f"(counts {hs} - {ps} = {hs - ps:+d} of {3 * N})")
