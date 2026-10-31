"""Average the V5_RESULT / V5_DIAG lines of several seed logs of the v5 confirmatory test (Lab 12, issue #21).
Modelled on tools/aggregate_v4.py. Numbers only, no verdict.
Usage: python tools/aggregate_v5.py seed42.log seed1.log seed2.log"""
import sys
from collections import defaultdict

N = 48  # unseen items per seed (Lab 12 asserts N_ITEMS == 48); a V5_DIAG line may carry its own n=
STAGES = (2, 3, 4)
res, diag, cfg = defaultdict(dict), defaultdict(dict), {}
seeds = []  # in order of first appearance
for path in sys.argv[1:]:
    for line in open(path, encoding="utf-8", errors="ignore"):
        if not line.startswith(("V5_RESULT ", "V5_DIAG ", "V5_CONFIG ")):
            continue
        kv = dict(x.split("=", 1) for x in line.split()[1:])
        seed = kv["seed"]
        if seed not in seeds:
            seeds.append(seed)
        if line.startswith("V5_CONFIG"):
            if seed in cfg and cfg[seed] != kv:
                sys.exit(f"ERROR: seed {seed} has two different V5_CONFIG lines: {cfg[seed]} vs {kv}")
            cfg[seed] = kv
            continue
        table, key = (res, kv["method"]) if line.startswith("V5_RESULT") else (diag, kv["name"])
        if seed in table[key]:
            sys.exit(f"ERROR: duplicate {line.split()[0]} for {key} seed {seed} ({path}); pass each seed's log once")
        table[key][seed] = kv

print("config:")
for s in seeds:
    c = cfg.get(s)
    print(f"  seed={s} " + (f"variant={c['variant']} mode={c['mode']} override={c.get('override', '?')}" if c
                            else "NO V5_CONFIG LINE (run incomplete?)"))
if len({(c["variant"], c["mode"]) for c in cfg.values()}) > 1:
    print("  WARNING: the seed logs use DIFFERENT HOP variant/mode settings")
if any(c.get("override") != "0" for c in cfg.values()):
    print("  WARNING: at least one log ran with an environment override (exploratory, not the registered config)")


def vals(d, key):
    """Per-seed integer values of `key` (seed order); None when missing or n/a (-1) in any seed."""
    v = [int(d[s][key]) for s in seeds if s in d]
    return v if v and min(v) >= 0 else None


def mean_pct(d, keys, n=N):
    v = [vals(d, k) for k in keys]
    if any(x is None for x in v):
        return None
    flat = [x for xs in v for x in xs]
    return 100.0 * sum(flat) / len(flat) / n


def fp(x, w=8):
    return f"{'-':>{w}}" if x is None else f"{x:{w - 1}.1f}%"


BR_K = [f"BR{k}" for k in STAGES]
DEC_K = [f"DEC{k}" for k in STAGES]
FL_K = [f"FLIP{k}" for k in STAGES]
print(f"\nper-method means across seeds (% of {N} unseen items; mean over the seeds present)")
cols = ["BR mean", "DEC mean", "BR-2", "BR-3", "BR-4", "DEC-2", "DEC-3", "DEC-4", "FLIP-2", "FLIP-3", "FLIP-4", "FLIP",
        "PREM"]
print(f"{'method':<16}{'seeds':>6}" + "".join(f"{c:>9}" for c in cols))
for m, d in res.items():
    row = [mean_pct(d, BR_K), mean_pct(d, DEC_K)]
    row += [mean_pct(d, [k]) for k in BR_K + DEC_K + FL_K]
    row += [mean_pct(d, FL_K), mean_pct(d, ["PREM"])]
    print(f"{m:<16}{len(d):>6}" + "".join(fp(x, 9) for x in row))

print(f"\nper-seed values (counts out of {N}; -1 = n/a)")
keys = BR_K + DEC_K + ["A2", "A3", "A4"] + FL_K + ["PREM"]
print(f"{'method':<16}{'seed':>6}" + "".join(f"{k:>7}" for k in keys) + f"{'BR':>9}{'DEC':>9}")
for m, d in res.items():
    for s in seeds:
        if s not in d:
            print(f"{m:<16}{s:>6}  (missing)")
            continue
        r = d[s]
        br = [int(r[k]) for k in BR_K]
        dec = [int(r[k]) for k in DEC_K]
        print(f"{m:<16}{s:>6}" + "".join(f"{int(r.get(k, -1)):>7}" for k in keys)
              + fp(100.0 * sum(br) / len(br) / N if min(br) >= 0 else None, 9)
              + fp(100.0 * sum(dec) / len(dec) / N if min(dec) >= 0 else None, 9))

print("\ndiagnostics (secondary):")
for name, d in diag.items():
    ns = {int(r.get("n", N)) for r in d.values()}
    n = ns.pop() if len(ns) == 1 else None
    if n is None:
        print(f"  {name:<28} ERROR: different n across seeds")
        continue
    parts = []
    for k in BR_K:
        v = vals(d, k)
        parts.append(f"{k}={'/'.join(map(str, v))} ({100 * sum(v) / len(v) / n:4.1f}%)" if v else f"{k}=-")
    m = mean_pct(d, BR_K, n)
    print(f"  {name:<28} n={n:<4} seeds={len(d)}  " + "  ".join(parts) + f"  mean {fp(m, 6).strip()}")
