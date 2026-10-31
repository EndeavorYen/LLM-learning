"""Aggregate the machine lines of Lab 14 (scripts/stage2_lab14_state_tracking.py; pre-registered on issue #23) over
the three registered seeds 42, 1, 2. Numbers only, no verdict.

Usage: python tools/aggregate_lab14.py seed42.log seed1.log seed2.log

Refuses (exit status 1):
  - a missing or extra seed (the seeds with an L14_CONFIG line must be exactly 42, 1, 2; every machine line must
    belong to one of them);
  - a data_sha that differs across seeds;
  - a duplicate (the same L14_CONFIG / L14_ACC / L14_SUM / L14_STEP / L14_FIT key twice, e.g. a log passed twice);
  - an incomplete set of lines (every method / mode / split / n / step of the script must be present for every seed);
  - totals per (split, n) or per step that differ across methods or seeds (every method is scored on the same stories);
  - an L14_SUM line that disagrees with its L14_ACC lines; L14_PARAMS that differ across logs.
A script_sha or env that differs across seeds is printed as a WARNING.

Every quantity is computed from exact counts summed over the 3 seeds: per n, acc_n = sum(correct) / sum(total);
SHORT = mean of acc_n over n = 1..4, LONG = mean over n = 6..10 (equal to the pooled count when every n has the same
number of stories, which is printed). Values in % with 2 decimals; differences in points.
"""
import sys
from collections import defaultdict

REQUIRED_SEEDS = ("42", "1", "2")
SPLITS = ("test", "val_seen")
ALL_N = tuple(range(1, 11))
SHORT_N, N5, LONG_N = (1, 2, 3, 4), 5, (6, 7, 8, 9, 10)
STEPS_T = tuple(range(0, 11))
HOPS = ("HOP-R896-d896", "HOP-R896-d64", "HOP-F64-d64", "HOP-F64-d896")
METHODS = [(h, m) for h in HOPS for m in ("free", "snap")] + [
    ("ONE-STEP-TF", "-"), ("ONE-STEP-QWEN", "-"), ("SYMBOLIC", "free"), ("SYMBOLIC", "snap"),
    ("BAG", "-"), ("LAST", "-"), ("MAJORITY", "-")]
STEP_METHODS = [mm for mm in METHODS if mm[0].startswith("HOP-") or mm[0] == "SYMBOLIC"]
PRIMARY = ("HOP-R896-d896", "free")
TAGS = ("L14_CONFIG", "L14_ACC", "L14_SUM", "L14_STEP", "L14_FIT", "L14_PARAMS")


def die(msg: str):
    print(f"ERROR: {msg}")
    sys.exit("aggregate_lab14: refused (see ERROR above)")


paths = sys.argv[1:]
if not paths or any(p.startswith("-") for p in paths):
    sys.exit(__doc__)

cfg = {}                                   # seed -> kv
acc = {}                                   # (seed, split, method, mode, n) -> (correct, total)
sums = {}                                  # (seed, split, method, mode) -> (short, n5, long)
steps = {}                                 # (seed, method, mode, t) -> (correct, total)
fit = {}                                   # (seed, method, mode) -> (correct, total, acc)
params = defaultdict(set)                  # method -> {params}


def put(table, key, val, line, path):
    if key in table:
        die(f"duplicate line {key} in {path}: {line.strip()}  (pass each seed's log once)")
    table[key] = val


for path in paths:
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            tag = line.split(" ", 1)[0]
            if tag not in TAGS:
                continue
            try:
                kv = dict(x.split("=", 1) for x in line.split()[1:])
            except ValueError:
                die(f"malformed line in {path}: {line.strip()}")
            if tag == "L14_PARAMS":
                params[kv["method"]].add(kv["params"])
                continue
            seed = kv["seed"]
            if tag == "L14_CONFIG":
                put(cfg, seed, kv, line, path)
            elif tag == "L14_ACC":
                put(acc, (seed, kv["split"], kv["method"], kv["mode"], int(kv["n"])),
                    (int(kv["correct"]), int(kv["total"])), line, path)
            elif tag == "L14_SUM":
                put(sums, (seed, kv["split"], kv["method"], kv["mode"]),
                    (float(kv["short"]), float(kv["n5"]), float(kv["long"])), line, path)
            elif tag == "L14_STEP":
                put(steps, (seed, kv["method"], kv["mode"], int(kv["t"])),
                    (int(kv["correct"]), int(kv["total"])), line, path)
            elif tag == "L14_FIT":
                put(fit, (seed, kv["method"], kv["mode"]),
                    (int(kv["correct"]), int(kv["total"]), float(kv["acc"])), line, path)

# ---- refusals ----
seeds_cfg = set(cfg)
seeds_any = seeds_cfg | {k[0] for k in acc} | {k[0] for k in sums} | {k[0] for k in steps} | {k[0] for k in fit}
missing = [s for s in REQUIRED_SEEDS if s not in seeds_cfg]
extra = sorted(seeds_any - set(REQUIRED_SEEDS))
if missing:
    die(f"missing seed(s) {missing}: no L14_CONFIG line (incomplete run or log not passed); required {REQUIRED_SEEDS}")
if extra:
    die(f"extra seed(s) {extra}; required exactly {REQUIRED_SEEDS}")
shas = {s: cfg[s].get("data_sha") for s in REQUIRED_SEEDS}
if len(set(shas.values())) != 1 or None in shas.values():
    die(f"data_sha differs across seeds: {shas}")
for k in ("script_sha", "env"):
    vs = {s: cfg[s].get(k, "?") for s in REQUIRED_SEEDS}
    if len(set(vs.values())) > 1:
        print(f"WARNING: L14_CONFIG {k} differs across seeds: {vs}")

problems = []
for s in REQUIRED_SEEDS:
    for (m, mo) in METHODS:
        for sp in SPLITS:
            problems += [f"L14_ACC seed={s} split={sp} {m}/{mo} n={n}" for n in ALL_N if (s, sp, m, mo, n) not in acc]
            if (s, sp, m, mo) not in sums:
                problems.append(f"L14_SUM seed={s} split={sp} {m}/{mo}")
        if (s, m, mo) not in fit:
            problems.append(f"L14_FIT seed={s} {m}/{mo}")
    for (m, mo) in STEP_METHODS:
        problems += [f"L14_STEP seed={s} {m}/{mo} t={t}" for t in STEPS_T if (s, m, mo, t) not in steps]
known = set(METHODS)
problems += [f"unexpected method/mode in L14_ACC: {k[2]}/{k[3]}" for k in acc if (k[2], k[3]) not in known]
problems += [f"unexpected split in L14_ACC: {k[1]}" for k in acc if k[1] not in SPLITS]
problems += [f"unexpected L14_STEP method/mode/t: {k[1:]}" for k in steps
             if (k[1], k[2]) not in set(STEP_METHODS) or k[3] not in STEPS_T]
if problems:
    for p in problems[:20]:
        print(f"ERROR: missing/unexpected {p}")
    die(f"{len(problems)} missing or unexpected line(s)")

for sp in SPLITS:  # same stories for every method and seed
    for n in ALL_N:
        tots = {acc[(s, sp, m, mo, n)][1] for s in REQUIRED_SEEDS for (m, mo) in METHODS}
        if len(tots) != 1:
            die(f"totals differ across methods/seeds for split={sp} n={n}: {sorted(tots)}")
for t in STEPS_T:
    tots = {steps[(s, m, mo, t)][1] for s in REQUIRED_SEEDS for (m, mo) in STEP_METHODS}
    if len(tots) != 1:
        die(f"per-step totals differ for t={t}: {sorted(tots)}")
for (s, sp, m, mo), (sh, f5, lg) in sums.items():  # L14_SUM must equal its L14_ACC lines
    fr = {n: acc[(s, sp, m, mo, n)][0] / acc[(s, sp, m, mo, n)][1] for n in ALL_N}
    ref = (sum(fr[n] for n in SHORT_N) / 4, fr[N5], sum(fr[n] for n in LONG_N) / 5)
    if any(abs(a - b) > 6e-5 for a, b in zip((sh, f5, lg), ref)):
        die(f"L14_SUM seed={s} split={sp} {m}/{mo} = {(sh, f5, lg)} disagrees with its L14_ACC lines {ref}")
for m, vs in params.items():
    if len(vs) != 1:
        die(f"L14_PARAMS differ across logs for {m}: {sorted(vs)}")

# ---- aggregation from exact counts ----
S = REQUIRED_SEEDS


def cnt(sp, mm, n):
    return (sum(acc[(s, sp, mm[0], mm[1], n)][0] for s in S), sum(acc[(s, sp, mm[0], mm[1], n)][1] for s in S))


def val(sp, mm, ns):
    """(mean over n of the seed-pooled per-n accuracy in %, pooled correct, pooled total)"""
    cs = [cnt(sp, mm, n) for n in ns]
    return 100.0 * sum(c / t for c, t in cs) / len(cs), sum(c for c, _ in cs), sum(t for _, t in cs)


def fmt(v) -> str:
    return f"{v[0]:.2f}% ({v[1]}/{v[2]})"


def mlabel(mm) -> str:
    return f"{mm[0]}/{mm[1]}"


print("config:")
for s in S:
    print(f"  seed={s} " + " ".join(f"{k}={cfg[s].get(k, '?')}" for k in ("data_sha", "script_sha", "env")))
print(f"seeds: {', '.join(S)} (3) | all quantities from exact counts summed over the 3 seeds")
for sp in SPLITS:
    per = [cnt(sp, METHODS[0], n)[1] for n in ALL_N]
    note = "equal for every n: mean over n = pooled count" if len(set(per)) == 1 else "NOT equal: mean over n != pooled"
    print(f"  {sp}: stories per n summed over seeds = {per} ({note})")

for sp in SPLITS:
    print(f"\n=== per-n final-state accuracy (%), split={sp} ({'unseen domains' if sp == 'test' else 'seen domains'}), "
          f"means over seeds ===")
    hdr = f"{'method':<16}{'mode':<6}" + "".join(f"{f'n={n}':>8}" for n in ALL_N)
    print(hdr + "\n" + "-" * len(hdr))
    for mm in METHODS:
        print(f"{mm[0]:<16}{mm[1]:<6}" + "".join(f"{100.0 * c / t:8.2f}" for c, t in (cnt(sp, mm, n) for n in ALL_N)))

print(f"\n=== summary: SHORT (n=1-4) / n5 / LONG (n=6-10), % with pooled counts, means over seeds ===")
hdr = f"{'method':<16}{'mode':<6}" + "".join(f"{f'{sp} {k}':>24}" for sp in SPLITS for k in ("SHORT", "n5", "LONG"))
print(hdr + "\n" + "-" * len(hdr))
for mm in METHODS:
    print(f"{mm[0]:<16}{mm[1]:<6}" + "".join(f"{fmt(val(sp, mm, ns)):>24}" for sp in SPLITS
                                             for ns in (SHORT_N, (N5,), LONG_N)))

print(f"\n=== per-step state accuracy (%), test stories with n = 10 (t = events consumed; t = 0 = after the init "
      f"sentence), means over seeds ===")
hdr = f"{'method':<16}{'mode':<6}" + "".join(f"{f't={t}':>8}" for t in STEPS_T)
print(hdr + "\n" + "-" * len(hdr))
for mm in STEP_METHODS:
    row = []
    for t in STEPS_T:
        c = sum(steps[(s, mm[0], mm[1], t)][0] for s in S)
        n = sum(steps[(s, mm[0], mm[1], t)][1] for s in S)
        row.append(f"{100.0 * c / n:8.2f}")
    print(f"{mm[0]:<16}{mm[1]:<6}" + "".join(row))
print(f"(stories per t summed over seeds: {sum(steps[(s, STEP_METHODS[0][0], STEP_METHODS[0][1], 0)][1] for s in S)})")

print(f"\n=== training fit (accuracy over every training prefix / step), means over seeds ===")
for mm in METHODS:
    c = sum(fit[(s, mm[0], mm[1])][0] for s in S)
    t = sum(fit[(s, mm[0], mm[1])][1] for s in S)
    print(f"  {mm[0]:<16}{mm[1]:<6}{100.0 * c / t:7.2f}% ({c}/{t})")
print("parameters: " + ", ".join(f"{m}={next(iter(v))}" for m, v in sorted(params.items())))

print(f"\n=== per-seed LONG on test (%, counts) ===")
print(f"{'method':<16}{'mode':<6}" + "".join(f"{f'seed {s}':>22}" for s in S))
for mm in METHODS:
    row = []
    for s in S:
        cs = [acc[(s, "test", mm[0], mm[1], n)] for n in LONG_N]
        row.append(f"{f'{100.0 * sum(c / t for c, t in cs) / len(cs):.2f}% ({sum(c for c, _ in cs)}/{sum(t for _, t in cs)})':>22}")
    print(f"{mm[0]:<16}{mm[1]:<6}" + "".join(row))


def diff(a, b) -> str:
    return f"{a[0] - b[0]:+.2f} points"


T_LONG = {mm: val("test", mm, LONG_N) for mm in METHODS}
print("\n=== registered quantities (issue #23; LONG = test, n = 6-10; numbers only, no verdict) ===")
tf, qw = T_LONG[("ONE-STEP-TF", "-")], T_LONG[("ONE-STEP-QWEN", "-")]
best = ("ONE-STEP-TF", "-") if tf[0] >= qw[0] else ("ONE-STEP-QWEN", "-")
print(f"P1  primary {mlabel(PRIMARY)} LONG {fmt(T_LONG[PRIMARY])} - max(ONE-STEP-TF {fmt(tf)}, ONE-STEP-QWEN "
      f"{fmt(qw)}) = {mlabel(best)} -> {diff(T_LONG[PRIMARY], T_LONG[best])}")
a, b = ("HOP-R896-d64", "free"), ("HOP-R896-d896", "free")
print(f"P2  {mlabel(a)} LONG {fmt(T_LONG[a])} - {mlabel(b)} LONG {fmt(T_LONG[b])} = {diff(T_LONG[a], T_LONG[b])}")
for mm in (a, b):
    v4, v10 = val("test", mm, (4,)), val("test", mm, (10,))
    print(f"    drop n=4 -> n=10, {mlabel(mm)}: {fmt(v4)} -> {fmt(v10)} = {v4[0] - v10[0]:+.2f} points")
print("P3  snap - free LONG per HOP variant:")
for h in HOPS:
    sn, fr = T_LONG[(h, "snap")], T_LONG[(h, "free")]
    print(f"    {h:<16} snap {fmt(sn)} - free {fmt(fr)} = {diff(sn, fr)}")
a, b = ("HOP-F64-d64", "free"), ("HOP-R896-d64", "free")
print(f"P4  {mlabel(a)} LONG {fmt(T_LONG[a])} - {mlabel(b)} LONG {fmt(T_LONG[b])} = {diff(T_LONG[a], T_LONG[b])}")

print("\n=== validity quantities (issue #23; numbers only, no verdict) ===")
print(f"V1  val_seen n=1-4: {mlabel(PRIMARY)} {fmt(val('val_seen', PRIMARY, SHORT_N))} | ONE-STEP-TF/- "
      f"{fmt(val('val_seen', ('ONE-STEP-TF', '-'), SHORT_N))}")
print(f"V2  test LONG: BAG/- {fmt(T_LONG[('BAG', '-')])}")
print(f"V3  blind judge: external (not in these logs)")
print(f"V4  val_seen LONG: SYMBOLIC/snap {fmt(val('val_seen', ('SYMBOLIC', 'snap'), LONG_N))}")
print(f"V5  leak check: asserted inside every run (each of the 3 logs reached its final L14_CONFIG line)")
