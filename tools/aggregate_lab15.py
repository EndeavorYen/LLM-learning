"""Aggregate the machine lines of Lab 15 (scripts/stage2_lab15_recognition.py; pre-registered on issue #25).
Numbers only: NO decision is applied here.

Usage: python tools/aggregate_lab15.py lab15.log [more.log ...]
  A complete run is one log (all 3 seeds inside). A run split with LAB15_MODELS gives several logs; they are merged.

Refuses (exit status 1):
  - a log without its final L15_CONFIG line (run incomplete);
  - a data sha that is not the pinned file, or seeds other than 42, 1, 2;
  - an encoder model covered by two logs, or a duplicate L15_CV / L15_STAGE / L15_ZS* key (e.g. a log passed twice);
  - L15_FOLD lines that differ across logs, or held-out totals that differ across encoders / recognizers / seeds for
    the same (data, k, fold) (every combination must be scored on the same held-out sentences);
  - a (data, encoder, recognizer, k) cell of a covered encoder without every fold x seed;
  - v5 L15_STAGE lines that do not sum to their L15_CV line; st event + init counts that do not sum to the total.
Warnings (printed): script_sha / env that differ across logs; a FAKE (test) environment; encoders not covered.

Accuracy = exact counts pooled over folds x seeds (sum correct / sum total). Every fold of a dataset has the same
number of held-out sentences, so this equals the mean accuracy over folds x seeds.
"""
import sys
from collections import defaultdict

PINNED = {"data_sha_st": "ed6d16bd3e625198bba74d28e7b20f6bdba9c399a3eb94b9a0d1f113f11e3108",
          "data_sha_v5": "d2a9e877595829b4ff1560be90cd04739acdb11be782b14a067c4b9c4d24f196"}
SEEDS = ("42", "1", "2")
MODELS = ("Q05", "Q15", "Q7", "BGE")
ENCODERS = ("Q05-mean", "Q05-prompt", "Q15-mean", "Q15-prompt", "Q7-mean", "Q7-prompt", "BGE")
RECOGNIZERS = ("NEAR", "LIN", "MLP", "FUNNEL", "XDOM")
DATASETS = {"st": dict(folds=4, ks=(4, 8, 12), reg_k=12), "v5": dict(folds=5, ks=(4, 8, 16), reg_k=16)}
TOP_N = 5
CHAIN_SENTENCES = 11  # a 10-step story = 1 init sentence + 10 event sentences (issue #25)


def die(msg: str):
    print(f"ERROR: {msg}")
    sys.exit("aggregate_lab15: refused (see ERROR above)")


def model_of(enc: str) -> str:
    return enc.split("-")[0]


paths = [p for p in sys.argv[1:]]
if not paths or any(p.startswith("-") for p in paths):
    sys.exit(__doc__)

configs = []                                  # (path, kv)
cv = defaultdict(dict)                        # (data, enc, rec, k) -> {(fold, seed): dict}
stage = defaultdict(dict)                     # (data, enc, rec, k, fold, seed) -> {stage: (c, t)}
zs, zspart, zscls = {}, {}, {}
folds = {}                                    # (data, fold) -> (heldout, pool)
model_log = {}                                # model -> path
warnings = []


def put(table, key, val, path, line):
    if key in table:
        die(f"duplicate line {key} in {path}: {line}")
    table[key] = val


def ints(kv, *names):
    return tuple(int(kv[n]) for n in names)


for path in paths:
    tags = []
    log_cv = set()
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = [ln.strip().replace("\r", "") for ln in fh]
    for line in lines:
        if not line.startswith("L15_"):
            continue
        tag, _, rest = line.partition(" ")
        try:
            kv = dict(p.split("=", 1) for p in rest.split())
        except ValueError:
            die(f"cannot parse {path}: {line}")
        tags.append(tag)
        if tag == "L15_CONFIG":
            configs.append((path, kv))
        elif tag == "L15_FOLD":
            key = (kv["data"], int(kv["fold"]))
            val = (kv["heldout"], kv["pool"])
            if key in folds and folds[key] != val:
                die(f"L15_FOLD {key} differs across logs ({path})")
            folds[key] = val
        elif tag == "L15_CV":
            key = (kv["data"], kv["enc"], kv["rec"], int(kv["k"]))
            fs = (int(kv["fold"]), kv["seed"])
            if fs in cv[key]:
                die(f"duplicate line {key + fs} in {path}: {line}")
            r = dict(zip(("c", "t", "ec", "et", "ic", "it"),
                         ints(kv, "correct", "total", "ev_correct", "ev_total", "init_correct", "init_total")))
            if "fit_correct" in kv:
                r.update(fit_c=int(kv["fit_correct"]), fit_t=int(kv["fit_total"]))
            cv[key][fs] = r
            log_cv.add(kv["enc"])
        elif tag == "L15_STAGE":
            key = (kv["data"], kv["enc"], kv["rec"], int(kv["k"]), int(kv["fold"]), kv["seed"])
            s = int(kv["stage"])
            if s in stage[key]:
                die(f"duplicate line {key} stage {s} in {path}: {line}")
            stage[key][s] = ints(kv, "correct", "total")
        elif tag == "L15_ZS":
            put(zs, kv["kind"], ints(kv, "correct", "total", "order1", "order2"), path, line)
        elif tag == "L15_ZSPART":
            put(zspart, kv["kind"], ints(kv, "correct", "total", "order1", "order2"), path, line)
        elif tag == "L15_ZSCLS":
            put(zscls, (kv["kind"], kv["cls"]), ints(kv, "correct", "total", "order1", "order2"), path, line)
    if not tags or tags[-1] != "L15_CONFIG" or tags.count("L15_CONFIG") != 1:
        die(f"{path}: the last L15 line must be its (only) L15_CONFIG line (incomplete or concatenated log?)")
    cfg = configs[-1][1]
    for m in cfg["models"].split(","):
        if m not in MODELS:
            die(f"{path}: unknown model {m!r}")
        if m in model_log:
            die(f"model {m} is covered by two logs: {model_log[m]} and {path}")
        model_log[m] = path
    if {model_of(e) for e in log_cv} != set(cfg["models"].split(",")):
        die(f"{path}: L15_CV encoders {sorted(log_cv)} do not match models={cfg['models']}")

# ---- configuration checks ----
for path, kv in configs:
    for k, v in PINNED.items():
        if kv.get(k) != v:
            die(f"{path}: {k}={kv.get(k)} is not the pinned file {v}")
    if tuple(kv["seeds"].split(",")) != SEEDS:
        die(f"{path}: seeds={kv['seeds']} (registered: {','.join(SEEDS)})")
    if "FAKE" in kv.get("env", ""):
        warnings.append(f"{path}: env={kv['env']} is a FAKE (test) environment: NOT a registered run")
for fld in ("script_sha", "env", "templates_sha"):
    vals = {kv.get(fld) for _p, kv in configs}
    if len(vals) > 1:
        warnings.append(f"{fld} differs across logs: {sorted(map(str, vals))}")
covered = [m for m in MODELS if m in model_log]
encs = [e for e in ENCODERS if model_of(e) in covered]
if covered != list(MODELS):
    warnings.append(f"the logs cover only the encoder models {covered}; missing {[m for m in MODELS if m not in covered]}"
                    f": every 'best' below is the best AMONG THE COVERED ENCODERS only")
zs_logs = [p for p, kv in configs if kv.get("zs") == "1"]
if "Q7" in covered and len(zs_logs) != 1:
    die(f"Q7 is covered but the zero-shot ran in {len(zs_logs)} logs")

# ---- completeness and consistency of the counts ----
for data, spec in DATASETS.items():
    for f in range(spec["folds"]):
        if (data, f) not in folds:
            die(f"missing L15_FOLD data={data} fold={f}")
    want = {(f, s) for f in range(spec["folds"]) for s in SEEDS}
    totals = {}
    for enc in encs:
        for rec in RECOGNIZERS:
            for k in spec["ks"]:
                cell = cv.get((data, enc, rec, k), {})
                if set(cell) != want:
                    die(f"incomplete cell data={data} enc={enc} rec={rec} k={k}: {len(cell)} of {len(want)} fold x seed")
                for (f, s), r in cell.items():
                    tk = (k, f)
                    tot = (r["t"], r["et"], r["it"], r.get("fit_t"))
                    if totals.setdefault(tk, tot) != tot:
                        die(f"held-out / training totals differ for data={data} k={k} fold={f}: {tot} vs {totals[tk]}")
                    if data == "st" and (r["ec"] + r["ic"] != r["c"] or r["et"] + r["it"] != r["t"]):
                        die(f"st event + init != total for {(enc, rec, k, f, s)}")
                    if data == "v5":
                        st_ = stage.get((data, enc, rec, k, f, s), {})
                        if set(st_) != set(range(5)) or sum(c for c, _t in st_.values()) != r["c"] \
                                or sum(t for _c, t in st_.values()) != r["t"]:
                            die(f"v5 L15_STAGE lines do not sum to L15_CV for {(enc, rec, k, f, s)}")
    extra = {key for key in cv if key[0] == data and key[1] not in encs}
    if extra:
        die(f"L15_CV lines for unexpected encoders: {sorted(extra)[:3]}")


# ---- report ----
def pooled(data, enc, rec, k, fields=("c", "t")):
    rs = cv[(data, enc, rec, k)].values()
    return tuple(sum(r[f] for r in rs) for f in fields)


def pct(c, t) -> str:
    return f"{100 * c / t:.2f}%" if t else "-"


def cellv(c, t) -> str:
    return f"{100 * c / t:7.2f}" if t else "      -"


print(f"Lab 15 aggregate | logs: {', '.join(paths)}")
for path, kv in configs:
    print(f"  {path}: models={kv['models']} zs={kv.get('zs')} script_sha={kv.get('script_sha')} env={kv.get('env')}")
for w in warnings:
    print(f"WARNING: {w}")
print("Accuracy = exact counts pooled over folds x seeds; held-out = domains never used to train that recognizer. "
      "NO decision is applied here.")

best_of = {}
for data, spec in DATASETS.items():
    n_f = spec["folds"]
    rk = spec["reg_k"]
    print(f"\n{'=' * 100}\n=== data={data}: mean CV accuracy (%) on held-out domains, pooled over {n_f} folds x "
          f"{len(SEEDS)} seeds ===")
    for k in spec["ks"]:
        c0, t0 = pooled(data, encs[0], RECOGNIZERS[0], k)
        print(f"--- k={k} training domains{' (registered k)' if k == rk else ''} | held-out sentences per "
              f"fold x seed: {t0 // (n_f * len(SEEDS))} ---")
        print(f"{'encoder':<12}" + "".join(f"{r:>8}" for r in RECOGNIZERS))
        for enc in encs:
            print(f"{enc:<12}" + "".join(f" {cellv(*pooled(data, enc, rec, k))}" for rec in RECOGNIZERS))
    if data == "st":
        for gname, flds in (("event sentences (5 classes)", ("ec", "et")), ("init sentences (8 classes)", ("ic", "it"))):
            print(f"--- k={rk}: {gname} ---")
            print(f"{'encoder':<12}" + "".join(f"{r:>8}" for r in RECOGNIZERS))
            for enc in encs:
                print(f"{enc:<12}" + "".join(f" {cellv(*pooled(data, enc, rec, rk, flds))}" for rec in RECOGNIZERS))

    combos = [(enc, rec) for enc in encs for rec in RECOGNIZERS]

    def acc(combo, k):
        c, t = pooled(data, combo[0], combo[1], k)
        return c / t

    # rank at the registered k: accuracy (exact fraction, same totals for every combo), ties keep the listing order
    ranked = sorted(combos, key=lambda cb: -pooled(data, cb[0], cb[1], rk)[0])
    best = ranked[0]
    bc, bt = pooled(data, *best, rk)
    ties = [cb for cb in ranked if pooled(data, *cb, rk)[0] == bc]
    best_of[data] = (best, bc, bt, ties)
    print(f"--- best combination at the registered k={rk}: {best[0]} + {best[1]} = {pct(bc, bt)} ({bc}/{bt})"
          + (f" | TIED with {', '.join(e + ' + ' + r for e, r in ties[1:])}" if len(ties) > 1 else "") + " ---")
    if data == "st":
        ec, et, ic, it = pooled(data, *best, rk, ("ec", "et", "ic", "it"))
        print(f"    event sentences {pct(ec, et)} ({ec}/{et}) | init sentences {pct(ic, it)} ({ic}/{it})")
    else:
        parts = []
        for s in range(5):
            c = sum(stage[(data, *best, rk, f, sd)][s][0] for f in range(n_f) for sd in SEEDS)
            t = sum(stage[(data, *best, rk, f, sd)][s][1] for f in range(n_f) for sd in SEEDS)
            parts.append(f"stage {s} {pct(c, t)} ({c}/{t})")
        print("    " + " | ".join(parts))
    cell = cv[(data, *best, rk)]
    per_fold = [(sum(cell[(f, s)]["c"] for s in SEEDS), sum(cell[(f, s)]["t"] for s in SEEDS)) for f in range(n_f)]
    per_seed = [(sum(cell[(f, s)]["c"] for f in range(n_f)), sum(cell[(f, s)]["t"] for f in range(n_f))) for s in SEEDS]
    print("    per held-out fold (pooled over seeds): " + " ".join(f"f{f}={pct(*per_fold[f])}" for f in range(n_f)))
    print("    per seed (pooled over folds): " + " ".join(f"{s}={pct(*per_seed[i])}" for i, s in enumerate(SEEDS)))
    if "fit_c" in next(iter(cell.values())):
        fc, ft = pooled(data, *best, rk, ("fit_c", "fit_t"))
        print(f"    training fit (eval mode): {pct(fc, ft)} ({fc}/{ft})")
    print(f"--- k-curves of the best {TOP_N} combinations at k={rk} (%): "
          + " / ".join(f"k={k}" for k in spec["ks"]) + " ---")
    for cb in ranked[:TOP_N]:
        print(f"  {cb[0] + ' + ' + cb[1]:<24}" + "".join(f" {cellv(*pooled(data, *cb, k))}" for k in spec["ks"]))

print(f"\n{'=' * 100}\n=== zero-shot reference (Qwen2.5-7B-Instruct, st, all train-domain sentences, no training; "
      f"correct = right under both option orders) ===")
if not zs:
    print("  (no L15_ZS lines in these logs)")
for kind in ("event", "init"):
    if kind in zs:
        c, t, o1, o2 = zs[kind]
        print(f"  {kind:<6}: {pct(c, t)} ({c}/{t}) | order1 {pct(o1, t)} ({o1}) | order2 {pct(o2, t)} ({o2})")
for kind, (c, t, o1, o2) in zspart.items():
    print(f"  {kind:<15}: {pct(c, t)} ({c}/{t}) | order1 {o1} | order2 {o2}")
if zs.get("event") and zs.get("init"):
    c = zs["event"][0] + zs["init"][0]
    t = zs["event"][1] + zs["init"][1]
    print(f"  all 13 classes: {pct(c, t)} ({c}/{t})")
for (kind, cls), (c, t, o1, o2) in zscls.items():
    print(f"    {kind:<6}{cls:<10} {c:>4}/{t:<4} order1 {o1:>4} order2 {o2:>4}")

(best, bc, bt, ties) = best_of["st"]
x = bc / bt
print(f"\n{'=' * 100}")
print(f"registered quantity: best st k={DATASETS['st']['reg_k']} = {100 * x:.2f}% ({bc}/{bt}) "
      f"[{best[0]} + {best[1]}]" + (f" (tied: {len(ties)} combinations)" if len(ties) > 1 else ""))
print(f"X^{CHAIN_SENTENCES} = {x ** CHAIN_SENTENCES:.4f}  (X = {x:.4f})")
for w in warnings:
    print(f"WARNING: {w}")
