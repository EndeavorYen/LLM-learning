"""Average the V4_RESULT / V4_DIAG lines of several seed logs (issue #21).
Usage: python tools/aggregate_v4.py seed42.log seed1.log seed2.log"""
import sys
from collections import defaultdict

N = 48
res, diag = defaultdict(dict), defaultdict(dict)
for path in sys.argv[1:]:
    for line in open(path, encoding="utf-8", errors="ignore"):
        if line.startswith(("V4_RESULT", "V4_DIAG")):
            kv = dict(x.split("=", 1) for x in line.split()[1:])
            (res[kv["method"]] if line.startswith("V4_RESULT") else diag[kv["name"]])[kv["seed"]] = kv


def fmt(d, keys):
    out = []
    for k in keys:
        vals = [int(s[k]) for s in d.values() if k in s]
        if not vals or min(vals) < 0:
            out.append(f"{'-':>18}")
            continue
        out.append(f"{'/'.join(map(str, vals)):>10} ({100 * sum(vals) / len(vals) / N:4.1f}%)")
    return "".join(out)


keys = ["BR2", "BR3", "BR4", "A2", "FLIP2", "PREM"]
print(f"{'method':<22}" + "".join(f"{k:>18}" for k in keys) + f"{'BR mean':>10}")
for m, d in res.items():
    br = [int(s[f"BR{k}"]) for s in d.values() for k in (2, 3, 4)]
    brm = f"{100 * sum(br) / len(br) / N:6.1f}%" if min(br) >= 0 else "     -"
    print(f"{m:<22}" + fmt(d, keys) + f"{brm:>10}")
print("\ndiagnostics:")
for m, d in diag.items():
    n = int(next(iter(d.values())).get("n", N))
    vals = {k: [int(s[k]) for s in d.values()] for k in ("BR2", "BR3", "BR4")}
    print(f"  {m:<32} " + "  ".join(f"{k}={'/'.join(map(str, v))} ({100 * sum(v) / len(v) / n:4.1f}%)" for k, v in vals.items()))
