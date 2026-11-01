"""Aggregate Lab 20 (capacity + amounts, issue #25) logs.

    python tools/aggregate_lab20.py eye_s42.log eye_s1.log eye_s2.log [--partial] [--script-verified]

Registered judgement (summed over seeds 42 / 1 / 2):
  EYE+EXEC normal LONG 8-12 >= 80 and amount accuracy (amount_v1 test IN / OUT sentences) >= 90 -> 規則可變;
  amount < 80 -> 數量讀不準; otherwise 部分可用.
  Learned reasoner: capacity-4 normal stories, |EYE+HOP - EYE+EXEC| <= 3 and ORACLE+HOP >= 95 -> 學到了容量.
"""
import re
import sys
from collections import defaultdict

EXPECTED = {"data_sha_story": "5cefc22664f607810c0932700fba739f36f533efea7f70bc612d877c70041a6c",
            "data_sha_amt_train": "b27131477c6bbb3a0c5b755b6ea72d396492c2fa198d6a6450ab13c07f09b1bb",
            "data_sha_amt_test": "4bda68a558894e5b9a7cbedb869eee472598a95d873ecf4ee1b1fa0b6ed37a81"}
SYSTEMS = ("ORACLE+EXEC", "ORACLE-AMT1+EXEC", "EYE+EXEC", "ORACLE+HOP", "EYE+HOP")
SEEDS = ("42", "1", "2")
N_STORIES = 22 * 4 * 12 * 3 + 22 * 20


def kv(line):
    return dict(re.findall(r"(\S+?)=(\S+)", line))


def pct(c):
    return 100.0 * c[0] / c[1] if c and c[1] else float("nan")


def add(dst, d):
    for k, v in d.items():
        if "/" in v and k not in ("seed", "system"):
            c, t = map(int, v.split("/"))
            dst[k][0] += c
            dst[k][1] += t


def main(argv):
    partial = "--partial" in argv
    files = [a for a in argv[1:] if not a.startswith("--")]
    res = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    per_seed = defaultdict(dict)
    sent, amt, dec, harm, hops, configs = (defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0]),
                                           defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0]), [], [])
    for fn in files:
        for line in open(fn, encoding="utf-8"):
            line = line.rstrip("\n")
            if line.startswith("L20_RES "):
                d = kv(line)
                add(res[d["system"]], d)
                per_seed[(d["system"], d["seed"])] = d
            elif line.startswith("L20_SENT "):
                add(sent, kv(line))
            elif line.startswith("L20_AMT "):
                add(amt, kv(line))
            elif line.startswith("L20_ERRS "):
                add(dec if " by_error_type " in line else harm, kv(line))
            elif line.startswith("L20_HOP "):
                hops.append(line)
            elif line.startswith("L20_CONFIG "):
                configs.append(kv(line))
    if not partial:
        if len(configs) != len(files):
            sys.exit("every log needs one L20_CONFIG line")
        if sorted(c.get("seed") for c in configs) != sorted(SEEDS):
            sys.exit(f"seeds {[c.get('seed') for c in configs]} != {SEEDS}")
        for c in configs:
            if "FAKE" in c.get("env", ""):
                sys.exit("stub run")
            for k, v in EXPECTED.items():
                if v is None or c.get(k) != v:
                    sys.exit(f"{k}: {c.get(k)} is not the registered sha ({v})")
        if len({c.get("script_sha") for c in configs}) != 1 and "--script-verified" not in argv:
            sys.exit("logs come from different script versions (use --script-verified after checking the job files)")
        for s in SEEDS:
            for sy in SYSTEMS:
                d = per_seed.get((sy, s))
                if not d or d.get("all", "0/0").split("/")[1] != str(N_STORIES):
                    sys.exit(f"{sy} seed {s}: not {N_STORIES} stories")
        # [review 7] amount coverage (22 domains x 8 groups x 4 IN / OUT forms per seed) and the oracle check
        if amt.get("registered", [0, 0])[1] != 22 * 8 * 4 * len(SEEDS):
            sys.exit(f"amount metric covers {amt.get('registered')} sentences, expected {22 * 8 * 4 * len(SEEDS)}")
        if res["ORACLE+EXEC"]["all"][0] != res["ORACLE+EXEC"]["all"][1]:
            sys.exit("ORACLE+EXEC is not 100%: rule or data error")
    else:
        print("!!! PARTIAL: not the registered result\n")

    print("Final-state accuracy (summed over seeds)")
    print("system      | SHORT | MID   | LONG  | traps | A1    | A2    | A3    | A4    | all")
    for s in SYSTEMS:
        r = res.get(s)
        if r:
            print(f"{s:11s} | " + " | ".join(f"{pct(r.get(k)):5.1f}" for k in
                                              ("SHORT", "MID", "LONG", "LONG_trap", "A1", "A2", "A3", "A4", "all")))
    print("\nnormal stories by capacity (all lengths | LONG 8-12):")
    for s in SYSTEMS:
        r = res.get(s)
        if r:
            print(f"{s:11s} " + "  ".join(f"cap{c}: {pct(r.get(f'ncap{c}')):5.1f} | {pct(r.get(f'nLONG_cap{c}')):5.1f}"
                                         for c in (2, 3, 4, 5)))
    print("\nnormal stories by length:")
    for s in SYSTEMS:
        r = res.get(s)
        if r:
            print(f"{s:11s} " + " ".join(f"{n}:{pct(r.get(f'nlen{n}')):5.1f}" for n in range(1, 13)))
    print("\nEye event accuracy by variant: " + " ".join(f"{k}={pct(v):.1f}" for k, v in sorted(sent.items())))
    print("Eye amount accuracy (gold IN / OUT sentences): " + " ".join(f"{k}={pct(v):.1f} ({v[0]}/{v[1]})"
                                                                     for k, v in sorted(amt.items())))
    print("Stories by kind of misreading (final right / stories): " + " ".join(
        f"{k}={v[0]}/{v[1]}" for k, v in sorted(dec.items())))
    print("Misreads by variant (stories made wrong / misreads): " + " ".join(
        f"{k}={v[0]}/{v[1]}" for k, v in sorted(harm.items(), key=lambda x: -x[1][1])))
    for h in hops:
        print(h)

    r = res.get("EYE+EXEC")
    if not r:
        return
    long_, a = pct(r.get("LONG")), pct(amt.get("registered"))
    verdict = "規則可變" if long_ >= 80 and a >= 90 else ("數量讀不準" if a < 80 else "部分可用")
    e4, h4, o4 = pct(res["EYE+EXEC"].get("ncap4")), pct(res["EYE+HOP"].get("ncap4")), pct(res["ORACLE+HOP"].get("ncap4"))
    hv = "學到了容量" if abs(h4 - e4) <= 3 and o4 >= 95 else "沒學到"
    tag = "  [PARTIAL]" if partial else ""
    print(f"\nContext [review 3/4]: amount on existing amount-1 sentences {pct(amt.get('k1_existing')):.2f}%, by gold amount "
          + " ".join(f"{k}={pct(amt.get(k)):.2f}" for k in ("amt1", "amt2", "amt3"))
          + f"; amount-blind reference (correct events, amounts read as 1) LONG {pct(res['ORACLE-AMT1+EXEC'].get('LONG')):.2f}%")
    print(f"\nREGISTERED: EYE+EXEC LONG {long_:.2f}%, amount {a:.2f}% -> {verdict}{tag}")
    print(f"REGISTERED (learned reasoner, capacity 4): EYE+HOP {h4:.2f} vs EYE+EXEC {e4:.2f}, ORACLE+HOP {o4:.2f} -> {hv}{tag}")


if __name__ == "__main__":
    main(sys.argv)
