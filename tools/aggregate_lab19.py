"""Aggregate Lab 19 (reading + reasoning joined, issue #25) logs.

    python tools/aggregate_lab19.py eye_s42.log eye_s1.log eye_s2.log direct.log think.log [--partial]

Registered judgement: EYE+EXEC on normal stories of length 8-12 (LONG, summed over seeds 42 / 1 / 2):
  >= 90% and not below QWEN3-8B-THINK on the same stories -> 讀＋推可用; < 75% -> 累積誤差太大; else 部分可用.
"Same stories" = the 220 normal LONG stories the thinking model answered (index 0-1 per domain and length); EYE+EXEC is
scored on them per seed from L19_STORY lines and averaged over the seeds.
"""
import re
import sys
from collections import defaultdict

EXPECTED_STORY = "69c5866bf9e4f4715bde1dfff4c9660d7fbc409d0498d73f7258cc0ab9650be6"
SYSTEMS = ("ORACLE+EXEC", "EYE+EXEC", "EYE+HOP", "7B-DIRECT", "QWEN3-8B-THINK")
SEEDS = ("42", "1", "2")


def kv(line):
    return dict(re.findall(r"(\S+?)=(\S+)", line))


def pct(c):
    return 100.0 * c[0] / c[1] if c and c[1] else float("nan")


def main(argv):
    partial = "--partial" in argv
    files = [a for a in argv[1:] if not a.startswith("--")]
    res = defaultdict(lambda: defaultdict(lambda: [0, 0]))      # system -> key -> [c, t] (summed over seeds)
    per_story = defaultdict(dict)                                # (system, seed) -> id -> ok
    extra, sent, errs, configs = {}, defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0]), []
    for fn in files:
        for line in open(fn, encoding="utf-8"):
            line = line.rstrip("\n")
            if line.startswith("L19_RES "):
                d = kv(line)
                for k, v in d.items():
                    if "/" in v and k not in ("seed", "system"):
                        c, t = map(int, v.split("/"))
                        res[d["system"]][k][0] += c
                        res[d["system"]][k][1] += t
                for k in ("no_answer", "gen_tokens", "sec"):
                    if k in d:
                        extra[(d["system"], k)] = d[k]
            elif line.startswith("L19_STORY "):
                d = kv(line)
                per_story[(d["system"], d["seed"])][d["id"]] = int(d["ok"])
            elif line.startswith("L19_SENT "):
                for k, v in kv(line).items():
                    if "/" in v:
                        c, t = map(int, v.split("/"))
                        sent[k][0] += c
                        sent[k][1] += t
            elif line.startswith("L19_ERRS ") and "harm_by_variant" in line:
                for k, v in kv(line).items():
                    if "/" in v:
                        c, t = map(int, v.split("/"))
                        errs[k][0] += c
                        errs[k][1] += t
            elif line.startswith("L19_CONFIG "):
                configs.append(kv(line))
    if len(configs) != len(files) and not partial:
        sys.exit("every log needs one L19_CONFIG line")
    for c in configs:
        if c.get("data_sha_story") != EXPECTED_STORY:
            sys.exit(f"story sha {c.get('data_sha_story')}")
        if "FAKE" in c.get("env", "") and not partial:
            sys.exit("stub run")
    jobs = sorted((c.get("job"), c.get("seed")) for c in configs)
    if not partial and jobs != sorted([("eye", s) for s in SEEDS] + [("direct", "0"), ("think", "0")]):
        sys.exit(f"jobs {jobs} != eye x 3 seeds + direct + think")
    if partial:
        print("!!! PARTIAL: not the registered result\n")

    print("Final-state accuracy (EYE systems summed over 3 seeds)")
    print("system | SHORT 1-4 | MID 5-7 | LONG 8-12 | traps (LONG) | T1 | T2 | T3 | T4 | all")
    for s in SYSTEMS:
        r = res.get(s)
        if r:
            print(f"{s:15s} | " + " | ".join(f"{pct(r.get(k)):6.2f}" for k in
                                            ("SHORT", "MID", "LONG", "LONG_trap", "T1", "T2", "T3", "T4", "all")))
    print("\nby length (normal + trap stories of that length):")
    for s in SYSTEMS:
        r = res.get(s)
        if r and s != "QWEN3-8B-THINK":
            print(f"{s:15s} " + " ".join(f"{n}:{pct(r.get(f'len{n}')):5.1f}" for n in range(1, 13)))
    if sent:
        print("\nEye sentence accuracy on the story pool (3 seeds): " + " ".join(
            f"{k}={pct(v):.1f}" for k, v in sorted(sent.items())))
    if errs:
        print("Misread sentences that made the story wrong, by variant (wrong stories / misreads): " + " ".join(
            f"{k}={v[0]}/{v[1]}" for k, v in sorted(errs.items(), key=lambda kv_: -kv_[1][1])))
    for k, v in sorted(extra.items()):
        print(f"{k[0]} {k[1]} = {v}")

    think = per_story.get(("QWEN3-8B-THINK", "0"), {})
    sub_normal = [i for i in think if "/N" in i]
    sub_trap = [i for i in think if "/T" in i]
    eye_seeds = [s for s in SEEDS if ("EYE+EXEC", s) in per_story]
    print("\nSame stories as the thinking model:")
    cmp = {}
    for name, ids in (("normal LONG (220)", sub_normal), ("traps (440)", sub_trap)):
        if not ids:
            continue
        t_acc = 100.0 * sum(think[i] for i in ids) / len(ids)
        e_acc = sum(100.0 * sum(per_story[("EYE+EXEC", s)][i] for i in ids) / len(ids) for s in eye_seeds) / max(len(eye_seeds), 1)
        d_acc = (100.0 * sum(per_story[("7B-DIRECT", "0")][i] for i in ids) / len(ids)
                 if ("7B-DIRECT", "0") in per_story else float("nan"))
        cmp[name] = (e_acc, t_acc)
        print(f"  {name}: EYE+EXEC {e_acc:.2f} | QWEN3-8B-THINK {t_acc:.2f} | 7B-DIRECT {d_acc:.2f}")
    r = res.get("EYE+EXEC")
    if not r:
        return
    long_ = pct(r.get("LONG"))
    e_sub, t_sub = cmp.get("normal LONG (220)", (float("nan"), float("nan")))
    if long_ >= 90 and e_sub >= t_sub:
        verdict = "讀＋推可用"
    elif long_ < 75:
        verdict = "累積誤差太大"
    else:
        verdict = "部分可用"
    print(f"\nREGISTERED: EYE+EXEC LONG {long_:.2f}% (same-story: eye {e_sub:.2f} vs think {t_sub:.2f}) -> {verdict}"
          + ("  [PARTIAL]" if partial else ""))


if __name__ == "__main__":
    main(sys.argv)
