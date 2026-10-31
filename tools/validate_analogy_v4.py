"""Acceptance check for benchmarks/analogy_v4.json (test data v4, "联想跳跃").

In v4 the two laws of a pair share the IDENTICAL stage-1 sentence in every domain; which
branch follows (stages 2-4) is fixed only by stage 0. Plain Python, no torch:
    python tools/validate_analogy_v4.py [path]

Pre-registered checks (spec section 资料验收 1-2; the 7B answerability check is run separately):
  V1  structure: 6 laws, pairs [[0,1],[2,3],[4,5]], 12 seen + 8 unseen domains (disjoint),
      5 stages per chain, stages 0-3 end with "，" and stage 4 with "。", stage-1 strings
      identical within each pair in every domain, no other duplicate sentence, 8-24 chars.
  V2  lexical: for every unseen chain and stage k in {2, 3, 4}, predict which law of the pair
      the sentence belongs to by char-unigram / char-bigram Jaccard to the SEEN sentences of
      (law, k), centroid (mean) and nearest-neighbour variants. Best variant <= 70%
      (chance 50%; ties count half). Stage-0 accuracy is reported, not constrained, since
      stage 0 is meant to state the branch.
  V3  no substring of >= 2 chars shared by every seen sentence of a (law, stage) cell and by
      half or more of that cell's unseen sentences, excluding generic stage markers shared by
      every seen sentence of that stage across all laws.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_analogy_v2 import PUNCT, ROOT, grams, jaccard  # noqa: E402

DEFAULT = ROOT / "benchmarks" / "analogy_v4.json"
N_LAWS, N_STAGES = 6, 5
PAIRS = [[0, 1], [2, 3], [4, 5]]
PARTNER = {a: b for a, b in PAIRS} | {b: a for a, b in PAIRS}
N_SEEN, N_UNSEEN = 12, 8
MIN_LEN, MAX_LEN = 8, 24
BRANCH_STAGES = (2, 3, 4)
V2_LIMIT = 0.70


def _strip(text):
    return "".join(c for c in text if c not in PUNCT)


def check_structure(data):
    errs = []
    laws = data.get("laws", [])
    if [l.get("id") for l in laws] != list(range(N_LAWS)):
        errs.append(f"expected laws with ids 0..{N_LAWS - 1}")
    for l in laws:
        if len(l.get("stages", [])) != N_STAGES:
            errs.append(f"law {l.get('id')}: {len(l.get('stages', []))} stage descriptions")
        if l.get("pair") != l.get("id", 0) // 2:
            errs.append(f"law {l.get('id')}: pair field {l.get('pair')}")
    if data.get("pairs") != PAIRS:
        errs.append(f"pairs must be {PAIRS}, got {data.get('pairs')}")
    seen, unseen = data.get("seen_domains", []), data.get("unseen_domains", [])
    if len(seen) != N_SEEN:
        errs.append(f"{len(seen)} seen domains (need {N_SEEN})")
    if len(unseen) != N_UNSEEN:
        errs.append(f"{len(unseen)} unseen domains (need {N_UNSEEN})")
    names = [d["name"] for d in seen + unseen]
    if len(set(names)) != len(names):
        errs.append(f"duplicate domain names: {sorted({n for n in names if names.count(n) > 1})}")
    counts = {}
    for dom in seen + unseen:
        ch = dom.get("chains", {})
        if sorted(ch) != [str(i) for i in range(N_LAWS)]:
            errs.append(f"{dom['name']}: chains keys {sorted(ch)}")
            continue
        for law in range(N_LAWS):
            chain = ch[str(law)]
            if len(chain) != N_STAGES or not all(isinstance(t, str) for t in chain):
                errs.append(f"{dom['name']} law {law}: need {N_STAGES} strings")
                continue
            for st, t in enumerate(chain):
                end = "。" if st == N_STAGES - 1 else "，"
                if not t.endswith(end) or "。" in t[:-1]:
                    errs.append(f"{dom['name']} law {law} stage {st}: punctuation ({t})")
                if not MIN_LEN <= len(t) <= MAX_LEN:
                    errs.append(f"{dom['name']} law {law} stage {st}: length {len(t)} ({t})")
                counts[t] = counts.get(t, 0) + 1
        for a, b in PAIRS:
            if ch[str(a)][1] != ch[str(b)][1]:
                errs.append(f"{dom['name']} pair {a}/{b}: stage-1 strings differ")
    shared = {dom["chains"][str(a)][1] for dom in seen + unseen for a, _ in PAIRS
              if dom.get("chains", {}).get(str(a))}
    dup = sorted(t for t, c in counts.items() if c > (2 if t in shared else 1))
    if dup:
        errs.append(f"duplicate sentences: {dup[:5]}")
    return errs


def cells(domains):
    out = {}
    for dom in domains:
        for law in range(N_LAWS):
            for st, t in enumerate(dom["chains"][str(law)]):
                out.setdefault((law, st), []).append(t)
    return out


def lexical(data, n, mode, stages):
    feats = {k: [grams(t, n) for t in v] for k, v in cells(data["seen_domains"]).items()}

    def sim(text, law, st):
        g = grams(text, n)
        s = [jaccard(g, f) for f in feats[(law, st)]]
        return max(s) if mode == "nn" else sum(s) / len(s)

    hit = {st: 0.0 for st in stages}
    tot = {st: 0 for st in stages}
    for dom in data["unseen_domains"]:
        for law in range(N_LAWS):
            for st in stages:
                text = dom["chains"][str(law)][st]
                a, b = sim(text, law, st), sim(text, PARTNER[law], st)
                hit[st] += 1.0 if a > b else 0.5 if a == b else 0.0
                tot[st] += 1
    return hit, tot


def _subs(text):
    s = _strip(text)
    return {s[i:j] for i in range(len(s)) for j in range(i + 2, len(s) + 1)}


def cue_leaks(data):
    seen, unseen = cells(data["seen_domains"]), cells(data["unseen_domains"])
    leaks = []
    for st in range(N_STAGES):
        generic = set.intersection(*(_subs(t) for law in range(N_LAWS) for t in seen[(law, st)]))
        for law in range(N_LAWS):
            common = set.intersection(*(_subs(t) for t in seen[(law, st)])) - generic
            un = unseen[(law, st)]
            for sub in sorted(common, key=len, reverse=True):
                hits = sum(sub in _strip(u) for u in un)
                if hits * 2 >= len(un):
                    leaks.append((law, st, sub, f"{hits}/{len(un)}"))
    return leaks


def main(path=DEFAULT):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    ok = True

    errs = check_structure(data)
    lens = [len(t) for d in data.get("seen_domains", []) + data.get("unseen_domains", [])
            for ch in d.get("chains", {}).values() for t in ch]
    print(f"V1 structure ({len(lens)} sentences, length {min(lens, default=0)}-{max(lens, default=0)}): "
          f"{'PASS' if not errs else 'FAIL'}")
    for e in errs[:20]:
        print("   -", e)
    ok &= not errs
    if errs:
        print("REJECTED")
        return 1

    best = 0.0
    for n in (1, 2):
        for mode in ("centroid", "nn"):
            hit, tot = lexical(data, n, mode, BRANCH_STAGES)
            h0, t0 = lexical(data, n, mode, (0,))
            acc = sum(hit.values()) / sum(tot.values())
            per = "  ".join(f"s{st} {hit[st] / tot[st]:.0%}" for st in BRANCH_STAGES)
            print(f"   lexical n={n} {mode:8s}: stages 2-4 {acc:.1%} ({per})   [stage 0 {h0[0] / t0[0]:.0%}]")
            best = max(best, acc)
    v2 = best <= V2_LIMIT
    print(f"V2 lexical pair accuracy, stages 2-4 (best variant) {best:.1%} <= {V2_LIMIT:.0%}: "
          f"{'PASS' if v2 else 'FAIL'}  (chance 50%)")
    ok &= v2

    leaks = cue_leaks(data)
    print(f"V3 shared cue substrings: {'PASS' if not leaks else 'FAIL'}")
    for law, st, sub, frac in leaks[:20]:
        print(f"   - law {law} stage {st}: '{sub}' in {frac} unseen sentences")
    ok &= not leaks

    print("ACCEPTED" if ok else "REJECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT))
