"""Acceptance check for benchmarks/analogy_v2.json (issue #7, test data v2).

The v1 test sets could be solved by matching cue words: every domain wrote the same
(law, stage) with the same phrase ("急剧飙升", "迅速枯竭" ...), so a character-overlap
classifier scored 8/8. A v2 data set is only accepted if a model with NO semantics cannot
recover the causal law of an unseen-domain sentence from surface overlap with the seen
domains. Plain Python, no torch:  python tools/validate_analogy_v2.py [path]

Pre-registered acceptance thresholds (set before any model is run on v2):
  T1  lexical law accuracy on unseen stage-2/3 clauses (stage given, 3-way, chance 33%)  <= 50%
  T2  lexical held-out pick (stage-2 or stage-3 target among the 6 never-trained
      stage-2/3 clauses of the same unseen domain, chance 17%)                          <= 33%
  T3  no substring of >= 2 characters shared by every seen phrasing of a (law, stage)
      cell and by half or more of that cell's unseen clauses (cue-word leak)
  T4  structure: 3 laws x 4 stages; every seen cell has >= 2 phrasings; >= 10 unseen domains
The best of the four lexical variants (char unigram / bigram x nearest-neighbour / centroid)
is used for T1 and T2, so the data must defeat all of them.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "benchmarks" / "analogy_v2.json"
PUNCT = set("，。,.；;：:！!？? 、")
N_LAWS, N_STAGES = 3, 4


def grams(text, n):
    s = "".join(c for c in text if c not in PUNCT)
    return set(s) if n == 1 else {s[i:i + n] for i in range(len(s) - n + 1)}


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 0.0


def load(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    seen = []  # (text, law, stage, domain)
    for dom in data["seen_domains"]:
        for law in range(N_LAWS):
            for st in range(N_STAGES):
                for text in dom["chains"][str(law)][st]:
                    seen.append((text, law, st, dom["name"]))
    unseen = []  # (domain, law, [s0, s1, s2, s3])
    for dom in data["unseen_domains"]:
        for law in range(N_LAWS):
            unseen.append((dom["name"], law, dom["chains"][str(law)]))
    return data, seen, unseen


def check_structure(data, seen, unseen):
    errs = []
    if len(data.get("laws", [])) != N_LAWS:
        errs.append(f"expected {N_LAWS} laws")
    for dom in data["seen_domains"]:
        for law in range(N_LAWS):
            cells = dom["chains"][str(law)]
            if len(cells) != N_STAGES:
                errs.append(f"seen {dom['name']} law {law}: {len(cells)} stages")
            for st, ph in enumerate(cells):
                if len(ph) < 2:
                    errs.append(f"seen {dom['name']} law {law} stage {st}: needs >= 2 phrasings")
    if len(data["unseen_domains"]) < 10:
        errs.append(f"only {len(data['unseen_domains'])} unseen domains (< 10)")
    for name, law, chain in unseen:
        if len(chain) != N_STAGES:
            errs.append(f"unseen {name} law {law}: {len(chain)} stages")
    texts = [t for t, *_ in seen] + [t for _, _, ch in unseen for t in ch]
    dup = {t for t in texts if texts.count(t) > 1}
    if dup:
        errs.append(f"duplicate sentences: {sorted(dup)[:5]}")
    return errs


def lexical_scores(seen, unseen, n, mode):
    feats = [(grams(t, n), law, st) for t, law, st, _ in seen]

    def law_given_stage(text, st):
        g = grams(text, n)
        if mode == "nn":
            best = max((f for f in feats if f[2] == st), key=lambda f: jaccard(g, f[0]))
            return best[1]
        per_law = {}
        for f in feats:
            if f[2] == st:
                per_law.setdefault(f[1], []).append(jaccard(g, f[0]))
        return max(per_law, key=lambda k: sum(per_law[k]) / len(per_law[k]))

    def class_sim(text, law, st):
        g = grams(text, n)
        sims = [jaccard(g, f[0]) for f in feats if f[1] == law and f[2] == st]
        return max(sims) if mode == "nn" else sum(sims) / len(sims)

    t1_hit = t1_n = t2_hit = t2_n = 0
    by_dom = {}
    for name, law, chain in unseen:
        by_dom.setdefault(name, {})[law] = chain
    for name, law, chain in unseen:
        for st in (2, 3):
            t1_hit += law_given_stage(chain[st], st) == law
            t1_n += 1
            # T2: which of the 6 held-out stage-2/3 clauses of this domain looks most like (law, st)?
            cands = [(l2, s2) for l2 in range(N_LAWS) for s2 in (2, 3)]
            pick = max(cands, key=lambda k: class_sim(by_dom[name][k[0]][k[1]], law, st))
            t2_hit += pick == (law, st)
            t2_n += 1
    return t1_hit, t1_n, t2_hit, t2_n


def _common_subs(texts):
    common = None
    for t in texts:
        g = set().union(*(grams(t, n) for n in range(2, 7)))
        common = g if common is None else common & g
    return common or set()


def cue_leaks(seen, unseen):
    """Law-discriminative leaks only. Generic stage markers shared by every law at that stage
    (e.g. a "当……时" frame for stage 1, "最终" for stage 3) carry no law information and are allowed."""
    leaks = []
    for law in range(N_LAWS):
        for st in range(N_STAGES):
            seen_texts = [t for t, l, s, _ in seen if l == law and s == st]
            generic = _common_subs([t for t, l, s, _ in seen if s == st])
            common = _common_subs(seen_texts) - generic
            un = [ch[st] for _, l, ch in unseen if l == law]
            for sub in sorted(common or [], key=len, reverse=True):
                hits = sum(sub in "".join(c for c in u if c not in PUNCT) for u in un)
                if hits * 2 >= len(un):
                    leaks.append((law, st, sub, f"{hits}/{len(un)}"))
    return leaks


def main(path=DEFAULT):
    data, seen, unseen = load(path)
    ok = True
    errs = check_structure(data, seen, unseen)
    print(f"T4 structure: {'PASS' if not errs else 'FAIL'}")
    for e in errs:
        print("   -", e)
    ok &= not errs

    best_t1 = best_t2 = 0.0
    for n in (1, 2):
        for mode in ("nn", "centroid"):
            a, an, b, bn = lexical_scores(seen, unseen, n, mode)
            print(f"   lexical n={n} {mode:8s}: law|stage {a}/{an} ({a/an:.0%})   held-out pick {b}/{bn} ({b/bn:.0%})")
            best_t1, best_t2 = max(best_t1, a / an), max(best_t2, b / bn)
    t1, t2 = best_t1 <= 0.50, best_t2 <= 0.33
    print(f"T1 lexical law accuracy (best variant) {best_t1:.0%} <= 50%: {'PASS' if t1 else 'FAIL'}  (chance 33%)")
    print(f"T2 lexical held-out pick (best variant) {best_t2:.0%} <= 33%: {'PASS' if t2 else 'FAIL'}  (chance 17%)")
    leaks = cue_leaks(seen, unseen)
    print(f"T3 shared cue substrings: {'PASS' if not leaks else 'FAIL'}")
    for law, st, sub, frac in leaks[:20]:
        print(f"   - law {law} stage {st}: '{sub}' in {frac} unseen clauses")
    ok &= t1 and t2 and not leaks
    print("ACCEPTED" if ok else "REJECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT))
