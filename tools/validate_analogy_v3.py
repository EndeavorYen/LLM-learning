"""Acceptance check for benchmarks/analogy_v3.json (stage 3, test data v3).

Reuses the v2 lexical checks unchanged (tools/validate_analogy_v2.py):
  T1  lexical law accuracy on unseen stage-2/3 clauses (best of 4 variants)   <= 50%
  T2  lexical held-out pick, law correct (best of 4 variants)                  <= 50%
  T3  no law-specific cue substring shared by a seen cell and >= half its unseen clauses
T4 is relaxed for v3, whose seen set adds the 12 v2 unseen domains with one phrasing each:
  T4  3 laws x 4 stages; every seen cell has >= 1 phrasing; >= 12 seen domains;
      exactly 12 unseen domains; no duplicate sentences.
Plain Python, no torch:  python tools/validate_analogy_v3.py [path]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_analogy_v2 import N_LAWS, N_STAGES, ROOT, cue_leaks, lexical_scores, load  # noqa: E402

DEFAULT = ROOT / "benchmarks" / "analogy_v3.json"


def check_structure(data, seen, unseen):
    errs = []
    if len(data.get("laws", [])) != N_LAWS:
        errs.append(f"expected {N_LAWS} laws")
    for law in data.get("laws", []):
        if len(law["stages"]) != N_STAGES:
            errs.append(f"law {law['id']}: {len(law['stages'])} stage descriptions")
    if len(data["seen_domains"]) < 12:
        errs.append(f"only {len(data['seen_domains'])} seen domains (< 12)")
    for dom in data["seen_domains"]:
        for law in range(N_LAWS):
            cells = dom["chains"][str(law)]
            if len(cells) != N_STAGES:
                errs.append(f"seen {dom['name']} law {law}: {len(cells)} stages")
            for st, ph in enumerate(cells):
                if len(ph) < 1:
                    errs.append(f"seen {dom['name']} law {law} stage {st}: needs >= 1 phrasing")
    if len(data["unseen_domains"]) != 12:
        errs.append(f"{len(data['unseen_domains'])} unseen domains (need exactly 12)")
    overlap = {d["name"] for d in data["seen_domains"]} & {d["name"] for d in data["unseen_domains"]}
    if overlap:
        errs.append(f"domains both seen and unseen: {sorted(overlap)}")
    for name, law, chain in unseen:
        if len(chain) != N_STAGES:
            errs.append(f"unseen {name} law {law}: {len(chain)} stages")
    texts = [t for t, *_ in seen] + [t for _, _, ch in unseen for t in ch]
    dup = {t for t in texts if texts.count(t) > 1}
    if dup:
        errs.append(f"duplicate sentences: {sorted(dup)[:5]}")
    return errs


def main(path=DEFAULT):
    data, seen, unseen = load(path)
    ok = True
    errs = check_structure(data, seen, unseen)
    print(f"T4 structure (v3): {'PASS' if not errs else 'FAIL'}  "
          f"({len(data['seen_domains'])} seen domains, {len(seen)} seen sentences, "
          f"{len(data['unseen_domains'])} unseen domains)")
    for e in errs:
        print("   -", e)
    ok &= not errs

    best_t1 = best_t2 = 0.0
    for n in (1, 2):
        for mode in ("nn", "centroid"):
            a, an, b, bn = lexical_scores(seen, unseen, n, mode)
            print(f"   lexical n={n} {mode:8s}: law|stage {a}/{an} ({a/an:.0%})   held-out pick law {b}/{bn} ({b/bn:.0%})")
            best_t1, best_t2 = max(best_t1, a / an), max(best_t2, b / bn)
    t1, t2 = best_t1 <= 0.50, best_t2 <= 0.50
    print(f"T1 lexical law accuracy (best variant) {best_t1:.0%} <= 50%: {'PASS' if t1 else 'FAIL'}  (chance 33%)")
    print(f"T2 lexical held-out pick, law correct (best variant) {best_t2:.0%} <= 50%: {'PASS' if t2 else 'FAIL'}  (chance 33%)")
    leaks = cue_leaks(seen, unseen)
    print(f"T3 shared cue substrings: {'PASS' if not leaks else 'FAIL'}")
    for law, st, sub, frac in leaks[:20]:
        print(f"   - law {law} stage {st}: '{sub}' in {frac} unseen clauses")
    ok &= t1 and t2 and not leaks
    print("ACCEPTED" if ok else "REJECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT))
