"""Acceptance check for benchmarks/analogy_v5.json (confirmatory test data v5, "联想跳跃").

v5 keeps the v4 format, laws and pairs. Its seen set is ALL 20 v4 domains (the 12 v4 seen
followed by the 8 v4 unseen, verbatim); its unseen set is 8 brand-new domains. Plain Python,
no torch:
    python tools/validate_analogy_v5.py [path]

Checks are those of tools/validate_analogy_v4.py (helpers imported from it), with the domain
counts changed to 20 seen + 8 unseen:
  V1  structure as in v4 (6 laws, pairs, 5 stages, punctuation, 8-24 chars, stage-1 strings
      identical within each pair, no other duplicate sentence), plus provenance: version
      "analogy_v5"; laws, pairs and seen_domains equal to analogy_v4.json (seen = v4 seen +
      v4 unseen, in order); no unseen domain name reused from analogy_v2/v3/v4.
  V2  lexical: char-unigram / char-bigram Jaccard (centroid and nearest-neighbour) to the
      SEEN sentences of (law, k) vs (partner, k), unseen stages 2-4. Best variant <= 70%
      (chance 50%; ties count half). Stage-0 accuracy is reported, not constrained.
  V3  no substring of >= 2 chars shared by every seen sentence of a (law, stage) cell and by
      half or more of that cell's unseen sentences (generic stage markers excluded).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_analogy_v4 as v4  # noqa: E402
from validate_analogy_v2 import ROOT  # noqa: E402

DEFAULT = ROOT / "benchmarks" / "analogy_v5.json"
V4_PATH = ROOT / "benchmarks" / "analogy_v4.json"
OLDER = [ROOT / "benchmarks" / f"analogy_v{i}.json" for i in (2, 3, 4)]
N_SEEN, N_UNSEEN = 20, 8


def check_structure(data):
    # v4.check_structure reads the domain counts from its module globals.
    saved = v4.N_SEEN, v4.N_UNSEEN
    v4.N_SEEN, v4.N_UNSEEN = N_SEEN, N_UNSEEN
    try:
        errs = v4.check_structure(data)
    finally:
        v4.N_SEEN, v4.N_UNSEEN = saved

    if data.get("version") != "analogy_v5":
        errs.append(f"version must be analogy_v5, got {data.get('version')}")
    ref = json.loads(V4_PATH.read_text(encoding="utf-8"))
    for key in ("laws", "pairs"):
        if data.get(key) != ref[key]:
            errs.append(f"{key} differ from analogy_v4.json")
    if data.get("seen_domains") != ref["seen_domains"] + ref["unseen_domains"]:
        errs.append("seen_domains must be analogy_v4 seen + unseen domains, verbatim and in order")
    old = set()
    for p in OLDER:
        d = json.loads(p.read_text(encoding="utf-8"))
        old |= {dom["name"] for part in ("seen_domains", "unseen_domains") for dom in d.get(part, [])}
    reused = sorted(dom["name"] for dom in data.get("unseen_domains", []) if dom["name"] in old)
    if reused:
        errs.append(f"unseen domain names reused from v2/v3/v4: {reused}")
    return errs


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
            hit, tot = v4.lexical(data, n, mode, v4.BRANCH_STAGES)
            h0, t0 = v4.lexical(data, n, mode, (0,))
            acc = sum(hit.values()) / sum(tot.values())
            per = "  ".join(f"s{st} {hit[st] / tot[st]:.0%}" for st in v4.BRANCH_STAGES)
            print(f"   lexical n={n} {mode:8s}: stages 2-4 {acc:.1%} ({per})   [stage 0 {h0[0] / t0[0]:.0%}]")
            best = max(best, acc)
    v2 = best <= v4.V2_LIMIT
    print(f"V2 lexical pair accuracy, stages 2-4 (best variant) {best:.1%} <= {v4.V2_LIMIT:.0%}: "
          f"{'PASS' if v2 else 'FAIL'}  (chance 50%)")
    ok &= v2

    leaks = v4.cue_leaks(data)
    print(f"V3 shared cue substrings: {'PASS' if not leaks else 'FAIL'}")
    for law, st, sub, frac in leaks[:20]:
        print(f"   - law {law} stage {st}: '{sub}' in {frac} unseen sentences")
    ok &= not leaks

    print("ACCEPTED" if ok else "REJECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT))
