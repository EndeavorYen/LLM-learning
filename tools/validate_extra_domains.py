"""Acceptance check for benchmarks/state_tracking_v1/extra_domains.json (Lab 16 part B, issue #25).

    python tools/validate_extra_domains.py [extra_domains.json] [templates.json] [--partial]

--partial skips the domain-count check of X1 (for checking a writer's part file of a few domains).

The extra file adds new TRAIN domains to state_tracking_v1 for the domain-count curve. It is checked with the very
same template rules as templates.json (T1-T3 of tools/validate_state_tracking.py, imported, not copied), applied to
the 24 registered domains + the extra ones together, so no extra string may repeat any registered string. In addition:
  X1  header: version "state_tracking_v1_extra", templates_sha256 == sha256(templates.json), 12 domains;
  X2  every extra domain has split "train" and exactly 7 / 7 / 6 / 6 / 6 IN / OUT / BREAK / FIX / NOOP paraphrases
      (the counts of every registered train domain, so per-domain class balance is unchanged);
  X3  no extra domain name equals or contains a registered domain name (or vice versa).
Exit status 1 on any error.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_state_tracking as V  # noqa: E402

N_EXTRA = 12
COUNTS = {"IN": 7, "OUT": 7, "BREAK": 6, "FIX": 6, "NOOP": 6}


def main(argv):
    partial = "--partial" in argv
    argv = [a for a in argv if a != "--partial"]
    extra_path = Path(argv[1]) if len(argv) > 1 else V.BENCH / "extra_domains.json"
    tpl_path = Path(argv[2]) if len(argv) > 2 else V.BENCH / "templates.json"
    tpl_raw = tpl_path.read_bytes()
    tpl = json.loads(tpl_raw)
    extra = json.loads(extra_path.read_bytes())
    errs = []
    if extra.get("version") != "state_tracking_v1_extra":
        errs.append(f"X1 version {extra.get('version')!r}")
    if extra.get("templates_sha256") != hashlib.sha256(tpl_raw).hexdigest():
        errs.append("X1 templates_sha256 does not match templates.json")
    doms = extra.get("domains", [])
    if len(doms) != N_EXTRA and not partial:
        errs.append(f"X1 need {N_EXTRA} domains, got {len(doms)}")
    reg_names = [d["name"] for d in tpl["domains"]]
    for d in doms:
        nm = d.get("name", "")
        if d.get("split") != "train":
            errs.append(f"X2 {nm}: split must be 'train'")
        ev = d.get("events", {})
        for e, n in COUNTS.items():
            if len(ev.get(e, [])) != n:
                errs.append(f"X2 {nm} {e}: need exactly {n} paraphrases, got {len(ev.get(e, []))}")
        for r in reg_names:
            if r in nm or nm in r:
                errs.append(f"X3 {nm}: overlaps the registered domain name {r}")
    combined = dict(tpl)
    combined["domains"] = tpl["domains"] + doms
    V.N_TRAIN_DOM = 16 + len(doms)          # the registered 16 + the extra ones; test stays 8
    errs += V.check_templates(combined)
    for e in errs:
        print("ERROR:", e)
    if errs:
        sys.exit(f"validate_extra_domains: {len(errs)} error(s)")
    print(f"PASS: {len(doms)} extra train domains ({', '.join(d['name'] for d in doms)}); T1-T3 on all "
          f"{len(combined['domains'])} domains; X1-X3")


if __name__ == "__main__":
    main(sys.argv)
