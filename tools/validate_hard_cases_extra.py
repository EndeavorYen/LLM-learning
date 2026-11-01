"""Acceptance check for benchmarks/state_tracking_v1/hard_cases_extra.json (Lab 17, issue #25): the hard-case
sentences of the 12 extra train domains (extra_domains.json).

    python tools/validate_hard_cases_extra.py [hard_cases_extra.json] [--partial]

The very same rules as tools/validate_hard_cases.py (H1-H5, imported, not copied), applied with the 12 extra
domains as the train domains, plus:
  E1  version "state_tracking_v1_hard_extra"; extra_domains_sha256 == sha256(extra_domains.json);
  E2  no text equals, contains or is contained in a string of templates.json OR extra_domains.json (H5 widened),
      and no text repeats a sentence of hard_cases.json.
--partial: the file may cover a subset of the extra domains (a writer's part file).
Exit status 1 on any error.
"""
import copy
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_hard_cases as V  # noqa: E402

BENCH = V.BENCH
VERSION_EXTRA = "state_tracking_v1_hard_extra"


def main(argv):
    partial = "--partial" in argv
    argv = [a for a in argv if a != "--partial"]
    path = Path(argv[1]) if len(argv) > 1 else BENCH / "hard_cases_extra.json"
    tpl_raw = (BENCH / "templates.json").read_bytes()
    ext_raw = (BENCH / "extra_domains.json").read_bytes()
    t, ext = json.loads(tpl_raw), json.loads(ext_raw)
    raw = path.read_bytes()
    h = json.loads(raw)
    errs = []
    if h.get("version") != VERSION_EXTRA:
        errs.append(f"E1 version {h.get('version')!r} != {VERSION_EXTRA!r}")
    if h.get("extra_domains_sha256") != hashlib.sha256(ext_raw).hexdigest():
        errs.append("E1 extra_domains_sha256 does not match extra_domains.json")
    names = {d.get("name") for d in h.get("domains", [])}
    # pseudo template set: the extra domains are "train", the registered test domains stay "test", the registered
    # train domains are kept (their strings count for the overlap check) under another split name
    pseudo = copy.deepcopy(t)
    for d in pseudo["domains"]:
        if d["split"] == "train":
            d["split"] = "registered_train"
    for d in ext["domains"]:
        if not partial or d["name"] in names:
            pseudo["domains"].append(dict(d, split="train"))
        else:
            pseudo["domains"].append(dict(d, split="extra_not_in_part"))
    hh = dict(h, version=V.VERSION, templates_sha256=hashlib.sha256(tpl_raw).hexdigest())
    errs += V.check(hh, raw.decode("utf-8"), pseudo, hashlib.sha256(tpl_raw).hexdigest())
    old = json.loads((BENCH / "hard_cases.json").read_bytes())
    old_texts = {it["text"] for d in old["domains"] for it in d["items"]}
    for d in h.get("domains", []):
        for it in d.get("items", []):
            if it.get("text") in old_texts:
                errs.append(f"E2 {it.get('id')}: repeats a hard_cases.json sentence")
    for e in errs:
        print("ERROR:", e)
    if errs:
        sys.exit(f"validate_hard_cases_extra: {len(errs)} error(s)")
    print(f"PASS: {len(names)} extra domains, {sum(len(d['items']) for d in h['domains'])} items; H1-H5, E1-E2")


if __name__ == "__main__":
    main(sys.argv)
