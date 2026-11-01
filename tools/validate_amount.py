"""Acceptance check for Lab 20 amount_v1 data (issue #25, registration "Lab 20（第二步 2b-1）「規則變活：容量＋數量」").

    python tools/validate_amount.py           (checks benchmarks/amount_v1/train.json and test.json)

A group = one real IN / OUT of exactly k units (k = 2 / 3) of THIS domain's system, in 5 forms:
BASE / NUMFORM / CHATTER_NUM -> (event, k); DIRSWAP -> (opposite, k); NEG -> NOOP.
R1  versions amount_v1_train / amount_v1_test; train domains = the 28 reading training domains, test domains = the 22
    story_v1 domains (same names).
R2  group ids (IN|OUT)(2|3)-i, i = 1..4 (train) / 1..2 (test); event / amount fields match the id; exactly 5 forms.
R3  every text: one sentence ending in '。', no Arabic digit, no line break, no hypothetical cue, no approximate amount
    word (两三 三两 几 若干 一些 一批 不少; 几 allowed in CHATTER_NUM asides).
R4  no test-split string of templates.json (domain / safeguard names, rules, event sentences) anywhere.
R5  no duplicate text within or across the two files; no text equal to a sentence of the existing data (templates,
    extra domains, hard cases, reading v1 / v2, probe, story_v1).
Exit status 1 on any error. Meaning (exact amount, direction, NEG) is checked by the independent blind check.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
B = ROOT / "benchmarks"
FORMS = ("BASE", "NUMFORM", "DIRSWAP", "NEG", "CHATTER_NUM")
HYPO = ("如果", "要是", "假如", "假设", "万一", "倘若", "假使", "若", "的话")
APPROX = ("两三", "三两", "几", "若干", "一些", "一批", "不少")


def strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from strings(v)


def main():
    errs = []
    tpl = json.loads((B / "state_tracking_v1" / "templates.json").read_bytes())
    forbid = set()
    for d in tpl["domains"]:
        if d["split"] == "test":
            forbid |= {d["name"], d["safeguard_name"], d["init"]["rules"]}
            forbid |= {t for lst in d["events"].values() for t in lst}
    existing = set()
    for p in ["state_tracking_v1/templates.json", "state_tracking_v1/extra_domains.json", "state_tracking_v1/hard_cases.json",
              "state_tracking_v1/hard_cases_extra.json", "reading_v1/train_groups.json", "reading_v1/test.json",
              "reading_v2/train_groups_v2.json", "reading_v2/test_v2.json", "comprehension_probe_v1/probe.json",
              "story_v1/stories.json"]:
        existing |= {s for s in strings(json.loads((B / p).read_bytes())) if s.endswith(("。", "？"))}
    want_doms = {"train": [d["name"] for d in json.loads((B / "reading_v2" / "train_groups_v2.json").read_bytes())["domains"]],
                 "test": [d["name"] for d in json.loads((B / "story_v1" / "stories.json").read_bytes())["domains"]]}
    seen = {}
    for kind, per in (("train", 4), ("test", 2)):
        data = json.loads((B / "amount_v1" / f"{kind}.json").read_bytes())
        if data.get("version") != f"amount_v1_{kind}":
            errs.append(f"{kind}: version {data.get('version')}")
        if sorted(d["name"] for d in data["domains"]) != sorted(want_doms[kind]):
            errs.append(f"{kind}: domains differ from the registered set")
        for d in data["domains"]:
            want = sorted(f"{e}{k}-{i}" for e in ("IN", "OUT") for k in (2, 3) for i in range(1, per + 1))
            if sorted(g["id"] for g in d["groups"]) != want:
                errs.append(f"{kind} {d['name']}: group ids")
            for g in d["groups"]:
                m = re.fullmatch(r"(IN|OUT)([23])-\d+", g["id"])
                if not m or g["event"] != m.group(1) or g["amount"] != int(m.group(2)):
                    errs.append(f"{kind} {d['name']} {g['id']}: event / amount fields")
                if sorted(g["forms"]) != sorted(FORMS):
                    errs.append(f"{kind} {d['name']} {g['id']}: forms")
                for f, s in g["forms"].items():
                    w = f"{kind} {d['name']} {g['id']} {f}: "
                    if not s or not s.endswith("。") or "\n" in s or s.count("。") != 1:
                        errs.append(w + "one sentence ending in 。")
                    if re.search(r"[0-9０-９]", s):
                        errs.append(w + "Arabic digit")
                    errs += [w + f"hypothetical cue {x}" for x in HYPO if x in s]
                    errs += [w + f"approximate word {x}" for x in APPROX if x in s and not (x == "几" and f == "CHATTER_NUM")]
                    errs += [w + "test-split string" for x in forbid if x in s][:1]
                    if s in existing:
                        errs.append(w + "equals an existing sentence")
                    if s in seen:
                        errs.append(w + f"duplicate of {seen[s]}")
                    seen[s] = f"{kind} {d['name']} {g['id']} {f}"
    print("\n".join(errs) if errs else f"PASS ({len(seen)} sentences)")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
