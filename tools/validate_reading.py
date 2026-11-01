"""Acceptance check for Lab 18a "reading breadth" data (issue #25, registration "Lab 18a「讀得廣」").

    python tools/validate_reading.py train <train_groups.json> [--partial]
    python tools/validate_reading.py test  <test.json> [--partial]
    python tools/validate_reading.py train2 <train_groups_v2.json> [--partial]   (Lab 18b: 1 group per event per domain,
        + CHATTER_B / POSNEG_B / OTHER_B / TRANSFER_B (IN / OUT); version reading_v2_train)
    python tools/validate_reading.py test2  <test_v2.json> [--partial]          (Lab 18b: as test; version reading_v2_test;
        names must not overlap the reading_v1 test domains either)

A group = one base sentence (an actual IN / OUT / BREAK / FIX of THIS system) + variants. Labels follow from the
variant type (label_of below), so the files carry text only.
R1  header: version, templates_sha256 / extra_domains_sha256 = the current files. train: the 28 training domains
    (16 of templates.json split train + 12 of extra_domains.json), names exactly (--partial: a subset). test: 8 new
    domains (--partial: any number) whose names do not overlap any templates / extra / probe domain name.
R2  train: per domain 8 groups, 2 per event; test: 12 groups, 3 per event, + noop (4) + support (3 per event type,
    NOOP included) + rules / measure / safeguard_name / safeguard {ok, broken}.
R3  variants: train = VARIANTS_TRAIN exactly (TRANSFER only in IN / OUT groups); test = the same + HYPO.
R4  every text: non-empty, one sentence ending in '。' (QUESTION: '？' or '吗。' style allowed: '？' or '。'), no
    Arabic digit, no '\n'.
R5  train texts never contain a hypothetical cue (HYPO_CUES); test HYPO texts contain at least one.
R6  no test-split domain name / safeguard name (templates.json) anywhere in the file.
R7  no duplicate text within the file; no text equal to a string of templates / extra / hard / hard_extra / probe
    (train texts are also checked against each other's cores).
Exit status 1 on any error.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks" / "state_tracking_v1"
PROBE = ROOT / "benchmarks" / "comprehension_probe_v1" / "probe.json"
EVENTS4 = ("IN", "OUT", "BREAK", "FIX")
VARIANTS_TRAIN = ("NEG", "INTENT", "QUESTION", "FUTURE", "POSNEG", "OTHER", "TRANSFER", "DIRSWAP", "PRONOUN", "CHATTER")
HYPO_CUES = ("如果", "要是", "假如", "假设", "万一", "倘若", "假使", "若", "的话")
OPP = {"IN": "OUT", "OUT": "IN", "BREAK": "FIX", "FIX": "BREAK"}


def label_of(event, variant):
    """Gold label of a variant of a group whose base is `event` (a '_B' suffix marks a second variant of that type)."""
    variant = variant[:-2] if variant.endswith("_B") else variant
    if variant in ("BASE", "POSNEG", "PRONOUN", "CHATTER", "TRANSFER"):
        return event
    if variant == "DIRSWAP":
        return OPP[event]
    return "NOOP"  # NEG, INTENT, QUESTION, FUTURE, OTHER, HYPO


def variants_for(event, test, extra_b=False):
    v = [x for x in VARIANTS_TRAIN if x != "TRANSFER" or event in ("IN", "OUT")]
    if extra_b:
        v += [x + "_B" for x in ("CHATTER", "POSNEG", "OTHER", "TRANSFER") if x != "TRANSFER" or event in ("IN", "OUT")]
    return v + (["HYPO"] if test else [])


def main(argv):
    partial = "--partial" in argv
    argv = [a for a in argv if a != "--partial"]
    kind_arg, path = argv[1], Path(argv[2])
    assert kind_arg in ("train", "test", "train2", "test2"), kind_arg
    v2 = kind_arg.endswith("2")
    kind = kind_arg.rstrip("2")
    tpl_raw, ext_raw = (BENCH / "templates.json").read_bytes(), (BENCH / "extra_domains.json").read_bytes()
    tpl, ext = json.loads(tpl_raw), json.loads(ext_raw)
    probe = json.loads(PROBE.read_bytes())
    raw = path.read_text(encoding="utf-8")
    p = json.loads(raw)
    errs = []
    want_v = ("reading_v2_" if v2 else "reading_v1_") + kind
    if p.get("version") != want_v:
        errs.append(f"R1 version {p.get('version')!r} != {want_v}")
    if p.get("templates_sha256") != hashlib.sha256(tpl_raw).hexdigest():
        errs.append("R1 templates_sha256")
    if p.get("extra_domains_sha256") != hashlib.sha256(ext_raw).hexdigest():
        errs.append("R1 extra_domains_sha256")
    train_names = sorted([d["name"] for d in tpl["domains"] if d["split"] == "train"] + [d["name"] for d in ext["domains"]])
    all_names = [d["name"] for d in tpl["domains"]] + [d["name"] for d in ext["domains"]] + [d["name"] for d in probe["domains"]]
    v1_test_path = ROOT / "benchmarks" / "reading_v1" / "test.json"
    v1_train_path = ROOT / "benchmarks" / "reading_v1" / "train_groups.json"
    v1_test = json.loads(v1_test_path.read_bytes()) if v2 and v1_test_path.exists() else {"domains": []}
    v1_train = json.loads(v1_train_path.read_bytes()) if v2 and v1_train_path.exists() else {"domains": []}
    all_names += [d["name"] for d in v1_test["domains"]]
    doms = p.get("domains", [])
    names = [d.get("name") for d in doms]
    if len(set(names)) != len(names):
        errs.append("R1 duplicate domain name")
    if kind == "train":
        if not set(names) <= set(train_names):
            errs.append(f"R1 not training domains: {sorted(set(names) - set(train_names))}")
        if not partial and sorted(names) != train_names:
            errs.append(f"R1 need all 28 training domains, missing {sorted(set(train_names) - set(names))}")
    else:
        if not partial and len(doms) != 8:
            errs.append(f"R1 need 8 test domains, got {len(doms)}")
        for n in names:
            if any(n in a or a in n for a in all_names):
                errs.append(f"R1 test domain name {n!r} overlaps an existing domain name")
    test_words = [w for d in tpl["domains"] if d["split"] == "test" for w in (d["name"], d["safeguard_name"])]
    for w in test_words:
        if w in raw:
            errs.append(f"R6 test-split name {w!r} occurs in the file")

    texts = []  # (where, text, variant)
    n_groups = (1 if v2 else 2) if kind == "train" else 3
    for d in doms:
        nm = d.get("name")
        gs = d.get("groups", [])
        cnt = {e: sum(g.get("event") == e for g in gs) for e in EVENTS4}
        if len(gs) != 4 * n_groups or any(c != n_groups for c in cnt.values()):
            errs.append(f"R2 {nm}: need {n_groups} groups per event, got {cnt} ({len(gs)} groups)")
        ids = [g.get("id") for g in gs]
        if len(set(ids)) != len(ids):
            errs.append(f"R2 {nm}: duplicate group id")
        for g in gs:
            e = g.get("event")
            if e not in EVENTS4:
                errs.append(f"R2 {nm} {g.get('id')}: event {e!r}")
                continue
            texts.append((f"{nm}/{g.get('id')}/BASE", g.get("base"), "BASE"))
            want = variants_for(e, kind == "test", extra_b=(v2 and kind == "train"))
            got = list(g.get("variants", {}))
            if sorted(got) != sorted(want):
                errs.append(f"R3 {nm} {g.get('id')}: variants {sorted(got)} != {sorted(want)}")
            for v, t in g.get("variants", {}).items():
                texts.append((f"{nm}/{g.get('id')}/{v}", t, v))
        if kind == "test":
            for k in ("rules", "measure", "safeguard_name"):
                if not isinstance(d.get(k), str) or not d.get(k):
                    errs.append(f"R2 {nm}: missing {k}")
            if set(d.get("safeguard", {})) != {"ok", "broken"}:
                errs.append(f"R2 {nm}: safeguard needs ok / broken")
            if len(d.get("noop", [])) != 4:
                errs.append(f"R2 {nm}: need 4 noop sentences")
            for i, t in enumerate(d.get("noop", [])):
                texts.append((f"{nm}/noop{i}", t, "NOOPSENT"))
            sup = d.get("support", {})
            for e in EVENTS4 + ("NOOP",):
                if len(sup.get(e, [])) != 3:
                    errs.append(f"R2 {nm}: support {e} needs 3 sentences")
                for i, t in enumerate(sup.get(e, [])):
                    texts.append((f"{nm}/support/{e}{i}", t, "SUPPORT"))
    known = []
    for d in tpl["domains"] + ext["domains"] + probe["domains"]:
        for lst in d["events"].values():
            known += lst
    for hf in ("hard_cases.json", "hard_cases_extra.json"):
        h = json.loads((BENCH / hf).read_bytes())
        known += [it["text"] for dd in h["domains"] for it in dd["items"]]
    for d in probe["domains"]:
        pr = d["probe"]
        for k in ("DIR", "SYS", "MENTION"):
            known += [q[s]["text"] for q in pr[k] for s in "ab"]
        known += [it["text"] for g in pr["REAL"] for it in g["items"]]
        known += [x["text"] for k in ("CUE", "STR") for x in pr[k]]
    for src in (v1_train, v1_test):  # Lab 18b: no repeat of a reading_v1 sentence
        for d in src["domains"]:
            for g in d["groups"]:
                known += [g["base"], *g["variants"].values()]
            known += d.get("noop", []) + [t for lst in d.get("support", {}).values() for t in lst]
    known = set(known)
    seen = {}
    for where, t, v in texts:
        if not isinstance(t, str) or not t.strip():
            errs.append(f"R4 {where}: empty")
            continue
        end_ok = t.endswith("。") or (v == "QUESTION" and t.endswith("？"))
        inner = t[:-1]
        if not end_ok or any(c in inner for c in "。？！!?") or "\n" in t:
            errs.append(f"R4 {where}: one sentence ending in '。' ({t})")
        if any(c.isdigit() and c.isascii() for c in t) or any(c in "０１２３４５６７８９" for c in t):
            errs.append(f"R4 {where}: Arabic digit ({t})")
        has_cue = any(c in t for c in HYPO_CUES)
        if kind == "train" and has_cue:
            errs.append(f"R5 {where}: hypothetical cue in a training text ({t})")
        if kind == "test" and v == "HYPO" and not has_cue:  # noqa
            errs.append(f"R5 {where}: HYPO text without a hypothetical cue ({t})")
        if t in seen:
            errs.append(f"R7 {where}: repeats {seen[t]} ({t})")
        seen.setdefault(t, where)
        if t in known:
            errs.append(f"R7 {where}: equals an existing sentence ({t})")
    for e in errs:
        print("ERROR:", e)
    if errs:
        sys.exit(f"validate_reading: {len(errs)} error(s)")
    print(f"PASS ({kind_arg}): {len(doms)} domains, {len(texts)} sentences; R1-R7")


if __name__ == "__main__":
    main(sys.argv)
