"""Acceptance check for benchmarks/comprehension_probe_v1/probe.json (Lab 17b comprehension probe, issue #25).

    python tools/validate_probe.py [probe.json] [--partial]

P1  header: version "comprehension_probe_v1"; templates_sha256 / extra_domains_sha256 = the current files; 6 domains,
    split "probe" (--partial: any number).
P2  the normal part of every probe domain passes the registered template rules T1-T3 (tools/validate_state_tracking.py,
    imported) together with the 24 registered + 12 extra domains, with exactly 7 / 7 / 6 / 6 / 6 event paraphrases.
P3  probe items per domain: DIR 4 pairs (2 IN/OUT, 2 BREAK/FIX); REAL 2 groups (actual IN, actual BREAK) with forms
    actual / negated / intended / hypothetical (the last three NOOP); SYS 2 pairs (IN or OUT vs NOOP); MENTION 2 pairs
    (BREAK or FIX vs NOOP); CUE 10 (2 per event type); STR 8 (IN 2, OUT 2, FIX 1, BREAK 1, NOOP 2).
P4  every probe text: one sentence ending in '。' (no ！？;), no digit, no number character other than 一, no 满/空/半.
    CUE: IN/OUT never contain '一' + measure, BREAK/FIX never contain the safeguard name. MENTION: the NOOP sentence
    names the safeguard.
P5  no test-domain name or test safeguard name anywhere; no probe text repeats, contains or is contained in any
    string of templates.json, extra_domains.json, hard_cases*.json or another probe text.
Exit status 1 on any error.
"""
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_state_tracking as V  # noqa: E402

BENCH = V.BENCH
EVENTS = ["IN", "OUT", "BREAK", "FIX", "NOOP"]
FORBID = set("零二两三四五六七八九十百千半满空0123456789０１２３４５６７８９")


def core(s):
    return s[:-1] if s.endswith("。") else s


def main(argv):
    partial = "--partial" in argv
    argv = [a for a in argv if a != "--partial"]
    path = Path(argv[1]) if len(argv) > 1 else BENCH.parent / "comprehension_probe_v1" / "probe.json"
    tpl_raw = (BENCH / "templates.json").read_bytes()
    ext_raw = (BENCH / "extra_domains.json").read_bytes()
    tpl, ext = json.loads(tpl_raw), json.loads(ext_raw)
    raw = path.read_text(encoding="utf-8")
    p = json.loads(raw)
    errs = []
    if p.get("version") != "comprehension_probe_v1":
        errs.append(f"P1 version {p.get('version')!r}")
    if p.get("templates_sha256") != hashlib.sha256(tpl_raw).hexdigest():
        errs.append("P1 templates_sha256")
    if p.get("extra_domains_sha256") != hashlib.sha256(ext_raw).hexdigest():
        errs.append("P1 extra_domains_sha256")
    doms = p.get("domains", [])
    if len(doms) != 6 and not partial:
        errs.append(f"P1 need 6 domains, got {len(doms)}")
    # P2: template rules on the normal part
    comb = copy.deepcopy(tpl)
    comb["domains"] = tpl["domains"] + [dict(d, split="train") for d in ext["domains"]]
    for d in doms:
        if d.get("split") != "probe":
            errs.append(f"P1 {d.get('name')}: split must be 'probe'")
        nd = {k: d[k] for k in ("name", "measure", "safeguard_name", "init", "events", "state_desc") if k in d}
        comb["domains"].append(dict(nd, split="train"))
        for e, n in zip(EVENTS, (7, 7, 6, 6, 6)):
            if len(d.get("events", {}).get(e, [])) != n:
                errs.append(f"P2 {d.get('name')} {e}: need {n} paraphrases")
    V.N_TRAIN_DOM = 16 + len(ext["domains"]) + len(doms)
    errs += [f"P2 {e}" for e in V.check_templates(comb)]
    # strings that probe texts must not overlap
    known = []
    for d in comb["domains"]:
        ini = d["init"]
        known += [ini["rules"], *ini["level"], *ini["safeguard"].values(), *d["state_desc"].values()]
        for lst in d["events"].values():
            known += lst
    for hf in ("hard_cases.json", "hard_cases_extra.json"):
        h = json.loads((BENCH / hf).read_bytes())
        known += [it["text"] for dd in h["domains"] for it in dd["items"]]
    known_c = [core(s) for s in known]
    test_words = [w for d in tpl["domains"] if d["split"] == "test" for w in (d["name"], d["safeguard_name"])]
    for w in test_words:
        if w in raw:
            errs.append(f"P5 test-split name '{w}' occurs in the file")
    seen = {}
    for d in doms:
        nm, pr = d.get("name"), d.get("probe", {})
        unit, sgn = "一" + d.get("measure", "?"), d.get("safeguard_name", "?")
        items = []  # (kind, id, text, label)

        def need_pairs(kind, n, ok):
            prs = pr.get(kind, [])
            if len(prs) != n:
                errs.append(f"P3 {nm} {kind}: need {n} pairs, got {len(prs)}")
            for q in prs:
                la, lb = q["a"]["label"], q["b"]["label"]
                if not ok(la, lb):
                    errs.append(f"P3 {nm} {kind} {q.get('id')}: labels {la}/{lb}")
                items.extend([(kind, q.get("id") + "a", q["a"]["text"], la), (kind, q.get("id") + "b", q["b"]["text"], lb)])
            return prs

        dirs = need_pairs("DIR", 4, lambda a, b: {a, b} in ({"IN", "OUT"}, {"BREAK", "FIX"}))
        if sorted(tuple(sorted({q["a"]["label"], q["b"]["label"]})) for q in dirs) != \
                [("BREAK", "FIX"), ("BREAK", "FIX"), ("IN", "OUT"), ("IN", "OUT")]:
            errs.append(f"P3 {nm} DIR: need 2 IN/OUT and 2 BREAK/FIX pairs")
        need_pairs("SYS", 2, lambda a, b: a in ("IN", "OUT") and b == "NOOP")
        men = need_pairs("MENTION", 2, lambda a, b: a in ("BREAK", "FIX") and b == "NOOP")
        for q in men:
            if sgn not in q["b"]["text"]:
                errs.append(f"P4 {nm} MENTION {q.get('id')}: the NOOP sentence must name the safeguard")
        groups = pr.get("REAL", [])
        if sorted(g["items"][0]["label"] for g in groups if g.get("items")) != ["BREAK", "IN"] or len(groups) != 2:
            errs.append(f"P3 {nm} REAL: need 2 groups with actual IN and actual BREAK")
        for g in groups:
            forms = [it["form"] for it in g.get("items", [])]
            if forms != ["actual", "negated", "intended", "hypothetical"]:
                errs.append(f"P3 {nm} REAL {g.get('id')}: forms {forms}")
            for it in g.get("items", []):
                if it["form"] != "actual" and it["label"] != "NOOP":
                    errs.append(f"P3 {nm} REAL {g.get('id')} {it['form']}: label must be NOOP")
                items.append(("REAL", f"{g.get('id')}-{it['form']}", it["text"], it["label"]))
        for kind, n, want in (("CUE", 10, {e: 2 for e in EVENTS}),
                              ("STR", 8, {"IN": 2, "OUT": 2, "FIX": 1, "BREAK": 1, "NOOP": 2})):
            lst = pr.get(kind, [])
            if len(lst) != n or Counter(x["label"] for x in lst) != Counter(want):
                errs.append(f"P3 {nm} {kind}: need {want}, got {dict(Counter(x['label'] for x in lst))}")
            for x in lst:
                items.append((kind, x.get("id"), x["text"], x["label"]))
                if kind == "CUE":
                    if x["label"] in ("IN", "OUT") and unit in x["text"]:
                        errs.append(f"P4 {nm} CUE {x.get('id')}: must not contain '{unit}'")
                    if x["label"] in ("BREAK", "FIX") and sgn in x["text"]:
                        errs.append(f"P4 {nm} CUE {x.get('id')}: must not name the safeguard")
        for kind, iid, txt, lab in items:
            if lab not in EVENTS:
                errs.append(f"P3 {nm} {iid}: label {lab}")
            if not isinstance(txt, str) or not txt.endswith("。") or txt.count("。") != 1 or set("！？；!?;") & set(txt):
                errs.append(f"P4 {nm} {iid}: one sentence ending in '。' ({txt})")
                continue
            bad = FORBID & set(txt)
            if bad:
                errs.append(f"P4 {nm} {iid}: forbidden characters {''.join(sorted(bad))} ({txt})")
            if txt in seen and not (kind == "REAL" and seen[txt][0] == "REAL"):
                errs.append(f"P5 {nm} {iid}: repeats {seen[txt][1]} ({txt})")
            seen.setdefault(txt, (kind, f"{nm}/{iid}"))
            c = core(txt)
            for s in known_c:
                if s and (s in c or c in s):
                    errs.append(f"P5 {nm} {iid}: overlaps an existing string '{s}'")
                    break
    for e in errs:
        print("ERROR:", e)
    if errs:
        sys.exit(f"validate_probe: {len(errs)} error(s)")
    n = sum(len(d["probe"].get(k, [])) * (2 if k in ("DIR", "SYS", "MENTION") else 1)
            for d in doms for k in ("DIR", "SYS", "MENTION", "CUE", "STR")) + 8 * len(doms)
    print(f"PASS: {len(doms)} probe domains ({', '.join(d['name'] for d in doms)}), {n} probe sentences; P1-P5")


if __name__ == "__main__":
    main(sys.argv)
