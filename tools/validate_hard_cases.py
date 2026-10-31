"""Acceptance check for benchmarks/state_tracking_v1/hard_cases.json (step 1b hard-case diagnostic, issue #25).

Plain Python, no torch:
    python tools/validate_hard_cases.py [hard_cases.json] [templates.json]

Errors (exit code 1):
  H1  header: version, templates_sha256 == sha256(templates.json), the 4 categories NEG / OTHER / TEMP / HEDGE.
  H2  domains: exactly the 16 train domains of templates.json, each once; no test-domain name or test
      safeguard name anywhere in the file.
  H3  items: 12 per domain, one per (category, surface) slot: NEG x IN/OUT/BREAK/FIX, OTHER x IN/OUT,
      TEMP x IN/OUT/BREAK/FIX, HEDGE x IN/BREAK; id == "<domain>-<category>-<surface>"; label NOOP for
      NEG and OTHER, the surface event for TEMP, UNCERTAIN for HEDGE.
  H4  sentence form, as for template events (tools/validate_state_tracking.py T2): one sentence ending
      in '。' with no ！？; no digit, no number word other than 一, no 满/空/半; IN/OUT-surface sentences
      contain "一" + measure exactly once, BREAK/FIX-surface ones never; BREAK/FIX-surface sentences name
      the domain's safeguard, IN/OUT-surface ones never.
  H5  uniqueness: no repeated text; no text equals, contains or is contained in any templates.json string
      (rules, level, safeguard, event, state_desc of all 24 domains; compared without the final '。').
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks" / "state_tracking_v1"
VERSION = "state_tracking_v1_hard"
CATEGORIES = ["NEG", "OTHER", "TEMP", "HEDGE"]
SLOTS = [("NEG", "IN"), ("NEG", "OUT"), ("NEG", "BREAK"), ("NEG", "FIX"),
         ("OTHER", "IN"), ("OTHER", "OUT"),
         ("TEMP", "IN"), ("TEMP", "OUT"), ("TEMP", "BREAK"), ("TEMP", "FIX"),
         ("HEDGE", "IN"), ("HEDGE", "BREAK")]
EVENT_FORBIDDEN = set("零二两三四五六七八九十百千半满空0123456789０１２３４５６７８９")
EXTRA_PUNCT = set("！？；!?;")


def want_label(cat, surface):
    return {"NEG": "NOOP", "OTHER": "NOOP", "TEMP": surface, "HEDGE": "UNCERTAIN"}[cat]


def template_strings(t):
    out = []
    for d in t["domains"]:
        ini = d["init"]
        out += [ini["rules"], *ini["level"], ini["safeguard"]["ok"], ini["safeguard"]["broken"]]
        for lst in d["events"].values():
            out += lst
        out += list(d["state_desc"].values())
    return out


def core(s):
    return s[:-1] if s.endswith("。") else s


def check(h, raw_text, t, tpl_sha):
    errs = []
    if h.get("version") != VERSION:
        errs.append(f"H1 version {h.get('version')!r} != {VERSION!r}")
    if h.get("templates_sha256") != tpl_sha:
        errs.append(f"H1 templates_sha256 {h.get('templates_sha256')} != sha256(templates.json) {tpl_sha}")
    cats = h.get("categories")
    if not isinstance(cats, dict) or sorted(cats) != sorted(CATEGORIES) \
            or not all(isinstance(v, str) and v for v in cats.values()):
        errs.append(f"H1 categories must describe exactly {CATEGORIES}")

    train = {d["name"]: d for d in t["domains"] if d["split"] == "train"}
    test = [d for d in t["domains"] if d["split"] == "test"]
    doms = h.get("domains", [])
    names = [d.get("name") for d in doms]
    if Counter(names) != Counter(list(train)):
        missing = sorted(set(train) - set(names))
        extra = sorted(set(names) - set(train))
        dup = sorted(n for n, c in Counter(names).items() if c > 1)
        errs.append(f"H2 domains must be the {len(train)} train domains once each: "
                    f"missing {missing}, not train {extra}, repeated {dup}")
    for d in test:
        for word in (d["name"], d["safeguard_name"]):
            if word in raw_text:
                errs.append(f"H2 test-split name '{word}' occurs in the file")

    tpl = [core(s) for s in template_strings(t)]
    seen = {}
    for dom in doms:
        nm = dom.get("name")
        td = train.get(nm)
        items = dom.get("items", [])
        slots = [(it.get("category"), it.get("surface")) for it in items]
        if len(items) != len(SLOTS) or Counter(slots) != Counter(SLOTS):
            errs.append(f"H3 {nm}: need 12 items, one per slot; got {len(items)}, "
                        f"{dict(Counter(f'{c}-{s}' for c, s in slots))}")
        if td is None:
            continue
        unit, sgn = "一" + td["measure"], td["safeguard_name"]
        for it in items:
            cat, surf, txt = it.get("category"), it.get("surface"), it.get("text")
            iid = it.get("id")
            if (cat, surf) not in SLOTS:
                errs.append(f"H3 {iid}: bad category/surface {cat}/{surf}")
                continue
            if iid != f"{nm}-{cat}-{surf}":
                errs.append(f"H3 {iid}: id must be '{nm}-{cat}-{surf}'")
            if it.get("label") != want_label(cat, surf):
                errs.append(f"H3 {iid}: label {it.get('label')!r} != {want_label(cat, surf)!r}")
            if not isinstance(txt, str) or not txt:
                errs.append(f"H4 {iid}: text missing")
                continue
            if not txt.endswith("。") or txt.count("。") != 1 or EXTRA_PUNCT & set(txt) or "\n" in txt:
                errs.append(f"H4 {iid}: must be one sentence ending in '。' ({txt})")
            bad = EVENT_FORBIDDEN & set(txt)
            if bad:
                errs.append(f"H4 {iid}: forbidden characters {''.join(sorted(bad))} ({txt})")
            if surf in ("IN", "OUT"):
                if txt.count(unit) != 1:
                    errs.append(f"H4 {iid}: must contain '{unit}' exactly once, found {txt.count(unit)} ({txt})")
                if sgn in txt:
                    errs.append(f"H4 {iid}: IN/OUT sentence must not name the safeguard '{sgn}' ({txt})")
            else:
                if unit in txt:
                    errs.append(f"H4 {iid}: BREAK/FIX sentence must not contain '{unit}' ({txt})")
                if sgn not in txt:
                    errs.append(f"H4 {iid}: BREAK/FIX sentence must name the safeguard '{sgn}' ({txt})")
            if txt in seen:
                errs.append(f"H5 {iid}: same text as {seen[txt]} ({txt})")
            seen.setdefault(txt, iid)
            c = core(txt)
            for s in tpl:
                if s in c or c in s:
                    errs.append(f"H5 {iid}: overlaps template string '{s}' ({txt})")
    return errs


def main(argv):
    hard_path = Path(argv[1]) if len(argv) > 1 else BENCH / "hard_cases.json"
    tpl_path = Path(argv[2]) if len(argv) > 2 else BENCH / "templates.json"
    tpl_raw = tpl_path.read_bytes()
    tpl_sha = hashlib.sha256(tpl_raw).hexdigest()
    t = json.loads(tpl_raw)
    raw = hard_path.read_bytes()
    raw_text = raw.decode("utf-8")
    h = json.loads(raw_text)
    print(f"templates:  {tpl_path}  sha256={tpl_sha}")
    print(f"hard cases: {hard_path}  sha256={hashlib.sha256(raw).hexdigest()}")
    errs = check(h, raw_text, t, tpl_sha)
    items = [it for d in h.get("domains", []) for it in d.get("items", [])]
    print(f"  {len(h.get('domains', []))} domains, {len(items)} items")
    print("  per category: " + " ".join(f"{c}={v}" for c, v in Counter(it.get("category") for it in items).items()))
    print("  per label:    " + " ".join(f"{c}={v}" for c, v in Counter(it.get("label") for it in items).items()))
    lens = [len(it["text"]) for it in items if isinstance(it.get("text"), str)]
    if lens:
        print(f"  text length {min(lens)}-{max(lens)} chars")
    if errs:
        print(f"\nFAIL: {len(errs)} error(s)")
        for e in errs[:80]:
            print("  " + e)
        return 1
    print("\nPASS: H1-H5 hard cases")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
