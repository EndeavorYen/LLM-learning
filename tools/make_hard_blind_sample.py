"""Blind-judge sample for hard_cases.json: all 192 sentences, shuffled, options A-F fixed; key kept outside."""
import json, random, sys
from collections import Counter
from pathlib import Path
BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "state_tracking_v1"
# usage: python tools/make_hard_blind_sample.py ITEMS_DIR KEY_PATH (the key must stay outside ITEMS_DIR)
ITEMS_DIR, KEY = Path(sys.argv[1]), Path(sys.argv[2])
assert ITEMS_DIR.resolve() not in KEY.resolve().parents, "key must be outside the items folder"
SEED = 20261031 + 25
QUESTION = "只根据这句话判断：对上面说明的这个系统来说，发生了什么？请从 A–F 中选一项。"
OPTIONS = {"A": "系统里的量增加一个单位", "B": "系统里的量减少一个单位", "C": "安全机制坏了",
           "D": "安全机制修好了", "E": "系统状态没有改变（与系统无关或没有发生）", "F": "无法确定是否发生"}
ANSWER = {"IN": "A", "OUT": "B", "BREAK": "C", "FIX": "D", "NOOP": "E", "UNCERTAIN": "F"}
t = json.loads((BENCH / "templates.json").read_text(encoding="utf-8"))
h = json.loads((BENCH / "hard_cases.json").read_text(encoding="utf-8"))
rules = {d["name"]: d["init"]["rules"] for d in t["domains"] if d["split"] == "train"}
rows = [(dom["name"], it) for dom in h["domains"] for it in dom["items"]]
rng = random.Random(SEED)
rng.shuffle(rows)
items, key = [], []
for i, (nm, it) in enumerate(rows, 1):
    items.append({"id": i, "system": rules[nm] + "。", "sentence": it["text"], "question": QUESTION, "options": OPTIONS})
    key.append({"id": i, "answer": ANSWER[it["label"]], "label": it["label"], "hard_id": it["id"],
                "domain": nm, "category": it["category"], "surface": it["surface"]})
ITEMS_DIR.mkdir(parents=True, exist_ok=True)
(ITEMS_DIR / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
KEY.write_text(json.dumps(key, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(len(items), "items;", "answers", dict(sorted(Counter(k["answer"] for k in key).items())))
print("first 3 hard ids:", [k["hard_id"] for k in key[:3]])
