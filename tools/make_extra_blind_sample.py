"""Blind answerability sample of benchmarks/state_tracking_v1/extra_domains.json (Lab 16 part B, issue #25).

    python tools/make_extra_blind_sample.py OUT_DIR

Per extra domain: one event sentence of each type (IN, OUT, BREAK, FIX, NOOP) and two initial-state sentences
(level + "，" + safeguard + "。", as in Lab 15's table), drawn with random.Random(20261101); 7 x 12 = 84 items.
Writes OUT_DIR/items.json (what the judge sees: domain, rules, sentence, question, options, in shuffled order) and
OUT_DIR/key.json (id -> correct letter, class). Score: python tools/make_extra_blind_sample.py --score OUT_DIR
(reads OUT_DIR/answers.json = {id: letter}).
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTRA = ROOT / "benchmarks" / "state_tracking_v1" / "extra_domains.json"
EVENTS = ["IN", "OUT", "BREAK", "FIX", "NOOP"]
EV_OPTS = ["系统里的量增加一个单位", "系统里的量减少一个单位", "安全机制坏了/失效", "安全机制修好/恢复", "与系统无关的事（量和安全机制都不变）"]
INIT_CLASSES = ["L0_ok", "L1_ok", "L2_ok", "L3_ok", "L0_broken", "L1_broken", "L2_broken", "L3_broken"]
LETTERS = "ABCDEFGH"
SEED = 20261101


def make(out_dir: Path):
    x = json.loads(EXTRA.read_text(encoding="utf-8"))
    rng = random.Random(SEED)
    items, key = [], {}
    for d in x["domains"]:
        rules = d["init"]["rules"]
        for ei, e in enumerate(EVENTS):
            s = rng.choice(d["events"][e])
            items.append(dict(domain=d["name"], rules=rules, sentence=s, question="这句话描述的是下面哪一种事件？",
                              options=EV_OPTS, _ans=ei, _cls=e))
        for c in rng.sample(INIT_CLASSES, 2):
            lv, sg = int(c[1]), c[3:]
            s = d["init"]["level"][lv] + "，" + d["init"]["safeguard"][sg] + "。"
            opts = [d["state_desc"][k] for k in INIT_CLASSES]
            items.append(dict(domain=d["name"], rules=rules, sentence=s, question="这句话描述的是系统目前的哪一种状态？",
                              options=opts, _ans=INIT_CLASSES.index(c), _cls=c))
    rng.shuffle(items)
    out = []
    for i, it in enumerate(items, 1):
        key[str(i)] = dict(letter=LETTERS[it["_ans"]], cls=it["_cls"], domain=it["domain"], sentence=it["sentence"])
        out.append(dict(id=i, domain=it["domain"], rules=it["rules"], sentence=it["sentence"], question=it["question"],
                        options={LETTERS[j]: o for j, o in enumerate(it["options"])}))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "items.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(out)} items -> {out_dir}")


def score(out_dir: Path):
    key = json.loads((out_dir / "key.json").read_text(encoding="utf-8"))
    ans = json.loads((out_dir / "answers.json").read_text(encoding="utf-8"))
    ans = {str(k): (v["answer"] if isinstance(v, dict) else v) for k, v in ans.items()}
    assert set(ans) == set(key), "answers.json must answer every item exactly once"
    wrong = [(i, key[i], ans[i]) for i in key if ans[i].strip().upper() != key[i]["letter"]]
    print(f"correct {len(key) - len(wrong)}/{len(key)} ({100 * (len(key) - len(wrong)) / len(key):.1f}%)")
    for i, k, a in wrong:
        print(f"  item {i}: {k['domain']} {k['cls']} key {k['letter']} answered {a}: {k['sentence']}")


if __name__ == "__main__":
    if sys.argv[1] == "--score":
        score(Path(sys.argv[2]))
    else:
        make(Path(sys.argv[1]))
