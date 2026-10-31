"""Blind-judge sample for the Lab 14 state-tracking test set (issue #23, validity check V3).

Plain Python, deterministic:
    python tools/make_state_blind_sample.py ITEMS_DIR KEY_PATH

Writes ITEMS_DIR/items.json (what the judge sees) and KEY_PATH (the answers). KEY_PATH must not be
inside ITEMS_DIR. 54 test items with n = 6..10, stratified: 6 per final class, each class's 6 items in
6 different test domains (6-7 items per domain), lengths cycling over 6..10 (10-11 items per length).
Each item: id, init_text, event_texts, the question, and the domain's 9 state_desc strings as options
A-I in a per-item shuffled order. Item order is shuffled too, so ids carry no class information.
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks" / "state_tracking_v1"
SEED = 20261031 + 3
N_PER_CLASS = 6
LENGTHS = list(range(6, 11))
LETTERS = "ABCDEFGHI"
QUESTION = "按照第一句给出的规则，依次发生上面这些事之后，这个系统最终处于哪种状态？请从 A–I 中选一个。"


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    items_dir, key_path = Path(argv[1]).resolve(), Path(argv[2]).resolve()
    if items_dir == key_path.parent or items_dir in key_path.parents:
        print(f"refusing: the key {key_path} must not be inside the items folder {items_dir}")
        return 2
    t = json.loads((BENCH / "templates.json").read_text(encoding="utf-8"))
    st = json.loads((BENCH / "stories.json").read_text(encoding="utf-8"))
    classes = st["classes"]
    doms = {d["name"]: d for d in t["domains"]}
    test_doms = [d["name"] for d in t["domains"] if d["split"] == "test"]
    pool = defaultdict(list)
    for s in st["splits"]["test"]:
        pool[(s["final"], s["domain"], s["n"])].append(s)
    rng = random.Random(SEED)
    picks = []
    for j in range(len(classes) * N_PER_CLASS):
        cls = classes[j // N_PER_CLASS]
        dom = test_doms[j % len(test_doms)]
        n = LENGTHS[j % len(LENGTHS)]
        picks.append(rng.choice(pool[(cls, dom, n)]))
    rng.shuffle(picks)
    items, key = [], []
    for i, s in enumerate(picks, 1):
        order = classes[:]
        rng.shuffle(order)
        sd = doms[s["domain"]]["state_desc"]
        items.append({"id": i, "init_text": s["init_text"], "event_texts": s["event_texts"],
                      "question": QUESTION, "options": {LETTERS[k]: sd[c] for k, c in enumerate(order)}})
        key.append({"id": i, "answer": LETTERS[order.index(s["final"])], "final": s["final"],
                    "story_id": s["id"], "domain": s["domain"], "n": s["n"]})
    items_dir.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    (items_dir / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    key_path.write_text(json.dumps(key, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(items)} items to {items_dir / 'items.json'} and the key to {key_path}")
    print("  per class:  " + " ".join(f"{c}={v}" for c, v in sorted(Counter(k['final'] for k in key).items(), key=lambda x: classes.index(x[0]))))
    print("  per domain: " + " ".join(f"{d}={v}" for d, v in Counter(k['domain'] for k in key).items()))
    print("  per n:      " + " ".join(f"n{n}={v}" for n, v in sorted(Counter(k['n'] for k in key).items())))
    print("  answers:    " + " ".join(f"{a}={v}" for a, v in sorted(Counter(k['answer'] for k in key).items())))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
