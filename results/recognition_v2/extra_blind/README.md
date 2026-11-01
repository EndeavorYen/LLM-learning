# Blind answerability check of the 12 extra training domains (Lab 16 part B, issue #25)

- **Data**: `benchmarks/state_tracking_v1/extra_domains.json`, sha256 `c26febd2…`. It passes
  `tools/validate_extra_domains.py`, which applies the registered template rules T1–T3 to all 36 domains, plus X1–X3.
- **Writers**: three independent writers, four domains each. They saw only a format specification and two
  registered training domains (水塔, 流浪狗收容所); none of them read `templates.json`.
- **Sample**: `tools/make_extra_blind_sample.py` drew 7 items per domain: one sentence per event type plus two
  initial-state sentences. That gives 84 items in shuffled order (`items.json`; the key is in `key.json`).
- **Judge**: an independent judge saw only `items.json`. Its answers are in `answers.json`.
- **Result: 84/84 correct**, above the registered threshold of ≥ 95%. The judge marked no item ambiguous.
- **Minor notes from the judge**, which did not change the result:
  - Five FIX sentences describe fitting a new safeguard without saying the old one was broken.
  - The zero-level phrases ("零管", "零辆" …) read stilted. The validator requires these phrases to contain 零, and
    the registered domains phrase them the same way (e.g. "此刻场内停着零辆车").
