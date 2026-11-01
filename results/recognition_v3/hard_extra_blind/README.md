# Blind check of the extra-domain hard sentences (Lab 17, issue #25)

- **Data**: `benchmarks/state_tracking_v1/hard_cases_extra.json`, 12 extra domains × 12 sentences = 144. It passes
  `tools/validate_hard_cases_extra.py`, which applies H1–H5 of `validate_hard_cases.py` plus E1–E2.
- **Sample**: all 144 sentences, shuffled by `tools/make_hard_blind_sample.py --extra`. The key was kept outside the
  folder the judge could read.
- **Result: 144/144 correct.**
- **Flags**: the judge marked 3 items as only slightly ambiguous. They were reworded before the sha was pinned:
  - 离心机 OTHER-OUT now says 另一台;
  - 书架 OTHER-OUT now says 另一个书架;
  - 杠铃架 TEMP-FIX now says the pin was broken.
- **Same commit, `extra_domains.json` v1 → v1.1**:
  - 渡轮's safeguard 超载报警器 was renamed to 限载感应器, because v1 contained the test-domain safeguard name 超载报警
    (电梯).
  - `tools/validate_extra_domains.py` now checks for this (X4).
  - Lab 16's results were computed with v1 (sha `c26febd2…`).
