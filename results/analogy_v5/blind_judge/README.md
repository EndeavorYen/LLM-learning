# v5 blind judgement (issue #21)

An independent judge that never saw the answer key read `items.json` only (domain, stage 0, stage 1,
step, options A/B in shuffled order) and wrote `answers.json`. `key.json` maps each id to the true
option, law, stage and domain.

Result: **144 / 144 correct, 0 items flagged ambiguous** (stage 2: 48/48, stage 3: 48/48, stage 4: 48/48;
every law 24/24). Bar (as for v4): >= 90% per stage with ambiguous items reviewed. The v5 unseen
domains are accepted as answerable from stage 0 + stage 1.
