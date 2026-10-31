# Blind answerability check, round 1 (V3, issue #23)

54 test items (n = 6-10, 6 per class, all 8 unseen domains), judged by an independent judge that saw
only `items.json`. Result: **54/54 correct**, 2 flagged ambiguous (items 23 and 36, both 民宿).

The flags were the same problem: the BREAK sentence "合作民宿的老板跟这边闹翻了，不再收介绍的客人。"
followed by a FIX sentence describing a different cause being resolved ("老板旅游回来，开门营业了").
The same cause-mismatch risk existed in other test domains (e.g. 垃圾站: the driver is ill, then
"换了新轮胎"). Fix, before any model run:
- every FIX sentence of the 8 test domains is now cause-neutral;
- the 民宿 "闹翻" BREAK is replaced by a closure cause.

Only test-split texts changed: event sequences, labels, train and val_seen are identical.
The corrected data are re-judged in round 2.
