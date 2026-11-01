# Blind annotation (amount_v1)

Each item names a system (`system`, e.g. 书架), its unit (`measure`, e.g. 箱) and one sentence (Simplified Chinese).
Decide what ACTUALLY happened to THIS system according to the sentence:

- `event`:
  - `IN`: units actually entered this system.
  - `OUT`: units actually left this system.
  - `BREAK`: its safeguard broke.
  - `FIX`: its safeguard was fixed.
  - `NOOP`: nothing actually changed in this system. This covers denials, plans, questions, things happening to
    another system, and unrelated talk.
- `amount`: how many units entered or left (1, 2 or 3). Use 0 when the event is not IN or OUT.
  - Count only this system's units. Numbers about other things (times, prices, people doing the moving, other
    objects) do not count.
  - Treat a domain's own unit as one unit. For example, 「一批乘客」 at a bus stop counts as 1 when the measure is 批,
    and 「一车油」 counts as 1 when the measure is 车.

Write a JSON object `{"<id>": {"event": "...", "amount": n}, ...}` covering every item, plus an optional
`"_notes"` entry: `{"<id>": "why unsure"}`. Judge each sentence on its own; do not look for or use any other file.
