# Blind annotation (story_v2)

Each item is a short story about one system. `start` says the system's capacity and its state at the beginning:
how many units are in it, and whether its safeguard works. Then `sentences` describe things in order. Work out the
system's state at the END.

Rules:
- Only things that ACTUALLY happen to THIS system count. Denials, plans, questions, hypotheticals, things that
  happen to another system, and unrelated talk change nothing.
- Units entering: if they fit within the capacity, the level goes up by that many. If they do not fit:
  - safeguard works → the extra units are kept out and the level becomes exactly the capacity;
  - safeguard broken → the system bursts (BURST). After a burst nothing changes any more.
- Units leaving: the level goes down by that many, but never below 0.
- The safeguard breaking or being fixed changes only the safeguard.
- Numbers about other things (times, prices, the people doing the moving) do not count.

Answer per item: `L<level>_ok` / `L<level>_broken` (for example `L3_ok`, `L0_broken`) or `BURST`.
Write `{"<id>": "<answer>", ...}` (plus an optional `"_notes": {"<id>": "..."}`). Judge each story on its own; do not
open any other file.
