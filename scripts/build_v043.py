#!/usr/bin/env python3
"""Build the v0.4.3 candidate: v0.4.2 + the MEASURED conservative day-6 routing table.

Step 2 of docs/proposal-2026-09-16-v043.md. The Step 1 panel measured the full
42-tape portfolio against the v0.4.2 chassis peer on paired seeds (both seats)
and produced logs/v043_draw_table.csv. This builder grafts the CONSERVATIVE
subset into the router and changes NOTHING else:

  * routes/tapes/layers/openings/endgame: byte-identical to v0.4.2;
  * the router reads ONE composed table: the incumbent's two static tables
    (EXP240 + YARN) merged, then the conservative overrides applied on top;
    every non-overridden pair keeps its incumbent assignment (incl. fallbacks);
  * the day-27 switch (route 2) is untouched — measured-correct in Step 3.

The splice is scripts/v043_common.make_table_variant, the exact mechanism the
Step 1 deep stage validated (452/452 episodes reproduced their constituent
cells; kept pairs matched the peer-vs-peer reference seat-for-seat).

The conservative subset is 19 assignments (the table's `conservative` column:
in-sample mean >= $1,000 AND >= 10/12 positive paired diffs, 19/20 retained
positive on held-out seeds 12-24). The 20th flip -- BRUNCH_SPOT/ICE_CREAM_SHOP
(105->110) -- measured -$1,105 HELD-OUT (6/6) despite passing the in-sample
filter, so per the held-out discipline it is DROPPED here (constant below).

Output: scratch/v0_4_3_candidate.py + a summary of what changed.
"""

from __future__ import annotations

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import v043_common as V  # noqa: E402

DRAW_TABLE = os.path.join(PROJECT_ROOT, "logs", "v043_draw_table.csv")
OUT = os.path.join(PROJECT_ROOT, "scratch", "v0_4_3_candidate.py")

# Measured 2026-09-16 (Step 1 held-out read): this conservative flip did not
# retain out-of-sample (BRUNCH_SPOT/ICE_CREAM_SHOP 105->110: held-out -$1,105,
# 6/6). Dropped from the shipped table.
DROPPED_RETRO_NEGATIVE = {("BRUNCH_SPOT", "ICE_CREAM_SHOP")}

PROVENANCE_NOTE = """v0.4.3: replaces 19 day-6 shop-pair assignments with the measured
  conservative routing table (logs/v043_draw_table.csv; Step 1 panel of
  docs/proposal-2026-09-16-v043.md). Method: paired-seed both-seat panel of the
  full 42-tape portfolio vs this chassis (v0.4.2 self-play), 11,688 screen+refine
  episodes, per-pair paired diffs vs the incumbent cell; assignments kept only
  at in-sample mean >= $1,000 and >= 10/12 positive, 19/20 retained positive on
  held-out seeds 12-24 (frequency-weighted EV +$1,038/game). Tapes, layers,
  openings and the day-27 endgame switch (route 2, measured-correct) are
  byte-identical to v0.4.2.

"""


def load_conservative_overrides() -> dict[tuple[str, str], int]:
    """{pair: route} for the conservative flips, minus retro-negative pairs."""
    overrides: dict[tuple[str, str], int] = {}
    with open(DRAW_TABLE) as f:
        for row in csv.DictReader(f):
            if row["conservative"] != "1":
                continue
            pair = (row["shop1"], row["shop2"])
            if pair in DROPPED_RETRO_NEGATIVE:
                continue
            overrides[pair] = int(row["route"])
    if not overrides:
        sys.exit("no conservative overrides found in the draw table")
    return overrides


def main() -> None:
    overrides = load_conservative_overrides()

    # Compose the shipped table: incumbent EXP + YARN tables, then overrides.
    # (greedy_table's Step-1 composition proved the merged lookup is
    # behaviorally identical for kept pairs; overrides only change their pairs.)
    inc_table, yarn_table = V.extract_tables_from_incumbent()
    composed: dict[tuple[str, str], int] = {}
    composed.update({tuple(k): int(v) for k, v in inc_table.items()})
    composed.update({tuple(k): int(v) for k, v in yarn_table.items()})
    n_yarn_overridden = sum(1 for p in overrides if "YARN_STORE" in p)
    composed.update(overrides)

    out = V.make_table_variant(composed, OUT, version_tag="0.4.3")

    # Provenance note in the header, after the existing v0.4.1 note.
    with open(out) as f:
        src = f.read()
    anchor = "WHY (measured 2026-09-14"
    assert src.count(anchor) == 1, "provenance anchor not found"
    src = src.replace(anchor, PROVENANCE_NOTE + anchor, 1)
    with open(out, "w") as f:
        f.write(src)

    print(f"wrote {out}")
    print(
        f"  composed table: {len(composed)} pairs ({len(inc_table)} EXP + {len(yarn_table)} YARN rows)"
    )
    print(f"  overrides applied: {len(overrides)} ({n_yarn_overridden} on YARN pairs)")
    print(f"  dropped retro-negative: {sorted(DROPPED_RETRO_NEGATIVE)}")
    for pair in sorted(overrides):
        print(f"    {pair[0]}/{pair[1]}: {composed[pair]}")


if __name__ == "__main__":
    main()
