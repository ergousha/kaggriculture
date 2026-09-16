#!/usr/bin/env python3
"""v0.4.3 Step 1 final analysis: write logs/v043_draw_table.csv + the verdict.

The full greedy table (60/64 flips, chosen as per-pair max of 4 candidates on
refine's 12 seeds) carries in-sample EV +$1,636 but held-out (seeds[12:24]) EV
+$1,042 -- below the proposal's +$1,500 gate, the expected winner's-curse
discount. This script quantifies a CONSERVATIVE subset selected on IN-SAMPLE
criteria only (mean paired diff >= $1,000 AND >= 10/12 positive diffs) and
reports its held-out retention honestly (the filter never saw the held-out
episodes, so its held-out EV is a clean out-of-sample read).
"""

from __future__ import annotations

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import statistics  # noqa: E402
from collections import defaultdict  # noqa: E402

from scripts import v043_common as V  # noqa: E402
from scripts.v043_panel import (  # noqa: E402
    DEEP_SEEDS,
    EV_GATE,
    REFINE_SEEDS,
    _row_pair,
    cells_index,
    incumbent_choice,
    load_reference,
    load_stage,
    pair_frequencies,
    paired_diffs,
)

CONSERVATIVE_MIN_MEAN = 1_000.0
CONSERVATIVE_MIN_WINS = 10


def main() -> int:
    inc_table, yarn_table = V.extract_tables_from_incumbent()
    tapes = V.load_tapes()
    uniq = V.unique_tapes(tapes)
    panel = load_stage("screen") + load_stage("refine")
    idx = cells_index(panel)
    deep = load_stage("deep")
    ref = load_reference()
    freq = pair_frequencies()
    by_pair = V.seeds_by_pair(V.load_chassis_draws(), min_per_pair=DEEP_SEEDS)

    # ---- in-sample paired evidence per pair ----
    insample: dict[tuple, dict] = {}
    pairs = sorted({k[0] for k in idx})
    for pair in pairs:
        diffs = paired_diffs(pair, idx, uniq, inc_table, yarn_table)
        incumbent = incumbent_choice(pair, inc_table, yarn_table)
        best_id, best_mean, best_n, best_w = incumbent, 0.0, 0, 0
        for tape, ds in diffs.items():
            if len(ds) >= REFINE_SEEDS:
                m = statistics.fmean(ds)
                if m > best_mean:
                    best_id, best_mean, best_n = tape, m, len(ds)
                    best_w = sum(1 for d in ds if d > 0)
        insample[pair] = {
            "incumbent": incumbent,
            "best": best_id,
            "mean": best_mean,
            "n": best_n,
            "wins": best_w,
        }

    # ---- held-out paired evidence per changed pair ----
    holdout: dict[tuple, list[float]] = defaultdict(list)
    for r in deep:
        pair = _row_pair(r)
        if pair is None:
            continue
        seeds = by_pair.get(pair, [])
        if len(seeds) <= REFINE_SEEDS or not (seeds[REFINE_SEEDS] <= r["seed"] <= seeds[-1]):
            continue
        incumbent = incumbent_choice(pair, inc_table, yarn_table)
        sel = insample.get(pair, {}).get("best")
        if sel is None or sel == incumbent:
            continue
        ref_row = ref.get(r["seed"])
        if ref_row is None:
            continue
        inc_me = ref_row["opp_cash"] if r["swap"] else ref_row["me_cash"]
        inc_opp = ref_row["me_cash"] if r["swap"] else ref_row["opp_cash"]
        holdout[pair].append((r["me_cash"] - r["opp_cash"]) - (inc_me - inc_opp))

    # ---- conservative subset (in-sample criteria ONLY) ----
    conservative = []
    for pair, e in insample.items():
        if e["best"] == e["incumbent"]:
            continue
        if e["mean"] >= CONSERVATIVE_MIN_MEAN and e["wins"] >= CONSERVATIVE_MIN_WINS:
            conservative.append(pair)

    def ev(pairs_subset, source):
        total = 0.0
        for pair in pairs_subset:
            ds = source.get(pair) or []
            if ds:
                total += freq.get(pair, 0.0) * statistics.fmean(ds)
        return total

    full_flips = [p for p in pairs if insample[p]["best"] != insample[p]["incumbent"]]
    full_ev = ev(full_flips, holdout)
    cons_ev = ev(conservative, holdout)
    cons_held = {p: holdout.get(p, []) for p in conservative}
    cons_retained = sum(
        1 for p in conservative if cons_held[p] and statistics.fmean(cons_held[p]) > 0
    )

    # ---- write the deliverable ----
    os.makedirs(os.path.dirname(V.DRAW_TABLE_CSV), exist_ok=True)
    with open(V.DRAW_TABLE_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "shop1",
                "shop2",
                "route",
                "incumbent_route",
                "changed",
                "conservative",
                "insample_mean_delta",
                "insample_n",
                "insample_wins",
                "holdout_mean_delta",
                "holdout_n",
                "holdout_wins",
            ]
        )
        for pair in sorted(insample):
            e = insample[pair]
            hd = holdout.get(pair, [])
            w.writerow(
                [
                    pair[0],
                    pair[1],
                    e["best"],
                    e["incumbent"],
                    int(e["best"] != e["incumbent"]),
                    int(pair in conservative),
                    round(e["mean"], 1),
                    e["n"],
                    e["wins"],
                    round(statistics.fmean(hd), 1) if hd else "",
                    len(hd),
                    sum(1 for d in hd if d > 0),
                ]
            )
    print(f"wrote {V.DRAW_TABLE_CSV}")

    print(
        f"\nSTEP 1 VERDICT (n={len(panel)} panel + {len(deep)} deep episodes, {len(ref)} reference)"
    )
    print(
        f"  full greedy table      : {len(full_flips)}/64 flips, held-out EV {full_ev:+,.0f}$/game  (gate >= +{EV_GATE:,.0f})"
    )
    print(
        f"  conservative subset   : {len(conservative)} flips "
        f"(in-sample mean >= ${CONSERVATIVE_MIN_MEAN:,.0f} and >= {CONSERVATIVE_MIN_WINS}/12 positive), "
        f"held-out EV {cons_ev:+,.0f}$/game, {cons_retained}/{len(conservative)} pairs retained positive"
    )
    print(
        f"  in-sample greedy EV was +1,636 -> full-table held-out retention ~{full_ev / 1636:.0%}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
