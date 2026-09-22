#!/usr/bin/env python3
"""Out-of-sample test of the four route cells the refit selected stably.

    uv run python scripts/v052_holdout.py --enumerate --workers 10
    uv run python scripts/v052_holdout.py --test --workers 10
    uv run python scripts/v052_holdout.py --report

WHY A SEPARATE SCRIPT. `scripts/v052_route_refit.py` swept 41 tapes across 19 draws
against real rank 1-12 agents and its leave-one-replay-out criterion returned
-$151/draw on 70 folds (37/70 positive) against an in-sample max-of-41 gain of
+$5,650/draw. That is the winner's curse, measured. Fifteen of the 19 cells flip their
pick between folds, which is noise by definition.

Four cells did NOT flip -- the same tape won in every fold:

    PET_CAFE/YARN_STORE        11 -> 12     in-sample +$19,634  (4 replays)
    PIZZA_SHOP/SMOOTHIE_SHOP  124 ->  2     in-sample  +$8,873  (3 replays)
    PIZZA_SHOP/YARN_STORE     128 -> 115    in-sample  +$8,706  (4 replays)
    BRUNCH_SPOT/BAKERY        103 -> 110    in-sample  +$1,438  (3 replays)

For a stable pick, leave-one-out gives NO independent check: each replay is held out
exactly once, so the LOO mean equals the in-sample gain by construction. Those four
numbers are still selection-on-the-same-data, i.e. exactly the evidence class that
produced v0.4.3.

So the four are frozen as PRE-SPECIFIED hypotheses -- chosen before this script ran, not
re-searched here -- and tested on replays downloaded AFTER the sweep and never used in
selection. Two arms per cell (shipped vs candidate), every fresh replay whose observed
draw matches, both seats. The draw is observed per pairing, never read from the replay:
only 10 of 40 pairings reproduce the replay's own step-144 draw, because weed spawns
consume RNG on empty tiles only and agent occupancy diverges the stream.

GATE: a cell promotes only if its fresh-replay margin beats the shipped tape. Four
independent tests, so a single nominal winner at p<0.05 is expected by chance ~19% of
the time; the write-up must say which of the four were tested, not just which passed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
FRESH_LIST = HERE / "logs" / "v052_fresh_replays.txt"
FRESH_DRAWS = HERE / "logs" / "v052_fresh_draws.csv"
HOLDOUT_JSON = HERE / "logs" / "v052_holdout.json"

# Frozen before any fresh replay was touched. Do not edit to chase a result.
HYPOTHESES: dict[tuple[str, str], tuple[int, int]] = {
    ("PET_CAFE", "YARN_STORE"): (11, 12),
    ("PIZZA_SHOP", "SMOOTHIE_SHOP"): (124, 2),
    ("PIZZA_SHOP", "YARN_STORE"): (128, 115),
    ("BRUNCH_SPOT", "BAKERY"): (103, 110),
}


def fresh_replays() -> list[tuple[str, int]]:
    """Replays listed in FRESH_LIST (written by --enumerate from the new downloads)."""
    import v052_route_refit as refit

    wanted = {line.strip() for line in FRESH_LIST.read_text().splitlines() if line.strip()}
    return [(p, s) for p, s in refit.kaggriculture_replays() if Path(p).name in wanted]


def enumerate_fresh(workers: int) -> None:
    import v052_route_refit as refit

    import local_arena as la

    replays = fresh_replays()
    print(f"{len(replays)} fresh replays never used in selection")
    jobs: list[dict] = []
    for path, seed in replays:
        for swap in (False, True):
            jobs.append(
                {
                    "agent": str(HERE / "main.py"),
                    "opponent": refit.shim_for(path),
                    "seed": seed,
                    "steps": la.DEFAULT_STEPS,
                    "swap": swap,
                    "decision_log": None,
                    "replay": None,
                }
            )
    print(f"observing draws: {len(jobs)} episodes on {workers} workers")
    results = refit._run(jobs, workers)
    with open(FRESH_DRAWS, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["replay", "seed", "swap", "shop1", "shop2"])
        for job, res in zip(jobs, results, strict=True):
            pair = res.get("observed_pair") or ("", "")
            writer.writerow([job["opponent"], job["seed"], int(job["swap"]), pair[0], pair[1]])
    hits = dict.fromkeys(HYPOTHESES, 0)
    for row in csv.DictReader(open(FRESH_DRAWS)):
        cell = (row["shop1"], row["shop2"])
        if cell in hits:
            hits[cell] += 1
    print(f"\nwrote {FRESH_DRAWS}")
    print("fresh pairings hitting each frozen hypothesis:")
    for cell, n in hits.items():
        print(f"  {str(cell):34s} {n} pairings")


def test(workers: int) -> None:
    import v052_route_refit as refit

    import local_arena as la

    rows = [r for r in csv.DictReader(open(FRESH_DRAWS)) if (r["shop1"], r["shop2"]) in HYPOTHESES]
    jobs: list[dict] = []
    index: list[tuple] = []
    for row in rows:
        cell = (row["shop1"], row["shop2"])
        for route in HYPOTHESES[cell]:
            jobs.append(
                {
                    "agent": refit.variant_for(cell, route),
                    "opponent": row["replay"],
                    "seed": int(row["seed"]),
                    "steps": la.DEFAULT_STEPS,
                    "swap": bool(int(row["swap"])),
                    "decision_log": None,
                    "replay": None,
                }
            )
            index.append((cell, route, row["replay"], int(row["swap"])))
    print(f"{len(rows)} matching fresh pairings -> {len(jobs)} episodes on {workers} workers")
    results = refit._run(jobs, workers)
    out = []
    for (cell, route, replay, swap), res in zip(index, results, strict=True):
        out.append(
            {
                "shop1": cell[0],
                "shop2": cell[1],
                "route": route,
                "replay": replay,
                "swap": swap,
                "me": res.get("me_cash"),
                "opp": res.get("opp_cash"),
                "pair": list(res.get("observed_pair") or ()),
                "error": res.get("harness_error") or "",
            }
        )
    HOLDOUT_JSON.write_text(json.dumps(out, indent=1))
    print(f"wrote {HOLDOUT_JSON}")
    report()


def report() -> None:
    rows = [
        r for r in json.loads(HOLDOUT_JSON.read_text()) if not r["error"] and r["me"] is not None
    ]
    rows = [r for r in rows if list(r["pair"]) == [r["shop1"], r["shop2"]]]
    print("\nFRESH-REPLAY TEST of the four frozen cells (never used in selection)\n")
    print(
        f"  {'draw':32s} {'ship':>5s} {'cand':>5s} {'n pair':>7s} "
        f"{'shipped':>11s} {'candidate':>11s} {'delta':>11s} {'wins':>7s}"
    )
    verdicts: list[tuple[tuple[str, str], int, int, float | None, int, int]] = []
    for cell, (cur, cand) in HYPOTHESES.items():
        per: dict[tuple, dict[int, float]] = {}
        for r in rows:
            if (r["shop1"], r["shop2"]) != cell:
                continue
            per.setdefault((r["replay"], r["swap"]), {})[r["route"]] = r["me"] - r["opp"]
        paired = [(v[cur], v[cand]) for v in per.values() if cur in v and cand in v]
        if not paired:
            print(f"  {str(cell):32s} {cur:>5d} {cand:>5d} {'0':>7s}   no fresh coverage")
            verdicts.append((cell, cur, cand, None, 0, 0))
            continue
        a = st.fmean(x for x, _ in paired)
        b = st.fmean(y for _, y in paired)
        wins = sum(1 for x, y in paired if y > x)
        print(
            f"  {str(cell):32s} {cur:>5d} {cand:>5d} {len(paired):>7d} "
            f"{a:>+11,.0f} {b:>+11,.0f} {b - a:>+11,.0f} {wins:>4d}/{len(paired)}"
        )
        verdicts.append((cell, cur, cand, b - a, wins, len(paired)))
    tested = [v for v in verdicts if v[3] is not None]
    passing = [v for v in tested if (v[3] or 0.0) > 0 and v[4] * 2 > v[5]]
    print(f"\n  tested {len(tested)} of {len(HYPOTHESES)} frozen cells on fresh replays")
    print(f"  cells beating the shipped tape out of sample: {len(passing)}")
    for cell, cur, cand, delta, wins, n in passing:
        print(f"    {str(cell):32s} {cur} -> {cand}   ${delta:+,.0f}  {wins}/{n} pairings")
    if not passing:
        print("    none. Ship nothing: the shipped table survives an honest holdout.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--enumerate", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    if args.enumerate:
        enumerate_fresh(args.workers)
    elif args.test:
        test(args.workers)
    elif args.report:
        report()
    else:
        ap.error("pick --enumerate, --test or --report")


if __name__ == "__main__":
    main()
