#!/usr/bin/env python3
"""Draw-stratified head-to-head: per-seed results joined to the day-6 shop draw.

The 120-seed aggregate gate hides per-draw behaviour, and the proposal's gate #4
is "no regression on adverse-draw seeds". This runner:

  1. loads the seed->draw table (scripts/enumerate_draws.py),
  2. runs N episodes of agent vs opponent on those seeds (alternating seats),
  3. reports per-draw-class win rate and mean cash delta.

Usage:
    uv run python scripts/draw_stratified.py --agent scratch/v0_4_1_candidate.py \
        --opponent opponents/v0_4_0.py --start 2000000 --count 60
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)

DRAWS_CSV = os.path.join(PROJECT_ROOT, "logs", "seed_draws.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--opponent", required=True)
    ap.add_argument("--start", type=int, default=2_000_000)
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    draws: dict[int, tuple[str, str]] = {}
    if os.path.exists(DRAWS_CSV):
        with open(DRAWS_CSV) as f:
            for row in csv.DictReader(f):
                draws[int(row["seed"])] = (row["shop1"], row["shop2"])
    seeds = [s for s in range(args.start, args.start + args.count) if s in draws]
    print(f"{len(seeds)}/{args.count} seeds have draws on file")

    # import the arena's episode runner (it installs instrumentation)
    import local_arena as la

    jobs = [
        {
            "agent": os.path.abspath(args.agent),
            "opponent": os.path.abspath(args.opponent),
            "seed": seed,
            "steps": 720,
            "swap": bool(i % 2),
            "decision_log": None,
            "replay": None,
        }
        for i, seed in enumerate(seeds)
    ]
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    with ctx.Pool(min(8, os.cpu_count() or 2)) as pool:
        results = pool.map(la.run_episode, jobs)

    per_seed = [
        {
            "seed": r["seed"],
            "draw": " ".join(draws[r["seed"]]),
            "yarn": "YARN_STORE" in draws[r["seed"]],
            "me": r["me_cash"],
            "opp": r["opp_cash"],
            "win": r["win"],
            "tie": r["tie"],
        }
        for r in results
    ]
    if args.json:
        with open(args.json, "w") as f:
            json.dump(per_seed, f, indent=1)

    by_draw: dict[str, list[dict]] = defaultdict(list)
    for row in per_seed:
        by_draw[row["draw"]].append(row)

    print(f"\n{'draw':<38} {'n':>3} {'W':>3} {'T':>3} {'L':>3} {'meanΔ$':>9}")
    yarn_rows = [r for r in per_seed if r["yarn"]]
    nonyarn_rows = [r for r in per_seed if not r["yarn"]]
    for draw, rows in sorted(by_draw.items(), key=lambda kv: -len(kv[1])):
        w = sum(r["win"] for r in rows)
        t = sum(r["tie"] for r in rows)
        l = len(rows) - w - t
        delta = sum(r["me"] - r["opp"] for r in rows) / len(rows)
        print(f"{draw:<38} {len(rows):>3} {w:>3} {t:>3} {l:>3} {delta:>+9,.0f}")

    def summarize(label: str, rows: list[dict]) -> None:
        w = sum(r["win"] for r in rows)
        t = sum(r["tie"] for r in rows)
        l = len(rows) - w - t
        delta = sum(r["me"] - r["opp"] for r in rows) / max(1, len(rows))
        print(f"{label:<38} {len(rows):>3} {w:>3} {t:>3} {l:>3} {delta:>+9,.0f}")

    print()
    summarize("ALL", per_seed)
    summarize("YARN draws (same routes both)", yarn_rows)
    summarize("non-YARN draws (EXP240 graft)", nonyarn_rows)

    losses = [r for r in per_seed if not r["win"] and not r["tie"]]
    if losses:
        print("\nworst 8 losses:")
        for r in sorted(losses, key=lambda r: r["me"] - r["opp"])[:8]:
            print(f"  seed {r['seed']} draw {r['draw']}: ${r['me']:,.0f} vs ${r['opp']:,.0f}")


if __name__ == "__main__":
    main()