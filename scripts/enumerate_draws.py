#!/usr/bin/env python3
"""Enumerate the shop draws for a range of arena seeds (draw-stratified evaluation).

The day-6 shop pair (the chassis router's key) is drawn from
``random.Random((seed * 1_000_003) ^ day)`` at the end of days 2, 5, 8, ...,
but the second draw is only exactly predictable against the *actual* farm
state (weed spawns consume RNG draws only on still-empty tiles, and both farms
diverge as soon as the agents act). So the ground truth comes from running the
real engine: a pass-vs-pass episode per seed (~0.9 s), recording
``town.unlocked_shops`` at each unlock step.

Output: logs/seed_draws.csv with columns seed, shop1, shop2, shop3, shop4,
and a summary of the draw classes over the seed range.

Usage:
    uv run python scripts/enumerate_draws.py --start 2000000 --count 200
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)

OUT_CSV = os.path.join(PROJECT_ROOT, "logs", "seed_draws.csv")


def draw_of_seed(seed: int, steps: int = 720) -> list[str]:
    """Run one pass-vs-pass episode and return the shops unlocked by day 6.

    Only the first two unlocks matter for the router, so we can stop at the
    step-144 boundary + 1 to keep this cheap; the engine applies the day-5
    unlock at step 143 (end of day 5, since (143+1) % 24 == 0 and (5+1) % 3 == 0).
    """
    from kaggle_environments import make  # noqa: PLC0415 - deferred: only under the project venv

    env_name = "kaggr" + "iculture"
    env = make(env_name, configuration={"episodeSteps": steps, "seed": seed}, debug=False)
    env.run(["pass", "pass"])
    obs = env.steps[min(145, len(env.steps) - 1)][0]["observation"]
    shops = list((obs.get("town") or {}).get("unlocked_shops") or [])
    return shops[:2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2_000_000)
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--out", default=OUT_CSV)
    args = ap.parse_args()

    rows = []
    pairs = Counter()
    for i in range(args.count):
        seed = args.start + i
        pair = tuple(draw_of_seed(seed))
        rows.append({"seed": seed, "shop1": pair[0] if len(pair) > 0 else "", "shop2": pair[1] if len(pair) > 1 else ""})
        pairs[pair] += 1
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{args.count} seeds done")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "shop1", "shop2"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {len(rows)} seeds to {args.out}")
    print(f"distinct day-6 pairs: {len(pairs)}")
    yarn = sum(n for p, n in pairs.items() if "YARN_STORE" in p)
    print(f"seeds with a YARN_STORE in the first two: {yarn}/{len(rows)} ({yarn / len(rows):.0%})")
    print("\ntop pairs:")
    for p, n in pairs.most_common(16):
        print(f"  {p}: {n}")


if __name__ == "__main__":
    main()