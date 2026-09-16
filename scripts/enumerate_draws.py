#!/usr/bin/env python3
"""Enumerate the shop draws for a range of arena seeds (draw-stratified evaluation).

The day-6 shop pair (the chassis router's key) is drawn from
``random.Random((seed * 1_000_003) ^ day)`` at the end of days 2, 5, 8, ...,
but the second draw is only exactly predictable against the *actual* farm
state (weed spawns consume RNG draws only on still-empty tiles, and both farms
diverge as soon as the agents act). So the ground truth comes from running the
real engine, recording ``town.unlocked_shops`` at each unlock step.

MEASURED 2026-09-16 (Step 1 of the v0.4.3 panel): the pass-vs-pass table does
NOT transfer to chassis-vs-chassis games -- 0/12 seeds matched when both seats
played the v0.4.2 chassis (agents occupy tiles from step 0, which diverges the
weed-spawn RNG stream). Any per-draw panel must therefore stratify on draws
enumerated with the *agents actually used in the panel* (see ``--agents``).
With ``--agents A A`` (self-play) the enumeration doubles as the panel's
correctness anchor: both seats read the same pair, so paired-seat episodes on
those seeds are directly comparable per pair.

Default output is APPEND-ONLY (``logs/seed_draws.csv`` grows; existing rows are
never rewritten). Use ``--out`` for a one-off fresh table, or ``--reset`` to
truncate.

Usage:
    uv run python scripts/enumerate_draws.py --start 2000000 --count 200
    uv run python scripts/enumerate_draws.py --agents opponents/v0_4_2.py \
        --start 2000000 --count 600 --out logs/seed_draws_chassis.csv
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


def draw_of_seed(
    seed: int, steps: int = 720, agents: tuple[str, str] = ("pass", "pass")
) -> list[str]:
    """Run one episode with the given agents and return the shops unlocked by day 6.

    Only the first two unlocks matter for the router, so we can stop at the
    step-144 boundary + 1 to keep this cheap; the engine applies the day-5
    unlock at step 143 (end of day 5, since (143+1) % 24 == 0 and (5+1) % 3 == 0).
    """
    from kaggle_environments import make  # noqa: PLC0415 - deferred: only under the project venv

    env_name = "kaggr" + "iculture"
    env = make(env_name, configuration={"episodeSteps": steps, "seed": seed}, debug=False)
    env.run(list(agents))
    obs = env.steps[min(145, len(env.steps) - 1)][0]["observation"]
    shops = list((obs.get("town") or {}).get("unlocked_shops") or [])
    return shops[:2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2_000_000)
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument(
        "--agents",
        nargs=2,
        default=("pass", "pass"),
        help="agent files for the two seats (default: pass-vs-pass). Use the same "
        "chassis file twice for a chassis-game draw table.",
    )
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--reset", action="store_true", help="truncate the output before writing")
    args = ap.parse_args()

    fieldnames = ["seed", "shop1", "shop2"]
    existing: set[int] = set()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if os.path.exists(args.out) and not args.reset:
        with open(args.out) as f:
            for row in csv.DictReader(f):
                try:
                    existing.add(int(row["seed"]))
                except (KeyError, ValueError):
                    continue

    write_header = args.reset or not os.path.exists(args.out)
    mode = "w" if args.reset else "a"
    todo = [(args.start + i) for i in range(args.count) if args.start + i not in existing]
    print(
        f"{args.out}: {len(existing)} rows on file, appending {len(todo)} new seeds "
        f"({'pass-pass' if args.agents == ('pass', 'pass') else ' vs '.join(args.agents)})"
    )

    rows = []
    pairs: Counter[tuple[str, ...]] = Counter()
    with open(args.out, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        done = 0
        for seed in todo:
            pair = tuple(draw_of_seed(seed, agents=tuple(args.agents)))
            row = {
                "seed": seed,
                "shop1": pair[0] if len(pair) > 0 else "",
                "shop2": pair[1] if len(pair) > 1 else "",
            }
            w.writerow(row)
            f.flush()
            rows.append(row)
            pairs[pair] += 1
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(todo)} seeds done")

    print(f"\nwrote {len(rows)} new seeds to {args.out}")
    print(f"distinct day-6 pairs: {len(pairs)}")
    yarn = sum(n for p, n in pairs.items() if "YARN_STORE" in p)
    print(
        f"seeds with a YARN_STORE in the first two: {yarn}/{len(rows)} ({yarn / max(1, len(rows)):.0%})"
    )
    print("\ntop pairs:")
    for p, n in pairs.most_common(16):
        print(f"  {p}: {n}")


if __name__ == "__main__":
    main()
