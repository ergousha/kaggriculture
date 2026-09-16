#!/usr/bin/env python3
"""v0.4.3 Step 1 verification: realized-revenue decomposition on panel pairs.

The proposal asks the losing pairs' mechanism be verified as route-choice, not
noise: compare the realized revenue decomposition (scripts/analyse_market_fills
Recorder) of the table's tape vs the losing tape on the same seed, both seats.
A real route-choice effect shows up as a systematic product-mix difference
(animal units, sell $/unit, failed orders); noise does not repeat across seeds.

Usage:
    uv run python scripts/v043_fills.py --pair PIZZA_SHOP FARMERS_MARKET \
        --tape 106 --seeds 4
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from scripts import v043_common as V  # noqa: E402
from scripts.analyse_market_fills import Recorder, instrument, report  # noqa: E402


def run_one(agent: str, opponent: str, seed: int, seat: int) -> dict:
    rec = Recorder()
    instrument(rec)
    from kaggle_environments import make

    env = make(
        "kaggriculture",
        configuration={"episodeSteps": 720, "seed": seed},
        debug=False,
    )
    players = [agent, opponent]
    if seat == 1:
        players = [opponent, agent]
    env.run(players)
    final = env.steps[-1]
    rec.rewards = [s.reward for s in final]
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, required=True)
    ap.add_argument("--tape", type=int, required=True, help="the LOSING tape id to compare")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--agent-seat", type=int, default=0)
    args = ap.parse_args()

    pair = tuple(args.pair)
    inc_table, yarn_table = V.extract_tables_from_incumbent()
    incumbent = (
        inc_table.get(pair, 100) if pair.count("YARN_STORE") <= 0 else yarn_table.get(pair, 0)
    )
    draws = V.load_chassis_draws()
    seeds = [s for s in sorted(draws) if draws[s] == pair][: args.seeds]
    print(f"pair {pair}: incumbent tape {incumbent}, comparing tape {args.tape}")
    print(f"seeds: {seeds}\n")

    losing = V.make_forced_variant(args.tape)
    # incumbent tape == peer table choice, so the peer IS the incumbent cell
    for seed in seeds:
        print(f"=== seed {seed} (agent seat {args.agent_seat}, agent = tape {args.tape}) ===")
        rec = run_one(losing, V.INCUMBENT, seed, args.agent_seat)
        report(rec, seat=args.agent_seat)
        print()


if __name__ == "__main__":
    main()
