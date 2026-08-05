#!/usr/bin/env python3
"""Deep Parameter Gym Sweep for Kaggriculture.

Runs parallel multi-episode parameter sweeps across candidate agent configurations
against baseline, adaptive, and leaderboard replay opponents to discover optimal
strategic hyper-parameters.

Usage:
    python gym_sweep.py --episodes 20 --workers 4
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
MAIN_PY = os.path.join(HERE, "main.py")


def run_candidate(config: dict[str, Any], episodes: int, opponent: str) -> dict[str, Any]:
    """Create temporary agent variant with config, run local_arena, return summary."""
    sys.path.insert(0, HERE)
    import local_arena

    # Create temporary modified copy of main.py
    var_name = f"_tmp_agent_{int(time.time() * 1000) % 1000000}.py"
    var_path = os.path.join(HERE, var_name)

    try:
        with open(MAIN_PY) as f:
            code = f.read()

        # Apply config parameter overrides
        for k, v in config.items():
            import re

            pattern = rf"^({k}\s*=\s*)([^\n]+)"

            def _repl(m, val=v):
                return f"{m.group(1)}{val!r}"

            code = re.sub(pattern, _repl, code, flags=re.MULTILINE)

        with open(var_path, "w") as f:
            f.write(code)

        opp_resolved = local_arena.resolve_opponent(opponent, var_path, LOG_DIR)
        seeds = [2000 + i for i in range(episodes)]
        results, logs = local_arena.run_set(
            var_path, opp_resolved, seeds, 720, max(1, (os.cpu_count() or 2) - 1), False, "sweep"
        )
        agg = local_arena.aggregate(results, logs)
        return {
            "config": config,
            "mean_cash": agg["mean_cash"],
            "opp_mean_cash": agg["opp_mean_cash"],
            "win_rate": agg["win_rate"],
            "turn_p95": agg["turn_p95"],
            "crashes": agg["crashes"],
        }
    finally:
        if os.path.exists(var_path):
            try:
                os.remove(var_path)
            except OSError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Gym Parameter Sweep")
    parser.add_argument(
        "--episodes", type=int, default=15, help="Episodes per candidate evaluation"
    )
    parser.add_argument(
        "--opponent", default="leaderboard", help="Opponent: baseline | adaptive | leaderboard"
    )
    args = parser.parse_args()

    print(
        f"[GymSweep] Starting parameter search ({args.episodes} episodes per candidate vs {args.opponent})..."
    )

    # Search space grid
    max_hands_grid = [12, 13, 14]
    hire_frac_grid = [0.12, 0.1453, 0.18]
    straw_target_grid = [40, 45, 50]
    straw_frac_grid = [0.4065, 0.45]

    candidates = []
    for mh, hf, st, sf in itertools.product(
        max_hands_grid, hire_frac_grid, straw_target_grid, straw_frac_grid
    ):
        candidates.append(
            {
                "MAX_HANDS": mh,
                "HIRE_CASH_FRACTION": hf,
                "STRAWBERRY_TILE_TARGET": st,
                "STRAWBERRY_LAND_FRACTION": sf,
            }
        )

    print(f"[GymSweep] Evaluating {len(candidates)} candidate configurations...")
    best_res = None
    best_cash = -1.0

    evaluations = []
    for idx, cand in enumerate(candidates, 1):
        print(f"[GymSweep] [{idx}/{len(candidates)}] Evaluating candidate: {cand}...")
        res = run_candidate(cand, args.episodes, args.opponent)
        evaluations.append(res)
        print(
            f"  -> Win Rate: {res['win_rate']:.0%}, Mean Cash: ${res['mean_cash']:,.2f} (opp ${res['opp_mean_cash']:,.2f})"
        )

        if res["crashes"] == 0 and res["mean_cash"] > best_cash:
            best_cash = res["mean_cash"]
            best_res = res

    print("\n" + "=" * 60)
    print("[GymSweep] Sweep Complete! Best Configuration:")
    print(json.dumps(best_res, indent=2))

    # Save best configuration
    out_path = os.path.join(LOG_DIR, "gym_best_config.json")
    with open(out_path, "w") as f:
        json.dump({"best_config": best_res, "all_evaluations": evaluations}, f, indent=2)
    print(f"[GymSweep] Saved best configuration results to {out_path}")


if __name__ == "__main__":
    main()
