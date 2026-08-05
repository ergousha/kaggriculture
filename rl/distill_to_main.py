"""Strategy Distillation & Main Agent Update Utility.

Evaluates RL-discovered strategy vectors against baseline main.py and distills
verified strategy constant improvements back into main.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from rl.kaggriculture_env import KaggricultureGymEnv
from rl.strategy_space import StrategySpace


class StrategyDistiller:
    """Evaluates and applies RL-discovered strategy parameters to main.py."""

    def __init__(self, main_path: str = "main.py") -> None:
        self.main_path = os.path.abspath(main_path)
        self.space = StrategySpace()

    def evaluate_strategy_file(
        self,
        strategy_file: str,
        opponents: list[str] | None = None,
        episodes: int = 20,
        base_seed: int = 500,
    ) -> dict[str, Any]:
        """Benchmark strategy parameters from a JSON file against baseline main.py."""
        if not os.path.exists(strategy_file):
            raise FileNotFoundError(f"Strategy file not found: {strategy_file}")

        with open(strategy_file, "r") as f:
            data = json.load(f)

        strategy_vec = data.get("strategy_vector")
        param_dict = data.get("param_dict")

        if strategy_vec is None and param_dict is not None:
            strategy_vec = self.space.dict_to_vector(param_dict)
        elif strategy_vec is not None and param_dict is None:
            param_dict = self.space.vector_to_dict(strategy_vec)

        if strategy_vec is None:
            raise ValueError("JSON file must contain 'strategy_vector' or 'param_dict'")

        opponents = opponents or ["baseline", "adaptive"]
        default_vec = self.space.get_default_vector()

        print(f"=== Distillation Evaluation ===")
        print(f"Candidate File: {strategy_file}")
        print(f"Testing against: {', '.join(opponents)} over {episodes} episodes...")

        cand_env = KaggricultureGymEnv(agent_base_path=self.main_path)
        
        cand_results = []
        base_results = []

        try:
            for opp in opponents:
                cand_env.set_opponent(opp)
                for i in range(episodes):
                    seed = base_seed + i
                    # Run candidate
                    res_cand = cand_env.evaluate_strategy(strategy_vec, seed=seed)
                    cand_results.append(res_cand)

                    # Run baseline
                    res_base = cand_env.evaluate_strategy(default_vec, seed=seed)
                    base_results.append(res_base)

            cand_mean_cash = sum(r["me_cash"] for r in cand_results) / len(cand_results)
            base_mean_cash = sum(r["me_cash"] for r in base_results) / len(base_results)
            delta = cand_mean_cash - base_mean_cash
            pct_change = (delta / max(1.0, base_mean_cash)) * 100.0

            cand_wins = sum(r["win"] for r in cand_results)
            base_wins = sum(r["win"] for r in base_results)

            print(f"\n--- Benchmark Results ---")
            print(f"Baseline Mean Cash:  ${base_mean_cash:,.2f} | Win Rate: {cand_wins/len(cand_results)*100:.1f}%")
            print(f"Candidate Mean Cash: ${cand_mean_cash:,.2f} | Win Rate: {base_wins/len(base_results)*100:.1f}%")
            print(f"Cash Advantage:      ${delta:+,.2f} ({pct_change:+.2f}%)\n")

            return {
                "cand_mean_cash": cand_mean_cash,
                "base_mean_cash": base_mean_cash,
                "delta": delta,
                "pct_change": pct_change,
                "param_dict": param_dict,
                "strategy_vec": strategy_vec,
                "improved": delta > 0,
            }
        finally:
            cand_env.close()

    def apply_to_main(self, param_dict: dict[str, Any]) -> None:
        """Apply parameter overrides directly to main.py."""
        print(f"[Distiller] Applying strategy parameters to {self.main_path}...")
        self.space.apply_to_file(self.main_path, self.main_path, param_dict)
        print("[Distiller] main.py successfully updated!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Strategy Distillation Utility")
    parser.add_argument("strategy_file", help="Path to RL strategy JSON file")
    parser.add_argument("--agent", default="main.py", help="Path to main.py")
    parser.add_argument("--episodes", type=int, default=10, help="Benchmark episodes")
    parser.add_argument("--apply", action="store_true", help="Apply to main.py if improved")
    args = parser.parse_args()

    distiller = StrategyDistiller(main_path=args.agent)
    res = distiller.evaluate_strategy_file(args.strategy_file, episodes=args.episodes)

    if args.apply:
        if res["improved"]:
            distiller.apply_to_main(res["param_dict"])
        else:
            print("[Distiller] Skipping update: Candidate did not outperform baseline.")


if __name__ == "__main__":
    main()
