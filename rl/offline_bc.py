"""Offline Behavior Cloning & Replay Dataset Analyzer for Kaggriculture.

Analyzes elite trajectories and decision log datasets (exported by EliteRecorder)
to extract empirical macro state-action patterns and warm-start strategy vectors.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from rl.strategy_space import StrategySpace


class OfflineBCAnalyzer:
    """Extracts strategic insights and strategy vectors from elite replay decision logs."""

    def __init__(self, dataset_path: str = "logs/elite_trajectories/dataset.jsonl") -> None:
        self.dataset_path = os.path.abspath(dataset_path)
        self.space = StrategySpace()

    def analyze_dataset(self) -> dict[str, Any]:
        """Analyze exported JSONL decision logs to extract macro strategy statistics."""
        if not os.path.exists(self.dataset_path):
            alt_path = os.path.abspath("logs/elite_dataset.jsonl")
            if os.path.exists(alt_path):
                self.dataset_path = alt_path

        if not os.path.exists(self.dataset_path):
            print(f"[OfflineBC] Dataset file not found at {self.dataset_path}", file=sys.stderr)
            return {}


        total_records = 0
        total_runs = set()
        day_cash: dict[int, list[float]] = {}
        day_hands: dict[int, list[int]] = {}

        with open(self.dataset_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    total_records += 1
                    meta = rec.get("meta", {})
                    run_id = meta.get("run_id", "unknown")
                    total_runs.add(run_id)

                    day = rec.get("day", 0)
                    money = rec.get("money", 0.0)
                    units = rec.get("n_units", 1)

                    day_cash.setdefault(day, []).append(money)
                    day_hands.setdefault(day, []).append(units)
                except json.JSONDecodeError:
                    continue

        avg_cash_by_day = {d: sum(vals) / len(vals) for d, vals in day_cash.items() if vals}
        max_hands_observed = max([max(v) for v in day_hands.values()]) if day_hands else 12

        print(f"=== Offline Behavior Cloning Analysis ===")
        print(f"Dataset Path:   {self.dataset_path}")
        print(f"Total Trajectories Analyzed: {len(total_runs)}")
        print(f"Total Turn Decision Records: {total_records}")
        print(f"Max Hands Observed in Elite Replays: {max_hands_observed}")
        if 10 in avg_cash_by_day:
            print(f"Avg Cash Day 10: ${avg_cash_by_day[10]:,.2f}")
        if 20 in avg_cash_by_day:
            print(f"Avg Cash Day 20: ${avg_cash_by_day[20]:,.2f}")
        if 30 in avg_cash_by_day:
            print(f"Avg Cash Day 30: ${avg_cash_by_day[30]:,.2f}")

        # Derive initial strategy overrides based on elite trajectories
        derived_strategy = {
            "MAX_HANDS": min(24, max(4, max_hands_observed)),
            "HIRE_CASH_FRACTION": 0.25,
            "LAND_CASH_BUFFER": 1000,
        }

        fit_vector = self.space.dict_to_vector(derived_strategy)

        return {
            "num_runs": len(total_runs),
            "total_records": total_records,
            "max_hands_observed": max_hands_observed,
            "derived_strategy": derived_strategy,
            "fit_vector": fit_vector,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline BC Dataset Analyzer")
    parser.add_argument(
        "--dataset",
        default="logs/elite_trajectories/dataset.jsonl",
        help="Path to exported decision log JSONL dataset",
    )
    args = parser.parse_args()

    analyzer = OfflineBCAnalyzer(dataset_path=args.dataset)
    analyzer.analyze_dataset()


if __name__ == "__main__":
    main()
