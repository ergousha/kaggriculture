"""Comprehensive Smoke Test Suite for Kaggriculture RL package."""

from __future__ import annotations

import os
import unittest

from rl.distill_to_main import StrategyDistiller
from rl.kaggriculture_env import KaggricultureGymEnv
from rl.offline_bc import OfflineBCAnalyzer
from rl.strategy_space import StrategySpace
from rl.train_strategy_rl import RLStrategyOptimizer


class TestRLSmoke(unittest.TestCase):
    """Smoke tests for RL package components."""

    def test_01_strategy_space_encoding(self) -> None:
        space = StrategySpace()
        self.assertGreater(space.dim, 0)

        default_vec = space.get_default_vector()
        self.assertEqual(len(default_vec), space.dim)
        for val in default_vec:
            self.assertGreaterEqual(val, -1.0)
            self.assertLessEqual(val, 1.0)

        param_dict = space.vector_to_dict(default_vec)
        self.assertIn("MAX_HANDS", param_dict)
        self.assertEqual(param_dict["MAX_HANDS"], 12)

        reconstructed_vec = space.dict_to_vector(param_dict)
        self.assertEqual(len(reconstructed_vec), space.dim)

        mutated = space.mutate(default_vec, scale=0.1, p_mutate=0.5)
        self.assertEqual(len(mutated), space.dim)

    def test_02_kaggriculture_gym_env(self) -> None:
        env = KaggricultureGymEnv(opponent_name="baseline")
        try:
            obs, info = env.reset(seed=42)
            self.assertIsNotNone(obs)
            self.assertIn("seed", info)

            # Evaluate strategy vector
            strategy_vec = env.strategy_space.get_default_vector()
            res = env.evaluate_strategy(strategy_vec, seed=42)
            self.assertIn("me_cash", res)
            self.assertIn("opp_cash", res)
            self.assertIn("win", res)
            self.assertEqual(res["status"], "DONE")
        finally:
            env.close()

    def test_03_rl_strategy_optimizer(self) -> None:
        optimizer = RLStrategyOptimizer(
            pop_size=2,
            elite_size=1,
            episodes_per_eval=2,
            opponents=["baseline"],
            output_dir="logs/rl_strategies_test",
        )
        res = optimizer.train(generations=1, base_seed=999)
        self.assertIn("fitness_score", res)
        self.assertIn("mean_cash", res)
        self.assertGreater(res["mean_cash"], 0.0)

    def test_04_offline_bc_analyzer(self) -> None:
        analyzer = OfflineBCAnalyzer()
        res = analyzer.analyze_dataset()
        self.assertIsInstance(res, dict)

    def test_05_strategy_distiller(self) -> None:
        best_file = "logs/rl_strategies/best_strategy.json"
        if os.path.exists(best_file):
            distiller = StrategyDistiller()
            res = distiller.evaluate_strategy_file(best_file, opponents=["baseline"], episodes=1)
            self.assertIn("cand_mean_cash", res)
            self.assertIn("base_mean_cash", res)


if __name__ == "__main__":
    unittest.main()
