"""Smoke tests for the macro-strategy search package."""

from __future__ import annotations

import os
import unittest

from search.cem import MacroCEM
from search.evolution import EvolutionSearch
from search.harness import MatchHarness
from search.objective import cvar, score_results, worst_episodes
from search.space import StrategySpace

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT_PATH = os.path.join(PROJECT_ROOT, "main.py")
# The searchable target. main.py became a route-replay agent in v0.2.0 and carries
# none of the tunable constants; the heuristic line's frozen head does.
HEURISTIC_PATH = os.path.join(PROJECT_ROOT, "opponents", "v0_1_1.py")


class TestStrategySpace(unittest.TestCase):
    def test_encoding_roundtrip(self) -> None:
        space = StrategySpace()
        self.assertGreater(space.dim, 0)

        default_vec = space.get_default_vector()
        self.assertEqual(len(default_vec), space.dim)
        for val in default_vec:
            self.assertGreaterEqual(val, -1.0)
            self.assertLessEqual(val, 1.0)

        param_dict = space.vector_to_dict(default_vec)
        self.assertIn("MAX_HANDS", param_dict)
        self.assertEqual(len(space.dict_to_vector(param_dict)), space.dim)
        self.assertEqual(len(space.mutate(default_vec, scale=0.1, p_mutate=0.5)), space.dim)

    def test_reads_live_agent_constants(self) -> None:
        """The searchable target is the heuristic agent family, not main.py.

        main.py became a route-replay agent in v0.2.0 and has none of these
        constants, so pointing this at main.py would assert that the submission is
        still a heuristic. The search stack is unchanged and still operates on the
        heuristic line; opponents/v0_1_1.py is its frozen head.
        """
        space = StrategySpace()
        target = HEURISTIC_PATH
        if not os.path.exists(target):
            self.skipTest(f"no heuristic agent snapshot at {target}")
        live = space.read_from_file(target)
        self.assertIn("MAX_HANDS", live)
        # Round-tripping the file's own values must not rewrite them.
        vec = space.get_file_vector(target)
        self.assertEqual(space.vector_to_dict(vec)["MAX_HANDS"], live["MAX_HANDS"])


class TestObjective(unittest.TestCase):
    def test_cvar_uses_only_the_tail(self) -> None:
        self.assertEqual(cvar([100.0, 200.0, 300.0, 400.0], alpha=0.25), 100.0)
        self.assertEqual(cvar([100.0, 200.0, 300.0, 400.0], alpha=0.5), 150.0)

    def test_tail_collapse_is_penalised(self) -> None:
        steady = [{"me_cash": 40_000.0, "opp_cash": 3_000.0, "win": 1} for _ in range(4)]
        spiky = [{"me_cash": 60_000.0, "opp_cash": 3_000.0, "win": 1} for _ in range(3)]
        spiky.append({"me_cash": 2_000.0, "opp_cash": 3_000.0, "win": 0})
        self.assertGreater(score_results(steady)["fitness"], score_results(spiky)["fitness"])
        self.assertGreater(score_results(spiky)["mean_cash"], score_results(steady)["mean_cash"])

    def test_worst_episodes_sorted(self) -> None:
        rows = [{"me_cash": v, "opp_cash": 0.0, "win": 1} for v in (5.0, 1.0, 3.0)]
        self.assertEqual([r["me_cash"] for r in worst_episodes(rows, k=2)], [1.0, 3.0])


class TestArena(unittest.TestCase):
    def test_harness_evaluates_a_match(self) -> None:
        env = MatchHarness(agent_base_path=AGENT_PATH, opponent_name="baseline")
        try:
            res = env.evaluate_strategy(env.strategy_space.get_file_vector(AGENT_PATH), seed=42)
            self.assertIn("me_cash", res)
            self.assertEqual(res["status"], "DONE")
        finally:
            env.close()

    def test_evolutionary_optimizer_reports_cvar(self) -> None:
        optimizer = EvolutionSearch(
            agent_base_path=AGENT_PATH,
            pop_size=2,
            elite_size=1,
            episodes_per_eval=2,
            opponents=["baseline"],
            output_dir="logs/evolution_search_test",
        )
        res = optimizer.train(generations=1, base_seed=999)
        for key in ("fitness", "mean_cash", "cvar_cash", "median_cash"):
            self.assertIn(key, res)
        self.assertGreater(res["mean_cash"], 0.0)

    def test_cem_diagnose_reports_tail(self) -> None:
        cem = MacroCEM(
            agent_base_path=AGENT_PATH,
            opponents=["baseline"],
            pop_size=2,
            episodes_per_eval=2,
            output_dir="logs/macro_search_test",
            n_workers=2,
        )
        res = cem.diagnose(episodes=2, base_seed=999)
        self.assertGreater(res["episodes"], 0)
        self.assertLessEqual(res["cvar_cash"], res["mean_cash"])


if __name__ == "__main__":
    unittest.main()
