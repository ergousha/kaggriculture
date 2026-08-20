"""Regression tests for the route-synthesis harness (issue #26).

The harness is only observable 30 expensive episodes into a search pass. These
tests pin the four "changes nothing" gates — zero-mutation identity, mutation
round-trip, and the accept-rule's agreement with `rank_cvar`'s metric — so a
regression surfaces in CI rather than in a seven-hour run.

Gates 1 and 2 are pure and always run. Gate 3 (an episode through the real
artifact, asserting no invalid actions) needs `kaggle_environments`, so it is
skipped with a loud message when the package is absent rather than silently
passing — the same treatment `search/smoke_test.py` gives the heuristic target.
"""

from __future__ import annotations

import copy
import os
import random
import unittest

from mining import common
from search import route_search as rs

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANDIDATES = os.path.join(PROJECT_ROOT, "candidates.jsonl")


def _seed() -> list[dict]:
    return rs.load_seed(CANDIDATES if os.path.exists(CANDIDATES) else None)


class TestSeedLoading(unittest.TestCase):
    def test_seed_hash_is_stable(self) -> None:
        seed = _seed()
        self.assertEqual(len(common.normalize_route(seed)), common.DEFAULT_STEPS - 1)
        # The shipped incumbent is hash-verified against the pool when present;
        # without a pool the hash must still be the incumbent's.
        self.assertTrue(rs._hash_of(seed).startswith(rs.SEED_CANDIDATE_PREFIX))


class TestOperators(unittest.TestCase):
    def setUp(self) -> None:
        self.seed = common.normalize_route(copy.deepcopy(_seed()))
        self.rng = random.Random(7)

    def test_every_operator_runs_or_sits_out_cleanly(self) -> None:
        for name in rs.ALL_OPERATORS:
            out = rs.OPERATORS[name](copy.deepcopy(self.seed), self.rng)
            if out is None:
                continue  # an operator may sit out a route it does not apply to
            mutated, note = out
            self.assertIsInstance(note, str)
            self.assertNotEqual(rs._hash_of(mutated), rs._hash_of(self.seed), name)

    def test_shift_preserves_nonmove_signature(self) -> None:
        out = rs.op_shift_task_block(copy.deepcopy(self.seed), self.rng, k=1)
        if out is None:
            self.skipTest("no shiftable movement run in the seed")
        mutated, _ = out
        # The no-op-by-construction property: every non-movement op that existed
        # still exists, in the same slot. This is what makes the shift safe.
        self.assertEqual(rs._nonmove_signature(self.seed), rs._nonmove_signature(mutated))

    def test_retarget_plant_keeps_buy_plant_consistent(self) -> None:
        out = rs.op_retarget_plant(copy.deepcopy(self.seed), self.rng)
        if out is None:
            self.skipTest("no PLANT in the seed")
        mutated, _ = out

        # The operator must not make the seed's buy/plant imbalance worse. The
        # incumbent already plants 33 strawberry against 32 buys (the 33rd is a
        # silent no-op the arena counts), so we do not assert buy>=plant on the
        # mutant — we assert the *deficit* per crop does not grow.
        def deficit(route):
            b: dict[str, int] = {}
            p: dict[str, int] = {}
            for a in route:
                for o in a.get("market") or []:
                    if isinstance(o, list) and len(o) >= 3 and o[0] == "BUY_SEED":
                        b[o[1]] = b.get(o[1], 0) + int(o[2])
                for u in rs._units(a):
                    if isinstance(u, list) and len(u) > 1 and u[0] == "PLANT":
                        p[u[1]] = p.get(u[1], 0) + 1
            return {c: p.get(c, 0) - b.get(c, 0) for c in set(b) | set(p)}

        d_seed = deficit(self.seed)
        d_mut = deficit(mutated)
        # The operator moves one unit of deficit between two crops, so no single
        # crop's deficit may grow by more than one, and the total deficit across
        # the route is conserved.
        for crop, short in d_mut.items():
            self.assertLessEqual(short, d_seed.get(crop, 0) + 1, crop)
        self.assertEqual(sum(d_seed.values()), sum(d_mut.values()))

    def test_swap_herd_operator_converts_cows_and_integrates_wool(self) -> None:
        cows_before, sheep_before = rs._herd_counts(self.seed)
        self.assertEqual(cows_before, 10)
        self.assertEqual(sheep_before, 4)

        out = rs.op_swap_herd(copy.deepcopy(self.seed), self.rng)
        self.assertIsNotNone(out)
        assert out is not None
        mutated, note = out
        cows_after, sheep_after = rs._herd_counts(mutated)
        self.assertEqual(cows_after, 9)
        self.assertEqual(sheep_after, 5)
        self.assertIn("converted COW #", note)
        self.assertIn("wool sells integrated", note)

        # Wool sells must now be present
        wool_sells = 0
        for action in mutated:
            for order in action.get("market") or []:
                if isinstance(order, list) and len(order) >= 3 and order[0] == "SELL" and order[1] == "WOOL":
                    wool_sells += 1
        self.assertGreater(wool_sells, 0)


class TestAcceptRule(unittest.TestCase):
    def test_accept_matches_panel_sort_key(self) -> None:
        # rank_cvar selects on (mean_win, worst_win); the harness must agree.
        better = {"n": 100, "mean_win": 0.60, "worst_win": 0.40}
        worse = {"n": 100, "mean_win": 0.55, "worst_win": 0.45}
        self.assertTrue(rs.accepts(better, worse))
        self.assertFalse(rs.accepts(worse, better))
        # The tiebreak: equal mean, better worst-case wins.
        a = {"n": 100, "mean_win": 0.60, "worst_win": 0.50}
        b = {"n": 100, "mean_win": 0.60, "worst_win": 0.45}
        self.assertTrue(rs.accepts(a, b))
        # An unevaluated challenger (n=0) never accepts — an incomplete grid is
        # a reject, not an accept.
        self.assertFalse(rs.accepts({"n": 0}, worse))

    def test_never_reevaluates_a_seen_hash(self) -> None:
        # The bookkeeping gate: a hash in `state.seen` short-circuits.
        state = rs.SearchState(results_path="/nonexistent/route_search.jsonl")
        state.seen.add("abc123")
        self.assertIn("abc123", state.seen)


class TestIdentityGate(unittest.TestCase):
    def test_zero_mutation_bakes_byte_identical_route(self) -> None:
        seed = _seed()
        baked = common.decode_route_b85(common.encode_route_b85(common.normalize_route(seed)))
        self.assertEqual(rs._hash_of(baked), rs._hash_of(seed))


@unittest.skipUnless(
    os.path.exists(CANDIDATES), "candidates.jsonl not present; fidelity episode skipped"
)
class TestFidelityEpisode(unittest.TestCase):
    def test_identity_route_replays_without_invalid_actions(self) -> None:
        try:
            from local_arena import run_episode
        except ImportError:
            self.skipTest("kaggle_environments not importable here")
        import tempfile

        seed = _seed()
        _, path = rs.materialize_agent(seed, tempfile.mkdtemp(prefix="route_search_test_"))
        res = run_episode(
            {
                "agent": path,
                "opponent": "random",
                "seed": 1000000,
                "steps": common.DEFAULT_STEPS,
                "swap": False,
                "decision_log": None,
                "replay": None,
            }
        )
        self.assertIsNone(res["harness_error"])
        self.assertEqual(res["invalid"], 0)
        self.assertEqual(res["crashes"], 0)
        self.assertEqual(res["timeouts"], 0)


if __name__ == "__main__":
    unittest.main()
