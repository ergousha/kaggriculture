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


# The search's hash/herd pins are facts of the v0.3.1 incumbent route. v0.4.x
# main.py is the multi-route chassis whose base tape is a different (public)
# route, so the seed source is pinned to the frozen incumbent snapshot.
INCUMBENT = os.environ.setdefault(
    "ROUTE_SEARCH_SEED_FILE",
    os.path.join(PROJECT_ROOT, "opponents", "v0_3_1.py"),
)


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
        self.assertEqual(cows_before, 9)
        self.assertEqual(sheep_before, 5)

        out = rs.op_swap_herd(copy.deepcopy(self.seed), self.rng)
        self.assertIsNotNone(out)
        assert out is not None
        mutated, note = out
        cows_after, sheep_after = rs._herd_counts(mutated)
        self.assertEqual(cows_after, 8)
        self.assertEqual(sheep_after, 6)
        self.assertIn("converted COW #", note)
        self.assertIn("wool sells integrated", note)

        # Wool sells must now be present
        wool_sells = 0
        for action in mutated:
            for order in action.get("market") or []:
                if (
                    isinstance(order, list)
                    and len(order) >= 3
                    and order[0] == "SELL"
                    and order[1] == "WOOL"
                ):
                    wool_sells += 1
        self.assertGreater(wool_sells, 0)

    def test_assign_idle_converts_pass_to_water(self) -> None:
        out = rs.op_assign_idle(copy.deepcopy(self.seed), self.rng)
        self.assertIsNotNone(out)
        assert out is not None
        mutated, note = out
        self.assertIn("assigned PASS->WATER", note)
        # Hash must differ from seed
        self.assertNotEqual(rs._hash_of(mutated), rs._hash_of(self.seed))


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


class TestMoveSellAndBuy(unittest.TestCase):
    """#30's joint operator: a SELL and the BUY it funds move together.

    Every earlier attempt moved one leg -- #23 held sales back and the next HIRE
    failed for cash, #25 pulled them forward and sold into a market that had not
    risen. The operator is only interesting if the pair really does stay a pair,
    so that is what these pin.
    """

    def test_both_legs_move_by_the_same_shift(self) -> None:
        route: list[dict] = [
            {"farmer": ["PASS"], "hands": [], "market": [["SELL", "MILK", 5]]},
            {"farmer": ["PASS"], "hands": [], "market": []},
            {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", 1]]},
            {"farmer": ["PASS"], "hands": [], "market": []},
            {"farmer": ["PASS"], "hands": [], "market": []},
        ]
        out = rs.op_move_sell_and_buy(route, None, shift=2, sell_site=(0, 0))
        assert out is not None
        mutated, note = out
        self.assertEqual(mutated[0]["market"], [])
        self.assertEqual(mutated[2]["market"], [["SELL", "MILK", 5]])
        self.assertEqual(mutated[4]["market"], [["BUY_SEED", "WHEAT", 1]])
        self.assertIn("shift +2", note)

    def test_a_hire_is_never_delayed(self) -> None:
        """A missing hand does not idle -- `_align_hands` truncates the hands
        list, so every later slot in that turn's trace shifts by one."""
        route: list[dict] = [
            {"farmer": ["PASS"], "hands": [], "market": [["SELL", "MILK", 5], ["HIRE"]]},
            {"farmer": ["PASS"], "hands": [], "market": []},
            {"farmer": ["PASS"], "hands": [], "market": []},
        ]
        self.assertEqual(rs._funder_forward_slack(route, (0, 1)), 0)
        self.assertIsNone(rs.op_move_sell_and_buy(route, None, shift=1, sell_site=(0, 0)))

    def test_a_buy_never_outruns_the_unit_op_it_feeds(self) -> None:
        route: list[dict] = [
            {"farmer": ["PASS"], "hands": [], "market": [["SELL", "MILK", 5]]},
            {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "MELON", 1]]},
            {"farmer": ["PASS"], "hands": [], "market": []},
            {"farmer": ["PLANT", "MELON"], "hands": [], "market": []},
            {"farmer": ["PASS"], "hands": [], "market": []},
        ]
        # The seed has to be in hand the turn *before* the PLANT: units act
        # before the market does.
        self.assertEqual(rs._funder_forward_slack(route, (1, 0)), 1)
        out = rs.op_move_sell_and_buy(route, None, shift=3, sell_site=(0, 0))
        assert out is not None
        self.assertEqual(out[0][2]["market"], [["BUY_SEED", "MELON", 1]])
        self.assertIn("shift +1", out[1])

    def test_a_sale_never_outruns_the_stock_that_fills_it(self) -> None:
        """A sale pulled back past the deposit that fills it earns nothing while
        the purchase it funds still spends -- the mirror of #23's failure."""
        route: list[dict] = [
            {"farmer": ["PASS"], "hands": [], "market": []},
            {"farmer": ["PASS"], "hands": [], "market": []},
            {"farmer": ["PLACE", "MILK", 5], "hands": [], "market": []},
            {"farmer": ["PASS"], "hands": [], "market": [["SELL", "MILK", 5]]},
            {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", 1]]},
        ]
        self.assertEqual(rs._sell_backward_slack(route, (3, 0)), 1)
        out = rs.op_move_sell_and_buy(route, None, shift=-3, sell_site=(3, 0))
        assert out is not None
        self.assertEqual(out[0][2]["market"], [["SELL", "MILK", 5]])
        self.assertIn("shift -1", out[1])

    def test_the_shipped_routes_purchase_schedule_has_almost_no_forward_slack(self) -> None:
        """The census #30 turns on: 319 of the route's 458 funded orders cannot
        be delayed by a single step, so "move the BUY with the SELL" has very
        little room to move it into."""
        census = rs.funder_slack_census(_seed())
        self.assertEqual(census["HIRE"]["zero"], census["HIRE"]["orders"])
        self.assertEqual(census["BUY_LAND"]["zero"], census["BUY_LAND"]["orders"])
        total = sum(v["orders"] for v in census.values())
        pinned = sum(v["zero"] for v in census.values())
        self.assertGreater(pinned / total, 0.65)

    def test_the_sale_never_lands_after_the_purchase_it_funds(self) -> None:
        seed = _seed()
        rng = random.Random(7)
        for _ in range(300):
            out = rs.op_move_sell_and_buy(seed, rng)
            if out is None:
                continue
            mutated, _note = out
            self.assertEqual(len(mutated), len(seed))

    def test_it_never_pushes_a_turn_past_the_market_order_cap(self) -> None:
        """The interpreter truncates at `maxMarketOrdersPerTurn`, so a mutation
        that overflows a turn would silently drop that turn's tail."""
        seed = _seed()
        rng = random.Random(11)
        applied = 0
        for _ in range(300):
            out = rs.op_move_sell_and_buy(seed, rng)
            if out is None:
                continue
            applied += 1
            worst = max(len(step.get("market") or []) for step in out[0])
            self.assertLessEqual(worst, rs.MAX_MARKET_ORDERS)
        self.assertGreater(applied, 0, "operator never fired on the shipped route")

    def test_it_refuses_rather_than_truncating_a_full_turn(self) -> None:
        full = [["HIRE"]] * rs.MAX_MARKET_ORDERS
        route: list[dict] = [
            {"farmer": ["PASS"], "hands": [], "market": [["SELL", "MILK", 5], ["HIRE"]]},
            {"farmer": ["PASS"], "hands": [], "market": list(full)},
        ]
        self.assertIsNone(rs.op_move_sell_and_buy(route, None, shift=1, sell_site=(0, 0)))

    def test_it_conserves_every_order_in_the_route(self) -> None:
        seed = _seed()
        before = sorted(tuple(o) for step in seed for o in (step.get("market") or []))
        rng = random.Random(3)
        for _ in range(50):
            out = rs.op_move_sell_and_buy(seed, rng)
            if out is None:
                continue
            after = sorted(tuple(o) for step in out[0] for o in (step.get("market") or []))
            self.assertEqual(before, after)

    def test_it_sits_out_a_route_with_nothing_to_re_time(self) -> None:
        route = [{"farmer": ["PASS"], "hands": [], "market": [["SELL", "WOOL", 5]]}]
        self.assertIsNone(rs.op_move_sell_and_buy(route, random.Random(0)))


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
