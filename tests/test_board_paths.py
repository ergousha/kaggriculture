"""Regression tests for the offline path re-planner (issue #29).

`search/board_paths.py` rewrites the movement stream of the route that ships, so
the only thing standing between a bug here and a silently broken submission is
this file. Three properties are pinned:

  * **The simulator is the env.** `simulate_positions` must agree with
    `kaggle_environments` on every unit's tile at every step of the real route —
    not approximately, exactly. This is the test that makes everything else
    meaningful, and it is the one that needs the package (skipped loudly, the
    same treatment `tests/test_route_search.py` gives its episode gate).
  * **The re-path is a no-op.** Every non-movement op still fires on its original
    step from its original tile, the market stream is untouched, and the route
    walks strictly less while idling strictly more.
  * **The gate has teeth.** A deliberately broken re-path must be *rejected*,
    because a verifier that never fails is not a verifier.
"""

from __future__ import annotations

import copy
import os
import unittest

from mining import common
from search import board_paths as bp
from search import route_search as rs

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANDIDATES = os.path.join(PROJECT_ROOT, "candidates.jsonl")


def _seed() -> list[dict]:
    return common.normalize_route(rs.load_seed(CANDIDATES if os.path.exists(CANDIDATES) else None))


class TestGeometry(unittest.TestCase):
    def test_shed_access_and_spawn_match_the_env(self) -> None:
        self.assertEqual(bp.shed_access_tiles(10), [(4, 4), (5, 4), (4, 5), (5, 5)])
        # Only the NW quadrant is unlocked at reset, so the farmer always starts
        # on the one shed-access tile inside it.
        self.assertEqual(bp.default_spawn(10), (4, 4))

    def test_hands_spawn_least_occupied_first_in_nwse_order(self) -> None:
        # The farmer holds the NW tile at every day's start, so the first hire
        # takes the next NWSE tile rather than doubling up.
        self.assertEqual(bp.spawn_hand((4, 4), [], 10), (5, 4))
        self.assertEqual(bp.spawn_hand((4, 4), [(5, 4)], 10), (4, 5))
        self.assertEqual(bp.spawn_hand((4, 4), [(5, 4), (4, 5)], 10), (5, 5))
        # Fifth unit: every tile holds one, so NWSE order breaks the tie.
        self.assertEqual(bp.spawn_hand((4, 4), [(5, 4), (4, 5), (5, 5)], 10), (4, 4))

    def test_shortest_path_is_manhattan_and_stays_on_the_board(self) -> None:
        for origin, destination in (((0, 0), (9, 9)), ((9, 0), (0, 9)), ((4, 4), (4, 4))):
            walk = bp.shortest_path_ops(origin, destination)
            self.assertEqual(len(walk), bp.manhattan(origin, destination))
            x, y = origin
            for op in walk:
                dx, dy = bp.MOVE_DELTAS[op]
                x, y = x + dx, y + dy
                self.assertTrue(0 <= x < 10 and 0 <= y < 10, (origin, destination, walk))
            self.assertEqual((x, y), destination)


class TestSimulator(unittest.TestCase):
    def setUp(self) -> None:
        self.route = _seed()

    def test_positions_agree_with_the_live_env(self) -> None:
        """The claim the whole module rests on: this simulator *is* the env."""
        try:
            from kaggle_environments import make
        except ImportError:
            self.skipTest("kaggle_environments not importable here")

        nominal = bp.simulate_positions(self.route)
        env = make(
            "kaggriculture",
            configuration={"episodeSteps": common.DEFAULT_STEPS, "seed": common.SEED_BASE},
            debug=False,
        )
        env.reset(2)
        no_op = {"farmer": ["PASS"], "hands": [], "market": []}
        for step, action in enumerate(self.route):
            farm = env.state[0].observation.farms[0]
            live = (tuple(farm["farmer"]), *[tuple(h) for h in farm["hands"]])
            self.assertEqual(live, nominal[step], f"step {step}")
            # The hands alignment the shipped agent applies (main._align_hands),
            # so what the env sees here is what it would see in a real episode.
            hands = [list(h) for h in (action.get("hands") or [])]
            n = len(farm["hands"])
            replay = {
                "farmer": list(action["farmer"]),
                "hands": hands[:n] + [["PASS"]] * max(0, n - len(hands)),
                "market": [list(o) for o in action["market"]],
            }
            env.step([replay, copy.deepcopy(no_op)])

    def test_off_board_moves_are_dropped_not_wrapped(self) -> None:
        route = [{"farmer": ["WEST"], "hands": [], "market": []} for _ in range(6)]
        positions = bp.simulate_positions(route)
        # Four WESTs reach x=0 from x=4; the fifth and sixth are no-ops.
        self.assertEqual(
            [p[0] for p in positions], [(4, 4), (3, 4), (2, 4), (1, 4), (0, 4), (0, 4)]
        )

    def test_day_boundary_resets_the_farmer_and_dismisses_hands(self) -> None:
        route: list[dict] = [
            {"farmer": ["PASS"], "hands": [], "market": []} for _ in range(bp.TURNS_PER_DAY + 1)
        ]
        route[0]["market"] = [["HIRE"]]
        route[1]["farmer"] = ["NORTH"]
        positions = bp.simulate_positions(route)
        self.assertEqual(len(positions[1]), 2)  # the hire acts from the next step
        self.assertEqual(positions[bp.TURNS_PER_DAY], ((4, 4),))


class TestSegments(unittest.TestCase):
    def setUp(self) -> None:
        self.route = _seed()

    def test_every_route_move_belongs_to_exactly_one_segment(self) -> None:
        """No move may be missed, or the slack report is understating the route."""
        report = bp.slack_report(self.route)
        census = bp.census(self.route)
        self.assertEqual(report["interior_moves"] + report["terminal_moves"], census["movement"])

    def test_no_segment_crosses_a_day_boundary(self) -> None:
        for seg in bp.segments(self.route):
            self.assertEqual(
                seg.start // bp.TURNS_PER_DAY,
                seg.end // bp.TURNS_PER_DAY,
                seg,
            )

    def test_segments_contain_no_anchors(self) -> None:
        for seg in bp.segments(self.route):
            for step in range(seg.start, seg.end + 1):
                op = bp.op_at(self.route, step, seg.slot)
                self.assertTrue(op in bp.MOVE_OPS or op == bp.IDLE_OP, (seg, step, op))

    def test_required_moves_never_exceed_the_moves_actually_walked(self) -> None:
        """Slack cannot be negative: the route's own walk is a feasible path."""
        for seg in bp.segments(self.route):
            self.assertGreaterEqual(seg.slack, 0, seg)


class TestRepath(unittest.TestCase):
    def setUp(self) -> None:
        self.route = _seed()

    def test_interior_repath_is_a_verified_no_op(self) -> None:
        mutated, stats = bp.repath(self.route)
        self.assertGreater(stats["turns_recovered"], 0)
        self.assertEqual(bp.verify_schedule(self.route, mutated), [])

    def test_full_repath_is_a_verified_no_op(self) -> None:
        mutated, stats = bp.repath(self.route, drop_terminal=True)
        self.assertGreater(stats["turns_recovered"], stats["segments_rewritten"] - 1)
        self.assertEqual(bp.verify_schedule(self.route, mutated), [])

    def test_repath_walks_less_and_idles_more(self) -> None:
        """#29's census gate: movement strictly down, PASS strictly up."""
        before = bp.census(self.route)
        for drop_terminal in (False, True):
            mutated, stats = bp.repath(self.route, drop_terminal=drop_terminal)
            after = bp.census(mutated)
            self.assertLess(after["movement"], before["movement"], drop_terminal)
            self.assertGreater(after["idle"], before["idle"], drop_terminal)
            # Every banked turn is a move that became a PASS and nothing else.
            self.assertEqual(before["movement"] - after["movement"], stats["turns_recovered"])
            self.assertEqual(after["idle"] - before["idle"], stats["turns_recovered"])
            self.assertEqual(after["unit_turns"] - before["unit_turns"], 0)

    def test_repath_is_idempotent(self) -> None:
        """A shortest path has no slack left, so a second pass must be a no-op."""
        once, _ = bp.repath(self.route, drop_terminal=True)
        twice, stats = bp.repath(once, drop_terminal=True)
        self.assertEqual(stats["turns_recovered"], 0)
        self.assertEqual(
            common.route_hash(common.normalize_route(once)),
            common.route_hash(common.normalize_route(twice)),
        )

    def test_repath_preserves_productive_and_market_streams(self) -> None:
        mutated, _ = bp.repath(self.route, drop_terminal=True)
        before, after = bp.census(self.route), bp.census(mutated)
        for op, n in before["counts"].items():
            if op in bp.MOVE_OPS or op == bp.IDLE_OP:
                continue
            self.assertEqual(after["counts"].get(op, 0), n, op)
        self.assertEqual(
            [a["market"] for a in self.route],
            [a["market"] for a in mutated],
        )


class TestVerifierHasTeeth(unittest.TestCase):
    """A gate that cannot fail proves nothing, so make it fail on purpose."""

    def setUp(self) -> None:
        self.route = _seed()

    def test_dropping_a_load_bearing_move_is_caught(self) -> None:
        segs = [s for s in bp.segments(self.route) if not s.terminal and s.required > 0]
        seg = segs[0]
        broken = copy.deepcopy(self.route)
        bp.write_unit(broken[seg.start], seg.slot, ["PASS"])
        violations = bp.verify_schedule(self.route, broken)
        self.assertTrue(violations)
        self.assertIn("fires at", violations[0])

    def test_moving_an_anchor_off_its_step_is_caught(self) -> None:
        step, slot = next(
            (step, slot)
            for step in range(len(self.route))
            for slot in range(len(bp.unit_ops(self.route[step])))
            if bp.op_at(self.route, step, slot) not in bp.MOVE_OPS
            and bp.op_at(self.route, step, slot) != bp.IDLE_OP
        )
        broken = copy.deepcopy(self.route)
        bp.write_unit(broken[step], slot, ["PASS"])
        self.assertTrue(bp.verify_schedule(self.route, broken))

    def test_touching_the_market_stream_is_caught(self) -> None:
        broken = copy.deepcopy(self.route)
        broken[0]["market"] = []
        self.assertIn("market stream changed", bp.verify_schedule(self.route, broken)[0])


class TestOperator(unittest.TestCase):
    """`op_repath` is what the search loop calls; it must never emit unverified."""

    def setUp(self) -> None:
        self.route = _seed()

    def test_operator_banks_turns_and_verifies(self) -> None:
        out = rs.op_repath(copy.deepcopy(self.route))
        self.assertIsNotNone(out)
        assert out is not None
        mutated, note = out
        self.assertIn("banked as PASS", note)
        self.assertEqual(bp.verify_schedule(self.route, mutated), [])
        self.assertNotEqual(rs._hash_of(mutated), rs._hash_of(self.route))

    def test_operator_sits_out_an_already_optimal_route(self) -> None:
        optimal, _ = bp.repath(self.route, drop_terminal=True)
        self.assertIsNone(rs.op_repath(optimal, drop_terminal=True))

    def test_scope_one_touches_a_single_site(self) -> None:
        import random

        out = rs.op_repath(copy.deepcopy(self.route), random.Random(3), scope="one")
        self.assertIsNotNone(out)
        assert out is not None
        mutated, note = out
        self.assertIn("slot", note)
        whole, whole_stats = bp.repath(self.route)
        one = bp.census(mutated)["movement"]
        self.assertGreater(one, bp.census(whole)["movement"])
        self.assertLess(one, bp.census(self.route)["movement"])
        self.assertEqual(bp.verify_schedule(self.route, mutated), [])


if __name__ == "__main__":
    unittest.main()
