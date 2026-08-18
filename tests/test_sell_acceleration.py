"""Regression tests for the SELL acceleration layer in `main.py`.

The layer sells a product that is already in the shed before the route's schedule gets
round to it. Two bounds make that safe, and both are load-bearing:

  * the per-item budget is the suffix sum of the route's own remaining SELL volume, so
    nothing the route did not plan to sell can ever be sold;
  * the units the route still lifts back out of the shed are reserved, because FEED and
    FERTILIZE consume from a unit's *carried* inventory and that inventory is filled by
    `PICKUP` from the shed. Selling a product the route later picks up starves the action
    that needed it. This route deposits fertilizer and picks 95 units of it back up; the
    first cut of the layer sold them and turned 9 no-op FERTILIZE calls per 30 episodes
    into 282, at a cost of ~$1,700 mean cash and 37 points of win rate. See
    docs/experiments.md, "v0.2.8 -- sell acceleration".

A third bound is inherited: route quantities are never shrunk. `obs.private.shed`
predates the turn's unit actions, so it is a lower bound and not the sellable quantity —
see tests/test_observation_ordering.py.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)


def _load_agent() -> Any:
    spec = importlib.util.spec_from_file_location(
        "main_agent", os.path.join(PROJECT_ROOT, "main.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AGENT = _load_agent()


def _obs(shed: dict[str, int], inventory: int = 10000) -> dict[str, Any]:
    items = list(AGENT._MARKET_PARAMS)
    return {
        "step": 0,
        "player": 0,
        "farms": [{"hands": [], "farmer": (0, 0), "tiles": []}],
        "market": {"inventory": dict.fromkeys(items, inventory)},
        "private": {"shed": dict(shed)},
    }


def _sold(market: list[list[Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for order in market:
        if AGENT._is_sell(order):
            out[order[1]] = out.get(order[1], 0) + int(order[2])
    return out


def test_reserves_what_the_route_picks_back_up() -> None:
    """The route lifts fertilizer and wheat back out of the shed; neither is sellable."""
    reserved = AGENT._PICKUP_SUFFIX_SUMS[0]
    assert reserved["FERTILIZER"] > 0 and reserved["WHEAT"] > 0

    shed = {"FERTILIZER": reserved["FERTILIZER"], "WHEAT": reserved["WHEAT"]}
    action = {"farmer": ["PASS"], "hands": [], "market": []}
    assert _sold(AGENT._accelerate_sells(_obs(shed), action, 0)["market"]) == {}


def test_accelerates_only_the_surplus_above_the_reserve() -> None:
    reserved = AGENT._PICKUP_SUFFIX_SUMS[0]["FERTILIZER"]
    shed = {"FERTILIZER": reserved + 7}
    action = {"farmer": ["PASS"], "hands": [], "market": []}
    assert _sold(AGENT._accelerate_sells(_obs(shed), action, 0)["market"]) == {"FERTILIZER": 7}


def test_never_sells_what_the_route_never_sells() -> None:
    """Animals sit in the shed too, and no budget exists for them."""
    shed = {"COW": 5, "SHEEP": 3, "EGG": 40}
    action = {"farmer": ["PASS"], "hands": [], "market": []}
    assert _sold(AGENT._accelerate_sells(_obs(shed), action, 0)["market"]) == {}


def test_never_sells_feed_wheat_even_with_a_wheat_budget() -> None:
    assert "WHEAT" not in AGENT._SELL_SUFFIX_SUMS[0]
    action = {"farmer": ["PASS"], "hands": [], "market": []}
    out = AGENT._accelerate_sells(_obs({"WHEAT": 900}), action, 0)
    assert _sold(out["market"]) == {}


def test_never_shrinks_a_route_order_when_the_shed_reads_empty() -> None:
    """`obs.private.shed` predates the turn, so an empty read must not truncate."""
    action = {"farmer": ["PASS"], "hands": [], "market": [["SELL", "MILK", 40]]}
    out = AGENT._accelerate_sells(_obs({}), action, 0)
    assert _sold(out["market"]) == {"MILK": 40}


def test_never_exceeds_the_routes_remaining_scheduled_volume() -> None:
    budget = AGENT._SELL_SUFFIX_SUMS[0]["MELON"]
    action = {"farmer": ["PASS"], "hands": [], "market": []}
    out = AGENT._accelerate_sells(_obs({"MELON": budget * 10}), action, 0)
    assert _sold(out["market"]) == {"MELON": budget}


def test_preserves_non_sell_orders_and_respects_the_order_cap() -> None:
    non_sells = [["HIRE"], ["BUY_SEED", "WHEAT", 1], ["BUY_LAND"]]
    action = {"farmer": ["PASS"], "hands": [], "market": list(non_sells)}
    shed = dict.fromkeys(AGENT._SELL_SUFFIX_SUMS[0], 500)
    out = AGENT._accelerate_sells(_obs(shed), action, 0)["market"]

    assert [o for o in out if not AGENT._is_sell(o)] == non_sells
    assert len(out) <= AGENT._MAX_MARKET_ORDERS
    # Ranked by price impact, largest first.
    impacts = [AGENT._impact(_obs(shed), o) for o in out if AGENT._is_sell(o)]
    assert impacts == sorted(impacts, reverse=True)


def test_out_of_range_steps_do_not_wrap_into_the_wrong_budget() -> None:
    """A negative step must not index the suffix table from the end."""
    action = {"farmer": ["PASS"], "hands": [], "market": []}
    for step in (-1, -len(AGENT._SELL_SUFFIX_SUMS), len(AGENT._SELL_SUFFIX_SUMS) + 5):
        out = AGENT._accelerate_sells(_obs({"MELON": 500}), dict(action), step)
        assert _sold(out["market"]) == {}
