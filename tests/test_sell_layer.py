"""What the SELL layer in `main.py` is and is not allowed to do.

`_rank_sells` reorders the SELL orders the route already contains among the slots they
already occupy. It never resizes one, never adds one, and never moves a non-SELL order.
Those three restraints are the whole reason the layer is safe, and each of them was
violated by the sell-acceleration layer that #25 proposed and PR #35 first implemented:

  * **Never size an order from the observation.** `obs.private.shed` predates the turn's
    unit actions, so it is a lower bound and not the sellable quantity. See
    tests/test_observation_ordering.py.
  * **Never sell a farm input.** FEED and FERTILIZE consume from a unit's *carried*
    inventory, which `PICKUP` fills from the shed. This route lifts 526 WHEAT and 95
    FERTILIZER back out of the shed after depositing them; selling either starves the
    action that needed it. The acceleration layer excluded WHEAT and not FERTILIZER, and
    turned 9 no-op FERTILIZE calls per 30 episodes into 282.
  * **Never add an order.** The market is capped at `maxMarketOrdersPerTurn` (10) and the
    interpreter truncates the tail, so an inserted SELL can push a HIRE off the turn.

The acceleration layer was measured and reverted -- it loses $1,680 on 30/30 paired seeds
against a peer-strength opponent. See docs/experiments.md, "v0.2.8 -- sell acceleration".
This file exists so that the next attempt has to argue with a test rather than with prose.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
MAX_MARKET_ORDERS = 10  # configuration.maxMarketOrdersPerTurn


def _load_agent() -> Any:
    spec = importlib.util.spec_from_file_location(
        "main_agent", os.path.join(PROJECT_ROOT, "main.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AGENT = _load_agent()


def _obs(shed: dict[str, int] | None = None, inventory: int = 10000) -> dict[str, Any]:
    return {
        "step": 0,
        "player": 0,
        "farms": [{"hands": [], "farmer": (0, 0), "tiles": []}],
        "market": {"inventory": dict.fromkeys(AGENT._MARKET_PARAMS, inventory)},
        "private": {"shed": dict(shed or {})},
    }


def _route_market_orders() -> list[list[Any]]:
    return [order for step in AGENT._ROUTE for order in (step.get("market") or [])]


def test_the_route_lifts_inputs_back_out_of_the_shed() -> None:
    """The premise of the "never sell an input" rule, pinned against the actual route.

    If a future route stops doing this the rule is cheap; if a future layer starts
    selling from the shed, this is the number it has to respect.
    """
    pickups: dict[str, int] = {}
    for step in AGENT._ROUTE:
        for unit in [step.get("farmer") or ["PASS"], *(step.get("hands") or [])]:
            if isinstance(unit, list) and len(unit) >= 2 and unit[0] == "PICKUP":
                n = int(unit[2]) if len(unit) >= 3 else 1
                pickups[unit[1]] = pickups.get(unit[1], 0) + n
    assert pickups["WHEAT"] > 0
    assert pickups["FERTILIZER"] > 0, "FERTILIZER is an input, not only a product"


def test_never_resizes_a_route_order() -> None:
    """Not even when the shed reads empty -- the observation predates the turn."""
    market = [["SELL", "MILK", 40], ["SELL", "MELON", 7]]
    action = {"farmer": ["PASS"], "hands": [], "market": [list(o) for o in market]}
    out = AGENT._rank_sells(_obs(shed={}), action)["market"]
    assert sorted(out) == sorted(market)


def test_never_adds_an_order_from_the_shed() -> None:
    shed = {"MILK": 500, "FERTILIZER": 500, "WHEAT": 500, "MELON": 500}
    action = {"farmer": ["PASS"], "hands": [], "market": []}
    assert AGENT._rank_sells(_obs(shed=shed), action)["market"] == []


def test_non_sell_orders_keep_their_slots() -> None:
    market: list[list[Any]] = [
        ["SELL", "MILK", 5],
        ["HIRE"],
        ["SELL", "MELON", 50],
        ["BUY_SEED", "WHEAT", 1],
        ["BUY_LAND"],
    ]
    action = {"farmer": ["PASS"], "hands": [], "market": [list(o) for o in market]}
    out = AGENT._rank_sells(_obs(), action)["market"]

    assert len(out) == len(market)
    for i, order in enumerate(market):
        if not AGENT._is_sell(order):
            assert out[i] == order


def test_sells_are_ordered_by_descending_price_impact() -> None:
    obs = _obs()
    market: list[list[Any]] = [
        ["SELL", "CARROT", 3],
        ["HIRE"],
        ["SELL", "MELON", 60],
        ["SELL", "MILK", 20],
    ]
    action = {"farmer": ["PASS"], "hands": [], "market": [list(o) for o in market]}
    out = AGENT._rank_sells(obs, action)["market"]

    impacts = [AGENT._impact(obs, o) for o in out if AGENT._is_sell(o)]
    assert impacts == sorted(impacts, reverse=True)


def test_the_route_never_exceeds_the_market_order_cap() -> None:
    """A route that already ran at the cap leaves the layer no room to add anything."""
    worst = max(len(step.get("market") or []) for step in AGENT._ROUTE)
    assert worst <= MAX_MARKET_ORDERS
