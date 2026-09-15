"""What the sell-metering layer in `main.py` is and is not allowed to do (#30).

`tests/test_sell_layer.py` pins the three restraints that make `_rank_sells`
safe. `_meter_sells` deliberately breaks one of them -- it *does* resize a route
order -- so it has to earn that with restraints of its own, and this file is
where they are pinned:

  * it is off unless `_METER_ITEMS` says otherwise, and off means untouched;
  * it never withholds while the farm holds less than the rest of the route has
    to spend. That envelope is the entire safety argument, and three previous
    attempts died without one (docs/experiments.md, issues #23 and #25);
  * it never touches a product that is not metered, so `WHEAT` and `FERTILIZER`
    -- which this route lifts back out of the shed to feed animals and fertilize
    tiles -- are out of reach by construction;
  * it never pushes a turn past `maxMarketOrdersPerTurn`, because the
    interpreter truncates the tail and an inserted SELL can drop a `HIRE`;
  * it holds a bounded number of units, and releases all of them on shed
    pressure and in the endgame, because unsold stock scores $0.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
MAX_MARKET_ORDERS = 10


def _load_agent() -> Any:
    # v0.4.x main.py is the multi-route public chassis (no metering layer);
    # these tests target the last agent that shipped one -- the v0.3.1 incumbent.
    agent_path = os.environ.get(
        "METER_LAYER_AGENT", os.path.join(PROJECT_ROOT, "opponents", "v0_3_1.py")
    )
    spec = importlib.util.spec_from_file_location("main_agent_meter", agent_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AGENT = _load_agent()

# Far enough above I0 that MILK and STRAWBERRY are both pinned at the $1 floor,
# which is the state the whole layer exists for.
GLUTTED = AGENT._MARKET_I0 + 400
RICH = 10_000_000.0  # more cash than the route can ever need
BROKE = 0.0


@pytest.fixture(autouse=True)
def _reset_meter_state():
    AGENT._METER_STATE = {0: {}, 1: {}}
    yield
    AGENT._METER_STATE = {0: {}, 1: {}}


@pytest.fixture
def metering(monkeypatch):
    """The layer switched on, with the shipped knobs."""
    monkeypatch.setattr(AGENT, "_METER_ITEMS", ("MILK", "STRAWBERRY"))
    return AGENT


def _obs(money: float = RICH, shed: dict[str, int] | None = None, inv: int = GLUTTED):
    return {
        "step": 400,
        "player": 0,
        "farms": [{"money": money, "hands": [], "farmer": (0, 0), "tiles": []}],
        "market": {"inventory": dict.fromkeys(AGENT._MARKET_PARAMS, inv)},
        "private": {"shed": dict(shed or {})},
    }


def _act(market: list[list[Any]]):
    return {"farmer": ["PASS"], "hands": [], "market": [list(o) for o in market]}


def _sells(action) -> dict[str, int]:
    out: dict[str, int] = {}
    for order in action["market"]:
        if AGENT._is_sell(order):
            out[order[1]] = out.get(order[1], 0) + int(order[2])
    return out


# ---------------------------------------------------------------------------
# Off means off
# ---------------------------------------------------------------------------


def test_the_shipped_default_is_a_no_op() -> None:
    """Whatever `_METER_ITEMS` ships as, an empty tuple has to be inert."""
    orig = AGENT._METER_ITEMS
    try:
        AGENT._METER_ITEMS = ()
        market: list[list[Any]] = [["SELL", "MILK", 40], ["HIRE"], ["SELL", "STRAWBERRY", 30]]
        action = _act(market)
        assert AGENT._meter_sells(_obs(), action, 400)["market"] == market
    finally:
        AGENT._METER_ITEMS = orig


# ---------------------------------------------------------------------------
# The envelope
# ---------------------------------------------------------------------------


def test_never_withholds_while_the_route_still_needs_the_cash(metering) -> None:
    """The whole layer, in one test. At step 200 the route still has thousands of
    dollars of HIRE and BUY ahead of it; a farm with none of it does not get to
    hold anything back, however bad the price is."""
    assert AGENT._cash_needed(200, 45.0) > 1000
    action = metering._meter_sells(_obs(money=BROKE), _act([["SELL", "MILK", 40]]), 200)
    assert _sells(action) == {"MILK": 40}
    assert AGENT._METER_STATE[0]["held"] == {}


def test_withholds_once_the_farm_covers_the_whole_remaining_schedule(metering) -> None:
    action = metering._meter_sells(_obs(money=RICH), _act([["SELL", "MILK", 40]]), 400)
    assert _sells(action).get("MILK", 0) < 40
    assert AGENT._METER_STATE[0]["held"]["MILK"] > 0


def test_a_price_above_the_floor_is_sold_not_held(metering) -> None:
    """Metering is a floor, not a hoard: at I0 the price is base and every unit
    the route asked to sell clears."""
    action = metering._meter_sells(
        _obs(money=RICH, inv=AGENT._MARKET_I0), _act([["SELL", "MILK", 4]]), 400
    )
    assert _sells(action) == {"MILK": 4}
    assert AGENT._METER_STATE[0]["held"] == {}


def test_the_requirement_falls_to_nothing_by_the_end_of_the_route() -> None:
    assert AGENT._cash_needed(len(AGENT._ROUTE), 45.0) == 0.0
    assert AGENT._cash_needed(0, 45.0) > AGENT._cash_needed(400, 45.0)


# ---------------------------------------------------------------------------
# Restraints
# ---------------------------------------------------------------------------


def test_never_touches_a_product_it_does_not_meter(metering) -> None:
    """WHEAT and FERTILIZER are farm inputs this route lifts back out of the
    shed; see tests/test_sell_layer.py."""
    market: list[list[Any]] = [
        ["SELL", "WHEAT", 60],
        ["SELL", "FERTILIZER", 40],
        ["SELL", "WOOL", 6],
    ]
    action = metering._meter_sells(_obs(money=RICH), _act(market), 400)
    assert action["market"] == market


def test_never_moves_or_drops_a_non_sell_order(metering) -> None:
    market: list[list[Any]] = [
        ["HIRE"],
        ["SELL", "MILK", 40],
        ["BUY_SEED", "WHEAT", 1],
        ["BUY_LAND"],
        ["BUY_PRODUCT", "WHEAT", 5],
    ]
    action = metering._meter_sells(_obs(money=RICH), _act(market), 400)
    kept = [o for o in action["market"] if not AGENT._is_sell(o)]
    assert kept == [o for o in market if not AGENT._is_sell(o)]


def test_never_exceeds_the_market_order_cap(metering) -> None:
    """A full turn plus a backlog in both metered products still fits."""
    AGENT._METER_STATE[0] = {"held": {"MILK": 30, "STRAWBERRY": 30}, "last_step": 399}
    market: list[list[Any]] = [["HIRE"]] * 9 + [["SELL", "WHEAT", 5]]
    action = metering._meter_sells(_obs(money=RICH), _act(market), 400)
    assert len(action["market"]) <= MAX_MARKET_ORDERS


def test_the_shipped_route_never_exceeds_the_cap_after_metering(metering) -> None:
    """The layer only appends into slots the route left free, so the census the
    sell-layer test pins survives it."""
    worst = 0
    for step, action in enumerate(AGENT._ROUTE):
        out = metering._meter_sells(_obs(money=RICH), AGENT._copy_action(action), step)
        worst = max(worst, len(out["market"]))
    assert worst <= MAX_MARKET_ORDERS


def test_holds_no_more_than_the_configured_shed_budget(metering, monkeypatch) -> None:
    monkeypatch.setattr(AGENT, "_METER_MAX_HOLD", 10)
    for step in range(400, 420):
        metering._meter_sells(_obs(money=RICH), _act([["SELL", "MILK", 40]]), step)
        assert sum(AGENT._METER_STATE[0]["held"].values()) <= 10


# ---------------------------------------------------------------------------
# The three releases
# ---------------------------------------------------------------------------


def test_shed_pressure_ends_withholding_and_flushes(metering) -> None:
    AGENT._METER_STATE[0] = {"held": {"MILK": 12}, "last_step": 399}
    shed = {"WOOL": AGENT._METER_SHED_RELEASE}
    action = metering._meter_sells(_obs(money=RICH, shed=shed), _act([["SELL", "MILK", 5]]), 400)
    assert _sells(action) == {"MILK": 17}
    assert AGENT._METER_STATE[0]["held"].get("MILK", 0) == 0


def test_the_endgame_flushes_even_with_room_and_money(metering) -> None:
    """Unsold stock scores $0, so the last two days sell at whatever is offered."""
    AGENT._METER_STATE[0] = {
        "held": {"MILK": 9, "STRAWBERRY": 4},
        "last_step": AGENT._METER_FLUSH_STEP - 1,
    }
    action = metering._meter_sells(_obs(money=RICH), _act([]), AGENT._METER_FLUSH_STEP)
    assert _sells(action) == {"MILK": 9, "STRAWBERRY": 4}
    assert not any(AGENT._METER_STATE[0]["held"].values())


def test_the_backlog_drains_into_the_floor_when_the_price_recovers(metering) -> None:
    """On a turn the route is not selling milk, a recovered market takes the
    backlog -- that is what "meter against the town's drain rate" means."""
    AGENT._METER_STATE[0] = {"held": {"MILK": 40}, "last_step": 399}
    action = metering._meter_sells(_obs(money=RICH, inv=AGENT._MARKET_I0), _act([]), 400)
    sold = _sells(action).get("MILK", 0)
    assert 0 < sold < 40
    assert AGENT._METER_STATE[0]["held"]["MILK"] == 40 - sold


def test_state_is_per_seat_and_resets_on_a_new_episode(metering) -> None:
    metering._meter_sells(_obs(money=RICH), _act([["SELL", "MILK", 40]]), 400)
    assert AGENT._METER_STATE[0]["held"]
    assert AGENT._METER_STATE[1] == {}
    metering._meter_sells(_obs(money=RICH), _act([]), 0)
    assert AGENT._METER_STATE[0]["held"] == {}
