"""The cash envelope has to be the interpreter's arithmetic, not a copy of it.

`search/cash_schedule.py` and the `_cash_requirement` baked into every emitted
agent both price the route's purchase schedule offline. `main.py` is
standard-library-only and cannot import the environment, so the catalogue lives
in three places; if any of them drifts, #30's safety argument --

    money(u) >= money(t) - spent(t..u) >= need(t) - spent(t..u) = need(u)

-- stops holding and the metering layer starts withholding cash a `HIRE` needed.
These tests pin all three copies against `kaggle_environments` itself.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from search import cash_schedule  # noqa: E402


def _load_agent():
    spec = importlib.util.spec_from_file_location(
        "main_agent_cash", os.path.join(PROJECT_ROOT, "main.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AGENT = _load_agent()
ENV = pytest.importorskip(
    "kaggle_environments.envs.kaggriculture.kaggriculture",
    reason="kaggle_environments not installed",
)


def test_catalogue_matches_the_interpreter() -> None:
    assert cash_schedule.SEED_COST == {c: p["seed"] for c, p in ENV.CROPS.items()}
    assert cash_schedule.ANIMAL_COST == {a: p["cost"] for a, p in ENV.ANIMALS.items()}
    assert list(cash_schedule.LAND_PRICES) == list(ENV.LAND_PRICES)
    assert cash_schedule.HIRE_MULT == ENV.FARM_HAND_COST_MULT


def test_the_agents_copy_matches_the_modules_copy() -> None:
    """`main.py` carries its own copy because it may not import the env."""
    assert AGENT._SEED_COST == cash_schedule.SEED_COST
    assert AGENT._ANIMAL_COST == cash_schedule.ANIMAL_COST
    assert tuple(AGENT._LAND_PRICES) == tuple(cash_schedule.LAND_PRICES)


def test_fib_matches_the_interpreters_hire_cost() -> None:
    for n in range(15):
        assert cash_schedule.fib(n) == ENV._fib(n)
        assert AGENT._fib(n) == ENV._fib(n)
        assert cash_schedule.fib(n) * cash_schedule.HIRE_MULT == ENV._hire_cost(n)


def test_baked_requirement_matches_the_offline_module() -> None:
    fixed, units = cash_schedule.requirement(AGENT._ROUTE)
    assert AGENT._CASH_FIXED == fixed
    assert AGENT._CASH_UNITS == units


def test_requirement_is_a_non_increasing_suffix_sum() -> None:
    fixed, units = cash_schedule.requirement(AGENT._ROUTE)
    per_fixed, per_units = cash_schedule.step_costs(AGENT._ROUTE)
    assert len(fixed) == len(AGENT._ROUTE) + 1
    assert fixed[-1] == 0.0 and units[-1] == 0
    assert fixed[0] == pytest.approx(sum(per_fixed))
    assert units[0] == sum(per_units)
    for t in range(len(fixed) - 1):
        assert fixed[t] >= fixed[t + 1]
        assert units[t] >= units[t + 1]


def test_the_hire_counter_resets_at_the_day_boundary() -> None:
    """Two hires on one day cost fib(0) + fib(1); one on each of two days costs
    fib(0) twice. Getting this wrong understates the requirement."""
    same_day = [{"market": [["HIRE"], ["HIRE"]]}] + [{"market": []}] * 47
    split = (
        [{"market": [["HIRE"]]}]
        + [{"market": []}] * 23
        + [{"market": [["HIRE"]]}]
        + [{"market": []}] * 23
    )
    assert cash_schedule.requirement(same_day)[0][0] == ENV._fib(0) + ENV._fib(1)
    assert cash_schedule.requirement(split)[0][0] == 2 * ENV._fib(0)


def test_the_land_ladder_is_walked_in_order_and_stops_at_three() -> None:
    route = [{"market": [["BUY_LAND"]]} for _ in range(4)]
    fixed, _ = cash_schedule.requirement(route)
    assert fixed[0] == sum(ENV.LAND_PRICES)  # the fourth buys nothing


def test_buy_product_is_reported_as_units_not_dollars() -> None:
    """Its price quotes against a shared inventory both seats move, so it is not
    knowable offline. Pretending otherwise is how an envelope stops being one."""
    route = [{"market": [["BUY_PRODUCT", "WHEAT", 7]]}]
    fixed, units = cash_schedule.requirement(route)
    assert fixed[0] == 0.0
    assert units[0] == 7


def test_census_reproduces_the_shipped_routes_purchase_schedule() -> None:
    c = cash_schedule.census(AGENT._ROUTE)
    assert c["orders"]["HIRE"] == 277
    assert c["orders"]["BUY_LAND"] == 2
    assert c["units"]["BUY_PRODUCT WHEAT"] == 522
    assert c["fixed_total"] > 0
