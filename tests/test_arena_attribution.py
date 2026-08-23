"""The arena's counters have to belong to the seat they claim (issue #30).

`shed_overflow_lost` and the failed-order counters are the only view we have of
things the observation cannot show, and #30's gate is stated directly in terms of
them: *"zero increase in shed overflow, and zero failed HIRE/BUY orders relative
to the incumbent -- instrument this explicitly, it is the failure mode."*

They were attributed by object identity, on the rule "the first distinct object
seen belongs to player 0", numbered by insertion order. That rule is wrong here:
`kaggle_environments` re-materialises the observation between steps, so a farm is
a different object on almost every turn -- 59 distinct farm ids in a 48-step
episode -- the map accumulated one stale entry per turn, and the seats silently
swapped whenever CPython recycled an id. Two runs of the *same* episode returned
different overflow counts.

These tests pin both halves of the replacement: the market-phase counters key on
a map rebuilt from `state` at the top of `_process_market`, and the unit-phase
counter keys on `idx == 0` being a seat boundary in `interpreter`'s call order.
"""

from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

pytest.importorskip("kaggle_environments", reason="kaggle_environments not installed")

import local_arena  # noqa: E402

STEPS = 48  # two in-game days: enough for an end-of-day drop and a market phase
COUNTERS = (
    "shed_overflow_lost",
    "orders_failed_hire",
    "orders_failed_land",
    "orders_failed_buy",
    "orders_failed_sell",
    "actions_total",
    "actions_pass",
)


def _run(agent: str, opponent: str, swap: bool = False, seed: int = 2_000_000) -> dict:
    return local_arena.run_episode(
        {
            "agent": agent,
            "opponent": opponent,
            "seed": seed,
            "steps": STEPS,
            "swap": swap,
            "decision_log": None,
            "replay": None,
        }
    )


MAIN = os.path.join(PROJECT_ROOT, "main.py")


def test_the_same_episode_twice_gives_the_same_counters() -> None:
    """The regression itself. Identical inputs, identical instrumentation."""
    first = _run(MAIN, "pass")
    second = _run(MAIN, "pass")
    assert {k: first[k] for k in COUNTERS} == {k: second[k] for k in COUNTERS}


@pytest.mark.parametrize("swap", [False, True])
def test_unit_actions_are_counted_for_our_seat_and_not_the_other(swap: bool) -> None:
    """`pass` acts once per turn with one unit; the route acts with up to
    thirteen. If the seats were swapped the count would be an order of magnitude
    out, in a direction the assertion can see."""
    ours = _run("pass", MAIN, swap=swap)
    assert ours["actions_total"] == ours["actions_pass"], "the pass agent only ever PASSes"
    assert ours["actions_total"] <= STEPS, "one unit, one action per turn"

    theirs = _run(MAIN, "pass", swap=swap)
    assert theirs["actions_total"] > 2 * STEPS, "the route drives many units per turn"
    assert theirs["actions_pass"] < theirs["actions_total"]
