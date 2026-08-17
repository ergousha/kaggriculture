"""Regression tests for the interpreter's within-turn ordering.

`interpreter` applies every unit action (kaggriculture.py L922-926) *before* it runs
`_process_market` (L928). So `obs.private.shed` — the shed an agent reads when it builds
its action — is the shed as it stood before this turn's `PLACE` / `DROP` land in it. It
is a lower bound on what the market loop will actually see, not the quantity available to
sell.

Any adaptive market layer that writes `qty = min(route_qty, observed_shed)` therefore
truncates every sale, every turn, for every product. The symptom is a collapse to near-$0
that reads as economic and is not. `main.py` is safe today only because it replays
recorded quantities verbatim; see docs/experiments.md, "Market layer: measured and
exhausted".
"""

from __future__ import annotations

import inspect
from typing import Any

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture

PASS_TURN: dict[str, Any] = {"farmer": ["PASS"], "hands": [], "market": []}


def _run_scripted(script: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """Run a two-player episode in which player 0 plays `script[step]`, else passes.

    Returns one snapshot per step of what player 0 observed *at the moment it was asked
    to act* — which is exactly the information an adaptive market layer would size from.
    """
    seen: list[dict[str, Any]] = []

    def scripted(obs: Any, config: Any) -> dict[str, Any]:
        step = obs["step"]
        seen.append(
            {
                "step": step,
                "shed": dict(obs["private"]["shed"]),
                "money": obs["farms"][obs["player"]]["money"],
                "market_wheat": obs["market"]["inventory"]["WHEAT"],
            }
        )
        return script.get(step, PASS_TURN)

    def idle(obs: Any, config: Any) -> dict[str, Any]:
        return PASS_TURN

    # +3 leaves at least one observed turn after the last scripted one: agents are polled
    # for steps 0..episodeSteps-2, so the aftermath of step N needs episodeSteps >= N + 3.
    steps = max(script) + 3
    env = make("kaggriculture", configuration={"episodeSteps": steps}, debug=True)
    env.run([scripted, idle])
    return seen


def test_observed_shed_is_a_lower_bound_on_what_the_market_sees() -> None:
    """A same-turn PLACE feeds a SELL the observation says is impossible.

    step 0  buy one WHEAT into the shed
    step 1  PICKUP it into the farmer's inventory, so the shed reads empty
    step 2  PLACE it back *and* sell it in the same turn
    """
    seen = _run_scripted(
        {
            0: {"farmer": ["PASS"], "hands": [], "market": [["BUY_PRODUCT", "WHEAT", 1]]},
            1: {"farmer": ["PICKUP", "WHEAT", 1], "hands": [], "market": []},
            2: {
                "farmer": ["PLACE", "WHEAT", 1],
                "hands": [],
                "market": [["SELL", "WHEAT", 1]],
            },
        }
    )

    assert seen[0]["shed"]["WHEAT"] == 0
    assert seen[1]["shed"]["WHEAT"] == 1, "the step-0 purchase should be visible by step 1"

    # The observation the sale was sized against says the shed holds no wheat...
    assert seen[2]["shed"]["WHEAT"] == 0
    # ...and the sale went through anyway, because PLACE ran first.
    assert seen[3]["money"] > seen[2]["money"], "SELL was starved by a stale observation"
    assert seen[3]["market_wheat"] == seen[2]["market_wheat"] + 1
    assert seen[3]["shed"]["WHEAT"] == 0

    # min(route_qty, observed_shed) would have sized this order to zero.
    assert min(1, seen[2]["shed"]["WHEAT"]) == 0


def test_observed_shed_is_not_an_upper_bound_either() -> None:
    """A same-turn PICKUP empties the shed under a SELL the observation permitted."""
    seen = _run_scripted(
        {
            0: {"farmer": ["PASS"], "hands": [], "market": [["BUY_PRODUCT", "WHEAT", 1]]},
            1: {
                "farmer": ["PICKUP", "WHEAT", 1],
                "hands": [],
                "market": [["SELL", "WHEAT", 1]],
            },
        }
    )

    assert seen[1]["shed"]["WHEAT"] == 1, "the observation says the wheat is sellable"
    assert seen[2]["money"] == seen[1]["money"], "but PICKUP ran first and the SELL no-opped"
    assert seen[2]["market_wheat"] == seen[1]["market_wheat"]


def test_interpreter_applies_unit_actions_before_the_market() -> None:
    """Pin the ordering itself, so a dependency bump that reorders it fails here."""
    src = inspect.getsource(kaggriculture.interpreter)
    unit = src.index("_apply_unit_action")
    market = src.index("_process_market")
    town = src.index("_town_consume")
    assert unit < market < town
