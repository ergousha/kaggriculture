"""The route's cash schedule: what it still has to pay for, step by step (#30).

Issue #30's premise, and the reason the three previous metering attempts failed:
the shipped route is **not** a production plan with a market layer attached, it is
a *cash schedule*. 277 `HIRE` orders, 10 `BUY_ANIMAL`, 91 `BUY_SEED` orders, 522
units of `BUY_PRODUCT WHEAT` and 2 `BUY_LAND` are each timed against money the
route expects to already have. Withhold a sale and the next `HIRE` silently does
not happen -- `_do_hire` returns without a word if `farm["money"] < cost` -- the
hand never materialises, and throughput collapses. The failure is a liquidity
failure, not an economic one.

The asset nobody had used is that **the route is fully known in advance**, so the
requirement is computable rather than guessable:

  * `HIRE` costs `farmHandCostMult * fib(hires_today)` with the counter reset by
    `_end_of_day`, so the n-th hire *of that day* costs fib(n) and the whole
    sequence follows from the route's own order stream;
  * `BUY_LAND` walks `LAND_PRICES = [1000, 2000, 4000]` in order;
  * `BUY_SEED` and `BUY_ANIMAL` are catalogue prices (`CROPS[c]["seed"]`,
    `ANIMALS[a]["cost"]`).

Those four are exact. `BUY_PRODUCT` is the one term that is *not* knowable
offline -- it quotes at `market_price(item, inventory - 1)` against a shared
inventory both seats move -- so this module reports it as a **unit count** and
leaves the pricing to whoever has a live observation. `main.py` prices the tail
from `obs.market` at a safety multiple; nothing here pretends to know it.

The result is `requirement(route)`: two arrays such that at step `t` the rest of
the episode costs `fixed[t] + price(WHEAT) * units[t]`. Hold that much cash and
**no future order in the route can fail**, because money only leaves the farm
through those orders:

    money(u) >= money(t) - spent(t..u) >= fixed(t) - spent(t..u) = fixed(u)

for every u >= t. That inequality is the whole safety argument for #30's
metering, and it is why metering inside the envelope is safe by construction
while the flat price floors measured in #23 ($26k / $23k / $1k against a $104k
baseline) were not.

    uv run python -m search.cash_schedule            # the incumbent's schedule
    uv run python -m search.cash_schedule --profile  # per-day requirement table
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Mirrors of the interpreter's catalogues. Duplicated rather than imported
# because `main.py` is standard-library-only and has to carry the same numbers;
# `tests/test_cash_schedule.py` asserts all three copies agree with the env.
SEED_COST = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
ANIMAL_COST = {"GOOSE": 300, "COW": 400, "SHEEP": 500}
LAND_PRICES = (1000, 2000, 4000)
TURNS_PER_DAY = 24
HIRE_MULT = 1


def fib(n: int) -> int:
    """`_fib(0) = 1, _fib(1) = 1, _fib(2) = 2, ...`, the env's own indexing."""
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def step_costs(
    route: list[dict],
    turns_per_day: int = TURNS_PER_DAY,
    hire_mult: int = HIRE_MULT,
) -> tuple[list[float], list[int]]:
    """Per step: (exactly-priced dollars, unpriced `BUY_PRODUCT` units).

    The hire counter resets at the day boundary because `_end_of_day` zeroes
    `farm["hires_today"]`, and a route that fires 12 hires on day 19 pays
    1+1+2+3+5+8+13+21+34+55+89+144 that day and starts again at 1 on day 20.
    """
    fixed: list[float] = []
    units: list[int] = []
    hires_today = 0
    day = -1
    lands = 0
    for t, action in enumerate(route):
        if t // turns_per_day != day:
            day = t // turns_per_day
            hires_today = 0
        cost = 0.0
        qty = 0
        for order in action.get("market") or []:
            if not isinstance(order, list) or not order:
                continue
            op = order[0]
            if op == "HIRE":
                cost += hire_mult * fib(hires_today)
                hires_today += 1
            elif op == "BUY_LAND":
                if lands < len(LAND_PRICES):
                    cost += LAND_PRICES[lands]
                    lands += 1
            elif len(order) >= 3:
                try:
                    n = int(order[2])
                except (TypeError, ValueError):
                    continue
                if n <= 0:
                    continue
                if op == "BUY_SEED":
                    cost += SEED_COST.get(order[1], 0) * n
                elif op == "BUY_ANIMAL":
                    cost += ANIMAL_COST.get(order[1], 0) * n
                elif op == "BUY_PRODUCT":
                    qty += n
        fixed.append(cost)
        units.append(qty)
    return fixed, units


def requirement(
    route: list[dict],
    turns_per_day: int = TURNS_PER_DAY,
    hire_mult: int = HIRE_MULT,
) -> tuple[list[float], list[int]]:
    """Cumulative *future* requirement: `(fixed, units)`, both length `len(route) + 1`.

    `fixed[t]` and `units[t]` cover every order at steps `>= t`, so `fixed[0]` is
    the whole episode's exactly-priced outlay and `fixed[len(route)]` is 0. The
    trailing zero entry is deliberate: a runtime layer indexes this by a step that
    can reach `len(route)` on the final turn.
    """
    per_fixed, per_units = step_costs(route, turns_per_day, hire_mult)
    n = len(route)
    fixed = [0.0] * (n + 1)
    units = [0] * (n + 1)
    for t in range(n - 1, -1, -1):
        fixed[t] = fixed[t + 1] + per_fixed[t]
        units[t] = units[t + 1] + per_units[t]
    return fixed, units


def census(route: list[dict]) -> dict[str, Any]:
    """Order counts and exact costs, the table #30 opens with."""
    counts: dict[str, int] = {}
    volumes: dict[str, int] = {}
    for action in route:
        for order in action.get("market") or []:
            if not isinstance(order, list) or not order:
                continue
            key = order[0] if order[0] in ("HIRE", "BUY_LAND") else f"{order[0]} {order[1]}"
            counts[key] = counts.get(key, 0) + 1
            if len(order) >= 3:
                try:
                    volumes[key] = volumes.get(key, 0) + int(order[2])
                except (TypeError, ValueError):
                    pass
    fixed, units = requirement(route)
    return {
        "steps": len(route),
        "orders": dict(sorted(counts.items())),
        "units": dict(sorted(volumes.items())),
        "fixed_total": fixed[0],
        "buy_product_units": units[0],
    }


def profile(route: list[dict], wheat_price: float = 45.0) -> list[dict[str, Any]]:
    """The per-day requirement table, one row per day."""
    fixed, units = requirement(route)
    rows = []
    for day in range(0, (len(route) + TURNS_PER_DAY - 1) // TURNS_PER_DAY):
        t = day * TURNS_PER_DAY
        rows.append(
            {
                "day": day,
                "step": t,
                "fixed": fixed[t],
                "buy_product_units": units[t],
                "requirement": fixed[t] + units[t] * wheat_price,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Route cash schedule (issue #30)")
    ap.add_argument("--profile", action="store_true", help="print the per-day requirement table")
    ap.add_argument(
        "--wheat-price",
        type=float,
        default=45.0,
        help="price used to value the unpriced BUY_PRODUCT tail in --profile",
    )
    args = ap.parse_args(argv)

    import main as main_module

    # v0.4.0+ ships the multi-route chassis (``_ROUTES``). The schedule tool
    # is route-shaped, so use the base tape (route 0).
    route = main_module._ROUTES[0]
    c = census(route)
    print(f"  route {c['steps']} steps")
    print(f"  {'order':<22} {'count':>6} {'units':>7}")
    for key, n in c["orders"].items():
        print(f"  {key:<22} {n:>6} {c['units'].get(key, ''):>7}")
    print(f"\n  exactly-priced outlay  ${c['fixed_total']:>10,.0f}")
    print(f"  BUY_PRODUCT units       {c['buy_product_units']:>10,}  (priced live)")

    if args.profile:
        print(f"\n  {'day':>4} {'step':>5} {'fixed':>10} {'units':>7} {'requirement':>13}")
        for row in profile(route, args.wheat_price):
            print(
                f"  {row['day']:>4} {row['step']:>5} {row['fixed']:>10,.0f} "
                f"{row['buy_product_units']:>7} {row['requirement']:>13,.0f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
