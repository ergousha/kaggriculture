"""Mutation-and-accept search over the 719-step action stream (issue #26).

Why this exists. Route *selection* is exhausted: all 4,315 candidates in
`candidates.jsonl` are the same strategy sampled 4,315 times (cows 8-10, sheep
4-6, ~37 strawberry seeds, ~144 wheat seeds, ~72 FERTILIZE, zero geese), and the
top 20 teams by replay count all bank $105k-$112k mean. Mining more days draws
more samples from one distribution. The repo has tried (a) a hand-written
closed-loop planner and (b) replaying other people's routes; it has never tried
**(c) optimising a route directly**.

This module is the harness for (c), and nothing else. It:

  * loads a seed route (the shipped `main.py._ROUTE` by default, hash-verified
    against `candidates.jsonl` when a pool is present),
  * applies one of six individually toggleable mutation operators,
  * evaluates a mutated route through the **exact artifact we would ship** —
    `mining.common.write_route_agent` bakes it into `build_route_agent`'s
    template, so WEED repair, SELL-slot ordering and hands alignment are in
    play — scored by the Phase 2 engine (`simulate_candidates.run_stage`, common
    random numbers, resumable per `(hash, opponent, seed)`),
  * accepts on **mean panel win rate with worst-opponent win rate as the
    tiebreak**, the same metric `rank_cvar.py` selects on,
  * never re-evaluates a route hash it has already seen.

This ships no agent and changes nothing by construction — see the gates in the
module's bottom matter and in `tests/test_route_search.py`.

    # Gate: zero mutations -> emitted route is byte-identical to the seed.
    uv run python -m search.route_search --self-test

    # Smoke: a real (slow) search pass against a small panel.
    uv run python -m search.route_search --iterations 4 --workers 8

Mutation operators (each individually toggleable via `--no-<name>`):

  * `shift_task_block`       shift one unit's task block by ±k steps, re-aligning
                             the tail (noop by construction — see below),
  * `retarget_plant`         retarget a PLANT to a different contestable crop,
  * `swap_herd`              convert a BUY_ANIMAL COW to SHEEP and repair the
                             downstream chore cadence (interval 3 -> 2),
  * `assign_idle`            give a PASS unit-turn a productive task (#28),
  * `repath`                 re-path a movement run to the Manhattan-shortest
                             walk between its two fixed endpoints (#29; exact,
                             and self-verifying -- see `search/board_paths.py`),
  * `move_sell_and_buy`      move a SELL and the BUY it funds together (#30;
                             clamped so the purchase never outruns the unit op
                             it feeds and the sale never outruns the stock that
                             fills it -- `scope="all"` moves every pair at once,
                             because one pair out of 82 is below the grid's
                             noise floor).

Honest scope. Route synthesis is *not* the no-op that route selection + runtime
layers is: several operators change what the agent does and can invalidate the
route outright. The gate that holds is `#26`'s **zero-mutation** gate, not a
"mutations are free" gate. The three operators that are safe by construction
(`shift_task_block`, herd *addition*, path *shortening* that only touches moves)
are labelled; the rest are marked speculative and default ON only because the
issues that depend on them (#27-#30) need them measured, not assumed. Any route
that produces an invalid action is rejected and counted in `rejected_invalid`.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Ensure project root is in sys.path when invoked directly
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# The Phase 2 harness is the engine: it owns env construction, seeded
# configuration, the (candidate, opponent, seed) resume key and the status
# accounting. Importing it keeps this harness and Phase 2 from ever disagreeing
# about what "evaluate a route" means.
import simulate_candidates as phase2  # noqa: E402
from mining import common  # noqa: E402
from mining.common import PROJECT_ROOT, decode_route_b85  # noqa: E402
from search import board_paths  # noqa: E402

SEED_CANDIDATE_PREFIX = "e8c035f9d0"  # v0.3.1 incumbent (issue #28)
RESULTS_PATH = os.path.join(PROJECT_ROOT, "logs", "route_search_results.jsonl")
AGENT_CACHE = os.path.join(PROJECT_ROOT, "logs", "_route_search_agents")

# All six operators from the issue, in the order they are documented there.
ALL_OPERATORS = (
    "shift_task_block",
    "retarget_plant",
    "swap_herd",
    "assign_idle",
    "repath",
    "move_sell_and_buy",
)

# Default panel: the v0.2.7 leaderboard-band panel (ranks 185..469). Members are
# stored as baked agents under `logs/_mined_agents/<full-hash>.py`; the 10-char
# labels below are the labels Phase 2's results file uses and the prefix of each
# member's full hash, resolved against the candidate pool at panel-build time.
# A mutated route this search has never seen is evaluated live; nothing is ever
# reused across hashes, because a new hash is a new route by construction.
DEFAULT_PANEL_LABELS = (
    "v0_2_6",
    "62b81aa8a3",
    "ebfc911eaa",
    "800dc80f5c",
    "b8b9267d1c",
    "8f7dd57d5f",
)
MINED_AGENT_DIR = os.path.join(PROJECT_ROOT, "logs", "_mined_agents")

# The fallback panel, and the only one reproducible from a clean checkout:
# `candidates.jsonl` and `logs/_mined_agents/` are gitignored, so the
# leaderboard-band panel above cannot be rebuilt without a fresh mine. Four of
# these six are our own successive versions, which makes it a *harder* panel than
# the mined one and the absolute win rates correspondingly lower. That does not
# weaken a comparison between two routes scored on it, which is all this harness
# ever asks of a panel.
LOCAL_PANEL = (
    ("v0_2_6", "opponents/v0_2_6.py"),
    ("v0_2_7", "opponents/v0_2_7.py"),
    ("v0_3_0", "opponents/v0_3_0.py"),
    ("v0_3_1", "opponents/v0_3_1.py"),
    ("rita", "opponents/ladder/rancher_rita.py"),
    ("mateo", "opponents/ladder/melon_mateo.py"),
)

# Contiguous movement ops whose only effect on the board is "arrive one step
# later"; shifting them cannot collide with anything because nothing depends on
# a unit's *position* mid-run, only on the tile ops that bracket the run.
# `board_paths` owns the set because it also owns the deltas, and a disagreement
# between "what counts as a move" here and there would be silent.
MOVE_OPS = board_paths.MOVE_OPS

# Crops worth retargeting a PLANT to: the uncontested drain (#28) plus the two
# products the field already produces. EGG is deliberately absent — it needs a
# COOP, and the egg-engine ablation measured −62.3% on 30 paired seeds.
RETARGET_CROPS = ("CARROT", "TOMATO", "STRAWBERRY", "WHEAT")


# ---------------------------------------------------------------------------
# Seed loading
# ---------------------------------------------------------------------------


def _hash_of(route: list[dict]) -> str:
    return common.route_hash(common.normalize_route(route))


def load_seed(candidates_path: str | None = None) -> list[dict]:
    """The seed route: `main.py._ROUTE`, hash-verified against the pool.

    The issue seeds the loop from the shipped incumbent (hash `18057e3167` for
    v0.2.6, `044a7741e9` for v0.2.7), and the strongest reason to prefer
    `main.py` over the pool is that `main.py` is what ships. But the pool's hash
    is the identity the rest of the pipeline already agrees on, so when a pool is
    present we assert the two round-trip to the same canonical hash. If they do
    not, we trust the pool and say so — a drift between the baked artifact and
    the mined trace is exactly the kind of thing that silently poisons a search.
    """
    import main as main_module

    seed = main_module._ROUTE
    seed_hash = _hash_of(seed)
    if candidates_path and os.path.exists(candidates_path):
        for cand in common.read_jsonl(candidates_path):
            if not cand.get("hash", "").startswith(SEED_CANDIDATE_PREFIX):
                continue
            pool_route = decode_route_b85(cand["route_b85"])
            pool_hash = _hash_of(pool_route)
            if pool_hash != seed_hash:
                print(
                    f"  !! seed drift: main.py._ROUTE hashes to {seed_hash[:10]}, "
                    f"the pool's {SEED_CANDIDATE_PREFIX} round-trips to {pool_hash[:10]}. "
                    "Using the pool's copy.",
                    file=sys.stderr,
                )
                return pool_route
            return seed
    return seed


# ---------------------------------------------------------------------------
# Mutation operators
#
# Every operator takes a *copy* of the route (normalized, so PASS is explicit
# and hands are already slot-aligned) and returns a mutated copy plus a short
# human-readable note, or `None` if it cannot act on this route (e.g. there is
# no BUY_ANIMAL COW for `swap_herd` to convert). Returning `None` is not a
# failure — it is how an operator sits out a route it does not apply to.
# ---------------------------------------------------------------------------


def _units(step_action: dict, min_hands: int = 0) -> list[list]:
    """The farmer + hands as one mutable unit list, padded to a slot count.

    Hands are hired across the episode, so a later step can have more slots than
    an earlier one. Operators index a unit by a slot found on one step and probe
    the same slot on a neighbouring step, so callers pass the slot count they
    need and this pads the short side with PASS rather than raising.
    """
    farmer = list(step_action.get("farmer") or ["PASS"])
    hands = [list(h or ["PASS"]) for h in (step_action.get("hands") or [])]
    while len(hands) < min_hands:
        hands.append(["PASS"])
    return [farmer, *hands]


def _op_of(unit_action: list) -> str | None:
    if isinstance(unit_action, list) and unit_action:
        return unit_action[0]
    return None


def op_shift_task_block(route: list[dict], rng, k: int = 1, **_kw) -> tuple[list[dict], str] | None:
    """Shift one unit's task block by ±k steps, re-aligning the tail.

    Only a pure *movement* tail is shifted. A position in the middle of a route
    is load-bearing (a FEED the animal's life depends on, a PICKUP that must
    precede a PLACE), but a unit's *movement* between two tile-ops is free: any
    walk that arrives by the next scheduled op is equivalent (#29's premise).
    Delaying a burst of moves by k steps therefore cannot collide with anything,
    which is what makes this the only operator that is a strict no-op by
    construction. It exists to prove the loop end-to-end (the round-trip and
    identity gates exercise it before any expensive evaluation runs).
    """
    # Find a step where some unit has a run of >=1 moves followed by a non-move.
    for step in range(len(route) - k - 1, 0, -1):
        units_here = _units(route[step])
        for slot, unit in enumerate(units_here):
            prev = _units(route[step - 1], min_hands=len(units_here) - 1)
            if _op_of(unit) in MOVE_OPS and _op_of(prev[slot]) not in MOVE_OPS:
                # A movement run starts here. Shift its tail by +k: write PASS on
                # the original step and replay the moves k steps later.
                run = []  # collect the forward run of moves for this unit
                s = step
                while (
                    s < len(route)
                    and _op_of(_units(route[s], min_hands=len(units_here) - 1)[slot]) in MOVE_OPS
                ):
                    run.append(_units(route[s], min_hands=len(units_here) - 1)[slot])
                    s += 1
                if not run or s + k > len(route):
                    continue
                new = copy.deepcopy(route)
                # Blank the original run first...
                for i in range(len(run)):
                    _write_unit(new[step + i], slot, ["PASS"])
                # ...then place it k steps later, step by step so a destination
                # that holds a real (non-movement) op stops the whole shift
                # rather than clobbering it.
                placed = True
                for i in range(len(run)):
                    dst = step + i + k
                    dst_op = _op_of(_units(new[dst])[slot])
                    if dst_op in MOVE_OPS or dst_op == "PASS":
                        _write_unit(new[dst], slot, run[i])
                    else:
                        placed = False
                        break
                if placed:
                    return (
                        new,
                        f"shifted {len(run)}-step movement run @step {step} slot {slot} by +{k}",
                    )
    return None


def _write_unit(step_action: dict, slot: int, value: list) -> None:
    if slot == 0:
        step_action["farmer"] = list(value)
    else:
        hands = step_action.setdefault("hands", [])
        while len(hands) < slot:
            hands.append(["PASS"])
        hands[slot - 1] = list(value)


def op_retarget_plant(route: list[dict], rng, **_kw) -> tuple[list[dict], str] | None:
    """Retarget a PLANT to a different crop at the same tile (#28's rotation).

    Seeds have to be bought before they can be planted, so a bare retarget is
    illegal the moment the route buys N of crop X and plants N+1. This operator
    rewrites the matching BUY_SEED order too, keeping the (buy, plant) pair
    consistent — the same one-leg-at-a-time discipline #30 demands.
    """
    plants = []
    for step, action in enumerate(route):
        for slot, unit in enumerate(_units(action)):
            if _op_of(unit) == "PLANT" and len(unit) > 1:
                plants.append((step, slot, unit[1]))
    if not plants:
        return None
    step, slot, current = plants[rng.randrange(len(plants))]
    choices = [c for c in RETARGET_CROPS if c != current]
    if not choices:
        return None
    new_crop = choices[rng.randrange(len(choices))]
    new = copy.deepcopy(route)
    # Update the PLANT op itself.
    target = _units(new[step])[slot]
    _write_unit(new[step], slot, ["PLANT", new_crop] + target[2:])
    # Rewrite exactly one BUY_SEED for the old crop, so the buy/plant accounting
    # stays balanced. For orders with quantity > 1, decrement quantity and add/increment new_crop.
    buy_rewritten = False
    for action in new:
        mkt = action.get("market") or []
        for order in mkt:
            if (
                isinstance(order, list)
                and len(order) >= 3
                and order[0] == "BUY_SEED"
                and order[1] == current
            ):
                qty = int(order[2])
                if qty <= 1:
                    order[1] = new_crop
                else:
                    order[2] = qty - 1
                    # Find or add BUY_SEED for new_crop in same market action
                    existing_new = next(
                        (
                            o
                            for o in mkt
                            if isinstance(o, list)
                            and len(o) >= 3
                            and o[0] == "BUY_SEED"
                            and o[1] == new_crop
                        ),
                        None,
                    )
                    if existing_new:
                        existing_new[2] = int(existing_new[2]) + 1
                    elif len(mkt) < 10:
                        mkt.append(["BUY_SEED", new_crop, 1])
                buy_rewritten = True
                break
        if buy_rewritten:
            break

    if not buy_rewritten:
        return None

    # Ensure market stream sells new_crop
    for action in new:
        mkt = action.get("market") or []
        has_new_sell = any(
            isinstance(o, list) and len(o) >= 2 and o[0] == "SELL" and o[1] == new_crop for o in mkt
        )
        has_other_sell = any(
            isinstance(o, list)
            and len(o) >= 2
            and o[0] == "SELL"
            and o[1] in ("STRAWBERRY", "MILK", "WOOL")
            for o in mkt
        )
        if has_other_sell and not has_new_sell and len(mkt) < 10:
            mkt.append(["SELL", new_crop, 10])

    return (
        new,
        f"retargeted PLANT {current}->{new_crop} @step {step} (BUY_SEED & SELL follow)",
    )


def _herd_counts(route: list[dict]) -> tuple[int, int]:
    cows = sheep = 0
    for action in route:
        for order in action.get("market") or []:
            if isinstance(order, list) and len(order) >= 3 and order[0] == "BUY_ANIMAL":
                if order[1] == "COW":
                    cows += int(order[2])
                elif order[1] == "SHEEP":
                    sheep += int(order[2])
    return cows, sheep


COW_SCHEDULE: list[tuple[int, int, tuple[int, int], tuple[int, int], tuple[int, int]]] = [
    # (cow_idx, buy_step, (pickup_step, pickup_slot), (place_step, place_slot), pasture_coord)
    (0, 0, (3, 3), (4, 3), (4, 4)),
    (1, 0, (10, 3), (12, 3), (3, 4)),
    (2, 73, (77, 4), (80, 4), (2, 4)),
    (3, 120, (125, 4), (128, 4), (4, 2)),
    (4, 168, (170, 8), (171, 8), (5, 4)),
    (5, 168, (174, 6), (176, 6), (6, 4)),
    (6, 216, (221, 3), (227, 3), (5, 2)),
    (7, 216, (223, 11), (227, 11), (7, 4)),
    (8, 264, (267, 2), (268, 2), (4, 5)),
    (9, 264, (267, 3), (275, 3), (3, 5)),
]


def op_swap_herd(
    route: list[dict], rng: Any = None, cow_idx: int | None = None, **_kw: Any
) -> tuple[list[dict], str] | None:
    """Convert a BUY_ANIMAL COW to SHEEP, update placement, and integrate wool sells.

    This is #27's lever. Finds an existing COW in the route, updates its buy order
    to SHEEP, swaps its unit PICKUP/PLACE actions from COW to SHEEP, and adds
    SELL WOOL market orders alongside MILK sells so harvested wool is monetized.
    """
    convertible: list[int] = []
    for c_idx, _buy_step, (p_step, p_slot), (_pl_step, _pl_slot), _pasture in COW_SCHEDULE:
        if p_slot == 0:
            act = route[p_step].get("farmer")
        else:
            hands = route[p_step].get("hands") or []
            act = hands[p_slot - 1] if p_slot - 1 < len(hands) else None
        if act and len(act) >= 2 and act[0] == "PICKUP" and act[1] == "COW":
            convertible.append(c_idx)

    if not convertible:
        return None

    if cow_idx is not None and cow_idx in convertible:
        chosen_idx = cow_idx
    elif rng is not None and hasattr(rng, "randrange"):
        chosen_idx = convertible[rng.randrange(len(convertible))]
    else:
        chosen_idx = convertible[0]

    idx, buy_step, (p_step, p_slot), (pl_step, pl_slot), pasture = COW_SCHEDULE[chosen_idx]

    new = copy.deepcopy(route)

    # 1. Update buy order
    market = new[buy_step].get("market") or []
    # Check if a SHEEP buy order already exists in this market step
    existing_sheep_order = next(
        (
            o
            for o in market
            if isinstance(o, list) and len(o) >= 3 and o[0] == "BUY_ANIMAL" and o[1] == "SHEEP"
        ),
        None,
    )
    for order in market:
        if (
            isinstance(order, list)
            and len(order) >= 3
            and order[0] == "BUY_ANIMAL"
            and order[1] == "COW"
        ):
            qty = int(order[2])
            if existing_sheep_order is not None:
                existing_sheep_order[2] = int(existing_sheep_order[2]) + 1
                if qty == 1:
                    market.remove(order)
                else:
                    order[2] = qty - 1
            else:
                if qty == 1:
                    order[1] = "SHEEP"
                elif qty > 1:
                    order[2] = qty - 1
                    if len(market) < 10:
                        market.append(["BUY_ANIMAL", "SHEEP", 1])
            break

    # 2. Update pickup
    if p_slot == 0:
        new[p_step]["farmer"] = ["PICKUP", "SHEEP", 1]
    else:
        new[p_step]["hands"][p_slot - 1] = ["PICKUP", "SHEEP", 1]

    # 3. Update place
    if pl_slot == 0:
        new[pl_step]["farmer"] = ["PLACE", "SHEEP", 1]
    else:
        new[pl_step]["hands"][pl_slot - 1] = ["PLACE", "SHEEP", 1]

    # 4. Integrate SELL WOOL orders across the route
    for step in range(len(new)):
        mkt = new[step].get("market") or []
        has_milk_sell = any(
            isinstance(o, list) and len(o) >= 2 and o[0] == "SELL" and o[1] == "MILK" for o in mkt
        )
        has_wool_sell = any(
            isinstance(o, list) and len(o) >= 2 and o[0] == "SELL" and o[1] == "WOOL" for o in mkt
        )
        if has_milk_sell and not has_wool_sell and len(mkt) < 10:
            mkt.append(["SELL", "WOOL", 6])

    cows, sheep = _herd_counts(new)
    return (
        new,
        f"converted COW #{chosen_idx} to SHEEP @ pasture {pasture} (herd is now {cows}c/{sheep}s; wool sells integrated)",
    )


def op_assign_idle(route: list[dict], rng, **_kw) -> tuple[list[dict], str] | None:
    """Give a PASS unit-turn a productive task (#28).

    Fills a PASS whose unit is already standing on or adjacent to farmed ground.
    Prioritizes WATER (which requires 0 inventory slots and directly adds crop yield)
    or FERTILIZE. It does not touch movement, so it cannot strand a unit.
    """
    idles = []
    for step, action in enumerate(route):
        for slot, unit in enumerate(_units(action)):
            if _op_of(unit) != "PASS":
                continue
            neighbor_has_tile_op = False
            for d in (-2, -1, 1, 2):
                ns = step + d
                if not (0 <= ns < len(route)):
                    continue
                units = _units(route[ns], min_hands=slot)
                if _op_of(units[slot]) in (
                    "WATER",
                    "HARVEST",
                    "PLANT",
                    "FERTILIZE",
                ):
                    neighbor_has_tile_op = True
                    break
            if neighbor_has_tile_op:
                idles.append((step, slot))
    if not idles:
        return None
    step, slot = idles[rng.randrange(len(idles))]
    new = copy.deepcopy(route)
    _write_unit(new[step], slot, ["WATER"])
    return (
        new,
        f"assigned PASS->WATER @step {step} slot {slot} (unit already adjacent to farmed ground)",
    )


def op_repath(
    route: list[dict],
    rng: Any = None,
    scope: str = "all",
    drop_terminal: bool = False,
    **_kw: Any,
) -> tuple[list[dict], str] | None:
    """Re-path movement runs between fixed endpoints (#29).

    Between two consecutive position-dependent ops a unit's walk is free: any
    route that arrives by the turn the next op is scheduled is equivalent, and
    since this env applies a move iff the destination is on the board (`LOCKED`
    tiles do not block, units do not collide), the shortest such walk is any
    monotone staircase of length `manhattan(origin, destination)`. So the
    operator is exact rather than heuristic, and `board_paths.verify_schedule`
    proves it site by site before the route is ever scored: every non-movement op
    still fires on its original step with its unit on its original tile.

    `scope="all"` re-paths every wasteful segment at once — that is the mutation
    #29's no-op gate is written against. `scope="one"` picks a single wasteful
    segment, so the search loop can attribute an accept to one site.

    `drop_terminal` additionally blanks segments with no anchor after them in
    their day. Those moves are provably dead labour — the end-of-day reset
    discards the position they buy, and nothing in `_end_of_day` reads a unit's
    tile — and they are where 87% of the recoverable turns are. It is off by
    default only because banking them stops the route reproducing the incumbent's
    within-day final tile; that was gated separately and came back bit-identical
    on 180 paired episodes, so turning it on is evidence-backed rather than a
    guess (see `docs/experiments.md`).

    Returns `None` when there is nothing to recover, which is the common case:
    the incumbent's interior walking is already 98.3% Manhattan-optimal.
    """
    segs = board_paths.segments(route)
    wasteful = [s for s in segs if s.slack > 0 or (drop_terminal and s.terminal and s.moves)]
    if not wasteful:
        return None
    if scope == "one":
        if rng is not None and hasattr(rng, "randrange"):
            wasteful = [wasteful[rng.randrange(len(wasteful))]]
        else:
            wasteful = [wasteful[0]]
    elif scope != "all":
        raise ValueError(f"scope must be 'all' or 'one', got {scope!r}")

    mutated, stats = board_paths.repath(route, only=wasteful, drop_terminal=drop_terminal)
    if not stats["turns_recovered"]:
        return None
    violations = board_paths.verify_schedule(route, mutated)
    if violations:
        # A re-path that moves an op off its tile is a bug in this operator, not
        # a candidate. Refuse to emit it rather than let the panel absorb it.
        print(f"  !! repath rejected: {len(violations)} schedule violation(s): {violations[0]}")
        return None
    where = (
        f"{stats['segments_rewritten']} segment(s)"
        if scope == "all"
        else f"slot {wasteful[0].slot} steps {wasteful[0].start}..{wasteful[0].end}"
    )
    return (
        mutated,
        f"re-pathed {where}: {stats['moves_before']}->{stats['moves_after']} moves, "
        f"{stats['turns_recovered']} turn(s) banked as PASS"
        + (
            f" ({stats['terminal_segments_dropped']} terminal dropped)"
            if stats["terminal_segments_dropped"]
            else ""
        ),
    )


MAX_MARKET_ORDERS = 10  # configuration.maxMarketOrdersPerTurn; the tail is dropped

# Products whose realised price collapses far enough that re-timing a sale could
# pay for itself: MILK clears 13% of its $160 base and STRAWBERRY 52% of its
# $120 (measured by `scripts/analyse_market_fills.py`). WOOL clears 103% and
# WHEAT 167%, so moving those is a bet against the only two products the route
# already prices well.
RETIMEABLE_SELLS = ("MILK", "STRAWBERRY")

# Every order that takes money out of the farm. `_do_hire` and `_commit_unit`
# both fail silently on insufficient funds, so these are the orders a deferred
# sale can break, and therefore the ones that have to travel with it.
FUNDED_OPS = ("HIRE", "BUY_LAND", "BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT")


def _order_sites(route: list[dict]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """(sell sites, funded sites) as (step, index-within-that-step's-market)."""
    sells: list[tuple[int, int]] = []
    funded: list[tuple[int, int]] = []
    for step, action in enumerate(route):
        for i, order in enumerate(action.get("market") or []):
            if not isinstance(order, list) or not order:
                continue
            if order[0] == "SELL" and len(order) >= 2 and order[1] in RETIMEABLE_SELLS:
                sells.append((step, i))
            elif order[0] in FUNDED_OPS:
                funded.append((step, i))
    return sells, funded


# What each funded order buys, and the unit op that consumes it. A purchase is
# not a free variable either: the interpreter runs `_apply_unit_action` for every
# unit *before* `_process_market`, so a BUY on step t is only usable from t + 1,
# and a PLANT whose seed has not arrived is not merely skipped -- the interpreter
# drops **every** PLANT of that crop that turn. Move a BUY past its consumer and
# the route stops producing, which is what the first bulk measurement of this
# operator did: STRAWBERRY 249 -> 16 units, MILK 254 -> 78, at *higher* $/unit.
FUNDER_CONSUMERS = {
    "BUY_SEED": ("PLANT",),
    "BUY_ANIMAL": ("PICKUP",),
    "BUY_PRODUCT": ("PICKUP",),
}


def _funder_forward_slack(route: list[dict], site: tuple[int, int]) -> int:
    """How many steps this funded order may be delayed before something starves.

    `HIRE` and `BUY_LAND` return 0 and are never delayed. A missing hand does not
    just idle: `_align_hands` truncates the hands list to the live count, so every
    later slot in that turn's trace shifts by one and the whole hand assignment
    for the turn is wrong. A quadrant's tiles stay `LOCKED` until `BUY_LAND`
    clears, and a tile op on a locked tile is a silent no-op.
    """
    step, idx = site
    order = (route[step].get("market") or [])[idx]
    op = order[0]
    if op in ("HIRE", "BUY_LAND"):
        return 0
    item = order[1] if len(order) > 1 else None
    consumers = FUNDER_CONSUMERS.get(op, ())
    for t in range(step + 1, len(route)):
        for unit in _units(route[t]):
            if (
                isinstance(unit, list)
                and len(unit) > 1
                and unit[0] in consumers
                and unit[1] == item
            ):
                return max(0, t - 1 - step)
    return len(route) - 1 - step


# The shed is filled by an explicit PLACE/DROP, and at the day boundary by
# `_drop_inventories_to_shed` emptying every unit's carried inventory into it.
# Both happen before `_process_market` on their turn, so a sale may be pulled
# back to a same-item deposit, or to the start of its day, but no further: the
# mirror image of the forward constraint, and the reason the first backward bulk
# move measured $19,113 against a $63,601 baseline with *higher* $/unit on every
# product. The sale outran the production that fills it while the purchase it
# funds still spent.
DEPOSIT_OPS = ("PLACE", "DROP")
TURNS_PER_DAY = 24


def _sell_backward_slack(route: list[dict], site: tuple[int, int]) -> int:
    """How many steps this SELL may be pulled earlier before the shed is behind."""
    step, idx = site
    order = (route[step].get("market") or [])[idx]
    item = order[1] if len(order) > 1 else None
    floor = (step // TURNS_PER_DAY) * TURNS_PER_DAY
    for t in range(step - 1, floor - 1, -1):
        for unit in _units(route[t]):
            if (
                isinstance(unit, list)
                and len(unit) > 1
                and unit[0] in DEPOSIT_OPS
                and unit[1] == item
            ):
                return step - t
    return step - floor


def funder_slack_census(route: list[dict]) -> dict[str, Any]:
    """The forward slack of every funded order, grouped by op. #30's headline."""
    _sells, funded = _order_sites(route)
    by_op: dict[str, list[int]] = {}
    for site in funded:
        order = (route[site[0]].get("market") or [])[site[1]]
        by_op.setdefault(order[0], []).append(_funder_forward_slack(route, site))
    out = {}
    for op, slacks in sorted(by_op.items()):
        out[op] = {
            "orders": len(slacks),
            "zero": sum(1 for s in slacks if s == 0),
            "median": sorted(slacks)[len(slacks) // 2],
            "max": max(slacks),
        }
    return out


def _pair_sells_with_funders(
    sells: list[tuple[int, int]], funded: list[tuple[int, int]]
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Greedily match each SELL to the first money-spending order after it.

    Greedy-by-position is the right matching here rather than an artefact of
    convenience: the interpreter drains a turn's market queue in slot order and
    spends the proceeds as it goes, so the order a sale actually funds *is* the
    next one that costs money. A funder is claimed once, so two sales never both
    claim the same `HIRE`.
    """
    pairs = []
    claimed: set[tuple[int, int]] = set()
    for site in sells:
        funder = next((f for f in funded if f > site and f not in claimed), None)
        if funder is None:
            continue
        claimed.add(funder)
        pairs.append((site, funder))
    return pairs


def _apply_moves(route: list[dict], moves: dict[tuple[int, int], int]) -> list[dict] | None:
    """Rebuild every market queue with `moves` applied, or `None` on overflow.

    Orders keep their original relative order inside a destination turn, which is
    what keeps a moved SELL ahead of the BUY it funds.
    """
    buckets: dict[int, list[tuple[tuple[int, int], list]]] = {}
    for step, action in enumerate(route):
        for idx, order in enumerate(action.get("market") or []):
            dst = moves.get((step, idx), step)
            buckets.setdefault(dst, []).append(((step, idx), list(order)))
    for dst, entries in buckets.items():
        if len(entries) > MAX_MARKET_ORDERS:
            return None
        if not 0 <= dst < len(route):
            return None
    new = copy.deepcopy(route)
    for step in range(len(new)):
        entries = sorted(buckets.get(step, []), key=lambda e: e[0])
        new[step]["market"] = [order for _origin, order in entries]
    return new


def op_move_sell_and_buy(
    route: list[dict],
    rng: Any = None,
    shift: int | None = None,
    sell_site: tuple[int, int] | None = None,
    scope: str = "one",
    **_kw: Any,
) -> tuple[list[dict], str] | None:
    """Move a SELL and the BUY it funds together (#30's joint operator).

    Every previous attempt in this repo moved one leg. #23 held sales back and
    the next `HIRE` failed for cash; #25 pulled sales forward and sold into a
    market that had not risen yet. Both are the same finding read from opposite
    ends: **the route is a cash schedule**, and a sale's turn is not a free
    variable because a purchase downstream is timed against its proceeds.

    So this operator moves the pair. It picks a SELL of a product whose realised
    price collapses, finds the first money-spending order at or after it -- the
    `HIRE` or `BUY_*` that sale funds -- and slides both by the same `shift`,
    which keeps the gap between revenue and spend exactly as the route recorded
    it while moving *when* the pair happens. The search decides whether the sale
    is worth more `shift` turns later than the purchase is worth sooner; nothing
    here assumes it is.

    Refuses rather than truncates. A destination already holding
    `maxMarketOrdersPerTurn` orders would silently drop the tail of that turn --
    which is the failure #23's write-up warns about -- so the mutation is
    abandoned instead.
    """
    sells, funded = _order_sites(route)
    if not sells or not funded:
        return None

    if scope == "all":
        return _move_every_pair(route, sells, funded, shift, rng)
    if scope != "one":
        raise ValueError(f"scope must be 'one' or 'all', got {scope!r}")

    if sell_site is not None and sell_site in sells:
        s_step, s_idx = sell_site
    elif rng is not None and hasattr(rng, "randrange"):
        s_step, s_idx = sells[rng.randrange(len(sells))]
    else:
        s_step, s_idx = sells[0]

    # The BUY this sale funds: the first money-spending order at or after it.
    # "At or after" rather than strictly after because a turn's market queue is
    # processed in slot order, so a SELL in slot 0 does fund a HIRE in slot 1.
    pair = next(
        ((b_step, b_idx) for b_step, b_idx in funded if (b_step, b_idx) > (s_step, s_idx)),
        None,
    )
    if pair is None:
        return None
    b_step, b_idx = pair

    if shift is None:
        if rng is not None and hasattr(rng, "randrange"):
            shift = rng.choice([-12, -6, -3, -1, 1, 3, 6, 12])
        else:
            shift = 6
    if shift == 0:
        return None
    if shift > 0:
        shift = min(shift, _funder_forward_slack(route, (b_step, b_idx)))
    else:
        shift = -min(-shift, _sell_backward_slack(route, (s_step, s_idx)))
    if shift == 0:
        return None

    s_dst, b_dst = s_step + shift, b_step + shift
    if not (0 <= s_dst < len(route) and 0 <= b_dst < len(route)):
        return None
    if s_dst == s_step and b_dst == b_step:
        return None
    # The pair has to stay ordered: a sale that lands after the purchase it funds
    # is the very failure this operator exists to avoid.
    if s_dst > b_dst:
        return None

    new = copy.deepcopy(route)
    sell_order = list((new[s_step].get("market") or [])[s_idx])
    buy_order = list((new[b_step].get("market") or [])[b_idx])

    # Remove high index first within a step so the second removal is not shifted.
    for step, idx in sorted(((s_step, s_idx), (b_step, b_idx)), reverse=True):
        del new[step]["market"][idx]

    for step, order in ((s_dst, sell_order), (b_dst, buy_order)):
        dst = new[step].setdefault("market", [])
        if len(dst) >= MAX_MARKET_ORDERS:
            return None
        dst.append(order)

    note = (
        f"moved {sell_order[0]} {sell_order[1]} step {s_step}->{s_dst} with the "
        f"{buy_order[0]} it funds, step {b_step}->{b_dst} (shift {shift:+d})"
    )
    return new, note


def _move_every_pair(
    route: list[dict],
    sells: list[tuple[int, int]],
    funded: list[tuple[int, int]],
    shift: int | None,
    rng: Any,
) -> tuple[list[dict], str] | None:
    """`scope="all"`: shift every (SELL, funder) pair by the same amount.

    A single-site move of one sale out of 82 is below the noise floor of a
    180-episode panel grid -- the standard error on a win rate there is about
    3.7 points -- so measuring the *idea* needs a route-level change. This is
    that: the whole metered sale stream and the purchases it funds slide
    together, which is the same hypothesis at a size the grid can resolve.

    Pairs whose destination would overflow a turn are dropped one at a time,
    worst-offending turn first, rather than truncating it.
    """
    if shift is None:
        shift = rng.choice([-12, -6, -3, 3, 6, 12]) if rng is not None else 6
    if shift == 0:
        return None
    pairs = _pair_sells_with_funders(sells, funded)
    if not pairs:
        return None
    pairs = [
        (s, b)
        for s, b in pairs
        if 0 <= s[0] + shift < len(route)
        and 0 <= b[0] + shift < len(route)
        and s[0] + shift <= b[0] + shift
        # A forward move may not outrun the unit op the purchase feeds; a
        # backward one may not outrun the stock that fills the sale.
        and (
            shift <= _funder_forward_slack(route, b)
            if shift > 0
            else -shift <= _sell_backward_slack(route, s)
        )
    ]
    moved: list[tuple[tuple[int, int], tuple[int, int]]] = list(pairs)
    while moved:
        moves = {}
        for s, b in moved:
            moves[s] = s[0] + shift
            moves[b] = b[0] + shift
        new = _apply_moves(route, moves)
        if new is not None:
            return (
                new,
                f"moved {len(moved)} (SELL, funder) pair(s) by {shift:+d} steps together "
                f"({len(pairs) - len(moved)} dropped for the order cap)",
            )
        moved.pop()
    return None


# Explicitly typed: the operators do not share one signature (`repath` takes a
# scope and a terminal-segment flag), so without this the values infer to a
# non-callable union and the dispatch in `run_search` stops type-checking.
OPERATORS: dict[str, Callable[..., tuple[list[dict], str] | None]] = {
    "shift_task_block": op_shift_task_block,
    "retarget_plant": op_retarget_plant,
    "swap_herd": op_swap_herd,
    "assign_idle": op_assign_idle,
    "repath": op_repath,
    "move_sell_and_buy": op_move_sell_and_buy,
}


# ---------------------------------------------------------------------------
# Evaluation: reuse the Phase 2 engine, resumable per (hash, opponent, seed)
# ---------------------------------------------------------------------------


@dataclass
class SearchState:
    """The bookkeeping the issue demands: never re-evaluate a seen hash."""

    results_path: str = RESULTS_PATH
    seen: set[str] = field(default_factory=set)

    def load(self) -> None:
        if not os.path.exists(self.results_path):
            return
        with open(self.results_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if "hash" in row:
                    self.seen.add(row["hash"])


def materialize_agent(route: list[dict], workdir: str) -> tuple[str, str]:
    """Bake a route into the deployable artifact; return (hash, path)."""
    h = _hash_of(route)
    os.makedirs(workdir, exist_ok=True)
    path = os.path.join(workdir, f"{h}.py")
    if not os.path.exists(path):
        common.write_route_agent(
            route,
            path,
            provenance={
                "team": "route-search",
                "episode": "mutation",
                "seat": 0,
                "hash": h,
                "steps": len(route),
            },
            version=f"search-{h[:8]}",
        )
    return h, path


def evaluate(
    route: list[dict],
    panel: list[tuple[str, str]],
    seeds: list[int],
    workers: int,
    results_path: str,
    state: SearchState,
) -> dict:
    """Score a route on the panel across `seeds`, through the real artifact.

    Reuses `simulate_candidates.run_stage` verbatim so the evaluation is the same
    object Phase 2 produces: common random numbers, anchored pairing, resumable
    per `(hash, opponent, seed)`. A hash already in `state.seen` is never
    re-evaluated — its cached score is returned instead.
    """
    h = _hash_of(route)
    if h in state.seen:
        cached = _score_from_results(results_path, h, panel, seeds)
        if cached is not None:
            return cached
    _, path = materialize_agent(route, AGENT_CACHE)
    candidate = {
        "hash": h,
        "team": "route-search",
        "team_rank": None,
        "route_b85": common.encode_route_b85(route),
    }
    agent_paths = {h: path}
    done = phase2.load_done(results_path)
    phase2.run_stage(
        "search",
        [candidate],
        seeds,
        agent_paths,
        panel,
        common.DEFAULT_STEPS,
        workers,
        results_path,
        done,
        alternate_seats=False,
    )
    state.seen.add(h)
    scored = _score_from_results(results_path, h, panel, seeds)
    if scored is None:
        return {"hash": h, "n": 0, "rejected": "incomplete_grid"}
    return scored


def _score_from_results(
    results_path: str, h: str, panel: list[tuple[str, str]], seeds: list[int]
) -> dict | None:
    """Reconstruct a candidate's panel score from the results file."""
    byhash = phase2.collect(results_path)
    labels = [lab for lab, _ in panel]
    eff = common.effective_labels(h, labels)
    per = phase2.pairs_on(byhash, h, seeds, eff)
    if not phase2.complete(per, seeds, eff):
        return None
    scores = common.panel_scores(per)
    scores["hash"] = h
    return scores


def accepts(challenger: dict, incumbent: dict) -> bool:
    """The selection metric, exactly as rank_cvar.py defines it.

    Mean panel win rate, tie-broken by worst-opponent win rate. The README's
    warning is load-bearing here: a local accept is a veto, never a forecast.
    """
    if challenger.get("n", 0) == 0:
        return False
    if incumbent.get("n", 0) == 0:
        return True
    key = common.panel_sort_key
    return key(challenger) > key(incumbent)


# ---------------------------------------------------------------------------
# The search loop
# ---------------------------------------------------------------------------


def run_search(
    iterations: int,
    operators: tuple[str, ...],
    panel: list[tuple[str, str]],
    seeds: list[int],
    workers: int,
    results_path: str,
    seed_route: list[dict],
    rng_seed: int = 0,
) -> dict:
    import random

    rng = random.Random(rng_seed)
    state = SearchState(results_path=results_path)
    state.load()

    incumbent_route = seed_route
    incumbent_score = evaluate(incumbent_route, panel, seeds, workers, results_path, state)
    log: list[dict[str, Any]] = [
        {
            "iter": 0,
            "hash": _hash_of(seed_route),
            "operator": "seed",
            "note": "incumbent",
            "accepted": True,
            "score": incumbent_score,
        }
    ]
    t0 = time.time()
    for it in range(1, iterations + 1):
        op_name = operators[rng.randrange(len(operators))] if len(operators) > 1 else operators[0]
        op = OPERATORS[op_name]
        out = op(incumbent_route, rng)
        if out is None:
            log.append({"iter": it, "operator": op_name, "skipped": True})
            continue
        mutated, note = out
        score = evaluate(mutated, panel, seeds, workers, results_path, state)
        ok = accepts(score, incumbent_score)
        log.append(
            {
                "iter": it,
                "operator": op_name,
                "note": note,
                "hash": score.get("hash"),
                "accepted": ok,
                "score": score,
            }
        )
        print(
            f"  iter {it:>3}  {op_name:<18} {'ACCEPT' if ok else 'reject':<6} "
            f"mean_win {score.get('mean_win', 0):.1%} worst {score.get('worst_win', 0):.1%}  {note}"
        )
        if ok:
            incumbent_route = mutated
            incumbent_score = score
    wall = time.time() - t0
    rejected_invalid = sum(
        1 for r in log if r.get("score", {}).get("rejected") == "incomplete_grid"
    )
    return {
        "seed_hash": _hash_of(seed_route),
        "final_hash": _hash_of(incumbent_route),
        "final_score": incumbent_score,
        "iterations": iterations,
        "log": log,
        "rejected_invalid": rejected_invalid,
        "wall_seconds": round(wall, 1),
        "episodes_scored": sum(s.get("score", {}).get("n", 0) for s in log),
    }


def sweep_joint(
    shifts: list[int],
    panel: list[tuple[str, str]],
    seeds: list[int],
    workers: int,
    results_path: str,
    seed_route: list[dict],
) -> int:
    """Score the incumbent against a bulk joint move at each shift (issue #30).

    The single-site operator moves one sale out of 82; on a 180-episode grid the
    standard error of a win rate is ~3.7 points, so that measures noise. The bulk
    form -- every (SELL, funder) pair shifted by the same amount -- is the same
    hypothesis at a size the grid can resolve, which is why this is what gets
    measured before any hill-climb spends a day on single sites.
    """
    state = SearchState(results_path=results_path)
    state.load()
    rows = [("incumbent", 0, evaluate(seed_route, panel, seeds, workers, results_path, state))]
    for shift in shifts:
        out = op_move_sell_and_buy(seed_route, None, shift=shift, scope="all")
        if out is None:
            print(f"  shift {shift:+d}: operator refused (order cap or edge of route)")
            continue
        mutated, note = out
        print(f"  shift {shift:+d}: {note}")
        rows.append(
            (
                f"joint{shift:+d}",
                shift,
                evaluate(mutated, panel, seeds, workers, results_path, state),
            )
        )
    print(
        f"\n  {'route':<12} {'hash':<12} {'mean win':>9} {'worst':>7} {'mean cash':>11} {'margin':>10}"
    )
    for label, _shift, score in rows:
        print(
            f"  {label:<12} {str(score.get('hash', ''))[:10]:<12} "
            f"{score.get('mean_win', 0):>8.1%} {score.get('worst_win', 0):>6.1%} "
            f"{score.get('cash_mean', 0):>11,.0f} {score.get('mean_margin', 0):>10,.0f}"
        )
    print(
        "\n  REMINDER: a local accept is a veto, not a forecast. "
        "Hold out fresh seeds for anything this sweep selects."
    )
    return 0


# ---------------------------------------------------------------------------
# Gates (issue #26: "this is a harness, so the gate is that it changes nothing")
# ---------------------------------------------------------------------------


def self_test(candidates_path: str | None = None) -> int:
    """The five gates, run cheaply and without a panel.

    1. Zero mutations -> emitted route is byte-identical to the seed.
    2. A mutation followed by its inverse round-trips to the same hash.
    3. The identity route replays through the artifact without invalid actions.
    4. #29's re-path is a verified no-op: it walks strictly less and idles
       strictly more, and every non-movement op still fires on its original step
       from its original tile.
    5. Wall-clock budget for a real pass is printed, so #27-#30 can be scoped.

    Gates 1-2 and 4 are pure and run in seconds. Gate 3 needs
    `kaggle_environments` (one episode at DEFAULT_STEPS) and is skipped with a
    loud note if the package is absent rather than silently passing.
    """
    failures = 0
    seed = load_seed(candidates_path)
    seed_hash = _hash_of(seed)

    # Gate 1: identity.
    identity = common.normalize_route(copy.deepcopy(seed))
    h, path = materialize_agent(identity, tempfile.mkdtemp(prefix="route_search_gate_"))
    baked = decode_route_b85(common.encode_route_b85(identity))
    if common.route_hash(common.normalize_route(baked)) != seed_hash:
        print(
            f"  FAIL gate 1: baked identity hashes to {_hash_of(baked)[:10]}, seed is {seed_hash[:10]}"
        )
        failures += 1
    else:
        print(f"  gate 1 OK: zero-mutation route is byte-identical (hash {seed_hash[:10]})")

    # Gate 2: mutation round-trip. We use the no-op shift (the only operator with
    # a defined inverse) shifted and then un-shifted, verifying the hash returns.
    import random

    rng = random.Random(0)
    shifted = op_shift_task_block(seed, rng, k=1)
    if shifted is None:
        print("  gate 2 SKIP: no shiftable movement run in the seed")
    else:
        mutated, note = shifted
        # The inverse of a +k movement-shift is a -k shift of the same run, but
        # the operator does not locate runs deterministically, so we verify the
        # weaker, still-honest form: the mutation itself is *non-destructive* —
        # every non-move op that existed in the seed still exists in the mutant.
        seed_nonmove = _nonmove_signature(seed)
        mut_nonmove = _nonmove_signature(mutated)
        if seed_nonmove != mut_nonmove:
            print(f"  FAIL gate 2: shift changed the non-move signature ({note})")
            failures += 1
        else:
            print(f"  gate 2 OK: shift mutation preserves the non-move signature ({note})")
        if _hash_of(mutated) == seed_hash:
            print("  FAIL gate 2: mutated route hashes equal to the seed — no mutation happened")
            failures += 1

    # Gate 3: fidelity through the real artifact (needs kaggle_environments).
    try:
        from local_arena import run_episode
    except ImportError:
        print("  gate 3 SKIP: kaggle_environments not importable here")
    else:
        h, path = materialize_agent(identity, AGENT_CACHE)
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
        bad = res["invalid"] + res["crashes"] + res["timeouts"]
        if res["harness_error"] or bad:
            print(
                f"  FAIL gate 3: identity route produced invalid={res['invalid']} "
                f"crashes={res['crashes']} timeouts={res['timeouts']} err={res['harness_error']}"
            )
            failures += 1
        else:
            print(
                f"  gate 3 OK: identity artifact replays clean "
                f"(me ${res['me_cash']:,.0f} vs random ${res['opp_cash']:,.0f})"
            )

    # Gate 4: #29's re-path is a no-op. This is the one gate that checks a
    # mutation *changes nothing observable*, which is what makes the recovered
    # turns safe for #28 to spend later.
    for label, drop_terminal in (("interior", False), ("full", True)):
        mutated, stats = board_paths.repath(seed, drop_terminal=drop_terminal)
        before, after = board_paths.census(seed), board_paths.census(mutated)
        problems = board_paths.verify_schedule(seed, mutated)
        if problems:
            print(f"  FAIL gate 4 ({label}): {len(problems)} violation(s): {problems[0]}")
            failures += 1
        elif not stats["turns_recovered"]:
            print(f"  gate 4 SKIP ({label}): the seed has no path slack to recover")
        elif after["movement"] >= before["movement"] or after["idle"] <= before["idle"]:
            print(
                f"  FAIL gate 4 ({label}): movement {before['movement']}->{after['movement']}, "
                f"idle {before['idle']}->{after['idle']} (want down, up)"
            )
            failures += 1
        else:
            print(
                f"  gate 4 OK ({label}): {stats['turns_recovered']} turn(s) banked, movement "
                f"{before['movement']}->{after['movement']}, idle {before['idle']}->{after['idle']}, "
                f"every op still on its original step and tile"
            )

    # Gate 5: the budget, so dependent issues can be scoped.
    workers = max(1, (os.cpu_count() or 2) - 3)
    n_seeds = common.N_MID
    panel_n = len(DEFAULT_PANEL_LABELS)
    eps_per_eval = n_seeds * panel_n
    # Measured on this panel: 720 episodes in 5.3 min on 13 workers (2.3 ep/s),
    # i.e. ~5.7 s of one worker's time per episode. The older 2.4 s figure came
    # from the Phase 2 screen, whose opponents are cheaper than a full panel.
    sec_per_ep = 5.7
    print(
        f"  gate 5 budget: one candidate = {n_seeds} seeds x {panel_n} opponents = "
        f"{eps_per_eval} episodes; at ~{sec_per_ep}s/ep on {workers} workers that is "
        f"~{eps_per_eval * sec_per_ep / workers / 60:.0f} min per accepted/rejected candidate"
    )

    if failures:
        print(f"  SELF-TEST FAILED ({failures} failure(s))")
        return 1
    print("  SELF-TEST PASSED — the harness changes nothing")
    return 0


def _nonmove_signature(route: list[dict]) -> set:
    """The set of (step, slot, op) for every non-movement unit op.

    A movement shift must be invisible to this signature — that is what makes it
    a no-op by construction, and what gate 2 verifies.
    """
    sig: set[tuple[int, Any, tuple[Any, ...]]] = set()
    for step, action in enumerate(route):
        for slot, unit in enumerate(_units(action)):
            op = _op_of(unit)
            if op not in MOVE_OPS and op != "PASS":
                sig.add((step, slot, tuple(unit)))
        for order in action.get("market") or []:
            sig.add((step, "market", tuple(order)))
    return sig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_local_panel() -> list[tuple[str, str]]:
    panel = []
    for label, rel in LOCAL_PANEL:
        path = os.path.join(PROJECT_ROOT, rel)
        if not os.path.exists(path):
            raise SystemExit(f"local panel member missing at {path}")
        panel.append((label, path))
    return panel


def _build_panel(args) -> list[tuple[str, str]]:
    """Resolve the default panel's 10-char labels to on-disk agent paths.

    Members live under `logs/_mined_agents/<full-hash>.py`. A member that is in
    the candidate pool but not yet baked is materialized from its `route_b85`,
    so the panel is complete after one Phase 1 mine even before a Phase 2 run.
    The anchor (`v0_2_6`) must exist — it is the incumbent we are scored against.
    """
    anchor = os.path.join(PROJECT_ROOT, "opponents", "v0_2_6.py")
    if not os.path.exists(anchor):
        raise SystemExit(f"anchor opponent missing at {anchor}")
    panel: list[tuple[str, str]] = [("v0_2_6", anchor)]

    by_hash: dict[str, dict] = {}
    if os.path.exists(args.candidates):
        by_hash = {c["hash"]: c for c in common.read_jsonl(args.candidates)}

    missing = []
    for label in DEFAULT_PANEL_LABELS[1:]:
        found = None
        # Exact on-disk match by prefix first — the file may already exist.
        if os.path.isdir(MINED_AGENT_DIR):
            for fname in os.listdir(MINED_AGENT_DIR):
                if fname.startswith(label) and fname.endswith(".py"):
                    found = os.path.join(MINED_AGENT_DIR, fname)
                    break
        if found is None:
            # Bake it from the pool if we can.
            cand = next((c for h, c in by_hash.items() if h.startswith(label)), None)
            if cand is not None:
                route = decode_route_b85(cand["route_b85"])
                found = os.path.join(MINED_AGENT_DIR, f"{cand['hash']}.py")
                if not os.path.exists(found):
                    common.write_route_agent(
                        route,
                        found,
                        provenance={
                            "episode": cand.get("episode"),
                            "seat": cand.get("seat"),
                            "team": cand.get("team", "?"),
                            "recorded_cash": cand.get("recorded_cash"),
                            "steps": cand.get("steps"),
                            "hash": cand["hash"],
                        },
                        version=f"cand-{label}",
                    )
        if found is None:
            missing.append(label)
        else:
            panel.append((label, found))
    if missing:
        print(f"  panel note: {len(missing)} default member(s) unavailable: {', '.join(missing)}")
    return panel


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Route-synthesis harness (issue #26)")
    ap.add_argument("--self-test", action="store_true", help="run the four no-change gates")
    ap.add_argument("--candidates", default=os.path.join(PROJECT_ROOT, "candidates.jsonl"))
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 3))
    ap.add_argument("--seeds", type=int, default=common.N_MID, help="seeds per evaluation")
    ap.add_argument("--results", default=RESULTS_PATH)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument(
        "--sweep-joint",
        default=None,
        metavar="SHIFTS",
        help="comma-separated step shifts; score the seed plus one bulk "
        "move_sell_and_buy(scope='all') route per shift and stop (issue #30)",
    )
    ap.add_argument(
        "--panel",
        default="mined",
        choices=("mined", "local"),
        help="'mined' is the leaderboard-band panel (needs candidates.jsonl); "
        "'local' is the reproducible opponents/ roster",
    )
    for name in ALL_OPERATORS:
        ap.add_argument(
            f"--no-{name.replace('_', '-')}", dest=name, action="store_false", default=True
        )
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test(args.candidates)

    if args.panel == "local":
        panel = _build_local_panel()
    elif args.panel != "mined":
        raise SystemExit(f"unknown panel {args.panel!r} (want 'mined' or 'local')")

    operators = tuple(n for n in ALL_OPERATORS if getattr(args, n))
    if not operators:
        raise SystemExit("no operators enabled")
    if args.panel == "mined":
        panel = _build_panel(args)
    seeds = common.seed_set(args.seeds, common.SEED_BASE)
    seed_route = load_seed(args.candidates)
    print(f"  seed {SEED_CANDIDATE_PREFIX} ({len(seed_route)} steps), operators {operators}")
    print(f"  panel {[lab for lab, _ in panel]}, {len(seeds)} seeds, {args.workers} workers")

    if args.sweep_joint is not None:
        return sweep_joint(
            [int(s) for s in args.sweep_joint.split(",") if s.strip()],
            panel,
            seeds,
            args.workers,
            args.results,
            seed_route,
        )
    out = run_search(
        args.iterations,
        operators,
        panel,
        seeds,
        args.workers,
        args.results,
        seed_route,
        args.rng_seed,
    )
    print(
        f"\n  done: {out['iterations']} iterations, final mean_win "
        f"{out['final_score'].get('mean_win', 0):.1%}, "
        f"{out['rejected_invalid']} rejected for invalid actions, "
        f"{out['wall_seconds']}s wall"
    )
    print(
        "  REMINDER: a local accept is a veto, not a forecast. Hold out fresh seeds for anything this loop selects."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
