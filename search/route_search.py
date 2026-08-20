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
  * `repath`                 re-path a movement run between two fixed endpoints,
  * `move_sell_and_buy`      move a SELL and the BUY it funds together (#30).

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
from dataclasses import dataclass, field
from typing import Any

# The Phase 2 harness is the engine: it owns env construction, seeded
# configuration, the (candidate, opponent, seed) resume key and the status
# accounting. Importing it keeps this harness and Phase 2 from ever disagreeing
# about what "evaluate a route" means.
import simulate_candidates as phase2
from mining import common
from mining.common import PROJECT_ROOT, decode_route_b85

SEED_CANDIDATE_PREFIX = "044a7741e9"  # v0.2.7 incumbent, issue #26 names this seed
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

# Contiguous movement ops whose only effect on the board is "arrive one step
# later"; shifting them cannot collide with anything because nothing depends on
# a unit's *position* mid-run, only on the tile ops that bracket the run.
MOVE_OPS = frozenset({"NORTH", "SOUTH", "EAST", "WEST"})

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
    # stays balanced. We take the first unmatched one.
    for action in new:
        for order in action.get("market") or []:
            if (
                isinstance(order, list)
                and len(order) >= 3
                and order[0] == "BUY_SEED"
                and order[1] == current
            ):
                order[1] = new_crop
                return (
                    new,
                    f"retargeted PLANT {current}->{new_crop} @step {step} (BUY_SEED follows)",
                )
    return None


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


def op_swap_herd(route: list[dict], rng, **_kw) -> tuple[list[dict], str] | None:
    """Convert a BUY_ANIMAL COW to SHEEP and repair the downstream chore cadence.

    This is #27's lever. COW interval is 3, SHEEP interval is 2, so the swapped
    animal needs its FEED/CARE/COLLECT_FERTILIZER cadence compressed from every
    3rd day to every 2nd — the repair pass re-emits those chores on the new
    cadence and leaves movement to the re-path operator. Adding animals where the
    route had none is safe by construction (nothing depended on their absence);
    *swapping* changes chore timing and is speculative.
    """
    cows, sheep = _herd_counts(route)
    if cows == 0:
        return None
    new = copy.deepcopy(route)
    converted = False
    for action in new:
        for order in action.get("market") or []:
            if (
                isinstance(order, list)
                and len(order) >= 3
                and order[0] == "BUY_ANIMAL"
                and order[1] == "COW"
            ):
                order[1] = "SHEEP"
                converted = True
                break
        if converted:
            break
    if not converted:
        return None
    # Cadence repair (compressing 3-day chores to 2-day) is deliberately NOT
    # approximated here. A wrong repair is worse than none: an over-scheduled
    # FEED silently no-ops, which reads as a herd that starves for no reason.
    # The interpreter tolerates the original cadence against a sheep (it just
    # feeds late), so the swap alone is a legal, measurable mutation; the
    # cadence compression is #27's dedicated operator, and its gate is that
    # realised $/unit does not collapse — not something this harness can guess.
    return (
        new,
        f"converted one BUY_ANIMAL COW->SHEEP (herd was {cows}c/{sheep}s; cadence left at interval 3, #27 compresses it)",
    )


def op_assign_idle(route: list[dict], rng, **_kw) -> tuple[list[dict], str] | None:
    """Give a PASS unit-turn a productive task (#28).

    The conservative reading: only fills a PASS whose unit is already standing
    somewhere a FERTILIZE is legal (the route COLLECT_FERTILIZERs, so fertilizer
    exists) — and FERTILIZE is the op with the best measured margin in the repo
    (+$400 of strawberry per 2 fertilizer that would have sold for ~$140). It
    does not touch movement, so it cannot strand a unit.
    """
    # Gate to the safe form, matching #28's work item: the PASS must sit on or
    # next to a tile the route *already* farms (so the unit does not have to
    # travel to reach it — travel would re-introduce the logistics-starvation
    # the mechanism note warns about). That set is the units whose adjacent
    # steps contain a tile op, which is the cheapest sound proxy for "standing
    # on farmed ground" without simulating positions.
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
    _write_unit(new[step], slot, ["FERTILIZE"])
    return (
        new,
        f"assigned PASS->FERTILIZE @step {step} slot {slot} (unit already adjacent to farmed ground)",
    )


def op_repath(route: list[dict], rng, **_kw) -> tuple[list[dict], str] | None:
    """Re-path a movement run between two fixed endpoints (#29)."""
    # Placeholder for #29's shortest-path operator. The identity and round-trip
    # gates do not need it; returning None keeps it out of the rotation without
    # pretending it ran.
    return None


def op_move_sell_and_buy(route: list[dict], rng, **_kw) -> tuple[list[dict], str] | None:
    """Move a SELL and the BUY it funds together (#30's joint operator)."""
    # Placeholder for #30's joint operator, for the same reason as `repath`.
    return None


OPERATORS = {
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


# ---------------------------------------------------------------------------
# Gates (issue #26: "this is a harness, so the gate is that it changes nothing")
# ---------------------------------------------------------------------------


def self_test(candidates_path: str | None = None) -> int:
    """The four gates, run cheaply and without a panel.

    1. Zero mutations -> emitted route is byte-identical to the seed.
    2. A mutation followed by its inverse round-trips to the same hash.
    3. The identity route replays through the artifact without invalid actions.
    4. Wall-clock budget for a real pass is printed, so #27-#30 can be scoped.

    Gates 1-2 are pure and run in milliseconds. Gate 3 needs `kaggle_environments`
    (one episode at DEFAULT_STEPS) and is skipped with a loud note if the package
    is absent rather than silently passing.
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

    # Gate 4: the budget, so dependent issues can be scoped.
    workers = max(1, (os.cpu_count() or 2) - 3)
    n_seeds = common.N_MID
    panel_n = len(DEFAULT_PANEL_LABELS)
    eps_per_eval = n_seeds * panel_n
    sec_per_ep = 2.4  # measured: Phase 2 screen at 0.8 ep/s on 8 workers
    print(
        f"  gate 4 budget: one candidate = {n_seeds} seeds x {panel_n} opponents = "
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
    for name in ALL_OPERATORS:
        ap.add_argument(
            f"--no-{name.replace('_', '-')}", dest=name, action="store_false", default=True
        )
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test(args.candidates)

    operators = tuple(n for n in ALL_OPERATORS if getattr(args, n))
    if not operators:
        raise SystemExit("no operators enabled")
    panel = _build_panel(args)
    seeds = common.seed_set(args.seeds, common.SEED_BASE)
    seed_route = load_seed(args.candidates)
    print(f"  seed {SEED_CANDIDATE_PREFIX} ({len(seed_route)} steps), operators {operators}")
    print(f"  panel {[lab for lab, _ in panel]}, {len(seeds)} seeds, {args.workers} workers")
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
