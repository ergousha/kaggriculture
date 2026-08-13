"""Verbatim trace replayer, used only by Phase 1's fidelity check.

This deliberately has *no* runtime layers -- no WEED repair, no SELL-slot
reordering, no hands alignment. The fidelity check reconstructs an original
episode closed-loop (both seats replaying their own recorded traces on the
episode's recovered `info.seed`), and the environment is deterministic, so the
reconstruction must reproduce the recorded rewards *exactly*. Any repair layer
would perturb an action and destroy that exactness, turning a precise format check
into a fuzzy one.

Both seats load from `KAGGRI_TRACE_0` / `KAGGRI_TRACE_1`.
"""

import json
import os

NO_OP = {"farmer": ["PASS"], "hands": [], "market": []}

_CACHE: dict = {}


def _load(seat):
    """Cache per (seat, path): a worker process runs many episodes."""
    path = os.environ.get(f"KAGGRI_TRACE_{seat}", "")
    key = (seat, path)
    if key in _CACHE:
        return _CACHE[key]
    try:
        with open(path) as f:
            route = json.load(f)
    except (OSError, ValueError):
        route = []
    _CACHE.clear()  # only ever two live traces; keeps memory flat across episodes
    _CACHE[key] = route
    return route


def _step_index(obs, config=None):
    """`step` is in the shared observation, so only seat 0 receives it; derive it
    from the per-seat clock otherwise (the env indexes turns as
    day * turnsPerDay + hour)."""
    step = obs.get("step")
    if step is not None:
        return int(step)
    turns_per_day = 24
    if isinstance(config, dict):
        turns_per_day = int(config.get("turnsPerDay") or 24)
    return int(obs.get("day", 0) or 0) * turns_per_day + int(obs.get("hour", 0) or 0)


def agent(obs, config=None):
    seat = 1 if int(obs.get("player", 0) or 0) == 1 else 0
    route = _load(seat)
    if not route:
        return NO_OP
    step = _step_index(obs, config)
    if step < 0 or step >= len(route):
        return NO_OP
    action = route[step] or {}
    return {
        "farmer": action.get("farmer") or ["PASS"],
        "hands": action.get("hands") or [],
        "market": action.get("market") or [],
    }
