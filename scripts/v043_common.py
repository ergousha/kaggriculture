#!/usr/bin/env python3
"""v0.4.3 Step 1: per-draw tape panel — shared machinery (docs/proposal-2026-09-16-v043.md).

The proposal's premise: the 64 day-6 shop-pair assignments in the inherited
`_EXP_SHOP_ROUTES`/YARN router tables are priors, not measurements. This module
supplies everything needed to measure them against the field-class peer
(v0.4.2 self-play) on the exact window the router controls:

    steps 0-143    every variant plays route 0 (identical)   -> cancels
    steps 144-647 the router's table choice                  -> MEASURED
    steps 648-718 every variant plays route 2 (identical)    -> cancels

Forced-tape variants are the incumbent file with a one-line router patch: the
router reads the *observed* pair (unchanged — day-6 unlock is in the
observation, so the variant behaves identically on every seed), then returns a
hard-coded tape id instead of consulting the static table. This measures one
tape per pair against the peer, which itself runs the incumbent table.

Key correctness anchor: a variant forced to the incumbent's own table choice
for a pair MUST tie the peer on every paired seed of that pair (identical
agents, identical seed -> identical episode). The panel runner reports any
non-tie as an error.

The two public-kernel third opponents (`metacounter`, `lynnsakurai`) are kept
as named opponents for the runner but are NOT part of the primary table build:
v0.4.2 self-play is the honest proxy for the chassis peer class the live
matchmaker actually pairs us against (the class plays the same tapes), and the
proposal's gate is mean cash vs the incumbent table on paired seeds.

Tape dedup: routes 101≡119, 109≡127, 105≡125 are byte-identical over the
measured window, so 41 chassis route ids collapse to 38 unique tapes; the
panel runs unique tapes only and expands back to ids when reporting.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

INCUMBENT = os.path.join(PROJECT_ROOT, "opponents", "v0_4_2.py")
CHASSIS_DRAWS_CSV = os.path.join(PROJECT_ROOT, "logs", "seed_draws_chassis.csv")
VARIANT_DIR = os.path.join(PROJECT_ROOT, "logs", "_v043_variants")
PANEL_DIR = os.path.join(PROJECT_ROOT, "logs", "v043_panel")
DRAW_TABLE_CSV = os.path.join(PROJECT_ROOT, "logs", "v043_draw_table.csv")

SHOPS = (
    "BAKERY",
    "BRUNCH_SPOT",
    "FARMERS_MARKET",
    "ICE_CREAM_SHOP",
    "PET_CAFE",
    "PIZZA_SHOP",
    "SMOOTHIE_SHOP",
    "YARN_STORE",
)
SWITCH_STEP = 144  # day-6 unlock is visible in the step-144 observation
ENDGAME_STEP = 648  # `step>=648` switches every draw to route 2
WINDOW = (SWITCH_STEP, ENDGAME_STEP)  # the window the router's table choice controls


def extract_tables_from_incumbent() -> tuple[dict, dict]:
    """Parse `_EXP_SHOP_ROUTES` and the YARN half of `_router` out of the
    incumbent source. Returns (exp_table, yarn_table), each {pair: route_id}."""
    with open(INCUMBENT) as f:
        src = f.read()
    i = src.index("_EXP_SHOP_ROUTES=")
    j = src.index("\n", i)
    exp_table = eval(  # noqa: S307 - literal dict of the repo's own file
        src[i + len("_EXP_SHOP_ROUTES=") : j], {"__builtins__": {}}
    )
    # The YARN table is an inline dict literal on the router's else-branch line:
    #     state['route']={('BAKERY', 'YARN_STORE'): 3, ...}.get(pair,0)
    # Anchor on BOTH delimiters of that exact line (a bare walk-back for '{'
    # first hits the '{}' inside the shops=_get(...) line two lines above).
    import re

    m = re.search(r"state\['route'\]=(\{.*\})\.get\(pair,0\)", src)
    if not m:
        raise SystemExit("YARN router table line not found in incumbent")
    yarn_table = eval(m.group(1), {"__builtins__": {}})  # noqa: S307
    return exp_table, yarn_table


def incumbent_route_for(pair: tuple[str, str]) -> int:
    exp_table, yarn_table = extract_tables_from_incumbent()
    if pair.count("YARN_STORE") <= 0:
        return exp_table.get(pair, 100)
    return yarn_table.get(pair, 0)


def load_tapes() -> dict[int, list]:
    """Exec the incumbent file and return its `_ROUTES` dict (the full 42-tape
    portfolio, ids 0-12 + 100-128)."""
    ns = {"__name__": "v043_tapes"}
    with open(INCUMBENT) as f:
        exec(compile(f.read(), INCUMBENT, "exec"), ns)  # noqa: S102
    return {int(rid): tape for rid, tape in ns["_ROUTES"].items()}


def window_fingerprint(tape: list, window: tuple[int, int] = WINDOW) -> str:
    """Fingerprint a tape over the window the router choice controls."""
    lo, hi = window
    blob = json.dumps(tape[lo:hi], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def unique_tapes(tapes: dict[int, list]) -> dict[int, list[int]]:
    """Map canonical representative id -> all ids sharing the identical window.

    The representative is the smallest id; the map preserves all members so a
    measured window can be attributed back to every table id that plays it.
    """
    groups: dict[str, list[int]] = {}
    for rid in sorted(tapes):
        groups.setdefault(window_fingerprint(tapes[rid]), []).append(rid)
    return {members[0]: members for members in groups.values()}


def make_forced_variant(
    tape_id: int,
    out_dir: str = VARIANT_DIR,
    incumbent_path: str = INCUMBENT,
    version_tag: str = "0.4.3-panel",
) -> str:
    """Write the incumbent file with the day-6 router branch replaced by a
    hard-coded tape id and return the path.

    The patch is textual and minimal: in `_router`, the line
        state['route']=_EXP_SHOP_ROUTES.get(pair,100)
    becomes
        state['route']=<TAPE_ID>
    so EXP240-branch variants measure that tape on every non-YARN draw, and the
    YARN branch (and the day-27 switch) are left byte-identical. Variants for
    YARN-pair tapes additionally replace the YARN dict literal with the same
    hard-coded id so the measurement covers YARN draws too.

    tape_id 100 forces the EXP240 fallback; the day-27 `state['route']=2` line is
    untouched, so the endgame window cancels as designed.
    """
    with open(incumbent_path) as f:
        src = f.read()
    exp_table, yarn_table = extract_tables_from_incumbent()

    token_exp = "state['route']=_EXP_SHOP_ROUTES.get(pair,100)"
    token_yarn_open = "state['route']={('BAKERY', 'YARN_STORE')"
    token_yarn_close = "}.get(pair,0)"
    assert src.count(token_exp) == 1, "EXP240 router token not found exactly once"
    assert src.count(token_yarn_open) == 1, "YARN router token not found exactly once"

    # Both branches forced: the measured quantity is "which tape plays steps
    # 144-647 for the observed pair", independent of which branch chose it.
    src = src.replace(token_exp, f"state['route']={int(tape_id)}")
    i = src.index(token_yarn_open)
    j = src.index(token_yarn_close, i) + len(token_yarn_close)
    src = src[:i] + f"state['route']={int(tape_id)}" + src[j:]

    src = src.replace('AGENT_VERSION = "0.4.2"', f'AGENT_VERSION = "{version_tag}"')

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"force_{int(tape_id)}.py")
    with open(out, "w") as f:
        f.write(src)
    return out


def make_endgame_variant(
    endgame_route: int,
    out_dir: str = VARIANT_DIR,
    incumbent_path: str = INCUMBENT,
    version_tag: str = "0.4.3-panel",
) -> str:
    """Force the day-27 switch (`step>=648`) to a given route instead of 2.

    Proposal Step 3 (parallel audit): the endgame tape is the one authored code
    path never re-validated against engine 1.32.7's price curves, and it is
    shared with the whole chassis class. This measures route 2 vs candidates
    (1, 5) on the 648-718 window, everything else byte-identical.
    """
    with open(incumbent_path) as f:
        src = f.read()
    token = "state['route']=2"
    assert src.count(token) == 1, "day-27 switch token not found exactly once"
    src = src.replace(token, f"state['route']={int(endgame_route)}")
    src = src.replace('AGENT_VERSION = "0.4.2"', f'AGENT_VERSION = "{version_tag}"')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"endgame_{int(endgame_route)}.py")
    with open(out, "w") as f:
        f.write(src)
    return out


def make_table_variant(
    table: dict[tuple[str, str], int],
    out_path: str,
    incumbent_path: str = INCUMBENT,
    version_tag: str = "0.4.3",
) -> str:
    """Write a candidate-file variant whose `_router` plays the given measured
    {pair: route_id} table (Step 2's build input, but also Step 1's deep-eval:
    the greedy table evaluated as it would actually ship)."""
    with open(incumbent_path) as f:
        src = f.read()
    token_exp = "state['route']=_EXP_SHOP_ROUTES.get(pair,100)"
    token_yarn_open = "state['route']={('BAKERY', 'YARN_STORE')"
    token_yarn_close = "}.get(pair,0)"
    assert src.count(token_exp) == 1
    assert src.count(token_yarn_open) == 1
    src = src.replace(token_exp, "state['route']=_V043_TABLE.get(pair,100)")
    i = src.index(token_yarn_open)
    j = src.index(token_yarn_close, i) + len(token_yarn_close)
    src = src[:i] + "state['route']=_V043_TABLE.get(pair,100)" + src[j:]
    table_src = repr({tuple(k): int(v) for k, v in table.items()})
    anchor = "_EXP_SHOP_ROUTES="
    assert src.count(anchor) == 1
    src = src.replace(anchor, "_V043_TABLE=" + table_src + "\n_EXP_SHOP_ROUTES=", 1)
    src = src.replace('AGENT_VERSION = "0.4.2"', f'AGENT_VERSION = "{version_tag}"')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(src)
    return out_path


def load_chassis_draws(csv_path: str = CHASSIS_DRAWS_CSV) -> dict[int, tuple[str, str]]:
    draws: dict[int, tuple[str, str]] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                draws[int(row["seed"])] = (row["shop1"], row["shop2"])
            except (KeyError, ValueError):
                continue
    return draws


def seeds_by_pair(
    draws: dict[int, tuple[str, str]],
    min_per_pair: int = 8,
    seed_range: tuple[int, int] | None = None,
) -> dict[tuple[str, str], list[int]]:
    """Group seeds by draw pair, keeping the first `min_per_pair` seeds per pair
    in ascending order (the panel is stratified, not exhaustive).

    seed_range defaults to the whole table: the enumeration grows over time
    (append-only), and later seeds must not be silently excluded -- refine and
    deep depend on per-pair depth BEYOND the first 600 seeds for the rare pairs.
    Positions 0..3 of each pair's list are stable once enumerated (seeds are
    appended in ascending order), so screen results stay valid as the table
    grows."""
    by_pair: dict[tuple[str, str], list[int]] = {}
    for seed in sorted(draws):
        if seed_range is not None and not (seed_range[0] <= seed < seed_range[1]):
            continue
        by_pair.setdefault(draws[seed], []).append(seed)
    return {pair: seeds[:min_per_pair] for pair, seeds in by_pair.items()}
