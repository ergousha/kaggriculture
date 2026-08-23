#!/usr/bin/env python3
"""Render and score sell-metering variants of the shipped agent (issue #30).

The metering layer lives in `AGENT_TEMPLATE` behind six module constants, so a
variant is *the shipped file with those constants rewritten* -- the same
one-file A/B #25 used, where the OFF arm was the real agent with one collection
emptied. Nothing else differs: same route, same WEED repair, same SELL-slot
ordering, same hands alignment.

Every variant is scored on one identical grid through `simulate_candidates`'
Phase 2 engine (common random numbers on both the seed and opponent axes), and
the per-episode rows now carry `shed_overflow_lost` and the failed-order
counters, so #30's gate --

    "Zero increase in shed overflow, and zero failed HIRE/BUY orders relative to
     the incumbent -- instrument this explicitly, it is the failure mode."

-- is read off the same run that produces the win rates, not asserted.

    uv run python scripts/sweep_meter.py --seeds 30
    uv run python scripts/sweep_meter.py --only off,f55_h24 --seeds 12
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
for _p in (PROJECT_ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import simulate_candidates as phase2  # noqa: E402
from mining import common  # noqa: E402

VARIANT_DIR = os.path.join(PROJECT_ROOT, "logs", "_meter_variants")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "logs", "meter_sweep_results.jsonl")

# Reproducible from this checkout: `candidates.jsonl` and `logs/_mined_agents/`
# are gitignored, so the leaderboard-band panel cannot be rebuilt here. Four of
# these six are our own successive versions, which makes the panel harder than
# the mined one and the absolute win rates lower; the comparison between
# variants is what this run is for, and that is panel-independent.
PANEL = (
    ("v0_2_6", "opponents/v0_2_6.py"),
    ("v0_2_7", "opponents/v0_2_7.py"),
    ("v0_3_0", "opponents/v0_3_0.py"),
    ("v0_3_1", "opponents/v0_3_1.py"),
    ("rita", "opponents/ladder/rancher_rita.py"),
    ("mateo", "opponents/ladder/melon_mateo.py"),
)

# Variant names are deliberately prefixed `m_` so that `common.effective_labels`
# never mistakes one for a panel member and drops the mirror: every variant
# faces all six opponents, including v0_3_1, which *is* the incumbent.
VARIANTS: dict[str, dict[str, object]] = {
    # The control. Byte-identical behaviour to v0.3.1 -- the layer returns before
    # it touches the action when `_METER_ITEMS` is empty.
    "m_off": {"_METER_ITEMS": ()},
    # Milk only, the product the issue is written about, at three floors.
    "m_milk_f40": {"_METER_ITEMS": ("MILK",), "_METER_FLOOR": 0.40},
    "m_milk_f55": {"_METER_ITEMS": ("MILK",), "_METER_FLOOR": 0.55},
    "m_milk_f80": {"_METER_ITEMS": ("MILK",), "_METER_FLOOR": 0.80},
    # Both crashing products.
    "m_both_f55": {"_METER_ITEMS": ("MILK", "STRAWBERRY"), "_METER_FLOOR": 0.55},
    # How much shed the hold is allowed to occupy is the other axis: the shed is
    # the binding constraint at 100/100, so this is not a free parameter.
    "m_both_f55_h8": {
        "_METER_ITEMS": ("MILK", "STRAWBERRY"),
        "_METER_FLOOR": 0.55,
        "_METER_MAX_HOLD": 8,
    },
    "m_both_f55_h48": {
        "_METER_ITEMS": ("MILK", "STRAWBERRY"),
        "_METER_FLOOR": 0.55,
        "_METER_MAX_HOLD": 48,
    },
    # Strawberry alone: it crashes later than milk (day 22 vs day 15), i.e. well
    # inside the window where the cash envelope is already open.
    "m_straw_f55": {"_METER_ITEMS": ("STRAWBERRY",), "_METER_FLOOR": 0.55},
    # A floor low enough that only units heading for the $1 floor are held. If
    # metering pays anywhere it should pay here, because the deferred unit is
    # worth almost nothing on the turn the route sells it.
    "m_both_f20": {"_METER_ITEMS": ("MILK", "STRAWBERRY"), "_METER_FLOOR": 0.20},
    # The unconstrained arm: hold as much as it takes, ignore shed pressure, and
    # let the town's drain lift the price back. This is the version of the idea
    # the issue describes; it is in the sweep to be measured, not to be shipped.
    "m_milk_unbounded": {
        "_METER_ITEMS": ("MILK",),
        "_METER_FLOOR": 0.55,
        "_METER_MAX_HOLD": 400,
        "_METER_SHED_RELEASE": 999,
    },
}


def render(name: str, overrides: dict[str, object], base: str) -> str:
    """`main.py` with the metering constants rewritten. Nothing else moves."""
    src = base
    for key, value in overrides.items():
        pattern = re.compile(rf"^{re.escape(key)} = .*$", re.M)
        if not pattern.search(src):
            raise SystemExit(f"{name}: no such constant {key} in main.py")
        src = pattern.sub(f"{key} = {value!r}", src, count=1)
    return src.replace('AGENT_VERSION = "', f'AGENT_VERSION = "{name}-', 1)


def materialize(names: list[str]) -> dict[str, str]:
    os.makedirs(VARIANT_DIR, exist_ok=True)
    with open(os.path.join(PROJECT_ROOT, "main.py")) as f:
        base = f.read()
    paths = {}
    for name in names:
        path = os.path.join(VARIANT_DIR, f"{name}.py")
        with open(path, "w") as f:
            f.write(render(name, VARIANTS[name], base))
        paths[name] = path
    return paths


def instrumentation(results_path: str, name: str) -> dict[str, float]:
    """Shed overflow and failed orders for one variant, summed over its grid."""
    keys = (
        "shed_overflow_lost",
        "orders_failed_hire",
        "orders_failed_land",
        "orders_failed_buy",
        "orders_failed_sell",
    )
    totals = dict.fromkeys(keys, 0)
    episodes = 0
    if not os.path.exists(results_path):
        return {**totals, "episodes": 0}
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("hash") != name:
                continue
            episodes += 1
            for k in keys:
                totals[k] += int(row.get(k, 0) or 0)
    return {**totals, "episodes": episodes}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sell-metering sweep (issue #30)")
    ap.add_argument("--seeds", type=int, default=common.N_MID)
    ap.add_argument("--seed-base", type=int, default=common.SEED_BASE)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 3))
    ap.add_argument("--results", default=RESULTS_PATH)
    ap.add_argument("--only", default=None, help="comma-separated variant names")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        for name, cfg in VARIANTS.items():
            print(f"  {name:<18} {cfg}")
        return 0

    names = [n.strip() for n in args.only.split(",")] if args.only else list(VARIANTS)
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variant(s): {', '.join(unknown)}")

    paths = materialize(names)
    for path in paths.values():
        check = subprocess.run(
            [sys.executable, "-m", "ruff", "check", path], capture_output=True, text=True
        )
        if check.returncode:
            raise SystemExit(check.stdout + check.stderr)

    panel = [(lab, os.path.join(PROJECT_ROOT, p)) for lab, p in PANEL]
    for _lab, p in panel:
        if not os.path.exists(p):
            raise SystemExit(f"panel member missing: {p}")
    seeds = common.seed_set(args.seeds, args.seed_base)
    candidates = [{"hash": n, "team": "meter-sweep", "team_rank": None} for n in names]

    print(f"  {len(names)} variant(s) x {len(seeds)} seeds x {len(panel)} opponents")
    done = phase2.load_done(args.results)
    phase2.run_stage(
        "meter",
        candidates,
        seeds,
        paths,
        panel,
        common.DEFAULT_STEPS,
        args.workers,
        args.results,
        done,
        alternate_seats=False,
    )

    byhash = phase2.collect(args.results)
    labels = [lab for lab, _ in panel]
    print(
        f"\n  {'variant':<18} {'mean win':>9} {'worst':>7} {'mean cash':>11} "
        f"{'margin':>10} {'cvar5':>10} {'shed lost':>10} {'fail hire':>10} {'fail buy':>9}"
    )
    rows = []
    for name in names:
        per = phase2.pairs_on(byhash, name, seeds, labels)
        scores = common.panel_scores(per)
        inst = instrumentation(args.results, name)
        rows.append((name, scores, inst))
        print(
            f"  {name:<18} {scores.get('mean_win', 0):>8.1%} {scores.get('worst_win', 0):>6.1%} "
            f"{scores.get('cash_mean', 0):>11,.0f} {scores.get('mean_margin', 0):>10,.0f} "
            f"{scores.get('margin_cvar5', 0):>10,.0f} {inst['shed_overflow_lost']:>10,} "
            f"{inst['orders_failed_hire']:>10,} {inst['orders_failed_buy']:>9,}"
        )
    print(
        "\n  REMINDER: a local accept is a veto, not a forecast. "
        "Hold out fresh seeds for anything this sweep selects."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
