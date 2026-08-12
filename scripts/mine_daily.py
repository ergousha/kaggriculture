#!/usr/bin/env python3
"""Mine Kaggle's daily episode dumps into per-seat strategy fingerprints.

Kaggle publishes the previous day's episodes as a dataset (~21 GB, ~700 episodes).
That is far too much to keep, and none of it is useful raw. What *is* useful is a
~1 KB fingerprint per player-seat, which compresses a day to well under a
megabyte and answers the only question worth asking of the dump: **what do the
agents above us do differently?**

Each fingerprint records, for one seat in one episode:

  * final cash, and whether that seat won;
  * realised revenue by product. This is not `quantity x quoted price`: the
    interpreter sells **per unit in a lockstep loop**, re-quoting after each unit,
    and `_commit_unit` aborts the order the moment the seat's shed runs dry. So the
    fill is clamped to the shed and the price is integrated over the curve.
    Counting ordered volume instead overstated our own fertilizer revenue at
    $76k on a $51k episode -- the agent orders far more than it holds;
  * `sell_units_unfillable` / `sell_orders_wasted_pct`, the volume ordered against
    an empty shed. Not free: it burns one of the 10 market-order slots per turn;
  * `BUY_PRODUCT` / `BUY_SEED` / `BUY_ANIMAL` / `HIRE` / `BUY_LAND` volumes;
  * the unit-op histogram (`WATER`, `HARVEST`, `FEED`, `FERTILIZE`, ... `PASS`),
    which is what separates the frontier from us: cash tracks productive ops
    almost linearly, and ops lost to walking and `PICKUP` are the real leak;
  * end-of-episode composition -- owned tiles, hands, crop tiles by type, animals
    by species, fertilized tile count;
  * unsold shed stock at the end, which scores $0 and is pure waste;
  * the episode's price trajectory per product (start / peak / end).

That last one is why this script exists at all. The market does **not** start
empty: inventory begins at 10,000 per product and town demand drains it all
episode, so prices *rise* -- median MILK 160 -> 328, STRAWBERRY 120 -> 294. Any
strategy derived from a "cumulative revenue into a virgin market" model is
derived from the wrong curve. See the README's economics section.

Usage:
    # mine everything already downloaded, write fingerprints, print the report
    uv run python scripts/mine_daily.py

    # sample a daily dump instead of pulling all 21.5 GB (~31 MB per episode)
    uv run python scripts/mine_daily.py --dataset kaggriculture-episodes-2026-08-10 \
        --sample 150 --sample-seed 810

    # mine a specific directory or file set, appending to the running CSV
    uv run python scripts/mine_daily.py logs/daily/2026-08-09 --append

    # fetch a daily dataset first (WARNING: ~21 GB), then mine it
    uv run python scripts/mine_daily.py --dataset kaggriculture-episodes-2026-08-09

    # re-print cohort report from an existing CSV without re-parsing replays
    uv run python scripts/mine_daily.py --report-only

Fingerprints are additive: run it daily with --append and the CSV becomes a
time series of what the frontier is doing, which is the input the macro search in
search/space.py should be seeded from.
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import math
import os
import random
import statistics
import sys
import zipfile
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
DEFAULT_SOURCES = [os.path.join(LOG_DIR, "leaderboard_replays")]
DEFAULT_OUT = os.path.join(LOG_DIR, "daily_fingerprints.csv")

# Kept explicit rather than discovered, so the CSV schema is stable across days
# and a column that stops appearing reads as a zero instead of vanishing.
PRODUCTS = [
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
]
SEEDS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMALS = ["GOOSE", "SHEEP", "COW"]
UNIT_OPS = [
    "PASS",
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "PICKUP",
    "PLACE",
    "DROP",
    "PLANT",
    "WATER",
    "HARVEST",
    "FERTILIZE",
    "COLLECT_FERTILIZER",
    "BUILD_COOP",
    "BUILD_PASTURE",
    "DIG",
    "FEED",
    "CARE",
]
# Ops that change the farm's state versus ops that only reposition a unit. The
# split is the point: the frontier and we spend a similar number of unit-turns,
# but a very different fraction of them on work.
PRODUCTIVE_OPS = [
    "PLANT",
    "WATER",
    "HARVEST",
    "FERTILIZE",
    "COLLECT_FERTILIZER",
    "BUILD_COOP",
    "BUILD_PASTURE",
    "DIG",
    "FEED",
    "CARE",
]
LOGISTICS_OPS = ["NORTH", "SOUTH", "EAST", "WEST", "PICKUP", "PLACE", "DROP"]
PRICE_TRACKED = ["WHEAT", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "CARROT"]

# Verbatim from kaggle_environments/envs/kaggriculture/kaggriculture.py. Needed
# because the interpreter sells **per unit in a lockstep loop**, re-quoting after
# every unit, so an order's revenue is an integral over the curve and not
# quantity x the quoted price.
MARKET_I0 = 10_000
PRICE_FLOOR = 1.0
MARKET_PARAMS: dict[str, dict[str, Any]] = {
    "WHEAT": {"base": 25, "T": 400, "below": ("sqrt", 0.80), "above": ("log", 0.20)},
    "CARROT": {"base": 35, "T": 450, "below": ("log", 0.20), "above": ("sqrt", 0.70)},
    "TOMATO": {"base": 60, "T": 200, "below": ("linear", 0.40), "above": ("sqrt", 0.60)},
    "STRAWBERRY": {"base": 120, "T": 100, "below": ("sqrt", 0.70), "above": ("linear", 1.60)},
    "MELON": {"base": 250, "T": 300, "below": ("log", 0.20), "above": ("sq", 3.60)},
    "EGG": {"base": 50, "T": 332, "below": ("linear", 0.40), "above": ("log", 0.20)},
    "MILK": {"base": 160, "T": 122, "below": ("sqrt", 0.60), "above": ("linear", 1.60)},
    "WOOL": {"base": 200, "T": 105, "below": ("log", 0.20), "above": ("sq", 3.20)},
    "FERTILIZER": {"base": 100, "T": 200, "below": ("linear", 0.40), "above": ("linear", 0.40)},
}


def _shape(func: str, x: float) -> float:
    x = max(0.0, x)
    if func == "sq":
        return x * x
    if func == "sqrt":
        return math.sqrt(x)
    if func == "log":
        return math.log(1.0 + x)
    if func == "log10":
        return math.log10(1.0 + x)
    return x


def price_at(item: str, inventory: float) -> float:
    """The env's own price curve, for integrating a multi-unit SELL."""
    p = MARKET_PARAMS.get(item)
    if p is None:
        return PRICE_FLOOR
    base, t = float(p["base"]), float(p["T"])
    func, target = p["below"] if inventory < MARKET_I0 else p["above"]
    amp = float(target) * base / _shape(func, t)
    if inventory < MARKET_I0:
        price = base + amp * _shape(func, MARKET_I0 - inventory)
    else:
        price = base - amp * _shape(func, inventory - MARKET_I0)
    return max(PRICE_FLOOR, round(price))


def sell_revenue(item: str, inventory: float, units: int) -> tuple[float, float]:
    """Revenue for selling `units` one at a time, and the resulting inventory.

    `_commit_unit` only raises market inventory when the unit cleared above $1,
    so a product parked on the floor stops depressing itself.
    """
    total = 0.0
    inv = float(inventory)
    for _ in range(int(units)):
        price = price_at(item, inv)
        total += price
        if price > PRICE_FLOOR:
            inv += 1
    return total, inv


def _fieldnames() -> list[str]:
    cols = ["episode_id", "source", "seat", "team", "cash", "won", "opp_cash", "steps"]
    cols += [f"rev_{p}" for p in PRODUCTS]
    cols += [f"sold_{p}" for p in PRODUCTS]
    cols += [f"bought_{p}" for p in PRODUCTS]
    cols += [f"seed_{s}" for s in SEEDS]
    cols += [f"animal_{a}" for a in ANIMALS]
    cols += ["hires", "land_buys"]
    cols += [f"op_{o}" for o in UNIT_OPS]
    cols += ["ops_productive", "ops_logistics", "ops_total"]
    cols += ["owned_tiles", "hands", "fertilized_tiles", "coops", "pastures", "weeds"]
    cols += [f"tile_{p}" for p in SEEDS]
    cols += [f"pen_{a}" for a in ANIMALS]
    cols += ["unsold_shed_units", "sell_units_unfillable", "sell_orders_wasted_pct"]
    cols += [f"price_{p}_{w}" for p in PRICE_TRACKED for w in ("start", "peak", "end")]
    return cols


def iter_replay_files(sources: list[str]) -> list[str]:
    """Every candidate replay path under `sources`, files and directories alike."""
    out = []
    for src in sources:
        if os.path.isfile(src):
            out.append(src)
        elif os.path.isdir(src):
            for root, _dirs, files in os.walk(src):
                if os.path.basename(root).startswith("_"):
                    continue
                for name in sorted(files):
                    if name.startswith("_"):
                        continue
                    if name.endswith((".json", ".json.gz")):
                        out.append(os.path.join(root, name))
        else:
            print(f"  skip (not found): {src}", file=sys.stderr)
    return out


def load_episode(path: str) -> dict | None:
    """Parse one replay, returning None for anything that is not kaggriculture.

    The 64 KB head check matters for throughput, not just correctness: a day's
    dump is thousands of files, and `name` sits in the first kilobyte while
    `steps` runs for megabytes.
    """
    try:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rb") as f:  # type: ignore[operator]
            head = f.read(65536)
            if b'"kaggriculture"' not in head:
                return None
            rest = f.read()
        data = json.loads(head + rest)
    except (OSError, ValueError):
        return None
    if data.get("name") != "kaggriculture":
        return None
    if not data.get("steps"):
        return None
    return data


def _tile_composition(farm: dict) -> dict:
    """Owned tiles, crop tiles, animals and fertilizer coverage from a final farm."""
    comp: dict[str, Any] = {
        "owned_tiles": 0,
        "fertilized_tiles": 0,
        "coops": 0,
        "pastures": 0,
        "weeds": 0,
        "tiles": collections.Counter(),
        "pens": collections.Counter(),
    }
    for row in farm.get("tiles") or []:
        for tile in row:
            if tile == "LOCKED":
                continue
            comp["owned_tiles"] += 1
            if not isinstance(tile, dict):
                continue
            kind = tile.get("kind")
            if kind == "PLANT":
                comp["tiles"][tile.get("crop")] += 1
                if (tile.get("fertilized_until_day") or -1) >= 0:
                    comp["fertilized_tiles"] += 1
            elif kind == "WEED":
                comp["weeds"] += 1
            elif kind in ("COOP", "PASTURE"):
                comp["coops" if kind == "COOP" else "pastures"] += 1
                animal = tile.get("animal")
                if animal:
                    comp["pens"][animal if isinstance(animal, str) else animal.get("species")] += 1
    return comp


def fingerprint(data: dict, source: str) -> list[dict]:
    """One fingerprint row per player-seat in one episode."""
    steps = data["steps"]
    info = data.get("info") or {}
    rewards = [float(r or 0.0) for r in (data.get("rewards") or [])]
    teams = info.get("TeamNames") or []
    n_seats = min(len(steps[0]), max(2, len(rewards)))

    rev: list[dict[str, float]] = [collections.defaultdict(float) for _ in range(n_seats)]
    sold: list[collections.Counter[str]] = [collections.Counter() for _ in range(n_seats)]
    bought: list[collections.Counter[str]] = [collections.Counter() for _ in range(n_seats)]
    seeds: list[collections.Counter[str]] = [collections.Counter() for _ in range(n_seats)]
    animals: list[collections.Counter[str]] = [collections.Counter() for _ in range(n_seats)]
    ops: list[collections.Counter[str]] = [collections.Counter() for _ in range(n_seats)]
    unfillable: list[collections.Counter[str]] = [collections.Counter() for _ in range(n_seats)]
    hires = [0] * n_seats
    land = [0] * n_seats

    price_seen: dict[str, list[float]] = {p: [] for p in PRICE_TRACKED}

    # `steps[t]["action"]` is the action that PRODUCED `steps[t]["observation"]`,
    # not the one taken from it: `steps[0]` carries an empty action, and a step's
    # money already includes that step's own sales. Verified on both a
    # locally-generated replay and a downloaded one. So an action at index t must
    # be priced against the market and shed at index t-1, and iteration starts at 1.
    for t in range(1, len(steps)):
        prev, step = steps[t - 1], steps[t]
        # Only seat 0 carries the full shared observation, so the market comes
        # from there and is authoritative for every seat at that step.
        market = (step[0].get("observation") or {}).get("market") or {}
        prices = market.get("prices") or {}
        for product in PRICE_TRACKED:
            value = prices.get(product)
            if value is not None:
                price_seen[product].append(float(value))

        pre_market = (prev[0].get("observation") or {}).get("market") or {}

        for seat in range(min(n_seats, len(step))):
            action = (step[seat] or {}).get("action") or {}
            if not isinstance(action, dict):
                continue

            # A SELL only draws from this seat's shed (`_commit_unit`), and the
            # order aborts the moment the shed runs dry. Ordering more than you
            # hold is therefore free of revenue but not free of cost: it burns one
            # of the 10 market-order slots per turn.
            seat_shed = dict(
                ((prev[seat] or {}).get("observation") or {}).get("private", {}).get("shed") or {}
            )
            inventory = dict(pre_market.get("inventory") or {})

            for order in action.get("market") or []:
                if not order:
                    continue
                op = order[0]
                item = order[1] if len(order) > 1 else None
                qty = 0
                if len(order) > 2:
                    try:
                        qty = int(order[2])
                    except (TypeError, ValueError):
                        qty = 0
                if op == "SELL" and item:
                    on_hand = int(seat_shed.get(item, 0) or 0)
                    filled = max(0, min(qty, on_hand))
                    unfillable[seat][item] += max(0, qty - filled)
                    if filled:
                        inv = float(inventory.get(item, MARKET_I0))
                        gained, inv_after = sell_revenue(item, inv, filled)
                        rev[seat][item] += gained
                        sold[seat][item] += filled
                        seat_shed[item] = on_hand - filled
                        inventory[item] = inv_after
                elif op == "BUY_PRODUCT" and item:
                    bought[seat][item] += qty
                elif op == "BUY_SEED" and item:
                    seeds[seat][item] += qty
                elif op == "BUY_ANIMAL" and item:
                    animals[seat][item] += qty
                elif op == "HIRE":
                    hires[seat] += 1
                elif op == "BUY_LAND":
                    land[seat] += 1

            units = [action.get("farmer")] + list(action.get("hands") or [])
            for unit in units:
                if unit:
                    ops[seat][unit[0]] += 1

    final_obs = (steps[-1][0].get("observation") or {}).get("farms") or []
    rows = []
    for seat in range(n_seats):
        cash = rewards[seat] if seat < len(rewards) else 0.0
        others = [r for i, r in enumerate(rewards) if i != seat]
        opp_cash = max(others) if others else 0.0
        farm = final_obs[seat] if seat < len(final_obs) else {}
        comp = _tile_composition(farm)

        shed = ((steps[-1][seat].get("observation") or {}).get("private") or {}).get("shed") or {}
        unsold = sum(int(v or 0) for v in shed.values())

        row = {
            "episode_id": info.get("EpisodeId") or data.get("id") or "",
            "source": os.path.basename(source),
            "seat": seat,
            "team": (teams[seat] if seat < len(teams) else "").strip('"'),
            "cash": round(cash, 2),
            "won": 1 if cash > opp_cash else 0,
            "opp_cash": round(opp_cash, 2),
            "steps": len(steps),
            "hires": hires[seat],
            "land_buys": land[seat],
            "owned_tiles": comp["owned_tiles"],
            "hands": len(farm.get("hands") or []),
            "fertilized_tiles": comp["fertilized_tiles"],
            "coops": comp["coops"],
            "pastures": comp["pastures"],
            "weeds": comp["weeds"],
            "unsold_shed_units": unsold,
            "sell_units_unfillable": sum(unfillable[seat].values()),
        }
        ordered = sum(sold[seat].values()) + sum(unfillable[seat].values())
        row["sell_orders_wasted_pct"] = (
            round(100.0 * sum(unfillable[seat].values()) / ordered, 1) if ordered else 0.0
        )
        for product in PRODUCTS:
            row[f"rev_{product}"] = round(rev[seat][product], 2)
            row[f"sold_{product}"] = sold[seat][product]
            row[f"bought_{product}"] = bought[seat][product]
        for seed in SEEDS:
            row[f"seed_{seed}"] = seeds[seat][seed]
            row[f"tile_{seed}"] = comp["tiles"][seed]
        for animal in ANIMALS:
            row[f"animal_{animal}"] = animals[seat][animal]
            row[f"pen_{animal}"] = comp["pens"][animal]
        for op in UNIT_OPS:
            row[f"op_{op}"] = ops[seat][op]
        row["ops_productive"] = sum(ops[seat][o] for o in PRODUCTIVE_OPS)
        row["ops_logistics"] = sum(ops[seat][o] for o in LOGISTICS_OPS)
        row["ops_total"] = sum(ops[seat].values())
        for product in PRICE_TRACKED:
            series = price_seen[product]
            row[f"price_{product}_start"] = series[0] if series else 0
            row[f"price_{product}_peak"] = max(series) if series else 0
            row[f"price_{product}_end"] = series[-1] if series else 0
        rows.append(row)
    return rows


def mine(sources: list[str], out_path: str, append: bool, limit: int | None) -> list[dict]:
    """Stream every replay under `sources` into fingerprint rows and a CSV."""
    paths = iter_replay_files(sources)
    print(f"candidate files: {len(paths)}")

    rows: list[dict] = []
    parsed = skipped = 0
    for path in paths:
        if limit and parsed >= limit:
            break
        data = load_episode(path)
        if data is None:
            skipped += 1
            continue
        try:
            rows.extend(fingerprint(data, path))
        except (KeyError, IndexError, TypeError) as exc:
            print(f"  skip (malformed): {os.path.basename(path)}: {exc}", file=sys.stderr)
            skipped += 1
            continue
        parsed += 1
        # A day's dump is thousands of multi-MB files; hold one at a time.
        del data
        if parsed % 100 == 0:
            print(f"  parsed {parsed} episodes...")

    print(f"parsed {parsed} kaggriculture episodes, skipped {skipped} other/malformed files")
    print(f"fingerprints: {len(rows)} player-seats")

    if rows:
        exists = os.path.exists(out_path)
        mode = "a" if (append and exists) else "w"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_fieldnames(), extrasaction="ignore")
            if mode == "w" or not exists:
                writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {out_path} ({mode})")
    return rows


def load_fingerprints(path: str) -> list[dict]:
    """Read a fingerprint CSV back, coercing numerics."""
    if not os.path.exists(path):
        raise SystemExit(f"no fingerprint CSV at {path}; run without --report-only first")
    rows = []
    with open(path, newline="") as f:
        for raw in csv.DictReader(f):
            row: dict = {}
            for key, value in raw.items():
                if key in ("team", "source", "episode_id"):
                    row[key] = value
                    continue
                try:
                    row[key] = float(value) if value not in ("", None) else 0.0
                except ValueError:
                    row[key] = value
            rows.append(row)
    return rows


def _cohort_report(label: str, grp: list[dict]) -> dict:
    """Print one cohort's cash, revenue mix, ops split, buys and composition."""
    n = len(grp)
    if not n:
        print(f"\n{label}: no rows")
        return {}
    mean_cash = statistics.mean(r["cash"] for r in grp)
    rev_total = sum(sum(r[f"rev_{p}"] for p in PRODUCTS) for r in grp)
    print(
        f"\n{label}  (n={n})  mean cash ${mean_cash:,.0f}  median ${statistics.median(r['cash'] for r in grp):,.0f}"
    )
    if rev_total:
        mix = sorted(
            ((p, sum(r[f"rev_{p}"] for r in grp) / rev_total) for p in PRODUCTS),
            key=lambda kv: -kv[1],
        )
        print("  revenue mix : " + "  ".join(f"{p}={100 * v:.1f}%" for p, v in mix if v >= 0.001))
    print(
        f"  ops/farm    : productive {statistics.mean(r['ops_productive'] for r in grp):,.0f}"
        f"  logistics {statistics.mean(r['ops_logistics'] for r in grp):,.0f}"
        f"  PASS {statistics.mean(r['op_PASS'] for r in grp):,.0f}"
        f"  FERTILIZE {statistics.mean(r['op_FERTILIZE'] for r in grp):,.1f}"
    )
    print(
        f"  buys/farm   : wheat {statistics.mean(r['bought_WHEAT'] for r in grp):,.0f}"
        f"  hires {statistics.mean(r['hires'] for r in grp):,.0f}"
        f"  land {statistics.mean(r['land_buys'] for r in grp):,.1f}"
        + "  "
        + " ".join(
            f"{a.lower()} {statistics.mean(r[f'animal_{a}'] for r in grp):.1f}" for a in ANIMALS
        )
    )
    print(
        f"  end farm    : owned {statistics.mean(r['owned_tiles'] for r in grp):.0f}"
        f"  hands {statistics.mean(r['hands'] for r in grp):.1f}"
        f"  fertilized {statistics.mean(r['fertilized_tiles'] for r in grp):.1f}"
        f"  pastures {statistics.mean(r['pastures'] for r in grp):.1f}"
        f"  coops {statistics.mean(r['coops'] for r in grp):.1f}"
        f"  unsold {statistics.mean(r['unsold_shed_units'] for r in grp):.0f}"
    )
    return {"n": n, "mean_cash": mean_cash}


GAP_COLUMNS = [
    ("ops_productive", "productive unit-ops"),
    ("ops_logistics", "unit-ops spent on logistics"),
    ("op_FERTILIZE", "FERTILIZE ops"),
    ("op_WATER", "WATER ops"),
    ("op_HARVEST", "HARVEST ops"),
    ("bought_WHEAT", "wheat bought"),
    ("animal_COW", "cows bought"),
    ("animal_SHEEP", "sheep bought"),
    ("animal_GOOSE", "geese bought"),
    ("land_buys", "land plots bought"),
    ("hires", "hires issued"),
    ("fertilized_tiles", "fertilized tiles at end"),
    ("coops", "coops at end"),
    ("unsold_shed_units", "unsold shed units (scores $0)"),
]


def report(rows: list[dict], me: str, top_frac: float) -> None:
    """Cohort comparison plus the ours-versus-frontier gap table."""
    rows = [r for r in rows if r.get("cash")]
    if not rows:
        raise SystemExit("no fingerprints to report on")
    rows.sort(key=lambda r: -r["cash"])

    cash = [r["cash"] for r in rows]
    quartiles = statistics.quantiles(cash, n=4) if len(cash) >= 4 else [0, 0, 0]
    print(f"\n{'=' * 78}\nfinal cash over {len(rows)} player-seats")
    print(
        f"  min ${min(cash):,.0f} | p25 ${quartiles[0]:,.0f} | median ${statistics.median(cash):,.0f}"
        f" | p75 ${quartiles[2]:,.0f} | max ${max(cash):,.0f}"
    )

    k_top = max(1, int(len(rows) * top_frac))
    k_quart = max(1, len(rows) // 4)
    top = rows[:k_top]
    ours = [r for r in rows if me and me.lower() in str(r.get("team", "")).lower()]

    _cohort_report(f"TOP {top_frac:.0%}", top)
    _cohort_report("TOP QUARTILE", rows[:k_quart])
    if ours:
        _cohort_report(f"OURS ({me})", ours)
    _cohort_report("BOTTOM QUARTILE", rows[-k_quart:])

    print(f"\n{'=' * 78}\nEXOGENOUS PRICE DRIFT (median start -> peak -> end)")
    print("  the market starts at 10,000 inventory per product and town demand drains it,")
    print("  so prices RISE; only a product the field dumps in bulk ends below its peak.")
    for product in PRICE_TRACKED:
        series = [
            (r[f"price_{product}_start"], r[f"price_{product}_peak"], r[f"price_{product}_end"])
            for r in rows
        ]
        series = [s for s in series if s[0]]
        if not series:
            continue
        start = statistics.median(s[0] for s in series)
        peak = statistics.median(s[1] for s in series)
        end = statistics.median(s[2] for s in series)
        flag = "  <-- crashes, the field dumps it" if end < peak * 0.97 else ""
        print(f"  {product:<11} {start:>6.0f} -> {peak:>6.0f} -> {end:>6.0f}{flag}")

    if not ours:
        print(f"\nno seats matched --me {me!r}; skipping the gap table")
        return

    print(f"\n{'=' * 78}\nGAP: ours vs top {top_frac:.0%}  (per farm; ratio >1 means they do more)")
    print(f"  {'metric':<32} {'ours':>10} {'top':>10} {'ratio':>8}")
    print(
        f"  {'final cash':<32} {statistics.mean(r['cash'] for r in ours):>10,.0f}"
        f" {statistics.mean(r['cash'] for r in top):>10,.0f}"
        f" {statistics.mean(r['cash'] for r in top) / max(1.0, statistics.mean(r['cash'] for r in ours)):>8.2f}x"
    )
    for col, label in GAP_COLUMNS:
        mine_v = statistics.mean(r[col] for r in ours)
        top_v = statistics.mean(r[col] for r in top)
        ratio = top_v / mine_v if mine_v else float("inf")
        ratio_s = "  inf" if ratio == float("inf") else f"{ratio:.2f}x"
        print(f"  {label:<32} {mine_v:>10,.1f} {top_v:>10,.1f} {ratio_s:>8}")


def download_dataset(ref: str, dest: str, sample: int = 0, sample_seed: int = 0) -> str:
    """Fetch a daily episode dataset via the Kaggle API. These are ~21 GB each.

    With `sample`, list the episode files and download a seeded random subset
    instead, which is how these dumps are actually usable on a laptop.
    """
    from submit import load_credentials

    load_credentials()
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    if "/" not in ref:
        ref = f"kaggle/{ref}"
    os.makedirs(dest, exist_ok=True)

    if not sample:
        print(f"downloading dataset {ref} -> {dest}")
        print("  WARNING: the daily dumps are ~21 GB; this will take a while and a lot of disk.")
        api.dataset_download_files(ref, path=dest, unzip=True, quiet=False)
        return dest

    # A daily dump is ~685 episodes at ~31 MB each. Cohort statistics converge long
    # before that, so sampling by episode is the default way to use these: 150
    # episodes is 300 player-seats for ~4.7 GB instead of 21.5 GB.
    names = []
    token = None
    while True:
        page = api.dataset_list_files(ref, page_token=token, page_size=200)
        batch = [f.name for f in page.files if f.name.endswith(".json")]
        names.extend(batch)
        token = getattr(page, "nextPageToken", None) or getattr(page, "next_page_token", None)
        if not token or not batch:
            break

    if not names:
        raise SystemExit(f"{ref} listed no .json episode files")

    # Seeded and sorted so the same --sample/--sample-seed pair always draws the
    # same episodes; a cohort comparison across days has to be reproducible.
    rng = random.Random(sample_seed)
    picked = sorted(rng.sample(names, min(sample, len(names))))
    print(f"{ref}: {len(names)} episodes available, sampling {len(picked)} (seed {sample_seed})")

    for i, name in enumerate(picked, 1):
        target = os.path.join(dest, name)
        if os.path.exists(target) and os.path.getsize(target) > 1000:
            continue
        try:
            api.dataset_download_file(ref, name, path=dest, quiet=True)
        except Exception as exc:  # noqa: BLE001 - one bad episode must not kill the run
            print(f"  [{i}/{len(picked)}] {name}: {exc}", file=sys.stderr)
            continue
        # The API sometimes lands the file zipped next to the requested name.
        zipped = target + ".zip"
        if os.path.exists(zipped):
            with zipfile.ZipFile(zipped) as zf:
                zf.extractall(dest)
            os.remove(zipped)
        if i % 10 == 0 or i == len(picked):
            done = sum(
                os.path.getsize(os.path.join(dest, n))
                for n in os.listdir(dest)
                if n.endswith(".json")
            )
            print(f"  [{i}/{len(picked)}] downloaded, {done / 1e9:.2f} GB on disk")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine Kaggle daily episode dumps into per-seat strategy fingerprints."
    )
    parser.add_argument(
        "sources",
        nargs="*",
        help=f"replay files or directories (default: {DEFAULT_SOURCES[0]})",
    )
    parser.add_argument("--dataset", help="Kaggle dataset ref or slug to download first (~21 GB)")
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="with --dataset, download only N random episodes (~31 MB each) instead of all ~21 GB",
    )
    parser.add_argument(
        "--sample-seed", type=int, default=0, help="seed for --sample, so a draw is reproducible"
    )
    parser.add_argument(
        "--dataset-dir",
        default=os.path.join(LOG_DIR, "daily"),
        help="where --dataset is unpacked",
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help="fingerprint CSV to write")
    parser.add_argument(
        "--append", action="store_true", help="append to the CSV instead of replacing"
    )
    parser.add_argument("--limit", type=int, help="stop after N episodes")
    parser.add_argument(
        "--report-only", action="store_true", help="report from --out, parse nothing"
    )
    parser.add_argument("--no-report", action="store_true", help="mine only, skip the report")
    parser.add_argument("--me", default="Ergin", help="substring identifying our own team name")
    parser.add_argument(
        "--top-frac", type=float, default=0.10, help="frontier cohort fraction (default 0.10)"
    )
    args = parser.parse_args()

    if args.report_only:
        report(load_fingerprints(args.out), args.me, args.top_frac)
        return

    sources = list(args.sources)
    if args.dataset:
        dest = os.path.join(args.dataset_dir, args.dataset.split("/")[-1])
        sources.append(download_dataset(args.dataset, dest, args.sample, args.sample_seed))
    if not sources:
        sources = list(DEFAULT_SOURCES)

    rows = mine(sources, args.out, args.append, args.limit)
    if not args.no_report:
        # Report on the whole CSV, not just this run, so --append accumulates.
        report(load_fingerprints(args.out) if args.append else rows, args.me, args.top_frac)


if __name__ == "__main__":
    main()
