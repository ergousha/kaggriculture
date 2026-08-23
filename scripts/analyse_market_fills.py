#!/usr/bin/env python3
"""Per-fill market attribution: what every sold unit actually realised.

Issue #30 asks for "realised MILK and STRAWBERRY $/unit at each accepted step",
and #27 produced the same table by hand in a scratchpad session. This script is
that measurement, committed: it drives one live episode and hooks the three
interpreter entry points that move money --

  * ``_commit_unit``  -- every SELL / BUY_SEED / BUY_PRODUCT / BUY_ANIMAL unit,
    with the price it cleared at and whether it committed at all,
  * ``_do_hire``      -- the Fibonacci hire, and whether it was affordable,
  * ``_do_buy_land``  -- the land ladder, and whether it was affordable,

-- attributing each to a seat by farm-object identity, which the interpreter
hands us directly (``obs0.farms`` is indexed by player id).

Two things it measures that nothing else in the repo can:

  * **realised $/unit**, i.e. revenue / units, per product per seat. This is the
    number that says MILK clears 11% of its $160 base while WOOL clears 126% of
    its $200 base.
  * **failed orders**. A HIRE that cannot be afforded is silently skipped by the
    interpreter and is invisible in the replay, the observation and the final
    cash. It is also the exact failure mode #30's metering has to avoid, so the
    gate for that work is "zero failed HIRE/BUY relative to the incumbent" and
    this is the instrument that reads it.

    uv run python scripts/analyse_market_fills.py --agent main.py \
        --opponent opponents/v0_3_1.py --seed 2000000
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TURNS_PER_DAY = 24


class Recorder:
    """Everything one episode's money movement, keyed by seat."""

    def __init__(self) -> None:
        self.fills: list[dict[str, Any]] = []
        self.hires: list[dict[str, Any]] = []
        self.lands: list[dict[str, Any]] = []
        self.cash: list[list[float]] = []
        self.shed: list[list[int]] = []
        self.inventory: list[dict[str, int]] = []
        self.step = 0
        self.seat_of: dict[int, int] = {}

    # -- attribution ------------------------------------------------------
    def seat(self, farm: Any) -> int:
        return self.seat_of.get(id(farm), -1)

    # -- aggregation ------------------------------------------------------
    def realised(self, seat: int, op: str = "SELL") -> dict[str, dict[str, float]]:
        agg: dict[str, list[float]] = defaultdict(list)
        for f in self.fills:
            if f["seat"] == seat and f["op"] == op and f["ok"]:
                agg[f["item"]].append(f["price"])
        out = {}
        for item, prices in sorted(agg.items()):
            out[item] = {
                "units": len(prices),
                "revenue": sum(prices),
                "per_unit": sum(prices) / len(prices),
            }
        return out

    def failures(self, seat: int) -> dict[str, int]:
        bad = {
            "hire": sum(1 for h in self.hires if h["seat"] == seat and not h["ok"]),
            "buy_land": sum(1 for land in self.lands if land["seat"] == seat and not land["ok"]),
        }
        for f in self.fills:
            if f["seat"] != seat or f["ok"] or f["op"] == "SELL":
                continue
            bad[f["op"].lower()] = bad.get(f["op"].lower(), 0) + 1
        bad["sell"] = sum(
            1 for f in self.fills if f["seat"] == seat and f["op"] == "SELL" and not f["ok"]
        )
        return bad

    def per_fill_trace(self, seat: int, item: str) -> list[tuple[int, int, float]]:
        """(day, units, mean price) for one product, one seat."""
        by_day: dict[int, list[float]] = defaultdict(list)
        for f in self.fills:
            if f["seat"] == seat and f["op"] == "SELL" and f["ok"] and f["item"] == item:
                by_day[f["step"] // TURNS_PER_DAY].append(f["price"])
        return [(d, len(p), statistics.fmean(p)) for d, p in sorted(by_day.items())]


def instrument(rec: Recorder) -> None:
    """Hook the interpreter. Idempotent per process; the hooks are pure observers."""
    from kaggle_environments.envs.kaggriculture import kaggriculture as K

    if getattr(K, "_fills_instrumented", False):
        K._fills_recorder = rec
        return
    K._fills_instrumented = True
    K._fills_recorder = rec

    orig_market = K._process_market
    orig_commit = K._commit_unit
    orig_hire = K._do_hire
    orig_land = K._do_buy_land

    def process_market(state, env):
        r: Recorder = K._fills_recorder
        obs0 = state[0].observation
        r.seat_of = {id(farm): i for i, farm in enumerate(obs0.farms)}
        r.step = int(K.get(obs0, "step", 0) or 0)
        while len(r.cash) <= r.step:
            r.cash.append([0.0, 0.0])
            r.shed.append([0, 0])
            r.inventory.append({})
        r.cash[r.step] = [float(f["money"]) for f in obs0.farms]
        r.shed[r.step] = [int(sum(s.observation.private["shed"].values())) for s in state]
        r.inventory[r.step] = dict(obs0.market["inventory"])
        return orig_market(state, env)

    def commit_unit(op, item, price, farm, private, market, shed_capacity=100):
        r: Recorder = K._fills_recorder
        ok = orig_commit(op, item, price, farm, private, market, shed_capacity)
        r.fills.append(
            {
                "step": r.step,
                "seat": r.seat(farm),
                "op": op,
                "item": item,
                "price": float(price),
                "ok": bool(ok),
            }
        )
        return ok

    def do_hire(farm, private, board_size, mult=K.FARM_HAND_COST_MULT):
        r: Recorder = K._fills_recorder
        cost = K._hire_cost(farm["hires_today"], mult)
        before = len(farm["hands"])
        out = orig_hire(farm, private, board_size, mult)
        r.hires.append(
            {
                "step": r.step,
                "seat": r.seat(farm),
                "cost": cost,
                "ok": len(farm["hands"]) > before,
                "money": float(farm["money"]),
            }
        )
        return out

    def do_buy_land(farm, board_size):
        r: Recorder = K._fills_recorder
        n_extra = len(farm["unlocked_quadrants"]) - 1
        cost = K.LAND_PRICES[n_extra] if n_extra < len(K.LAND_PRICES) else None
        before = len(farm["unlocked_quadrants"])
        out = orig_land(farm, board_size)
        r.lands.append(
            {
                "step": r.step,
                "seat": r.seat(farm),
                "cost": cost,
                # A fourth BUY_LAND has nothing left to buy; that is the route's
                # problem, not a liquidity failure, so it is not counted as one.
                "ok": len(farm["unlocked_quadrants"]) > before or cost is None,
                "money": float(farm["money"]),
            }
        )
        return out

    K._process_market = process_market
    K._commit_unit = commit_unit
    K._do_hire = do_hire
    K._do_buy_land = do_buy_land


def run(agent: str, opponent: str, seed: int, steps: int = 720) -> Recorder:
    rec = Recorder()
    instrument(rec)
    from kaggle_environments import make

    env = make(
        "kaggriculture",
        configuration={"episodeSteps": steps, "seed": seed},
        debug=False,
    )
    env.run([agent, opponent])
    final = env.steps[-1]
    rec.rewards = [s.reward for s in final]  # type: ignore[attr-defined]
    return rec


BASES = {
    "WHEAT": 25,
    "CARROT": 35,
    "TOMATO": 60,
    "STRAWBERRY": 120,
    "MELON": 250,
    "EGG": 50,
    "MILK": 160,
    "WOOL": 200,
    "FERTILIZER": 100,
}


def report(rec: Recorder, seat: int = 0, trace_items: tuple[str, ...] = ("MILK", "STRAWBERRY")):
    ours = rec.realised(seat)
    theirs = rec.realised(1 - seat)
    print(f"\n  realised $/unit (seat {seat} = us)")
    print(
        f"  {'product':<12} {'units':>6} {'revenue':>11} {'$/unit':>9} {'base':>6} "
        f"{'vs base':>8} {'opp $/u':>9}"
    )
    for item in sorted(set(ours) | set(theirs)):
        a = ours.get(item)
        b = theirs.get(item)
        base = BASES.get(item, 0)
        if a is None:
            print(
                f"  {item:<12} {'-':>6} {'-':>11} {'-':>9} {base:>6} {'-':>8} "
                f"{b['per_unit'] if b else 0:>9.1f}"
            )
            continue
        print(
            f"  {item:<12} {a['units']:>6} {a['revenue']:>11,.0f} {a['per_unit']:>9.1f} "
            f"{base:>6} {a['per_unit'] / base:>7.0%} "
            f"{(b['per_unit'] if b else 0):>9.1f}"
        )

    print(f"\n  failed orders (seat {seat})")
    fails = rec.failures(seat)
    print("   ", ", ".join(f"{k}={v}" for k, v in sorted(fails.items())) or "none")

    for item in trace_items:
        trace = rec.per_fill_trace(seat, item)
        if not trace:
            continue
        print(f"\n  per-day {item} fills (day: units @ mean $)")
        print("    " + "  ".join(f"d{d}: {n}@${p:,.0f}" for d, n, p in trace))

    peak = max((s[seat] for s in rec.shed), default=0)
    print(f"\n  shed peak {peak}/100, final cash {getattr(rec, 'rewards', ['?', '?'])[seat]}")
    return {
        "realised": ours,
        "opponent_realised": theirs,
        "failures": fails,
        "shed_peak": peak,
        "rewards": getattr(rec, "rewards", None),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Per-fill market attribution (issue #30)")
    ap.add_argument("--agent", default=os.path.join(PROJECT_ROOT, "main.py"))
    ap.add_argument("--opponent", default=os.path.join(PROJECT_ROOT, "opponents", "v0_3_1.py"))
    ap.add_argument("--seed", type=int, default=2000000)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--json", default=None, help="write the full record here")
    args = ap.parse_args(argv)

    rec = run(args.agent, args.opponent, args.seed, args.steps)
    summary = report(rec, args.seat)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "summary": summary,
                    "fills": rec.fills,
                    "hires": rec.hires,
                    "lands": rec.lands,
                    "cash": rec.cash,
                    "shed": rec.shed,
                },
                f,
            )
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
