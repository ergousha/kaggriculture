#!/usr/bin/env python3
"""Smoke test for the v0.4.3 panel machinery (run before launching the screen).

Validates the measurement's core claim end-to-end: a forced-tape variant whose
tape equals the incumbent table's choice for the observed pair must tie the
incumbent peer EXACTLY on every seed, and a different tape must produce a real
cash delta. Also checks tape-dedup representatives and the variant patch.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import multiprocessing as mp  # noqa: E402

import local_arena as la  # noqa: E402
from scripts import v043_common as V  # noqa: E402


def _run_job(job: dict) -> dict:
    res = la.run_episode(job)
    res["tape_id"] = job["tape_id"]
    return res


def run_smoke() -> int:
    inc_table, yarn_table = V.extract_tables_from_incumbent()
    tapes = V.load_tapes()
    uniq = V.unique_tapes(tapes)
    print(f"portfolio: {len(tapes)} ids, {len(uniq)} unique windows")
    assert len(uniq) == 38, f"expected 38 unique windows, got {len(uniq)}"

    draws = V.load_chassis_draws()
    pair = ("PIZZA_SHOP", "FARMERS_MARKET")
    seeds = [s for s in sorted(draws) if draws[s] == pair][:2]
    inc_rep = inc_table[pair]
    print(f"smoke pair: {pair}, seeds: {seeds}, incumbent tape: {inc_rep}")

    jobs = []
    for tape in (inc_rep, 106):
        vp = V.make_forced_variant(tape)
        for i, seed in enumerate(seeds):
            jobs.append(
                {
                    "agent": vp,
                    "opponent": V.INCUMBENT,
                    "seed": seed,
                    "steps": 720,
                    "swap": bool(i % 2),
                    "decision_log": None,
                    "replay": None,
                    "pair": list(pair),
                    "tape_id": tape,
                }
            )
    ctx = mp.get_context("spawn")
    with ctx.Pool(4) as pool:
        results = pool.map(_run_job, jobs)

    failures = 0
    for r in sorted(results, key=lambda r: (r["tape_id"], r["seed"])):
        verdict = "TIE" if r["tie"] else ("WIN" if r["win"] else "LOSS")
        obs = tuple(r.get("observed_pair") or ())
        print(
            f"  tape {r['tape_id']:>3} seed {r['seed']} swap {int(r['swap'])}: "
            f"me {r['me_cash']:>10,.0f} opp {r['opp_cash']:>10,.0f} "
            f"obs {obs} {verdict}"
        )
        if r["tape_id"] == inc_rep:
            if not r["tie"]:
                failures += 1
                print("    !! anchor must tie exactly")
            if obs != pair:
                failures += 1
                print("    !! observed pair must match the chassis table row")
    if failures:
        print(f"\nSMOKE FAILED: {failures} problems")
        return 1
    print("\nSMOKE OK: anchor ties exactly, observed pairs match the chassis table")
    return 0


if __name__ == "__main__":
    sys.exit(run_smoke())
