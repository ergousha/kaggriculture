#!/usr/bin/env python3
"""v0.4.3 Step 3 (parallel audit): day-27 endgame tape A/B (proposal §Step 3).

The one authored code path never re-validated against engine 1.32.7: at
``step >= 648`` the router switches EVERY draw to route 2 (inherited from the
metacounter lineage; the whole chassis class shares it). This A/Bs the incumbent
endgame (route 2) against the top endgame candidates (routes 1 and 5) on the
648-718 window with everything else byte-identical.

Design:
  * paired seeds, both seats (seat alternates per seed) vs the v0.4.2 peer;
  * route-2-forced variant is the anchor: it must tie the peer EXACTLY;
  * draw-stratified: uses the chassis draw table; reports per-YARN/non-YARN;
  * resumable append-only JSONL (logs/v043_panel/endgame.jsonl).

Usage:
    uv run python scripts/v043_endgame.py --workers 12 --seeds 60
    uv run python scripts/v043_endgame.py --report
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import v043_common as V  # noqa: E402

PEER = V.INCUMBENT
CANDIDATES = (1, 5)  # proposal's top-3 endgame candidates minus the incumbent 2
OUT = os.path.join(V.PANEL_DIR, "endgame.jsonl")


def _run_job(job: dict) -> dict:
    import local_arena as la

    res = la.run_episode(job)
    res["endgame"] = job["endgame"]
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--seeds", type=int, default=60, help="paired seeds per variant")
    ap.add_argument("--start", type=int, default=2_000_000)
    args = ap.parse_args()

    os.makedirs(V.PANEL_DIR, exist_ok=True)
    draws = V.load_chassis_draws()
    seed_list = [s for s in sorted(draws) if args.start <= s < args.start + 10_000][: args.seeds]
    print(f"endgame audit: {len(seed_list)} paired seeds, candidates {CANDIDATES} vs incumbent 2")

    jobs = []
    for route in (2, *CANDIDATES):  # 2 is the anchor
        vp = V.make_endgame_variant(route)
        for i, seed in enumerate(seed_list):
            jobs.append(
                {
                    "agent": vp,
                    "opponent": PEER,
                    "seed": seed,
                    "steps": 720,
                    "swap": bool(i % 2),
                    "decision_log": None,
                    "replay": None,
                    "endgame": route,
                }
            )

    done: set[tuple] = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add((rec["endgame"], rec["seed"], rec["swap"]))
                except Exception:
                    continue
    jobs = [j for j in jobs if (j["endgame"], j["seed"], j["swap"]) not in done]
    print(f"{len(jobs)} episodes to run ({len(done)} on file)")
    if jobs:
        t0 = time.time()
        ctx = mp.get_context("spawn")
        n = len(jobs)
        with open(OUT, "a") as sink, ctx.Pool(args.workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_run_job, jobs, chunksize=1)):
                sink.write(json.dumps(res) + "\n")
                sink.flush()
                if (i + 1) % 50 == 0:
                    el = time.time() - t0
                    print(f"  {i + 1}/{n}  ({el / 60:.0f}m elapsed)", flush=True)

    # report
    recs = []
    with open(OUT) as f:
        for line in f:
            try:
                recs.append(json.loads(line))
            except Exception:
                continue
    by_route: dict[int, list[dict]] = defaultdict(list)
    for r in recs:
        by_route[r["endgame"]].append(r)
    print(
        f"\n{'route':>5} {'n':>4} {'W':>3} {'T':>3} {'L':>3} {'Δμ$':>9} {'Δμ yarn':>9} {'Δμ nonY':>9}"
    )
    for route in sorted(by_route):
        rows = by_route[route]
        deltas = [r["me_cash"] - r["opp_cash"] for r in rows]
        yarn = [
            r["me_cash"] - r["opp_cash"]
            for r in rows
            if "YARN_STORE" in tuple(r.get("observed_pair") or ())
        ]
        nony = [
            r["me_cash"] - r["opp_cash"]
            for r in rows
            if "YARN_STORE" not in tuple(r.get("observed_pair") or ())
        ]
        w = sum(r["win"] for r in rows)
        t = sum(r["tie"] for r in rows)
        losses = len(rows) - w - t
        print(
            f"{route:>5} {len(rows):>4} {w:>3} {t:>3} {losses:>3} "
            f"{statistics.fmean(deltas):>+9,.0f} "
            f"{statistics.fmean(yarn):>+9,.0f} "
            f"{statistics.fmean(nony):>+9,.0f}"
        )
    anchor = by_route.get(2, [])
    bad_anchors = [r for r in anchor if not r["tie"]]
    if bad_anchors:
        print(f"\n!! {len(bad_anchors)} anchor non-ties -- audit is VOID")


if __name__ == "__main__":
    main()
