#!/usr/bin/env python3
"""Refit the day-6 routing table against REAL leaderboard agents, not self-play.

    uv run python scripts/v052_route_refit.py --enumerate --workers 10
    uv run python scripts/v052_route_refit.py --coverage
    uv run python scripts/v052_route_refit.py --sweep --min-replays 2 --workers 10
    uv run python scripts/v052_route_refit.py --report

WHY. The agent's only real degree of freedom is the day-6 table: 64 shop-pair draws, one
route tape each, out of 42 tapes. v0.4.3 fitted 19 of those cells against
`opponents/v0_4_2.py` -- i.e. against ourselves -- and the criterion was in-sample cash.
`docs/finding-2026-09-22-v052-routing-revert-rejected.md` shows what that bought: the
table is load-bearing (reverting it costs $4,143/draw against real opponents) but its
individual assignments have never been tested against anything except our own snapshots.
This script re-fits them against the recorded action streams of the rank 1-12 teams.

TWO ENGINE FACTS THIS HARNESS HAS TO RESPECT.

  1. A replay is only coherent on the board it was recorded against, so every episode
     runs at the replay's OWN seed, recovered from `info.seed`.

  2. The day-6 draw is NOT a function of the seed alone. `scripts/enumerate_draws.py`
     measured that a pass-vs-pass draw table transfers to chassis-vs-chassis play on
     0/12 seeds: agents occupy tiles from step 0, and weed spawns only consume RNG on
     still-empty tiles, so the two farms diverge the draw stream. The draw a pairing
     ACTUALLY produces therefore has to be observed, per (replay, seat), by running it
     -- which is what --enumerate does, reading `observed_pair` back out of the arena.
     Classifying draws from the replay's own step-144 observation, as
     scripts/v052_replay_ab.py did, is only valid for its exact-tie anchor.

PARALLELISM. `KAGGRICULTURE_REPLAY_PATH` is read inside the worker at agent-load time,
and a spawn Pool's workers inherit the parent environment once, at pool creation -- so
setting it per job does not reach an existing worker. Instead each replay gets a tiny
generated opponent shim with its path baked in, which makes the replay part of the job
tuple and lets every episode go into ONE pool. The first version of this work set the
variable in the parent and called run_set per replay, which serialised the run to two
episodes at a time.
"""

from __future__ import annotations

import argparse
import ast
import csv
import glob
import json
import os
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
WORK = HERE / "logs" / "_v052_refit"
DRAWS_CSV = HERE / "logs" / "v052_replay_draws.csv"
SWEEP_JSON = HERE / "logs" / "v052_route_sweep.json"

REPLAY_DIRS = ("logs/leaderboard_replays", "logs/september_spy")


def kaggriculture_replays() -> list[tuple[str, int]]:
    """(path, seed) for every kaggriculture replay whose seed is in the file head.

    Both are read from a 4 KB head: `name` sits at ~byte 860 and `seed` just after, while
    `steps` starts at ~byte 6300 and runs for megabytes.
    """
    out: list[tuple[str, int]] = []
    for directory in REPLAY_DIRS:
        for path in sorted(glob.glob(str(HERE / directory / "episode-*.json"))):
            head = Path(path).read_bytes()[:4096].decode("utf-8", "ignore")
            if '"kaggriculture"' not in head:
                continue
            match = re.search(r'"seed"\s*:\s*(\d+)', head)
            if match:
                out.append((path, int(match.group(1))))
    seen: dict[int, tuple[str, int]] = {}
    for path, seed in out:
        seen.setdefault(seed, (path, seed))  # one replay per seed; duplicates add nothing
    return sorted(seen.values(), key=lambda r: r[1])


def shim_for(replay: str) -> str:
    """An opponent file that is leaderboard_replay.py pinned to one replay."""
    WORK.mkdir(parents=True, exist_ok=True)
    dest = WORK / f"opp_{Path(replay).stem}.py"
    if not dest.exists():
        real = HERE / "opponents" / "leaderboard_replay.py"
        dest.write_text(
            "import os\n"
            f"os.environ['KAGGRICULTURE_REPLAY_PATH'] = {replay!r}\n"
            f"_src = open({str(real)!r}).read()\n"
            "_ns = {}\n"
            f"exec(compile(_src, {str(real)!r}, 'exec'), _ns)\n"
            "agent = _ns['agent']\n"
        )
    return str(dest)


def table() -> dict[tuple[str, str], int]:
    src = (HERE / "main.py").read_text()
    match = re.search(r"^_V043_TABLE=(\{.*\})$", src, re.M)
    assert match is not None
    return ast.literal_eval(match.group(1))


def route_ids() -> list[int]:
    """Every tape id the shipped agent can actually route to.

    Read from the imported module rather than by regex: `_ROUTES` is assembled from a
    dict literal plus an `_EXP_ROUTES` update plus per-id opening rewrites, so a pattern
    over the source misses the generic 1-12 tapes that the YARN cells route to.
    """
    import main

    return sorted(main._ROUTES.keys())


def variant_for(cell: tuple[str, str], route: int) -> str:
    """main.py with one table cell overridden, via local_arena's own rewriter."""
    import local_arena as la

    WORK.mkdir(parents=True, exist_ok=True)
    dest = WORK / f"agent_{cell[0]}_{cell[1]}_{route}.py"
    if not dest.exists():
        patched = dict(table())
        patched[cell] = route
        la.make_variant(str(HERE / "main.py"), str(dest), consts={"_V043_TABLE": patched})
    return str(dest)


def _run(jobs: list[dict], workers: int) -> list[dict]:
    import multiprocessing as mp

    import local_arena as la

    if workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            return pool.map(la.run_episode, jobs)
    return [la.run_episode(j) for j in jobs]


def enumerate_draws(workers: int) -> None:
    import local_arena as la

    replays = kaggriculture_replays()
    print(f"{len(replays)} kaggriculture replays with a distinct seed")
    jobs: list[dict] = []
    for path, seed in replays:
        for swap in (False, True):
            jobs.append(
                {
                    "agent": str(HERE / "main.py"),
                    "opponent": shim_for(path),
                    "seed": seed,
                    "steps": la.DEFAULT_STEPS,
                    "swap": swap,
                    "decision_log": None,
                    "replay": None,
                }
            )
    print(f"running {len(jobs)} episodes on {workers} workers to observe the real draws...")
    results = _run(jobs, workers)
    with open(DRAWS_CSV, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["replay", "seed", "swap", "shop1", "shop2", "me_cash", "opp_cash", "error"]
        )
        for job, res in zip(jobs, results, strict=True):
            pair = res.get("observed_pair") or ("", "")
            writer.writerow(
                [
                    job["opponent"],
                    job["seed"],
                    int(job["swap"]),
                    pair[0],
                    pair[1],
                    res.get("me_cash"),
                    res.get("opp_cash"),
                    res.get("harness_error") or "",
                ]
            )
    print(f"wrote {DRAWS_CSV}")
    coverage()


def coverage() -> None:
    if not DRAWS_CSV.exists():
        raise SystemExit("run --enumerate first")
    rows = list(csv.DictReader(open(DRAWS_CSV)))
    ok = [r for r in rows if r["shop1"] and not r["error"]]
    print(f"\nepisodes: {len(rows)}   with an observed draw and no error: {len(ok)}")
    by: dict[tuple[str, str], list[dict]] = {}
    for r in ok:
        by.setdefault((r["shop1"], r["shop2"]), []).append(r)
    shipped = table()
    print(f"distinct draws observed: {len(by)} of 64\n")
    print(f"  {'draw':34s} {'episodes':>9s} {'replays':>8s} {'shipped':>8s} {'margin':>11s}")
    for cell, group in sorted(by.items(), key=lambda kv: -len(kv[1])):
        margins = [float(r["me_cash"]) - float(r["opp_cash"]) for r in group]
        print(
            f"  {str(cell):34s} {len(group):>9d} {len({r['replay'] for r in group}):>8d} "
            f"{shipped.get(cell, 0):>8d} {st.fmean(margins):>+11,.0f}"
        )
    usable = {c: g for c, g in by.items() if len({r["replay"] for r in g}) >= 2}
    print(f"\ndraws with >=2 distinct replays (fittable with a held-out split): {len(usable)}")
    print("candidate tapes available:", len(route_ids()))


def _usable(min_replays: int) -> dict[tuple[str, str], list[dict]]:
    rows = [r for r in csv.DictReader(open(DRAWS_CSV)) if r["shop1"] and not r["error"]]
    by: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        by.setdefault((r["shop1"], r["shop2"]), []).append(r)
    return {c: g for c, g in by.items() if len({r["replay"] for r in g}) >= min_replays}


def sweep(min_replays: int, workers: int, only: list[str] | None) -> None:
    """For every fittable draw, play every candidate tape against its own replays."""
    import local_arena as la

    usable = _usable(min_replays)
    if only:
        usable = {c: g for c, g in usable.items() if f"{c[0]}/{c[1]}" in only}
    routes = route_ids()
    shipped = table()
    jobs: list[dict] = []
    index: list[tuple] = []
    for cell, group in sorted(usable.items()):
        for route in routes:
            agent = variant_for(cell, route)
            for row in group:
                jobs.append(
                    {
                        "agent": agent,
                        "opponent": row["replay"],
                        "seed": int(row["seed"]),
                        "steps": la.DEFAULT_STEPS,
                        "swap": bool(int(row["swap"])),
                        "decision_log": None,
                        "replay": None,
                    }
                )
                index.append((cell, route, row["replay"], int(row["swap"])))
    print(
        f"{len(usable)} fittable draws x {len(routes)} tapes -> {len(jobs)} episodes "
        f"on {workers} workers"
    )
    print("shipped cells under test:", {f"{c[0]}/{c[1]}": shipped.get(c) for c in sorted(usable)})
    results = _run(jobs, workers)
    out = []
    for (cell, route, replay, swap), res in zip(index, results, strict=True):
        out.append(
            {
                "shop1": cell[0],
                "shop2": cell[1],
                "route": route,
                "replay": replay,
                "swap": swap,
                "me": res.get("me_cash"),
                "opp": res.get("opp_cash"),
                "pair": list(res.get("observed_pair") or ()),
                "error": res.get("harness_error") or "",
            }
        )
    SWEEP_JSON.write_text(json.dumps(out, indent=1))
    print(f"wrote {SWEEP_JSON}")
    report(min_replays)


def report(min_replays: int) -> None:
    """Leave-one-replay-out cross-validation of the REFIT PROCEDURE itself.

    Picking the best of 41 tapes from a handful of episodes is exactly how the v0.4.3
    table was produced, and the winner's curse does the rest: with per-episode margins
    spanning +/-$150k, the max of 41 noisy cell estimates is mostly noise. Reporting that
    maximum as an improvement would repeat the error with a different oracle.

    So the criterion is pre-registered and out-of-sample. For each draw and each held-out
    replay: choose the best tape using ONLY the other replays, then score
    (chosen - shipped) on the held-out replay. Averaged over every fold and draw, that is
    an honest estimate of what refitting the table would actually buy. If it is <= 0, the
    refit does not work and nothing ships, however good the in-sample maximum looks.
    """
    if not SWEEP_JSON.exists():
        raise SystemExit("run --sweep first")
    rows = [r for r in json.loads(SWEEP_JSON.read_text()) if not r["error"] and r["me"] is not None]
    rows = [r for r in rows if list(r["pair"]) == [r["shop1"], r["shop2"]]]
    shipped = table()

    # draw -> tape -> replay -> mean margin over the two seats
    cells: dict[tuple, dict[int, dict[str, list[float]]]] = {}
    for r in rows:
        cell = (r["shop1"], r["shop2"])
        cells.setdefault(cell, {}).setdefault(r["route"], {}).setdefault(r["replay"], []).append(
            r["me"] - r["opp"]
        )
    folds: list[tuple] = []
    insample_gain = []
    print("\nPER-DRAW REFIT, leave-one-replay-out (pre-registered criterion)\n")
    print(f"  {'draw':32s} {'ship':>5s} {'in-sample best':>22s} {'LOO out-of-sample':>20s}")
    for cell, per_tape in sorted(cells.items()):
        mean = {t: {rp: st.fmean(v) for rp, v in d.items()} for t, d in per_tape.items()}
        cur = shipped.get(cell)
        if cur not in mean:
            continue
        replays = sorted(mean[cur])
        if len(replays) < min_replays:
            continue
        # in-sample maximum, for contrast only
        best_t = max(mean, key=lambda t: st.fmean([mean[t][rp] for rp in replays if rp in mean[t]]))
        best_v = st.fmean([mean[best_t][rp] for rp in replays if rp in mean[best_t]])
        cur_v = st.fmean([mean[cur][rp] for rp in replays])
        insample_gain.append(best_v - cur_v)
        # leave-one-replay-out
        deltas = []
        for held in replays:
            fit = [rp for rp in replays if rp != held]
            pick = max(
                (t for t in mean if held in mean[t] and all(rp in mean[t] for rp in fit)),
                key=lambda t: st.fmean([mean[t][rp] for rp in fit]),
                default=None,
            )
            if pick is None or held not in mean[cur]:
                continue
            deltas.append(mean[pick][held] - mean[cur][held])
            folds.append((cell, held, pick, deltas[-1]))
        loo = st.fmean(deltas) if deltas else float("nan")
        print(
            f"  {str(cell):32s} {cur:>5d}  {best_t:>4d} ${best_v - cur_v:>+13,.0f}"
            f"   {len(deltas)} folds ${loo:>+10,.0f}"
        )

    print("\n" + "=" * 78)
    if insample_gain:
        print(
            f"in-sample gain from taking the max of {len(route_ids())} tapes per draw: "
            f"${st.fmean(insample_gain):+,.0f}/draw  <-- the winner's curse, not a result"
        )
    if folds:
        d = [f[3] for f in folds]
        wins = sum(1 for x in d if x > 0)
        se = st.pstdev(d) / (len(d) ** 0.5) or float("inf")
        print(
            f"OUT-OF-SAMPLE (leave-one-replay-out, {len(d)} folds across "
            f"{len({f[0] for f in folds})} draws):"
        )
        print(
            f"  refit minus shipped: ${st.fmean(d):+,.0f}/draw   median ${st.median(d):+,.0f}   "
            f"{wins}/{len(d)} folds positive   t={st.fmean(d) / se:+.2f}"
        )
        if st.fmean(d) <= 0 or st.fmean(d) / se < 1.65:
            print("\n  VERDICT: the refit does NOT beat the shipped table out of sample.")
            print("  Ship nothing. The shipped cells are not improvable on this corpus.")
        else:
            print("\n  VERDICT: the refit beats the shipped table out of sample.")
            print("  Promote only cells whose every LOO fold is positive:")
            per: dict[tuple, list] = {}
            for cell, _, pick, delta in folds:
                per.setdefault(cell, []).append((pick, delta))
            for cell, fs in sorted(per.items()):
                picks = {p for p, _ in fs}
                if len(picks) == 1 and all(x > 0 for _, x in fs):
                    print(
                        f"    {str(cell):32s} {shipped.get(cell)} -> {picks.pop()}  "
                        f"({len(fs)}/{len(fs)} folds positive)"
                    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--enumerate", action="store_true")
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--min-replays", type=int, default=2)
    ap.add_argument("--only", nargs="*", help="limit to SHOP1/SHOP2 draws")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    if args.enumerate:
        enumerate_draws(args.workers)
    elif args.coverage:
        coverage()
    elif args.sweep:
        sweep(args.min_replays, args.workers, args.only)
    elif args.report:
        report(args.min_replays)
    else:
        ap.error("pick --enumerate, --coverage, --sweep or --report")


if __name__ == "__main__":
    main()
