#!/usr/bin/env python3
"""Phase 2 -- stress-test candidate routes against market randomness.

    uv run python simulate_candidates.py --candidates candidates.jsonl --workers 12
    uv run python simulate_candidates.py --stage screen --workers 12    # one stage only
    uv run python simulate_candidates.py --resume                       # continue a killed run

Reuses `local_arena.run_episode` verbatim as the engine wrapper (it owns the env
construction, the seeded configuration, the instrumentation and the status
accounting) and drives it with `multiprocessing` following `local_arena.run_set`'s
pattern.

COMMON RANDOM NUMBERS. Every candidate is scored on the *identical* seed list,
fixed once in `mining/common.seed_sets()`. Without paired seeds a CVaR comparison
between two routes is mostly a comparison of their seed luck. The seed sets are
nested supersets -- screen (12) subset of mid (50) subset of final (500) -- so each
stage reuses the episodes its predecessor already paid for, and the Phase 3 holdout
(500) is disjoint from all of them.

THREE-STAGE SIEVE. Deduplication recovers only ~1.4% of this corpus, so the pool
stays at ~3,300 routes and a flat 50-seed screen would cost ~13 h. Instead:

    stage    candidates          seeds   purpose
    screen   all (~3,300)           12   cheap cut; drop routes that cannot survive
                                         12 shared markets at all
    mid      top --k-mid (400)      50   the spec's screening resolution
    final    top --k-final (20)    500   the CVaR_5 estimate Phase 3 ranks on

Ranking within a stage uses CVaR, but the *screen* uses a wider tail
(`--screen-alpha`, default 0.25 = mean of the worst 3 of 12) because CVaR_5 on 12
samples degenerates to the single worst episode and would discard good routes on
one unlucky draw. Selection is still CVaR_5 on the 500-seed set, in Phase 3.

WHAT IS SCORED. Each candidate is evaluated as the artifact we would actually
deploy: the route baked into build_route_agent's template, so main.py's three
runtime layers (WEED repair, SELL-slot ordering, hands alignment) are in play.
Scoring the bare trace would measure something we do not ship.

SEAT AND OPPONENT. The candidate plays seat 0 against a fixed opponent in seat 1.
The market's inventory is shared state between seats, so the opponent genuinely
perturbs the economy and must be held constant across candidates -- it is. Traces
are seat-portable (verified: swapping both seats' traces swaps their scores
exactly), so fixing seat 0 costs nothing and keeps the pairing trivially clean.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time

from mining import common
from mining.common import PROJECT_ROOT, cvar, decode_route_b85, describe

AGENT_CACHE = os.path.join(PROJECT_ROOT, "logs", "_mined_agents")
DEFAULT_OPPONENT = os.path.join("opponents", "v0_2_4.py")
STAGES = ("screen", "mid", "final")


def run_job(job: dict) -> dict:
    """One episode via local_arena's own runner, tagged with the candidate hash."""
    from local_arena import run_episode

    res = run_episode(job)
    return {
        "hash": job["hash"],
        "seed": res["seed"],
        "me_cash": res["me_cash"],
        "opp_cash": res["opp_cash"],
        "swap": res["swap"],
        "invalid": res["invalid"],
        "crashes": res["crashes"],
        "timeouts": res["timeouts"],
        "noop": res["actions_noop"],
        "actions": res["actions_total"],
        "harness_error": res["harness_error"],
    }


def materialize_agents(candidates: list[dict], workdir: str) -> tuple[dict[str, str], int]:
    """Write one deployable agent per candidate, cached by hash."""
    os.makedirs(workdir, exist_ok=True)
    paths = {}
    written = 0
    for cand in candidates:
        path = os.path.join(workdir, f"{cand['hash']}.py")
        paths[cand["hash"]] = path
        if os.path.exists(path):
            continue
        route = decode_route_b85(cand["route_b85"])
        common.write_route_agent(
            route,
            path,
            provenance={
                "episode": cand.get("episode"),
                "seat": cand.get("seat"),
                "team": cand.get("team", "?"),
                "recorded_cash": cand.get("recorded_cash"),
                "steps": cand.get("steps"),
                "hash": cand["hash"],
            },
            version=f"cand-{cand['hash'][:8]}",
        )
        written += 1
    return paths, written


def load_done(path: str) -> set[tuple[str, int]]:
    """(hash, seed) pairs already simulated, for --resume."""
    done: set[tuple[str, int]] = set()
    if not os.path.exists(path):
        return done
    import json

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if "hash" in row and "seed" in row:
                done.add((row["hash"], int(row["seed"])))
    return done


def run_stage(
    stage: str,
    candidates: list[dict],
    seeds: list[int],
    agent_paths: dict[str, str],
    opponent: str,
    steps: int,
    workers: int,
    results_path: str,
    done: set[tuple[str, int]],
    alternate_seats: bool,
) -> dict[str, dict[int, float]]:
    """Run every (candidate, seed) pair not already done; append results as they land."""
    import json

    jobs = []
    for cand in candidates:
        h = cand["hash"]
        for i, seed in enumerate(seeds):
            if (h, seed) in done:
                continue
            jobs.append(
                {
                    "hash": h,
                    "agent": agent_paths[h],
                    "opponent": opponent,
                    "seed": seed,
                    "steps": steps,
                    # Fixed seat 0 by default. Seats are provably symmetric here, and
                    # a constant seat keeps every candidate's pairing identical.
                    "swap": bool(i % 2) if alternate_seats else False,
                    "decision_log": None,
                    "replay": None,
                }
            )

    print(
        f"\n  stage {stage:<6} {len(candidates):>5,} candidates x {len(seeds):>4} seeds"
        f"  -> {len(jobs):,} new episodes ({len(candidates) * len(seeds) - len(jobs):,} reused)"
    )

    t0 = time.time()
    n = 0
    bad = 0
    with open(results_path, "a") as sink:
        if jobs:
            ctx = mp.get_context("spawn")
            with ctx.Pool(workers) as pool:
                for res in pool.imap_unordered(run_job, jobs, chunksize=4):
                    res["stage"] = stage
                    sink.write(json.dumps(res, separators=(",", ":")) + "\n")
                    # Mark the pair done immediately: the seed sets are nested, so
                    # the next stage must reuse these episodes rather than repay
                    # for them.
                    done.add((res["hash"], int(res["seed"])))
                    n += 1
                    if res["harness_error"] or res["crashes"] or res["invalid"]:
                        bad += 1
                    if n % 500 == 0:
                        sink.flush()
                        rate = n / (time.time() - t0)
                        eta = (len(jobs) - n) / rate if rate else 0
                        print(
                            f"        {n:,}/{len(jobs):,}  {rate:.1f} ep/s  "
                            f"eta {eta / 60:.0f}m  bad {bad}",
                            flush=True,
                        )
    wall = time.time() - t0
    if jobs:
        print(
            f"        done {n:,} episodes in {wall / 60:.1f}m ({n / max(wall, 1e-9):.1f} ep/s), bad {bad}"
        )
    if bad:
        print(f"        !! {bad} episodes had a crash / invalid status / harness error")
    return collect(results_path, stage=None)


def collect(results_path: str, stage: str | None = None) -> dict[str, dict[int, float]]:
    """Per-candidate cash lists, restricted to a seed set by the caller."""
    import json

    out: dict[str, dict[int, float]] = {}
    if not os.path.exists(results_path):
        return {}
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if stage and row.get("stage") != stage:
                continue
            out.setdefault(row["hash"], {})[int(row["seed"])] = float(row["me_cash"])
    return out


def scores_on(byhash: dict[str, dict[int, float]], h: str, seeds: list[int]) -> list[float]:
    """Cash for exactly `seeds`, in seed order. Missing seeds are omitted, so a
    partial candidate never silently outranks a complete one on a shorter tail."""
    got = byhash.get(h, {})
    return [got[s] for s in seeds if s in got]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase 2: simulate candidate routes")
    ap.add_argument("--candidates", default="candidates.jsonl")
    ap.add_argument("--results", default="logs/simulation_results.jsonl")
    ap.add_argument("--opponent", default=DEFAULT_OPPONENT)
    ap.add_argument("--steps", type=int, default=common.DEFAULT_STEPS)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 3))
    ap.add_argument("--k-mid", type=int, default=400, help="candidates promoted to the 50-seed run")
    ap.add_argument("--k-final", type=int, default=20, help="finalists for the 500-seed run")
    ap.add_argument(
        "--screen-alpha", type=float, default=0.25, help="CVaR tail for the 12-seed screen"
    )
    ap.add_argument("--seed-mode", choices=("sequential", "random31"), default="sequential")
    ap.add_argument("--stage", choices=STAGES, help="run only this stage")
    ap.add_argument(
        "--resume", action="store_true", help="skip (candidate, seed) pairs already done"
    )
    ap.add_argument("--limit", type=int, default=0, help="cap candidates (smoke tests)")
    ap.add_argument(
        "--alternate-seats",
        action="store_true",
        help="alternate seats across seeds instead of pinning the candidate to seat 0",
    )
    args = ap.parse_args(argv)

    cand_path = (
        args.candidates
        if os.path.isabs(args.candidates)
        else os.path.join(PROJECT_ROOT, args.candidates)
    )
    if not os.path.exists(cand_path):
        raise SystemExit(f"no candidate pool at {cand_path} (run mine_replays.py first)")
    candidates = common.read_jsonl(cand_path)
    if args.limit:
        candidates = candidates[: args.limit]
    if not candidates:
        raise SystemExit("candidate pool is empty")

    unvalidated = [c for c in candidates if c.get("fidelity") != "exact"]
    if unvalidated:
        print(
            f"  !! {len(unvalidated):,} of {len(candidates):,} candidates did not pass the "
            "Phase 1 fidelity gate; results downstream are not trustworthy"
        )

    opponent = (
        args.opponent if os.path.isabs(args.opponent) else os.path.join(PROJECT_ROOT, args.opponent)
    )
    if not os.path.exists(opponent):
        raise SystemExit(f"opponent not found: {opponent}")

    results_path = (
        args.results if os.path.isabs(args.results) else os.path.join(PROJECT_ROOT, args.results)
    )
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    if not args.resume and os.path.exists(results_path):
        os.rename(results_path, results_path + ".bak")
        print(f"  moved previous results to {os.path.basename(results_path)}.bak")

    seeds = common.seed_sets(args.seed_mode)
    print(f"Phase 2: {len(candidates):,} candidates vs {os.path.relpath(opponent, PROJECT_ROOT)}")
    print(f"  workers {args.workers}   steps {args.steps}   seed mode {args.seed_mode}")
    print(
        f"  CRN seed sets: screen {len(seeds['screen'])}, mid {len(seeds['mid'])}, "
        f"final {len(seeds['final'])} (nested); holdout {len(seeds['holdout'])} (disjoint, Phase 3)"
    )
    print(f"  seat: {'alternating' if args.alternate_seats else 'candidate pinned to seat 0'}")

    agent_paths, written = materialize_agents(candidates, AGENT_CACHE)
    print(f"  materialized {written:,} new agent files (cache {AGENT_CACHE})")

    done = load_done(results_path) if args.resume else set()
    if done:
        print(f"  resume: {len(done):,} (candidate, seed) pairs already simulated")

    stages_to_run = [args.stage] if args.stage else list(STAGES)
    pool = candidates
    summary = {}

    for stage in stages_to_run:
        seed_list = seeds[stage]
        byhash = run_stage(
            stage,
            pool,
            seed_list,
            agent_paths,
            opponent,
            args.steps,
            args.workers,
            results_path,
            done,
            args.alternate_seats,
        )
        alpha = args.screen_alpha if stage == "screen" else 0.05
        ranked = []
        for cand in pool:
            vals = scores_on(byhash, cand["hash"], seed_list)
            if len(vals) < len(seed_list):
                continue
            ranked.append((cvar(vals, alpha), cand, vals))
        ranked.sort(key=lambda r: -r[0])
        label = f"CVaR{int(alpha * 100)}"
        print(f"        top 5 by {label} on {len(seed_list)} shared seeds:")
        for score, cand, vals in ranked[:5]:
            d = describe(vals)
            print(
                f"          ${score:>10,.0f}  {cand['hash'][:10]}  {cand['team'][:16]:<16} "
                f"mean ${d['mean']:>9,.0f}  min ${d['min']:>9,.0f}  rec ${cand['recorded_cash']:>9,.0f}"
            )
        summary[stage] = {
            "seeds": len(seed_list),
            "candidates": len(pool),
            "ranked": len(ranked),
            "alpha": alpha,
            "top": [
                {"hash": c["hash"], f"cvar{int(alpha * 100)}": round(s, 1), **describe(v)}
                for s, c, v in ranked[:10]
            ],
        }
        if stage == "screen":
            pool = [c for _, c, _ in ranked[: args.k_mid]]
        elif stage == "mid":
            pool = [c for _, c, _ in ranked[: args.k_final]]
        if not pool:
            print("        no candidate completed this stage; stopping")
            break

    import json

    summary_path = os.path.join(PROJECT_ROOT, "logs", "simulation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "opponent": os.path.relpath(opponent, PROJECT_ROOT),
                "seed_mode": args.seed_mode,
                "seed_sets": {k: [v[0], v[-1], len(v)] for k, v in seeds.items()},
                "alternate_seats": args.alternate_seats,
                "stages": summary,
                "finalists": [c["hash"] for c in pool],
            },
            f,
            indent=2,
        )
    print(f"\n  results  -> {results_path}")
    print(f"  summary  -> {summary_path}")
    print(f"  finalists: {len(pool)}  (feed to rank_cvar.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
