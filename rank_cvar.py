#!/usr/bin/env python3
"""Phase 3 -- rank finalists by CVaR_5 and validate the winner on held-out seeds.

    uv run python rank_cvar.py --workers 12
    uv run python rank_cvar.py --no-holdout          # report the table only

Two things happen here.

RANKING. Finalists are ranked by CVaR_5 on the 500-seed evaluation set from
Phase 2. Mean, median, 5th percentile, min and max are reported alongside for
context, but selection is by CVaR_5 alone. CVaR_5 is the mean of the worst 5% of
outcomes (25 of 500), not the 5th-percentile score itself -- it is the stricter of
the two and it distinguishes "bad seeds earn $80k" from "bad seeds earn $20k".

WINNER'S-CURSE GUARD. Taking the maximum of ~3,300 noisy CVaR estimates overfits
the evaluation seed set: the top route is partly the luckiest estimate rather than
the most robust route. So the top pick and the v0.2.4 baseline are both re-run on a
FRESH 500-seed set, disjoint from every seed used in Phase 2, and the decision is
made on those held-out numbers. A winner that cannot beat v0.2.4 on held-out
CVaR_5 is reported as a failure, not shipped.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
import time

from mining import common
from mining.common import PROJECT_ROOT, describe
from simulate_candidates import AGENT_CACHE, DEFAULT_OPPONENT, collect, run_job, scores_on

SEED_CAVEAT = """
  ASSUMPTION -- seed distribution. Every number above comes from local episodes
  seeded with {mode} integers ({lo}..{hi}). The engine derives all market
  randomness from `random.Random((seed * 1000003) ^ day)`, so these seeds do span
  the shop-draw space, but they are NOT drawn the way Kaggle draws them: Kaggle
  assigns each episode a seed we cannot observe or reproduce, and
  `resolve_episode_seed` falls back to a random 31-bit integer. If Kaggle's
  seeding differs materially in distribution from this set, the offline CVaR
  ordering need not transfer to the ladder. Nothing in this pipeline can verify
  that; it is an assumption, and it is the largest single risk in the result.
  (`--seed-mode random31` samples the same 31-bit range as the engine's own
  fallback and is the closer analogue if you want a second read.)
"""


def evaluate_on(
    label: str,
    agent_path: str,
    seeds: list[int],
    opponent: str,
    steps: int,
    workers: int,
    results_path: str,
    tag: str,
    alternate_seats: bool,
) -> list[float]:
    """Run one agent over a seed set, appending rows to a results file."""
    jobs = [
        {
            "hash": tag,
            "agent": agent_path,
            "opponent": opponent,
            "seed": seed,
            "steps": steps,
            "swap": bool(i % 2) if alternate_seats else False,
            "decision_log": None,
            "replay": None,
        }
        for i, seed in enumerate(seeds)
    ]
    print(f"    {label}: {len(jobs)} held-out episodes", flush=True)
    t0 = time.time()
    out = []
    with open(results_path, "a") as sink:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            for res in pool.imap_unordered(run_job, jobs, chunksize=4):
                res["stage"] = "holdout"
                sink.write(json.dumps(res, separators=(",", ":")) + "\n")
                out.append(res["me_cash"])
    print(f"      {len(out)} episodes in {(time.time() - t0) / 60:.1f}m", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase 3: CVaR ranking + held-out validation")
    ap.add_argument("--candidates", default="candidates.jsonl")
    ap.add_argument("--results", default="logs/simulation_results.jsonl")
    ap.add_argument("--holdout-results", default="logs/holdout_results.jsonl")
    ap.add_argument("--summary", default="logs/simulation_summary.json")
    ap.add_argument("--report", default="logs/cvar_report.json")
    ap.add_argument("--baseline", default=DEFAULT_OPPONENT, help="route to beat (v0.2.4)")
    ap.add_argument("--opponent", default=DEFAULT_OPPONENT)
    ap.add_argument("--steps", type=int, default=common.DEFAULT_STEPS)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 3))
    ap.add_argument("--seed-mode", choices=("sequential", "random31"), default="sequential")
    ap.add_argument("--top", type=int, default=20, help="rows to print in the table")
    ap.add_argument("--no-holdout", action="store_true", help="skip the winner's-curse guard")
    ap.add_argument(
        "--holdout-n",
        type=int,
        default=common.N_HOLDOUT,
        help="held-out seed count (default 500; smaller weakens the guard)",
    )
    ap.add_argument("--alternate-seats", action="store_true")
    args = ap.parse_args(argv)

    def abspath(p):
        return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)

    candidates = {c["hash"]: c for c in common.read_jsonl(abspath(args.candidates))}
    results_path = abspath(args.results)
    if not os.path.exists(results_path):
        raise SystemExit(f"no simulation results at {results_path} (run simulate_candidates.py)")

    seeds = common.seed_sets(args.seed_mode, n_holdout=args.holdout_n)
    final_seeds = seeds["final"]
    holdout_seeds = seeds["holdout"]
    overlap = set(final_seeds) & set(holdout_seeds)
    if overlap:
        raise SystemExit(f"holdout is not disjoint from evaluation seeds ({len(overlap)} shared)")

    byhash = collect(results_path)

    # Finalists are those with a complete 500-seed evaluation record.
    rows = []
    for h, cand in candidates.items():
        vals = scores_on(byhash, h, final_seeds)
        if len(vals) < len(final_seeds):
            continue
        d = describe(vals)
        rows.append({"hash": h, "candidate": cand, "scores": vals, **d})
    if not rows:
        raise SystemExit(
            "no candidate has a complete 500-seed record; run simulate_candidates.py --stage final"
        )
    rows.sort(key=lambda r: -r["cvar5"])

    print(f"Phase 3: {len(rows)} finalists ranked by CVaR_5 on {len(final_seeds)} shared seeds")
    print(
        f"  (CVaR_5 = mean of the worst {max(1, int(len(final_seeds) * 0.05))} of {len(final_seeds)} outcomes)"
    )
    print(
        f"\n  {'rank':>4}  {'CVaR_5':>10}  {'mean':>10}  {'median':>10}  {'p5':>10}  "
        f"{'min':>10}  {'max':>10}  {'recorded':>10}  hash / team"
    )
    for i, r in enumerate(rows[: args.top], 1):
        c = r["candidate"]
        print(
            f"  {i:>4}  {r['cvar5']:>10,.0f}  {r['mean']:>10,.0f}  {r['median']:>10,.0f}  "
            f"{r['p5']:>10,.0f}  {r['min']:>10,.0f}  {r['max']:>10,.0f}  "
            f"{c['recorded_cash']:>10,.0f}  {r['hash'][:10]} {c['team'][:18]}"
        )

    # Recorded cash vs measured robustness: worth stating explicitly, because it
    # is the reason the $85k threshold is set low.
    if len(rows) > 2:
        rec = [r["candidate"]["recorded_cash"] for r in rows]
        cv = [r["cvar5"] for r in rows]
        try:
            corr = statistics.correlation(rec, cv)
            print(
                f"\n  correlation(recorded cash, CVaR_5) over finalists = {corr:+.2f}"
                "  -- recorded cash is a weak proxy for robustness"
            )
        except statistics.StatisticsError:
            pass

    report = {
        "seed_mode": args.seed_mode,
        "evaluation_seeds": [final_seeds[0], final_seeds[-1], len(final_seeds)],
        "holdout_seeds": [holdout_seeds[0], holdout_seeds[-1], len(holdout_seeds)],
        "cvar_definition": "mean of the worst floor(0.05*n) outcomes",
        "ranking": [
            {
                "rank": i,
                "hash": r["hash"],
                "team": r["candidate"]["team"],
                "episode": r["candidate"]["episode"],
                "seat": r["candidate"]["seat"],
                "recorded_cash": r["candidate"]["recorded_cash"],
                **{k: r[k] for k in ("cvar5", "mean", "median", "p5", "min", "max", "stdev", "n")},
            }
            for i, r in enumerate(rows, 1)
        ],
    }

    if args.no_holdout:
        print("\n  holdout SKIPPED (--no-holdout); no winner selected")
        report["holdout"] = None
    else:
        winner = rows[0]
        agent_path = os.path.join(AGENT_CACHE, f"{winner['hash']}.py")
        if not os.path.exists(agent_path):
            raise SystemExit(f"winner agent missing: {agent_path}")
        baseline_path = abspath(args.baseline)
        opponent = abspath(args.opponent)
        hpath = abspath(args.holdout_results)
        os.makedirs(os.path.dirname(hpath), exist_ok=True)

        print(
            f"\n  winner's-curse guard: re-running the top pick and v0.2.4 on "
            f"{len(holdout_seeds)} FRESH seeds disjoint from Phase 2"
        )
        win_vals = evaluate_on(
            f"winner {winner['hash'][:10]}",
            agent_path,
            holdout_seeds,
            opponent,
            args.steps,
            args.workers,
            hpath,
            f"holdout-{winner['hash']}",
            args.alternate_seats,
        )
        base_vals = evaluate_on(
            "baseline v0.2.4",
            baseline_path,
            holdout_seeds,
            opponent,
            args.steps,
            args.workers,
            hpath,
            "holdout-baseline-v0_2_4",
            args.alternate_seats,
        )

        w, b = describe(win_vals), describe(base_vals)
        delta = w["cvar5"] - b["cvar5"]
        beats = delta > 0

        # The opponent is held constant at v0.2.4 for every run, which is what
        # common random numbers require -- but it makes the baseline row, and only
        # the baseline row, a MIRROR match (v0.2.4 vs itself). Two identical routes
        # bid for the same shared market inventory at the same instants, which is a
        # materially different economy from a candidate facing a route unlike its
        # own. So the comparison is like-for-like in opponent but not in pairing.
        mirror = os.path.abspath(baseline_path) == os.path.abspath(opponent)
        if mirror:
            print(
                "\n    NOTE the baseline is a mirror match: v0.2.4 is both the agent and\n"
                "    the opponent, while candidates face v0.2.4 as a dissimilar route. The\n"
                "    opponent is constant (CRN holds), but the baseline's pairing is not\n"
                "    comparable to the candidates'. On the ladder we never face ourselves,\n"
                "    so treat this baseline as a self-play reference, not a field estimate."
            )

        print(f"\n  held-out results on {len(holdout_seeds)} shared seeds")
        print(
            f"    {'':<22} {'CVaR_5':>10} {'mean':>10} {'median':>10} {'p5':>10} {'min':>10} {'max':>10}"
        )
        for name, d in (("winner", w), ("v0.2.4 baseline", b)):
            print(
                f"    {name:<22} {d['cvar5']:>10,.0f} {d['mean']:>10,.0f} {d['median']:>10,.0f} "
                f"{d['p5']:>10,.0f} {d['min']:>10,.0f} {d['max']:>10,.0f}"
            )
        print(f"\n    held-out CVaR_5 delta: {delta:+,.0f}")

        # Shrinkage between the selection estimate and the held-out one is the size
        # of the winner's curse; report it rather than hiding it.
        shrink = winner["cvar5"] - w["cvar5"]
        print(
            f"    selection CVaR_5 {winner['cvar5']:,.0f} -> held-out {w['cvar5']:,.0f} "
            f"({-shrink:+,.0f} = winner's-curse shrinkage)"
        )

        if len(win_vals) == len(base_vals):
            from local_arena import paired_significance

            sig = paired_significance(win_vals, base_vals)
            print(
                f"    paired mean delta {sig['mean_diff']:+,.0f}  "
                f"winner better on {sig['a_better']}/{sig['n']} seeds  p~{sig['p_approx']}"
            )
            report["holdout_significance"] = sig

        print(
            f"\n  VERDICT: {'PASS' if beats else 'FAIL'} -- winner's held-out CVaR_5 "
            f"{'exceeds' if beats else 'does NOT exceed'} v0.2.4's"
        )
        if not beats:
            print(
                "  Do not ship. Either the pool contains no route more robust than v0.2.4,\n"
                "  or the selection overfit the evaluation seeds. Widen the pool or the seed sets."
            )

        report["holdout"] = {
            "winner_hash": winner["hash"],
            "winner": w,
            "baseline": b,
            "baseline_path": os.path.relpath(baseline_path, PROJECT_ROOT),
            "cvar5_delta": round(delta, 1),
            "winner_beats_baseline": beats,
            "selection_cvar5": winner["cvar5"],
            "shrinkage": round(shrink, 1),
        }

    print(
        SEED_CAVEAT.format(
            mode=args.seed_mode, lo=final_seeds[0], hi=max(final_seeds[-1], holdout_seeds[-1])
        )
    )

    report_path = abspath(args.report)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report["seed_caveat"] = " ".join(SEED_CAVEAT.split())
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  report -> {report_path}")

    if not args.no_holdout and not report["holdout"]["winner_beats_baseline"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
