#!/usr/bin/env python3
"""Phase 3 -- rank finalists by panel win rate and validate on held-out seeds.

    uv run python rank_cvar.py --workers 12
    uv run python rank_cvar.py --no-holdout          # report the table only

RANKING. Finalists are ranked by **mean win rate across the opponent panel**, with
the worst-opponent win rate as the robustness tiebreak. Margin-CVaR5, cash-CVaR5
and mean cash are reported alongside as diagnostics, but they do not select.

Why win rate and not cash. The competition scores a skill rating driven by wins,
and cash is overwhelmingly common-mode: measured on live episodes,
corr(our cash, opponent cash) = +0.86, because both seats draw from one shared
market. Own-cash CVaR therefore mostly measures "was this a good seed" -- a factor
that moves both players together and cancels in the head-to-head that sets the
rating. The first run optimised own-cash CVaR, improved it by a predicted +$1,610,
and delivered -$1,453 live while the win rate (never optimised) carried the gain.

Why a panel and not one opponent. That same run shipped a route that beat its
single opponent 97.0% of the time and won ~47% live. At 97% the metric is also
saturated, so it cannot rank the finalists at all. See mining/panel.py.

WINNER'S-CURSE GUARD. The top pick and the incumbent are both re-run on a FRESH
seed set disjoint from Phase 2, against the same panel. A winner that cannot beat
the incumbent held out is reported as a failure, not shipped.
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
from mining.common import PROJECT_ROOT
from simulate_candidates import (
    AGENT_CACHE,
    DEFAULT_OPPONENT,
    collect,
    complete,
    pairs_on,
    run_job,
)

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

PANEL_CAVEAT = """
  ASSUMPTION -- panel representativeness. The panel is drawn from routes mined out
  of past leaderboard replays, so it represents the field as it PLAYED, not the
  field as it will play. The previous run measured the cost of getting this wrong:
  a route selected against a single opponent won 97% offline and ~47% live. A
  6-opponent panel narrows that gap but does not close it -- the live field has
  hundreds of distinct routes and improves daily. Read the per-opponent breakdown,
  not just the mean: a winner that is strong against five members and weak against
  one is still an exploit, and the live field will contain that one.
"""


def evaluate_on(
    label: str,
    agent_path: str,
    seeds: list[int],
    opponents: list[tuple[str, str]],
    steps: int,
    workers: int,
    results_path: str,
    tag: str,
    alternate_seats: bool,
) -> dict:
    """Run one agent over the full (seed x opponent) grid; return per-opponent pairs."""
    jobs = []
    for opp_label, opp_path in opponents:
        for i, seed in enumerate(seeds):
            jobs.append(
                {
                    "hash": tag,
                    "opp_label": opp_label,
                    "agent": agent_path,
                    "opponent": opp_path,
                    "seed": seed,
                    "steps": steps,
                    "swap": bool(i % 2) if alternate_seats else False,
                    "decision_log": None,
                    "replay": None,
                }
            )
    print(f"    {label}: {len(jobs)} held-out episodes ({len(seeds)}x{len(opponents)})", flush=True)
    t0 = time.time()
    per: dict[str, list[tuple[float, float]]] = {lab: [] for lab, _ in opponents}
    with open(results_path, "a") as sink:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            for res in pool.imap_unordered(run_job, jobs, chunksize=4):
                res["stage"] = "holdout"
                sink.write(json.dumps(res, separators=(",", ":")) + "\n")
                per[res["opp_label"]].append((res["me_cash"], res["opp_cash"]))
    print(f"      done in {(time.time() - t0) / 60:.1f}m", flush=True)
    return per


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Phase 3: panel win-rate ranking + held-out validation"
    )
    ap.add_argument("--candidates", default="candidates.jsonl")
    ap.add_argument("--results", default="logs/simulation_results.jsonl")
    ap.add_argument("--holdout-results", default="logs/holdout_results.jsonl")
    ap.add_argument("--summary", default="logs/simulation_summary.json")
    ap.add_argument("--report", default="logs/cvar_report.json")
    ap.add_argument("--baseline", default=DEFAULT_OPPONENT, help="route to beat (currently live)")
    ap.add_argument("--steps", type=int, default=common.DEFAULT_STEPS)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 3))
    ap.add_argument("--seed-mode", choices=("sequential", "random31"), default="sequential")
    ap.add_argument("--top", type=int, default=12, help="rows to print in the table")
    ap.add_argument("--no-holdout", action="store_true", help="skip the winner's-curse guard")
    ap.add_argument("--holdout-n", type=int, default=common.N_HOLDOUT)
    ap.add_argument("--alternate-seats", action="store_true")
    args = ap.parse_args(argv)

    def abspath(p):
        return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)

    candidates = {c["hash"]: c for c in common.read_jsonl(abspath(args.candidates))}
    results_path = abspath(args.results)
    if not os.path.exists(results_path):
        raise SystemExit(f"no simulation results at {results_path} (run simulate_candidates.py)")
    summary_path = abspath(args.summary)
    if not os.path.exists(summary_path):
        raise SystemExit(f"no Phase 2 summary at {summary_path}")
    with open(summary_path) as f:
        summary = json.load(f)

    labels = summary.get("panel_labels") or []
    if not labels:
        raise SystemExit("Phase 2 summary has no panel; rerun simulate_candidates.py")

    # Rebuild the panel paths: the anchor from the repo, mined members from the cache.
    anchor_label = (summary.get("panel") or {}).get("anchor")
    panel: list[tuple[str, str]] = []
    for lab in labels:
        if lab == anchor_label:
            panel.append((lab, abspath(summary["anchor"])))
        else:
            member = next(
                (m for m in (summary.get("panel") or {}).get("members", []) if m["label"] == lab),
                None,
            )
            if member is None:
                raise SystemExit(f"panel member {lab} missing from summary")
            panel.append((lab, os.path.join(AGENT_CACHE, f"{member['hash']}.py")))
    for lab, path in panel:
        if not os.path.exists(path):
            raise SystemExit(f"panel agent missing: {lab} -> {path}")

    seeds = common.seed_sets(args.seed_mode, n_holdout=args.holdout_n)
    final_seeds = seeds["final"]
    holdout_seeds = seeds["holdout"]
    overlap = set(final_seeds) & set(holdout_seeds)
    if overlap:
        raise SystemExit(f"holdout is not disjoint from evaluation seeds ({len(overlap)} shared)")

    byhash = collect(results_path)
    rows = []
    for h, cand in candidates.items():
        eff = common.effective_labels(h, labels)
        per = pairs_on(byhash, h, final_seeds, eff)
        if not complete(per, final_seeds, eff):
            continue
        rows.append({"hash": h, "candidate": cand, **common.panel_scores(per)})
    if not rows:
        raise SystemExit("no candidate has a complete (seed x panel) record; run the final stage")
    rows.sort(key=common.panel_sort_key, reverse=True)

    print(
        f"Phase 3: {len(rows)} finalists ranked by mean win rate across a "
        f"{len(labels)}-opponent panel on {len(final_seeds)} shared seeds "
        f"({len(final_seeds) * len(labels)} games each)"
    )
    print("  selection = mean win rate; worst-opponent win rate is the robustness tiebreak")
    print(
        f"\n  {'rank':>4}  {'mean win':>9}  {'worst':>7}  {'worst opp':>10}  "
        f"{'margin CVaR5':>13}  {'cash CVaR5':>11}  {'cash mean':>10}  hash / team"
    )
    for i, r in enumerate(rows[: args.top], 1):
        c = r["candidate"]
        print(
            f"  {i:>4}  {r['mean_win']:>9.1%}  {r['worst_win']:>7.1%}  "
            f"{str(r['worst_opponent'])[:10]:>10}  ${r['margin_cvar5']:>+12,.0f}  "
            f"${r['cash_cvar5']:>10,.0f}  ${r['cash_mean']:>9,.0f}  "
            f"{r['hash'][:10]} {str(c['team'])[:16]}"
        )

    # Does the panel discriminate? A saturated top is what produced v0.2.5.
    spread = rows[0]["mean_win"] - rows[min(len(rows) - 1, args.top - 1)]["mean_win"]
    print(f"\n  win-rate spread across the printed finalists: {spread:.1%}")
    if rows[0]["mean_win"] > 0.95:
        print(
            "  !! the leader exceeds 95% against this panel -- the metric is near saturation "
            "and the panel is probably still too narrow"
        )
    if len(rows) > 2:
        rec = [r["candidate"]["recorded_cash"] for r in rows]
        try:
            print(
                f"  correlation(recorded cash, mean win rate) = "
                f"{statistics.correlation(rec, [r['mean_win'] for r in rows]):+.2f}"
            )
        except statistics.StatisticsError:
            pass

    report = {
        "seed_mode": args.seed_mode,
        "panel": labels,
        "evaluation_seeds": [final_seeds[0], final_seeds[-1], len(final_seeds)],
        "holdout_seeds": [holdout_seeds[0], holdout_seeds[-1], len(holdout_seeds)],
        "metric": "mean win rate across opponent panel (tiebreak: worst-opponent win rate)",
        "ranking": [
            {
                "rank": i,
                "hash": r["hash"],
                "team": r["candidate"]["team"],
                "episode": r["candidate"]["episode"],
                "seat": r["candidate"]["seat"],
                "recorded_cash": r["candidate"]["recorded_cash"],
                **{
                    k: r[k]
                    for k in (
                        "mean_win",
                        "worst_win",
                        "worst_opponent",
                        "per_opponent",
                        "margin_cvar5",
                        "cash_cvar5",
                        "cash_mean",
                        "n",
                    )
                },
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
        hpath = abspath(args.holdout_results)
        os.makedirs(os.path.dirname(hpath), exist_ok=True)

        # Both sides must face the SAME opponents or the comparison is not
        # apples-to-apples: drop any panel member that is the winner itself, and the
        # anchor when the anchor IS the incumbent (a mirror it can only draw).
        common_panel = [
            (lab, path)
            for lab, path in panel
            if not winner["hash"].startswith(lab)
            and os.path.abspath(path) != os.path.abspath(baseline_path)
        ]
        if not common_panel:
            raise SystemExit("no opponent is common to both the winner and the incumbent")
        dropped = [lab for lab, _ in panel if (lab, _) not in common_panel]
        if dropped:
            print(
                f"    scoring both sides on the {len(common_panel)} panel members common to "
                f"both (dropped self-matches: {', '.join(dropped)})"
            )

        print(
            f"\n  winner's-curse guard: re-running the top pick and the incumbent on "
            f"{len(holdout_seeds)} FRESH seeds x {len(common_panel)} opponents, disjoint from Phase 2"
        )
        win_per = evaluate_on(
            f"winner {winner['hash'][:10]}",
            agent_path,
            holdout_seeds,
            common_panel,
            args.steps,
            args.workers,
            hpath,
            f"holdout-{winner['hash']}",
            args.alternate_seats,
        )
        base_per = evaluate_on(
            "incumbent",
            baseline_path,
            holdout_seeds,
            common_panel,
            args.steps,
            args.workers,
            hpath,
            "holdout-baseline",
            args.alternate_seats,
        )
        w = common.panel_scores(win_per)
        b = common.panel_scores(base_per)
        delta = w["mean_win"] - b["mean_win"]
        beats = delta > 0

        print(
            f"\n  held-out results on {len(holdout_seeds)} shared seeds "
            f"x {len(common_panel)} common opponents"
        )
        print(f"    {'':<12} {'mean win':>9} {'worst':>7} {'margin CVaR5':>13} {'cash mean':>11}")
        for name, d in (("winner", w), ("incumbent", b)):
            print(
                f"    {name:<12} {d['mean_win']:>9.1%} {d['worst_win']:>7.1%} "
                f"${d['margin_cvar5']:>+12,.0f} ${d['cash_mean']:>10,.0f}"
            )
        print("\n    per-opponent win rate:")
        for lab, _path in common_panel:
            print(
                f"      {lab:<12} winner {w['per_opponent'].get(lab, float('nan')):>6.1%}   "
                f"incumbent {b['per_opponent'].get(lab, float('nan')):>6.1%}"
            )
        print(f"\n    held-out mean win-rate delta: {delta:+.1%}")
        # A tie at the ceiling is not evidence the winner is no better -- it means
        # the common panel cannot tell them apart. Say so rather than reporting a
        # bare FAIL that reads like a measured loss.
        if abs(delta) < 1e-9 and w["mean_win"] > 0.95 and b["mean_win"] > 0.95:
            print(
                "    !! both sides are at the ceiling against every common opponent: this\n"
                "       panel cannot discriminate them. Widen the panel before deciding."
            )
        shrink = winner["mean_win"] - w["mean_win"]
        print(
            f"    selection {winner['mean_win']:.1%} -> held-out {w['mean_win']:.1%} "
            f"({-shrink:+.1%} = winner's-curse shrinkage)"
        )
        print(
            f"\n  VERDICT: {'PASS' if beats else 'FAIL'} -- winner's held-out mean win rate "
            f"{'exceeds' if beats else 'does NOT exceed'} the incumbent's"
        )
        if not beats:
            print(
                "  Do not ship. Either the pool contains no route stronger than the incumbent\n"
                "  against this panel, or the selection overfit the evaluation seeds."
            )
        report["holdout"] = {
            "winner_hash": winner["hash"],
            "winner": w,
            "baseline": b,
            "baseline_path": os.path.relpath(baseline_path, PROJECT_ROOT),
            "win_delta": round(delta, 4),
            "winner_beats_baseline": beats,
            "selection_mean_win": winner["mean_win"],
            "shrinkage": round(shrink, 4),
        }

    print(
        SEED_CAVEAT.format(
            mode=args.seed_mode, lo=final_seeds[0], hi=max(final_seeds[-1], holdout_seeds[-1])
        )
    )
    print(PANEL_CAVEAT)

    report_path = abspath(args.report)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report["seed_caveat"] = " ".join(SEED_CAVEAT.split())
    report["panel_caveat"] = " ".join(PANEL_CAVEAT.split())
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  report -> {report_path}")

    if not args.no_holdout and not report["holdout"]["winner_beats_baseline"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
