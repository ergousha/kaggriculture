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

WHERE THE PANEL COMES FROM (`--panel-source`, changed in v0.2.7). The v0.2.6 run
drew panel members from the top `--panel-from-top` *screen* performers, and the
screen ranks by win rate against the incumbent anchor -- so every member beat the
incumbent ~100% by construction, and the panel was really "the routes that counter
us". Worse, the pool it drew from is whoever happened to appear in the daily replay
dumps: a sample of the whole field weighted by episode volume, not by strength. We
are matched by rating, so that optimised win rate against the *median* of the field
while the ladder pays for beating the band above us.

`--panel-source leaderboard-top` instead draws members from candidates whose team
sits in the rank window `--panel-rank-min .. --panel-team-top`, ordered by that
rank. Selection within the eligible set is unchanged -- the same greedy max-min
action-distance diversity and distinct-team bias from `mining/panel.py` -- and the
incumbent is still the anchor, so results stay comparable. `screen-top` reproduces
v0.2.6.

It is a WINDOW, not a top-N, and that is a measured choice. Issue #22 scored the
opponents the top 5 teams actually played and found rank-5 `peikopon` (2987) never
matches anyone above 3000: isolation into the >3000 pool is a consequence of
crossing 3000, not the way up. Selecting against the leaders would optimise for
games we are not matched into. `--panel-rank-min` excludes them, so the panel is
the band around and modestly above our own rating, and it advances as we climb.

Requires `team_rank` on every candidate, which Phase 1 stamps from
`scripts/fetch_team_ranks.py`. A pool mined before that existed will be rejected
rather than silently falling back.

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
from mining.common import PROJECT_ROOT, decode_route_b85

AGENT_CACHE = os.path.join(PROJECT_ROOT, "logs", "_mined_agents")
# The incumbent is what we must beat, and it is what is live on the ladder.
DEFAULT_OPPONENT = os.path.join("opponents", "v0_2_6.py")
STAGES = ("screen", "mid", "final")
PANEL_SOURCES = ("screen-top", "leaderboard-top")


def run_job(job: dict) -> dict:
    """One episode via local_arena's own runner, tagged with the candidate hash."""
    from local_arena import run_episode

    res = run_episode(job)
    return {
        "hash": job["hash"],
        "opp_label": job["opp_label"],
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


def load_done(path: str) -> set[tuple[str, str, int]]:
    """(hash, opponent, seed) triples already simulated, for --resume."""
    done: set[tuple[str, str, int]] = set()
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
                done.add((row["hash"], row.get("opp_label") or "v0_2_4", int(row["seed"])))
    return done


def run_stage(
    stage: str,
    candidates: list[dict],
    seeds: list[int],
    agent_paths: dict[str, str],
    opponents: list[tuple[str, str]],
    steps: int,
    workers: int,
    results_path: str,
    done: set[tuple[str, str, int]],
    alternate_seats: bool,
) -> dict:
    """Run every (candidate, opponent, seed) triple not already done.

    `opponents` is a list of (label, path). A candidate's sample size is
    len(seeds) * len(opponents), and every candidate sees the *identical* grid --
    common random numbers now span both axes, so two candidates differ only in
    their route, never in which markets or which opponents they faced.
    """
    import json

    jobs = []
    for cand in candidates:
        h = cand["hash"]
        for label, path in opponents:
            for i, seed in enumerate(seeds):
                if (h, label, seed) in done:
                    continue
                jobs.append(
                    {
                        "hash": h,
                        "opp_label": label,
                        "agent": agent_paths[h],
                        "opponent": path,
                        "seed": seed,
                        "steps": steps,
                        # Fixed seat 0 by default. Seats are provably symmetric here,
                        # and a constant seat keeps every pairing identical.
                        "swap": bool(i % 2) if alternate_seats else False,
                        "decision_log": None,
                        "replay": None,
                    }
                )

    total = len(candidates) * len(seeds) * len(opponents)
    print(
        f"\n  stage {stage:<6} {len(candidates):>5,} candidates x {len(seeds):>3} seeds"
        f" x {len(opponents)} opponents  -> {len(jobs):,} new episodes"
        f" ({total - len(jobs):,} reused)"
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
                    # Nested seed sets and a panel that contains the screen opponent
                    # mean the next stage must reuse these, not repay for them.
                    done.add((res["hash"], res["opp_label"], int(res["seed"])))
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
            f"        done {n:,} episodes in {wall / 60:.1f}m "
            f"({n / max(wall, 1e-9):.1f} ep/s), bad {bad}"
        )
    if bad:
        print(f"        !! {bad} episodes had a crash / invalid status / harness error")
    return collect(results_path)


def collect(results_path: str) -> dict:
    """{hash: {opp_label: {seed: (me_cash, opp_cash)}}}."""
    import json

    out: dict = {}
    if not os.path.exists(results_path):
        return out
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            label = row.get("opp_label") or "v0_2_4"
            out.setdefault(row["hash"], {}).setdefault(label, {})[int(row["seed"])] = (
                float(row["me_cash"]),
                float(row["opp_cash"]),
            )
    return out


def pairs_on(byhash: dict, h: str, seeds: list[int], labels: list[str]) -> dict:
    """Per-opponent (me, opp) pairs on exactly `seeds`. A candidate missing any cell
    of the grid is reported short, so it is dropped rather than ranked on a
    favourable subset."""
    got = byhash.get(h, {})
    out = {}
    for label in labels:
        d = got.get(label, {})
        out[label] = [d[s] for s in seeds if s in d]
    return out


def complete(per_opp: dict, seeds: list[int], labels: list[str]) -> bool:
    return all(len(per_opp.get(label, [])) == len(seeds) for label in labels)


def in_band(candidate: dict, rank_min: int, rank_max: int) -> bool:
    """Is this candidate's team inside the panel's rating window?

    An unranked team (left the board, or renamed since the snapshot) is *unknown*,
    not top-ranked -- `None` must never sort to the front of the ladder, so it is
    mapped past every real rank rather than to zero.
    """
    rank = candidate.get("team_rank")
    return rank is not None and rank_min <= rank <= rank_max


def describe_band(rank_min: int, rank_max: int) -> str:
    """`ranks 185..469 (2300.4..2600.1)`, when the rank map is available.

    Ranks are what `candidates.jsonl` carries, but every argument about *which*
    band to aim at is made in rating points (see issue #22), so print both.
    """
    window = f"ranks {rank_min}..{rank_max}"
    ranks, _ = common.load_team_ranks()
    if not ranks:
        return window
    scores: dict[int, float] = {}
    import json as _json

    with open(common.TEAM_RANKS_PATH, encoding="utf-8") as f:
        for meta in (_json.load(f).get("teams") or {}).values():
            scores[int(meta["rank"])] = float(meta["score"])
    inside = [s for r, s in scores.items() if rank_min <= r <= rank_max]
    if not inside:
        return window
    return f"{window} ({min(inside):.1f}..{max(inside):.1f} rating)"


def check_panel_source(
    candidates: list[dict], source: str, rank_min: int, rank_max: int, panel_size: int
) -> None:
    """Fail before the screen, not after it.

    The screen is ~85% of a seven-hour run, and every reason a leaderboard panel
    cannot be built is knowable from `candidates.jsonl` alone. Discovering a pool
    with no `team_rank` after the screen finishes wastes six hours.
    """
    if source != "leaderboard-top":
        return
    if not any("team_rank" in c for c in candidates):
        raise SystemExit(
            "--panel-source leaderboard-top needs `team_rank` on the candidates, and this "
            "pool has none.\nRun scripts/fetch_team_ranks.py --refresh, then re-run "
            "mine_replays.py (Phase 1) to stamp it in."
        )
    if rank_min > rank_max:
        raise SystemExit(f"--panel-rank-min {rank_min} is worse than --panel-team-top {rank_max}")
    band = [c for c in candidates if in_band(c, rank_min, rank_max)]
    teams = {c["team"] for c in band}
    if len(teams) < panel_size - 1:
        raise SystemExit(
            f"only {len(teams)} distinct teams in the corpus sit in {describe_band(rank_min, rank_max)}, "
            f"but the panel needs {panel_size - 1} mined members.\nWiden the band "
            "(--panel-rank-min / --panel-team-top) or lower --panel-size."
        )
    print(
        f"  panel source: leaderboard-top, {describe_band(rank_min, rank_max)} "
        f"-> {len(band):,} eligible candidates from {len(teams)} teams"
    )


def build_panel_entries(ranked: list[tuple[dict, dict]], args) -> list[dict]:
    """Ordered pool that `select_panel` seeds and greedily extends.

    `ranked` is every screened candidate, strongest-first by win rate against the
    anchor. The two sources differ only in which entries are eligible and in what
    "strongest" means for the seed pick:

      screen-top       top N screen performers, ordered by win rate vs the anchor.
      leaderboard-top  candidates from teams ranked <= N, ordered by that rank;
                       ties broken by screen win rate, which only chooses *which*
                       of one team's routes to use and so cannot re-introduce the
                       anti-incumbent bias across the panel.
    """
    if args.panel_source == "screen-top":
        pool = list(ranked[: args.panel_from_top])
    else:
        pool = [
            (sc, cand)
            for sc, cand in ranked
            if in_band(cand, args.panel_rank_min, args.panel_team_top)
        ]
        pool.sort(key=lambda r: (r[1]["team_rank"], -r[0]["mean_win"]))
    return [
        {
            "hash": cand["hash"],
            "team": cand.get("team"),
            "rank": cand.get("team_rank"),
            "route": common.decode_route_b85(cand["route_b85"]),
            "win": sc["mean_win"],
        }
        for sc, cand in pool
    ]


PANEL_MIN_SPREAD = 0.30


def _warn_on_panel_quality(panelmod, picked: list[dict], panel_size: int) -> None:
    """Print the two ways a panel silently stops being a panel.

    The runbook gates on both by eye; printing them means a `grep '!!'` over the
    log finds them, and a resumed run that quietly picked fewer members than asked
    is not mistaken for a full one.
    """
    if len(picked) < panel_size - 1:
        print(
            f"        !! panel is short: asked for {panel_size - 1} mined members, "
            f"got {len(picked)}. The eligible pool ran out of routes far enough apart."
        )
    tight = [
        (p["hash"][:10], d)
        for p, d in zip(picked, panelmod.min_distances(picked), strict=True)
        if d < PANEL_MIN_SPREAD
    ]
    if tight:
        print(
            f"        !! {len(tight)} panel member(s) sit closer than "
            f"{PANEL_MIN_SPREAD:.2f} to an earlier member "
            f"({', '.join(f'{h} {d:.2f}' for h, d in tight)}); the panel is narrower "
            "than it looks."
        )
    teams = {p.get("team") for p in picked}
    if len(teams) < len(picked):
        print(f"        !! {len(picked)} mined members span only {len(teams)} distinct teams")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase 2: simulate candidate routes")
    ap.add_argument("--candidates", default="candidates.jsonl")
    ap.add_argument("--results", default="logs/simulation_results.jsonl")
    ap.add_argument("--opponent", default=DEFAULT_OPPONENT, help="screen opponent / panel anchor")
    ap.add_argument("--steps", type=int, default=common.DEFAULT_STEPS)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 3))
    ap.add_argument("--k-mid", type=int, default=150, help="candidates promoted to the panel run")
    ap.add_argument("--k-final", type=int, default=12, help="finalists for the full-seed run")
    ap.add_argument(
        "--panel-size", type=int, default=6, help="opponents in the panel (incl. anchor)"
    )
    ap.add_argument(
        "--panel-source",
        choices=PANEL_SOURCES,
        default="leaderboard-top",
        help="where panel members are drawn from (screen-top reproduces v0.2.6)",
    )
    ap.add_argument(
        "--panel-from-top",
        type=int,
        default=120,
        help="screen-top only: draw panel members from the top N screen performers",
    )
    ap.add_argument(
        "--panel-team-top",
        type=int,
        default=30,
        help="leaderboard-top only: draw panel members from teams ranked <= N",
    )
    ap.add_argument(
        "--panel-rank-min",
        type=int,
        default=1,
        help=(
            "leaderboard-top only: exclude teams ranked better than N, making the panel a "
            "rating BAND rather than the top of the ladder (see issue #22)"
        ),
    )
    ap.add_argument(
        "--panel-min-distance",
        type=float,
        default=0.15,
        help="minimum per-step action distance between panel members",
    )
    ap.add_argument("--seed-mode", choices=("sequential", "random31"), default="sequential")
    ap.add_argument("--stage", choices=STAGES, help="run only this stage")
    ap.add_argument("--resume", action="store_true", help="skip triples already done")
    ap.add_argument("--limit", type=int, default=0, help="cap candidates (smoke tests)")
    ap.add_argument(
        "--alternate-seats",
        action="store_true",
        help="alternate seats across seeds instead of pinning the candidate to seat 0",
    )
    args = ap.parse_args(argv)

    def abspath(p):
        return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)

    cand_path = abspath(args.candidates)
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

    anchor = abspath(args.opponent)
    if not os.path.exists(anchor):
        raise SystemExit(f"opponent not found: {anchor}")

    check_panel_source(
        candidates,
        args.panel_source,
        args.panel_rank_min,
        args.panel_team_top,
        args.panel_size,
    )

    results_path = abspath(args.results)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    if not args.resume and os.path.exists(results_path):
        os.rename(results_path, results_path + ".bak")
        print(f"  moved previous results to {os.path.basename(results_path)}.bak")

    seeds = common.seed_sets(args.seed_mode)
    print(f"Phase 2: {len(candidates):,} candidates")
    print(f"  workers {args.workers}   steps {args.steps}   seed mode {args.seed_mode}")
    print(
        f"  CRN seed sets: screen {len(seeds['screen'])}, mid {len(seeds['mid'])}, "
        f"final {len(seeds['final'])} (nested); holdout {len(seeds['holdout'])} (disjoint)"
    )
    print(f"  seat: {'alternating' if args.alternate_seats else 'candidate pinned to seat 0'}")

    agent_paths, written = materialize_agents(candidates, AGENT_CACHE)
    print(f"  materialized {written:,} new agent files (cache {AGENT_CACHE})")

    done = load_done(results_path) if args.resume else set()
    if done:
        print(f"  resume: {len(done):,} (candidate, opponent, seed) triples already simulated")

    anchor_label = os.path.splitext(os.path.basename(anchor))[0]
    screen_panel = [(anchor_label, anchor)]
    panel = screen_panel

    stages_to_run = [args.stage] if args.stage else list(STAGES)
    pool = candidates
    summary = {}
    panel_meta = None

    for stage in stages_to_run:
        seed_list = seeds[stage]
        opponents = screen_panel if stage == "screen" else panel
        labels = [lab for lab, _ in opponents]

        byhash = run_stage(
            stage,
            pool,
            seed_list,
            agent_paths,
            opponents,
            args.steps,
            args.workers,
            results_path,
            done,
            args.alternate_seats,
        )

        ranked = []
        for cand in pool:
            eff = common.effective_labels(cand["hash"], labels)
            per = pairs_on(byhash, cand["hash"], seed_list, eff)
            if not complete(per, seed_list, eff):
                continue
            ranked.append((common.panel_scores(per), cand))
        ranked.sort(key=lambda r: common.panel_sort_key(r[0]), reverse=True)
        if not ranked:
            print("        no candidate completed this stage; stopping")
            break

        print(
            f"        top 5 by mean win rate vs {len(labels)} opponent(s) "
            f"on {len(seed_list)} shared seeds:"
        )
        for sc, cand in ranked[:5]:
            print(
                f"          win {sc['mean_win']:>6.1%}  worst {sc['worst_win']:>6.1%}  "
                f"margin-CVaR5 ${sc['margin_cvar5']:>+9,.0f}  cash ${sc['cash_mean']:>9,.0f}  "
                f"{cand['hash'][:10]} {str(cand['team'])[:14]}"
            )
        # Saturation is the failure that produced the shipped v0.2.5: at a 97% win
        # rate the metric cannot separate the leaders, so the pick is arbitrary.
        top_rates = [sc["mean_win"] for sc, _ in ranked[: max(2, args.k_final)]]
        if top_rates and min(top_rates) > 0.95:
            print(
                f"        !! SATURATED: the top {len(top_rates)} candidates all exceed 95% "
                "win rate; this metric cannot discriminate between them. Widen the panel."
            )

        summary[stage] = {
            "seeds": len(seed_list),
            "candidates": len(pool),
            "ranked": len(ranked),
            "opponents": labels,
            "top": [{"hash": c["hash"], "team": c.get("team"), **sc} for sc, c in ranked[:10]],
        }

        if stage == "screen":
            pool = [c for _, c in ranked[: args.k_mid]]
            # Build the panel from structurally distinct routes -- never from
            # recorded cash, which is uncorrelated with strength. `--panel-source`
            # decides whether "distinct and strong" is read off the screen (v0.2.6)
            # or off the ladder (v0.2.7); see the module docstring.
            from mining import panel as panelmod

            entries = build_panel_entries(ranked, args)
            picked = panelmod.select_panel(
                entries, max(1, args.panel_size - 1), args.panel_min_distance
            )
            panel = [(anchor_label, anchor)] + [
                (p["hash"][:10], agent_paths[p["hash"]]) for p in picked
            ]
            panel_meta = {
                "anchor": anchor_label,
                "source": args.panel_source,
                "team_top": args.panel_team_top if args.panel_source == "leaderboard-top" else None,
                "rank_min": args.panel_rank_min if args.panel_source == "leaderboard-top" else None,
                "band": (
                    describe_band(args.panel_rank_min, args.panel_team_top)
                    if args.panel_source == "leaderboard-top"
                    else None
                ),
                "from_top": args.panel_from_top if args.panel_source == "screen-top" else None,
                "eligible": len(entries),
                "members": [
                    {
                        "label": p["hash"][:10],
                        "hash": p["hash"],
                        "team": p.get("team"),
                        "team_rank": p.get("rank"),
                        "screen_win": round(p["win"], 4),
                    }
                    for p in picked
                ],
            }
            print(
                f"\n  opponent panel ({len(panel)} members, anchor + {len(picked)} mined, "
                f"source {args.panel_source}, {len(entries):,} eligible):"
            )
            print(f"    1. {anchor_label:<10}  {'(anchor, incumbent)':<18}")
            print(panelmod.describe_panel(picked, start=2))
            _warn_on_panel_quality(panelmod, picked, args.panel_size)
        elif stage == "mid":
            pool = [c for _, c in ranked[: args.k_final]]

    import json

    summary_path = os.path.join(PROJECT_ROOT, "logs", "simulation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "anchor": os.path.relpath(anchor, PROJECT_ROOT),
                "panel": panel_meta,
                "panel_source": args.panel_source,
                "panel_labels": [lab for lab, _ in panel],
                "seed_mode": args.seed_mode,
                "seed_sets": {k: [v[0], v[-1], len(v)] for k, v in seeds.items()},
                "alternate_seats": args.alternate_seats,
                "metric": "mean win rate across panel; worst-opponent win rate as tiebreak",
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
