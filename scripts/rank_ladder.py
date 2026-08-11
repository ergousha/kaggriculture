#!/usr/bin/env python3
"""Rank an agent against the Kaggriculture reference ladder.

This exists because every other gate in this repo has failed to predict the live
score. `--opponent baseline` measures survival against a bot that barely produces.
Paired seeds against our own frozen previous version catch regressions but share
our blind spots -- v0.1.0 won that gate 23W-7L and moved the live rating by
nothing. A downloaded leaderboard replay is open-loop and goes bankrupt when the
market diverges. What was missing was a *calibrated* opponent set: a spread of
agents whose relative strength is already known and documented, so a result places
us on a scale instead of just comparing us to ourselves.

Rungs 0-5 (`opponents/ladder/`) are Rayk Kretzschmar's reference ladder, MIT, and
they share a byte-identical scheduler -- the only difference between them is a
`POLICY` dict, so a gap between rungs is an economic decision and nothing else.
Rungs 6-9 embed the shared public meta line and are deliberately NOT vendored; the
dataset's own NOTICE asks that submissions be built on 0-5. They are still the most
informative opponents available, so this script picks them up from
`reference/ladder/` when that directory exists (see --fetch), uses them for
measurement only, and never treats them as a base for our own agent.

    uv run python scripts/rank_ladder.py --fetch            # download the dataset
    uv run python scripts/rank_ladder.py --episodes 6       # rank main.py
    uv run python scripts/rank_ladder.py --agent opponents/v0_1_0.py --tiers 0-5

Every rung is played on the same seed set and on **both seats** (common random
numbers), so a difference between rungs is strategy and not luck. That matters more
here than usual: an episode's shop unlocks are drawn at random and largely decide
how much cash both players make, so unpaired cash comparisons are mostly noise.
See docs/experiments.md.

Attribution: the ladder, its measured league table and the tier commentary are from
https://www.kaggle.com/datasets/raykkretzschmar/kaggriculture-reference-agents and
the notebook https://www.kaggle.com/code/raykkretzschmar/kaggriculture-rank-your-agent
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)

LADDER_DIR = os.path.join(PROJECT_ROOT, "opponents", "ladder")
REFERENCE_DIR = os.path.join(PROJECT_ROOT, "reference", "ladder")
DATASET = "raykkretzschmar/kaggriculture-reference-agents"

# tier -> (slug, headline). Tiers 6-9 are measurement-only and are read from
# reference/ladder/ if present; see the module docstring.
RUNGS = [
    (0, "fallow_finn", "Never plants anything. The reward floor."),
    (1, "wheat_walter", "One farmer, wheat only, harvests too early."),
    (2, "rotation_rosa", "Hires help and runs a three-crop rotation."),
    (3, "homestead_hana", "Buys one quadrant and scales staples with a real crew."),
    (4, "melon_mateo", "Farms the premium crop and refuses to dump it."),
    (5, "rancher_rita", "Livestock at scale, on a wheat feed chain."),
    (6, "broker_bea", "Meta field plan with opportunistic wheat timing."),
    (7, "ledger_lena", "Meta field plan, different sell-ordering trade-off."),
    (8, "slotter_silas", "Meta field plan with a reordered SELL layer."),
    (9, "closer_cleo", "Meta field plan, sells reordered in place."),
]


def fetch_dataset() -> None:
    """Download the reference-agent dataset into reference/ladder/."""
    from submit import load_credentials

    load_credentials()
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    os.makedirs(REFERENCE_DIR, exist_ok=True)
    print(f"downloading {DATASET} -> {REFERENCE_DIR}")
    api.dataset_download_files(DATASET, path=REFERENCE_DIR, unzip=True, quiet=False)


def resolve_rung(slug: str) -> str | None:
    """Vendored copy first, then the downloaded dataset."""
    for directory in (LADDER_DIR, REFERENCE_DIR):
        path = os.path.join(directory, f"{slug}.py")
        if os.path.exists(path):
            return path
    return None


def parse_tiers(spec: str) -> set[int]:
    wanted: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            wanted.update(range(int(lo), int(hi) + 1))
        else:
            wanted.add(int(part))
    return wanted


def play(agent_path: str, opponent_path: str, seed: int, swap: bool, steps: int) -> dict:
    """One match. Returns our cash, theirs, and our status."""
    from kaggle_environments import make

    # actTimeout 2.0, not the competition's 1.0: this runs many episodes in a row on
    # a laptop and a timeout here would be a measurement artefact, not a real fault.
    # submit.py's pre-flight is what gates real turn compute.
    env = make(
        "kaggriculture",
        configuration={"episodeSteps": steps, "seed": seed, "actTimeout": 2.0},
        debug=False,
    )
    agents = [opponent_path, agent_path] if swap else [agent_path, opponent_path]
    me = 1 if swap else 0
    try:
        env.run(agents)
        final = env.steps[-1]
        rewards = [s.get("reward", 0.0) or 0.0 for s in final]
        statuses = [s.get("status", "DONE") for s in final]
        return {
            "me": float(rewards[me]),
            "opp": float(rewards[1 - me]),
            "status": statuses[me],
            "seed": seed,
            "swap": swap,
        }
    except Exception as exc:  # noqa: BLE001 - one bad match must not kill the sweep
        return {"me": 0.0, "opp": 0.0, "status": f"ERROR: {exc}", "seed": seed, "swap": swap}


def load_reference_league() -> list[dict]:
    """The dataset's own measured pairwise results, for context in the report."""
    for directory in (LADDER_DIR, REFERENCE_DIR):
        path = os.path.join(directory, "baseline_league.csv")
        if os.path.exists(path):
            with open(path, newline="") as f:
                return list(csv.DictReader(f))
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank an agent against the reference ladder.")
    parser.add_argument("--agent", default="main.py")
    parser.add_argument("--episodes", type=int, default=6, help="matches per rung (seats split)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument("--tiers", default="0-9", help="e.g. 0-5 or 4,5,6")
    parser.add_argument("--fetch", action="store_true", help="download the dataset and exit")
    args = parser.parse_args()

    if args.fetch:
        fetch_dataset()
        return

    agent_path = os.path.abspath(args.agent)
    if not os.path.exists(agent_path):
        raise SystemExit(f"agent not found: {agent_path}")

    wanted = parse_tiers(args.tiers)
    rungs = []
    for tier, slug, headline in RUNGS:
        if tier not in wanted:
            continue
        path = resolve_rung(slug)
        if path is None:
            print(
                f"  tier {tier} {slug}: not available (run --fetch for tiers 6-9)", file=sys.stderr
            )
            continue
        rungs.append((tier, slug, headline, path))

    if not rungs:
        raise SystemExit("no rungs available; try --fetch")

    print(
        f"ranking {os.path.basename(agent_path)} over {len(rungs)} rungs, "
        f"{args.episodes} episodes each, seats alternated"
    )

    results = []
    for tier, slug, headline, path in rungs:
        rows = []
        for i in range(args.episodes):
            rows.append(play(agent_path, path, args.seed + i, bool(i % 2), args.steps))
        wins = sum(1 for r in rows if r["me"] > r["opp"])
        losses = sum(1 for r in rows if r["me"] < r["opp"])
        ties = len(rows) - wins - losses
        errors = sum(1 for r in rows if str(r["status"]).startswith("ERROR"))
        margin = statistics.mean(r["me"] - r["opp"] for r in rows)
        results.append(
            {
                "tier": tier,
                "slug": slug,
                "headline": headline,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "errors": errors,
                "mean_cash": statistics.mean(r["me"] for r in rows),
                "opp_cash": statistics.mean(r["opp"] for r in rows),
                "margin": margin,
            }
        )
        flag = "  <-- ERRORS" if errors else ""
        print(
            f"  tier {tier} {slug:<16} {wins}W-{losses}L-{ties}T  "
            f"ours ${results[-1]['mean_cash']:>9,.0f}  theirs ${results[-1]['opp_cash']:>9,.0f}  "
            f"margin ${margin:>+10,.0f}{flag}"
        )

    print(f"\n{'=' * 78}")
    beaten = [r for r in results if r["wins"] > r["losses"]]
    lost_to = [r for r in results if r["wins"] < r["losses"]]
    highest_beaten = max((r["tier"] for r in beaten), default=None)
    lowest_lost = min((r["tier"] for r in lost_to), default=None)

    # Phrase the placement relative to the tiers actually played, not to the whole
    # ladder: `--tiers 6-9` losing everything means "below tier 6", not "below tier 0".
    played = sorted(r["tier"] for r in results)
    if highest_beaten is None:
        print(
            f"RUNG: below tier {played[0]} — lost to every rung played ({played[0]}-{played[-1]})."
        )
    elif lowest_lost is None:
        print(f"RUNG: at or above tier {highest_beaten} — beat every rung played.")
    else:
        print(f"RUNG: between tier {highest_beaten} and tier {lowest_lost}.")
    total_w = sum(r["wins"] for r in results)
    total_g = sum(r["wins"] + r["losses"] + r["ties"] for r in results)
    print(f"overall {total_w}/{total_g} wins across the ladder")
    if any(r["errors"] for r in results):
        print("WARNING: some matches errored; treat those rungs as unmeasured")

    league = load_reference_league()
    if league and highest_beaten is not None and lowest_lost is not None:
        gap = [
            row
            for row in league
            if {row["agent_a"], row["agent_b"]}
            == {
                next(r["slug"] for r in results if r["tier"] == highest_beaten),
                next(r["slug"] for r in results if r["tier"] == lowest_lost),
            }
        ]
        if gap:
            row = gap[0]
            print(
                f"reference league for that step: {row['agent_a']} vs {row['agent_b']} "
                f"{row['wins_a']}-{row['wins_b']}, mean margin {float(row['mean_margin_a']):+,.0f}"
            )


if __name__ == "__main__":
    main()
