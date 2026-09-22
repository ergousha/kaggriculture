#!/usr/bin/env python3
"""Paired live margin per version, bucketed by the OPPONENT's leaderboard rating.

    uv run python scripts/audit_opponent_bands.py
    uv run python scripts/audit_opponent_bands.py --versions v0.4.1 v0.5.1
    uv run python scripts/audit_opponent_bands.py --offline   # use the cached pull

WHY THIS EXISTS. `public_score` cannot rank two versions. It is a skill rating that
climbs while a submission plays and freezes when a newer submission evicts it, so it
reports *maturity* at least as much as strength. The repo has measured a 633.7-point
spread on byte-identical code (docs/finding-2026-09-21-live-score-not-comparable.md),
and the effect is visible in one line: v0.5.0 read 1772.9 while winning 54.4% of its
episodes, and byte-identical v0.4.3 read 1785.8 while winning 38.5%. The higher score
belonged to the version that was losing.

What IS comparable is the paired cash margin against opposition of a known strength.
`api.competition_list_episodes(submission_id)` returns, per episode, each agent's
`reward` and `team_id` -- no replay download, so a 600-episode version costs one API
walk instead of 18 GB. Joining `team_id` to the public leaderboard CSV gives the
opponent's rating, and bucketing by it removes the confound that a mature submission
has been matched into a harder pool than a fresh one.

The band that matters is 2000+. Everything in this repo's lineage beats sub-1600
opposition and is roughly at parity in its own 1600-2000 band; the leaderboard is
decided by whether a version can hold its own against the teams above it.

CONFOUNDS, stated because they are not resolved:
  * the rating is TODAY's leaderboard snapshot, not the rating the opponent held when
    the episode was played. A team that has since climbed or fallen is misfiled.
  * versions played on different calendar days against a field that grows daily.
  * this is an observational read of live play, not a randomised A/B. Use it to reject
    a change, and to rank candidates -- not to claim a calibrated effect size.
Submissions with the same runtime hash are pooled, because they are the same agent.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
CACHE = HERE / "logs" / "opponent_bands.json"
BANDS = ((0, 1200), (1200, 1600), (1600, 2000), (2000, 2400), (2400, 10**6))
DEFAULT_VERSIONS = ("v0.4.0", "v0.4.1", "v0.4.3", "v0.5.0", "v0.5.1", "v0.5.2")


def leaderboard_ratings() -> dict[str, float]:
    """team_id -> score from the newest leaderboard CSV in scratch/ or logs/."""
    paths = glob.glob(str(HERE / "scratch" / "*publicleaderboard*.csv")) + glob.glob(
        str(HERE / "logs" / "*publicleaderboard*.csv")
    )
    if not paths:
        raise SystemExit(
            "no leaderboard CSV found; run "
            "`uv run kaggle competitions leaderboard kaggriculture --download -p scratch` first"
        )
    newest = max(paths, key=os.path.getmtime)
    print(f"leaderboard: {os.path.basename(newest)}")
    with open(newest, newline="") as handle:
        return {str(r["TeamId"]): float(r["Score"]) for r in csv.DictReader(handle)}


def _reward(agent) -> float:
    raw = agent.get("reward") if isinstance(agent, dict) else getattr(agent, "reward", None)
    return 0.0 if raw is None else float(raw)


def _ident(agent, attr: str, key: str):
    """kagglesdk defaults these ids to 0, not None; treat 0 as unset."""
    value = agent.get(key) if isinstance(agent, dict) else getattr(agent, attr, None)
    return value if value else None


def pull(versions: list[str]) -> dict[str, list[dict]]:
    from submit import load_credentials

    load_credentials()
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    subs = list(api.competition_submissions("kaggriculture", page_size=100))
    out: dict[str, list[dict]] = {}
    for version in versions:
        pattern = re.compile(rf"(?<![\w.]){re.escape(version)}(?![\w.])")
        matches = [
            s for s in subs if getattr(s, "description", None) and pattern.search(s.description)
        ]
        records: list[dict] = []
        for sub in matches:
            sub_id = int(sub.ref)
            for episode in api.competition_list_episodes(sub_id) or []:
                if "COMPLETED" not in str(getattr(episode, "state", "")):
                    continue
                agents = (
                    getattr(episode, "agents", None)
                    or (episode.get("agents") if isinstance(episode, dict) else None)
                    or getattr(episode, "_agents", [])
                )
                mine = opp = None
                opp_team = None
                for agent in agents:
                    if str(_ident(agent, "submission_id", "submissionId")) == str(sub_id):
                        mine = _reward(agent)
                    else:
                        opp = _reward(agent)
                        opp_team = str(_ident(agent, "team_id", "teamId") or "")
                if mine is None or opp is None:
                    continue
                records.append({"me": mine, "opp": opp, "team": opp_team})
        out[version] = records
        print(f"  {version}: {len(matches)} submission(s), {len(records)} scored episodes")
    return out


def report(data: dict[str, list[dict]], ratings: dict[str, float]) -> None:
    for low, high in BANDS:
        rows = []
        for version, records in data.items():
            margins = [
                r["me"] - r["opp"]
                for r in records
                if r["team"] in ratings and low <= ratings[r["team"]] < high
            ]
            if len(margins) >= 5:
                se = st.pstdev(margins) / math.sqrt(len(margins)) or float("inf")
                rows.append(
                    (
                        version,
                        len(margins),
                        100 * sum(1 for m in margins if m > 0) / len(margins),
                        st.fmean(margins),
                        st.fmean(margins) / se,
                    )
                )
        if not rows:
            continue
        label = f"{low}-{high}" if high < 10**6 else f"{low}+"
        star = "   <-- the band that decides the leaderboard" if low == 2000 else ""
        print(f"\nOPPONENTS RATED {label}{star}")
        print(f"  {'version':10s} {'n':>5s} {'win rate':>9s} {'margin':>11s} {'t':>7s}")
        for version, n, wr, margin, t in rows:
            print(f"  {version:10s} {n:>5d} {wr:>8.1f}% {margin:>+11,.0f} {t:>+7.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions", nargs="*", default=list(DEFAULT_VERSIONS))
    parser.add_argument("--offline", action="store_true", help="reuse logs/opponent_bands.json")
    args = parser.parse_args()

    if args.offline:
        if not CACHE.exists():
            raise SystemExit("no cache; run once without --offline")
        data = json.loads(CACHE.read_text())
    else:
        data = pull(args.versions)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(data, indent=1))

    report(data, leaderboard_ratings())
    print(
        "\nRead this as a veto, not as an effect size: the opponent rating is a "
        "present-day\nsnapshot and the versions did not play the same field on the same day."
    )


if __name__ == "__main__":
    main()
