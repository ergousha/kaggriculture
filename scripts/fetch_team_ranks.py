#!/usr/bin/env python3
"""Persist a `team name -> ladder rank` map for the mining pipeline.

    uv run python scripts/fetch_team_ranks.py            # reuse the newest snapshot
    uv run python scripts/fetch_team_ranks.py --refresh  # re-download first

Why this exists. Phase 1 records the team that played each mined replay, but not
how good that team is. Without a rank the pipeline can only draw its opponent
panel from *whoever happened to appear in the daily dumps* -- a sample of the whole
field weighted by episode volume, not by strength. We are matched by rating, so
that optimises win rate against the median of the field while the ladder pays for
beating the band above us.

The public leaderboard CSV is the join. Team names are unique on it (checked: zero
duplicates across 4,927 rows) and the replay `info.TeamNames` carries the same
string, so name is a sound key -- Kaggle does not put team ids in a replay.

CAVEAT, and it is a real one. The rank is a snapshot of *today*, while a mined
replay was played days earlier by whatever agent that team had running then. A
team that climbed since is credited for strength its mined route did not have, and
one that fell is under-credited. The snapshot filename is stored in the artifact so
any downstream claim can be dated.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import zipfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SCRATCH_DIR = os.path.join(REPO_ROOT, "scratch")
LOGS_DIR = os.path.join(REPO_ROOT, "logs")
SNAPSHOT_GLOB = "kaggriculture-publicleaderboard*.csv"
OUR_USERNAME = "erginakin"


def leaderboard_snapshot(refresh: bool = False) -> str:
    """Path to the newest leaderboard CSV, downloading one if asked or missing.

    Snapshots are timestamp-named, so the last sorted entry is the newest. Callers
    that never pass `refresh` would otherwise pin themselves to the first CSV ever
    written and silently pair a stale leaderboard with live data.
    """
    pattern = os.path.join(SCRATCH_DIR, SNAPSHOT_GLOB)
    files = sorted(glob.glob(pattern))
    if refresh or not files:
        from submit import load_credentials

        load_credentials()
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        print("Downloading leaderboard snapshot...")
        os.makedirs(SCRATCH_DIR, exist_ok=True)
        api.competition_leaderboard_download("kaggriculture", SCRATCH_DIR)
        zip_path = os.path.join(SCRATCH_DIR, "kaggriculture.zip")
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(SCRATCH_DIR)
        files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no leaderboard CSV in {SCRATCH_DIR}; pass --refresh to download one")
    return files[-1]


def read_leaderboard(path: str) -> list[dict]:
    # utf-8-sig: Kaggle's export carries a BOM, which otherwise lands in the first
    # column name and makes row["Rank"] a KeyError.
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def build_map(rows: list[dict]) -> dict:
    teams: dict[str, dict] = {}
    collisions = 0
    for row in rows:
        name = row["TeamName"]
        if name in teams:
            # Names are unique in practice; if that ever changes, keeping the better
            # rank is the conservative choice and the count is reported.
            collisions += 1
            if int(row["Rank"]) >= teams[name]["rank"]:
                continue
        teams[name] = {
            "rank": int(row["Rank"]),
            "score": float(row["Score"]),
            "team_id": int(row["TeamId"]),
        }
    return {"teams": teams, "collisions": collisions}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="write logs/team_ranks.json from the public leaderboard"
    )
    ap.add_argument("--refresh", action="store_true", help="re-download the leaderboard first")
    ap.add_argument("--out", default=os.path.join(LOGS_DIR, "team_ranks.json"))
    args = ap.parse_args(argv)

    path = leaderboard_snapshot(args.refresh)
    rows = read_leaderboard(path)
    built = build_map(rows)
    teams = built["teams"]
    print(f"leaderboard snapshot {os.path.basename(path)}: {len(teams):,} teams")
    if built["collisions"]:
        print(f"  !! {built['collisions']} duplicate team names collapsed (kept the better rank)")

    ours = next(
        (r for r in rows if OUR_USERNAME.lower() in (r.get("TeamMemberUserNames") or "").lower()),
        None,
    )
    if ours is None:
        # Not fatal -- the map is still usable -- but every "the band above us"
        # claim downstream is anchored on this row, so say it is missing.
        print(f"  !! our team ({OUR_USERNAME}) is not on this snapshot")
    else:
        print(f"  us: rank {ours['Rank']}, score {ours['Score']} ({ours['TeamName']})")

    top = sorted(teams.items(), key=lambda kv: kv[1]["rank"])[:10]
    print("  top 10:")
    for name, meta in top:
        print(f"    {meta['rank']:>4}  {meta['score']:>7.1f}  {name}")

    payload = {
        "snapshot": os.path.basename(path),
        "n_teams": len(teams),
        "our_team": (
            None
            if ours is None
            else {
                "name": ours["TeamName"],
                "rank": int(ours["Rank"]),
                "score": float(ours["Score"]),
            }
        ),
        "teams": teams,
    }
    out = args.out if os.path.isabs(args.out) else os.path.join(REPO_ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
