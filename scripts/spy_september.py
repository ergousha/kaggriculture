#!/usr/bin/env python3
"""Spy the September leaders: download + fingerprint fresh episodes of the top teams.

Phase 1 of docs/proposal-2026-09-14-endgame.md, chassis-class edition: the daily
dumps 403, but the per-episode replay path works (verified by examine_agent.py
and calculate_converged_winrate.py). This script:

  1. reads the fresh leaderboard snapshot (research_leaderboard.py --refresh),
  2. for the top N teams, lists their active submissions' episodes and picks the
     most recent K per team,
  3. downloads each replay (reusing examine_agent.py's download_replay helper),
  4. parses it with mine_daily.py's replay miner to a per-seat fingerprint row,
  5. appends the rows to logs/september_spy.csv and prints a cohort report.

Usage:
    uv run python scripts/spy_september.py --top 12 --per-team 6
    uv run python scripts/spy_september.py --top 12 --per-team 6 --fingerprint-only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)

SCRATCH_DIR = os.path.join(PROJECT_ROOT, "scratch")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
SPIES_DIR = os.path.join(LOGS_DIR, "september_spy")
OUT_CSV = os.path.join(LOGS_DIR, "september_spy.csv")
LB_JSON = os.path.join(LOGS_DIR, "leaderboard_research.json")


def agent_team_id(a):
    value = a.get("teamId") if isinstance(a, dict) else getattr(a, "team_id", None)
    return value if value else None


def episode_id(ep):
    value = ep.get("id") if isinstance(ep, dict) else getattr(ep, "id", None)
    if not value and hasattr(ep, "_id"):
        value = ep._id
    return value or None


def with_retry(fn, *args, what: str, attempts: int = 4, base_delay: float = 2.0):
    import time

    for attempt in range(attempts):
        try:
            return fn(*args)
        except Exception as exc:  # noqa: BLE001
            if "429" in str(exc) or attempt == attempts - 1:
                raise
            delay = base_delay * (2**attempt)
            print(f"  ! {what} failed ({exc}); retry in {delay:.0f}s")
            time.sleep(delay)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=12, help="how many top teams to spy")
    ap.add_argument("--per-team", type=int, default=6, help="recent episodes per team")
    ap.add_argument(
        "--fingerprint-only", action="store_true", help="re-mine already-downloaded replays"
    )
    args = ap.parse_args()

    os.makedirs(SPIES_DIR, exist_ok=True)

    if not os.path.exists(LB_JSON):
        print("Run scripts/research_leaderboard.py --refresh first.")
        sys.exit(1)
    with open(LB_JSON) as f:
        research = json.load(f)

    # team name -> rank/score from the research run
    teams = [
        (r["Rank"], r["TeamName"], r["Score"], r["TeamId"]) for r in research["results"][: args.top]
    ]

    # Reuse the daily miner for parsing; import after path setup.
    import mine_daily as md  # noqa: TID252 - scripts share a repo root
    from examine_agent import download_replay

    from submit import load_credentials

    load_credentials()
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    wanted: dict[int, dict] = {}  # episode_id -> meta
    if not args.fingerprint_only:
        for rank, name, score, team_id in teams:
            try:
                subs = with_retry(
                    api.competition_team_submissions, team_id, what=f"{name} submissions"
                )
                if not subs:
                    continue
                eps_all: dict[int, Any] = {}
                for sub in subs:
                    sub_id = getattr(sub, "id", None) or int(sub.ref)
                    eps = with_retry(
                        api.competition_list_episodes, sub_id, what=f"{name} eps ({sub_id})"
                    )
                    for ep in eps or []:
                        eid = episode_id(ep)
                        if eid:
                            eps_all[eid] = ep
                recent = sorted(eps_all)[-args.per_team :]
                for eid in recent:
                    wanted[eid] = {"team": name, "rank": rank, "score": score, "team_id": team_id}
                print(f"[{rank}] {name}: {len(eps_all)} eps, taking {len(recent)}")
            except Exception as exc:  # noqa: BLE001
                print(f"[{rank}] {name} FAILED: {exc}")

        # download the replays we don't have yet
        have = {f for f in os.listdir(SPIES_DIR) if f.endswith(".json")}
        todo = [eid for eid in wanted if f"episode-{eid}-replay.json" not in have]
        print(f"\ndownloading {len(todo)} replays...")
        for i, eid in enumerate(sorted(todo)):
            print(f"  {i + 1}/{len(todo)} ep {eid} ({wanted[eid]['team']})")
            download_replay(api, eid, SPIES_DIR)
    else:
        # rebuild `wanted` from whatever is already on disk plus a team map saved earlier
        meta_path = os.path.join(SPIES_DIR, "_meta.json")
        with open(meta_path) as f:
            wanted = {int(k): v for k, v in json.load(f).items()}

    if not args.fingerprint_only:
        with open(os.path.join(SPIES_DIR, "_meta.json"), "w") as f:
            json.dump({str(k): v for k, v in wanted.items()}, f, indent=1)

    # ---- fingerprint every replay on disk -----------------------------------
    rows = []
    files = [f for f in os.listdir(SPIES_DIR) if f.endswith(".json")]
    for fname in sorted(files):
        path = os.path.join(SPIES_DIR, fname)
        try:
            data = md.load_episode(path)
            if data is None:
                continue
            info = data.get("info") or {}
            eid = str(info.get("EpisodeId") or data.get("id") or "")
            meta = wanted.get(int(eid), {}) if eid.isdigit() else {}
            for row in md.fingerprint(data, fname):
                row["spy_team"] = meta.get("team", "?")
                row["spy_rank"] = meta.get("rank", 0)
                row["spy_score"] = meta.get("score", 0.0)
                row["replay_file"] = fname
                rows.append(row)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! parse {fname}: {exc}")

    if not rows:
        print("no rows; nothing to write")
        return

    fieldnames: list[str] = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    write_header = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} seat-fingerprints to {OUT_CSV}")

    # ---- cohort report -------------------------------------------------------
    df = pd.DataFrame(rows)
    print("\n=== top-team seat behaviour (mean per seat) ===")
    cols = [
        "cash",
        "won",
        "hires",
        "land_buys",
        "owned_tiles",
        "hands",
        "rev_WHEAT",
        "rev_STRAWBERRY",
        "rev_MELON",
        "rev_MILK",
        "rev_WOOL",
        "rev_TOMATO",
        "rev_CARROT",
        "rev_EGG",
        "rev_FERTILIZER",
        "sold_WHEAT",
        "sold_STRAWBERRY",
        "sold_MELON",
        "sold_MILK",
        "sold_WOOL",
        "sold_TOMATO",
        "sold_CARROT",
        "sold_EGG",
        "animal_SHEEP",
        "animal_COW",
        "animal_GOOSE",
        "ops_productive",
        "unsold_shed_units",
        "fertilized_tiles",
    ]
    have_cols = [c for c in cols if c in df.columns]
    grouped = df.groupby("spy_team")[have_cols].mean().sort_values("cash", ascending=False)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(grouped.round(1).to_string())


if __name__ == "__main__":
    from typing import Any  # noqa: E402

    main()
