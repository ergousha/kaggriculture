#!/usr/bin/env python3
"""Examine Agent Failures.

Fetches matches for a specific Kaggle submission (identified by description, e.g., 'v0.0.5'),
finds the episodes where the agent failed (lost or errored), and downloads those replays
for detailed examination.

Usage:
    python examine_agent.py v0.0.5
    python examine_agent.py v0.0.5 --limit 10
"""

import argparse
import os
import shutil
import sys
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from leaderboard_crawler import load_kaggle_api  # noqa: E402


def download_replay(api: Any, episode_id: int, target_dir: str) -> str | None:
    """Download episode replay JSON to the specified directory."""
    target_path = os.path.join(target_dir, f"episode-{episode_id}-replay.json")
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
        return target_path

    tmp_dir = os.path.join(target_dir, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        api.competition_episode_replay(episode_id, path=tmp_dir, quiet=True)
        expected = f"episode-{episode_id}-replay.json"
        src = os.path.join(tmp_dir, expected)
        if not os.path.exists(src):
            candidates = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.endswith(".json")]
            if not candidates:
                return None
            src = candidates[0]
        shutil.move(src, target_path)
        return target_path
    except Exception as exc:
        if "429" in str(exc):
            raise
        print(f"Error downloading replay {episode_id}: {exc}", file=sys.stderr)
        return None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        cwd_file = f"episode-{episode_id}-replay.json"
        if os.path.exists(cwd_file):
            try:
                os.remove(cwd_file)
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(description="Download failing episodes for a specific agent version.")
    parser.add_argument("version", help="The description/version of the submission (e.g. 'v0.0.5')")
    parser.add_argument("--limit", type=int, default=10, help="Max failed replays to download")
    parser.add_argument("--competition", type=str, default="kaggriculture", help="Competition name")

    args = parser.parse_args()

    api = load_kaggle_api()
    if not api:
        print("Failed to authenticate Kaggle API.")
        sys.exit(1)

    print(f"Fetching submissions for competition '{args.competition}'...")
    try:
        subs = api.competition_submissions(args.competition)
    except Exception as e:
        print(f"Error fetching submissions: {e}")
        sys.exit(1)

    target_sub = None
    for s in subs:
        desc = getattr(s, "description", None)
        if desc and desc.strip() == args.version:
            target_sub = s
            break

    if not target_sub:
        print(f"Could not find any submission with description: {args.version}")
        print("Available submissions (top 10):")
        for s in subs[:10]:
            print(f" - ID: {getattr(s, 'ref', 'N/A')}, Desc: {getattr(s, 'description', 'N/A')}")
        sys.exit(1)

    sub_id = int(target_sub.ref)
    print(f"Found submission '{args.version}' with ID {sub_id}")

    print("Fetching episodes...")
    try:
        eps = api.competition_list_episodes(sub_id)
    except Exception as e:
        print(f"Error fetching episodes: {e}")
        sys.exit(1)

    failed_eps = []

    for ep in eps:
        ep_id = getattr(ep, "id", None) or (ep.get("id") if isinstance(ep, dict) else None)
        agents = getattr(ep, "agents", []) or (ep.get("agents", []) if isinstance(ep, dict) else [])

        my_agent = None
        opp_agent = None

        for a in agents:
            a_sub_id = getattr(a, "submission_id", None) or (a.get("submissionId") if isinstance(a, dict) else None)
            if str(a_sub_id) == str(sub_id):
                my_agent = a
            else:
                opp_agent = a

        if my_agent and opp_agent:
            my_reward = getattr(my_agent, "reward", None)
            opp_reward = getattr(opp_agent, "reward", None)

            # Handle dictionary case if needed
            if isinstance(my_agent, dict):
                my_reward = my_agent.get("reward")
            if isinstance(opp_agent, dict):
                opp_reward = opp_agent.get("reward")

            if my_reward is None:
                my_reward = 0.0
            if opp_reward is None:
                opp_reward = 0.0

            if float(my_reward) < float(opp_reward):
                failed_eps.append((ep_id, float(my_reward), float(opp_reward)))

    print(f"Found {len(failed_eps)} matches where {args.version} failed/lost.")

    if not failed_eps:
        print("No failures to download.")
        sys.exit(0)

    target_dir = os.path.join(HERE, "logs", f"failures_{args.version}")
    os.makedirs(target_dir, exist_ok=True)
    print(f"Downloading up to {args.limit} replays to {target_dir}...")

    downloaded = 0
    import time
    for ep_id, my_r, opp_r in failed_eps[:args.limit]:
        print(f"Downloading episode {ep_id} (Reward: {my_r} vs {opp_r})...")
        retries = 3
        for attempt in range(retries):
            try:
                path = download_replay(api, int(ep_id), target_dir)
                if path:
                    print(f" -> Saved to {path}")
                    downloaded += 1
                break
            except Exception as e:
                if "429" in str(e):
                    wait_time = 15 * (attempt + 1)
                    print(f" -> Rate limited (429). Waiting {wait_time}s before retry {attempt + 1}/{retries}...")
                    time.sleep(wait_time)
                else:
                    print(f" -> Error: {e}")
                    break
        time.sleep(5)

    print(f"\nDone! Downloaded {downloaded} failed replays.")

if __name__ == "__main__":
    main()
