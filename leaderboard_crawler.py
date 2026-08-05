#!/usr/bin/env python3
"""Leaderboard Intelligence Crawler & Replay Mining Pipeline for Kaggriculture.

Continuously scans Kaggle competition leaderboard matches, streams 720-step episode
replays, parses turn-by-turn strategic decisions (opening build orders, hiring curves,
land expansion triggers, animal purchases, market selling thresholds), and builds a
structured strategy intelligence database.

Usage:
    python leaderboard_crawler.py                         # Single scan & report
    python leaderboard_crawler.py --limit 10              # Scan up to 10 episodes
    python leaderboard_crawler.py --top-teams 5            # Focus on top 5 teams
    python leaderboard_crawler.py --interval 600          # Run continuously every 10 min
    python leaderboard_crawler.py --import-hall-of-fame   # Ingest top replays to EliteRecorder
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sys
import time
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
REPLAY_DIR = os.path.join(LOG_DIR, "leaderboard_replays")
INTEL_JSON = os.path.join(LOG_DIR, "leaderboard_intelligence.json")
INTEL_MD = os.path.join(LOG_DIR, "leaderboard_intelligence.md")


def load_kaggle_api() -> Any:
    """Authenticate and return KaggleApi instance using kaggle_credentials.py."""
    sys.path.insert(0, HERE)
    try:
        import kaggle_credentials as creds  # type: ignore

        token = getattr(creds, "KAGGLE_API_TOKEN", "").strip()
        user = getattr(creds, "KAGGLE_USERNAME", "").strip()
        key = getattr(creds, "KAGGLE_KEY", "").strip()

        if token:
            os.environ["KAGGLE_API_TOKEN"] = token
        elif user and key:
            os.environ["KAGGLE_USERNAME"] = user
            os.environ["KAGGLE_KEY"] = key
    except ImportError:
        pass

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as exc:
        print(f"[LeaderboardCrawler] Failed to authenticate Kaggle API: {exc}", file=sys.stderr)
        return None


class LeaderboardCrawler:
    """Streams and analyzes Kaggle Leaderboard replays for strategic patterns."""

    def __init__(self, competition: str = "kaggriculture") -> None:
        self.competition = competition
        self.api = load_kaggle_api()
        os.makedirs(REPLAY_DIR, exist_ok=True)

    def fetch_leaderboard(self) -> list[dict[str, Any]]:
        """Fetch current leaderboard teams."""
        if not self.api:
            return []
        try:
            lb = self.api.competition_leaderboard_view(self.competition)
            results = []
            for item in lb:
                t_id = getattr(item, "team_id", None) or getattr(item, "_team_id", None)
                t_name = getattr(item, "team_name", None) or getattr(item, "_team_name", None)
                score = getattr(item, "score", None) or getattr(item, "_score", None)
                date = getattr(item, "submission_date", None) or getattr(item, "_submission_date", None)
                results.append(
                    {
                        "team_id": t_id,
                        "team_name": t_name,
                        "score": float(score) if score else 0.0,
                        "submission_date": str(date) if date else "",
                    }
                )
            return sorted(results, key=lambda x: x["score"], reverse=True)
        except Exception as exc:
            print(f"[LeaderboardCrawler] Error fetching leaderboard: {exc}", file=sys.stderr)
            return []

    def fetch_episodes_for_team(self, submission_id: int) -> list[dict[str, Any]]:
        """Fetch episode list for a specific submission ID."""
        if not self.api:
            return []
        try:
            eps = self.api.competition_list_episodes(submission_id)
            ep_list = []
            for ep in eps:
                ep_id = getattr(ep, "id", None) or (ep.get("id") if isinstance(ep, dict) else None)
                agents = getattr(ep, "agents", []) or (ep.get("agents", []) if isinstance(ep, dict) else [])
                if ep_id:
                    ep_list.append(
                        {
                            "id": ep_id,
                            "agents": [
                                {
                                    "submission_id": getattr(a, "submission_id", None) or (a.get("submissionId") if isinstance(a, dict) else None),
                                    "team_name": getattr(a, "team_name", None) or (a.get("teamName") if isinstance(a, dict) else None),
                                    "reward": getattr(a, "reward", 0.0) or (a.get("reward", 0.0) if isinstance(a, dict) else 0.0),
                                }
                                for a in agents
                            ],
                        }
                    )
            return ep_list
        except Exception as exc:
            print(f"[LeaderboardCrawler] Error fetching episodes for submission {submission_id}: {exc}", file=sys.stderr)
            return []

    def download_replay(self, episode_id: int) -> str | None:
        """Download episode replay JSON to REPLAY_DIR."""
        if not self.api:
            return None
        target_path = os.path.join(REPLAY_DIR, f"episode-{episode_id}-replay.json")
        if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
            return target_path

        tmp_dir = os.path.join(REPLAY_DIR, "_tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        try:
            self.api.competition_episode_replay(episode_id, path=tmp_dir, quiet=True)
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
            print(f"[LeaderboardCrawler] Error downloading replay {episode_id}: {exc}", file=sys.stderr)
            return None
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            cwd_file = f"episode-{episode_id}-replay.json"
            if os.path.exists(cwd_file):
                try:
                    os.remove(cwd_file)
                except OSError:
                    pass

    def parse_episode_trajectory(self, replay_path: str) -> dict[str, Any] | None:
        """Dissect a full 720-step replay file into player strategy metrics."""
        try:
            with open(replay_path, "r") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"[LeaderboardCrawler] Error reading replay {replay_path}: {exc}", file=sys.stderr)
            return None

        steps = data.get("steps", [])
        if not steps:
            return None

        ep_id = data.get("id", 0)
        players_data = []

        for p_idx in (0, 1):
            team_name = "Player_" + str(p_idx)
            if p_idx < len(steps[-1]):
                agent_info = steps[-1][p_idx]
                team_name = agent_info.get("teamName") or f"Player_{p_idx}"

            # Trace trajectory
            opening_crops: dict[str, int] = {}
            livestock_purchases: list[dict[str, Any]] = []
            land_expansions: list[dict[str, Any]] = []
            hand_curve: list[int] = []
            cash_curve: list[float] = []

            prev_animals: dict[str, int] = {}
            prev_quadrants: int = 1

            for s_idx, step_state in enumerate(steps):
                if p_idx >= len(step_state):
                    continue
                p_data = step_state[p_idx]
                obs = p_data.get("observation") or {}
                farms = obs.get("farms") or []
                if p_idx >= len(farms):
                    continue
                farm = farms[p_idx]

                money = float(farm.get("money", 0.0))
                hands = len(farm.get("hands") or []) + 1
                quads = len(farm.get("quadrants") or [])
                animals = farm.get("animals") or {}

                if s_idx % 24 == 0 or s_idx == len(steps) - 1:
                    cash_curve.append(money)
                    hand_curve.append(hands)

                # Day 1-15 opening crop detection (from tiles)
                day = s_idx // 24
                if day <= 15:
                    tiles = farm.get("tiles") or []
                    if isinstance(tiles, list):
                        for row in tiles:
                            if isinstance(row, list):
                                for tile in row:
                                    if isinstance(tile, dict):
                                        c_type = tile.get("crop_type") or tile.get("crop")
                                        if c_type:
                                            opening_crops[c_type] = opening_crops.get(c_type, 0) + 1

                # Detect new animals purchased
                for a_type, count in animals.items():
                    prev_c = prev_animals.get(a_type, 0)
                    if count > prev_c:
                        livestock_purchases.append(
                            {
                                "animal": a_type,
                                "day": day,
                                "count": count - prev_c,
                            }
                        )
                prev_animals = dict(animals)

                # Detect land expansion
                if quads > prev_quadrants:
                    land_expansions.append({"day": day, "quadrants": quads})
                    prev_quadrants = quads

            final_reward = float(steps[-1][p_idx].get("reward", 0.0) or 0.0)

            # Summarize top crops
            sorted_crops = sorted(opening_crops.items(), key=lambda x: x[1], reverse=True)
            primary_crop = sorted_crops[0][0] if sorted_crops else "UNKNOWN"

            players_data.append(
                {
                    "player_index": p_idx,
                    "team_name": team_name,
                    "final_cash": final_reward,
                    "primary_opening_crop": primary_crop,
                    "livestock_purchases": livestock_purchases,
                    "land_expansions": land_expansions,
                    "max_hands": max(hand_curve) if hand_curve else 1,
                    "cash_day15": cash_curve[15] if len(cash_curve) > 15 else 0.0,
                    "cash_day30": cash_curve[30] if len(cash_curve) > 30 else 0.0,
                }
            )

        return {
            "episode_id": ep_id,
            "parsed_at": datetime.datetime.now().isoformat(),
            "players": players_data,
        }

    def run_scan(
        self,
        limit_episodes: int = 10,
        top_teams: int = 5,
        import_hall_of_fame: bool = False,
    ) -> dict[str, Any]:
        """Execute a full scan of leaderboard matches and build intelligence payload."""
        print(f"[LeaderboardCrawler] Starting scan (top {top_teams} teams, limit {limit_episodes} episodes)...")
        lb_teams = self.fetch_leaderboard()
        print(f"[LeaderboardCrawler] Fetched {len(lb_teams)} teams from leaderboard.")

        parsed_episodes: list[dict[str, Any]] = []
        scanned_count = 0

        # Discover submission IDs from our submissions and top leaderboard team entries
        target_sub_ids = []
        my_subs = []
        try:
            my_subs = self.api.competition_submissions(self.competition) if self.api else []
        except Exception:
            pass

        for s in my_subs[:5]:
            ref = getattr(s, "ref", None) or (s.get("ref") if isinstance(s, dict) else None)
            if ref:
                target_sub_ids.append(int(ref))

        # Also attempt to pull episodes from top team entries
        for team in lb_teams[:top_teams]:
            t_id = team.get("team_id")
            if t_id:
                try:
                    # Search episodes by team ID if available
                    eps_team = self.api.competition_list_episodes(int(t_id))
                    for ep_info in eps_team:
                        if scanned_count >= limit_episodes:
                            break
                        ep_id = getattr(ep_info, "id", None) or (ep_info.get("id") if isinstance(ep_info, dict) else None)
                        if ep_id:
                            r_file = self.download_replay(int(ep_id))
                            if r_file:
                                parsed = self.parse_episode_trajectory(r_file)
                                if parsed:
                                    parsed_episodes.append(parsed)
                                    scanned_count += 1
                                    if import_hall_of_fame:
                                        try:
                                            from elite_recorder import EliteRecorder
                                            rec = EliteRecorder()
                                            rec.import_kaggle_episode(int(ep_id), player_index=0)
                                            rec.import_kaggle_episode(int(ep_id), player_index=1)
                                        except Exception:
                                            pass
                except Exception:
                    pass

        for sub_id in target_sub_ids:
            if scanned_count >= limit_episodes:
                break
            eps = self.fetch_episodes_for_team(sub_id)
            for ep_info in eps:
                if scanned_count >= limit_episodes:
                    break
                ep_id = ep_info["id"]
                r_file = self.download_replay(ep_id)
                if r_file:
                    parsed = self.parse_episode_trajectory(r_file)
                    if parsed:
                        parsed_episodes.append(parsed)
                        scanned_count += 1

                if import_hall_of_fame:
                    try:
                        from elite_recorder import EliteRecorder

                        rec = EliteRecorder()
                        rec.import_kaggle_episode(ep_id, player_index=0)
                        rec.import_kaggle_episode(ep_id, player_index=1)
                    except Exception as exc:
                        print(f"[LeaderboardCrawler] Error importing ep {ep_id} to Hall of Fame: {exc}")

        crop_counts: dict[str, int] = {}
        high_score_players: list[dict[str, Any]] = []

        for ep in parsed_episodes:
            for p in ep["players"]:
                c = p["primary_opening_crop"]
                crop_counts[c] = crop_counts.get(c, 0) + 1
                if p["final_cash"] >= 30000.0:
                    high_score_players.append(p)

        high_score_players.sort(key=lambda x: x["final_cash"], reverse=True)

        intel_payload = {
            "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scanned_episodes_count": len(parsed_episodes),
            "top_opening_crops": crop_counts,
            "top_performers": high_score_players[:10],
            "episodes": parsed_episodes,
        }

        with open(INTEL_JSON, "w") as f:
            json.dump(intel_payload, f, indent=2)

        self._write_markdown_dashboard(intel_payload)

        print(f"[LeaderboardCrawler] Scan complete! Parsed {len(parsed_episodes)} episodes.")
        print(f"[LeaderboardCrawler] Intelligence saved to {INTEL_JSON} and {INTEL_MD}")

        return intel_payload

    def _write_markdown_dashboard(self, payload: dict[str, Any]) -> None:
        """Write human-readable leaderboard intelligence dashboard."""
        lines = [
            "# Leaderboard Intelligence & Strategy Dashboard",
            "",
            f"**Last Updated**: {payload.get('last_updated', '-')}",
            f"**Total Scanned Matches**: {payload.get('scanned_episodes_count', 0)}",
            "",
            "## 1. Meta Strategy Distribution (Opening Crops)",
            "",
            "| Crop Strategy | Count | Percentage |",
            "| --- | --- | --- |",
        ]

        total_ep = (payload.get("scanned_episodes_count", 1) * 2) or 1
        for crop, cnt in sorted(payload.get("top_opening_crops", {}).items(), key=lambda x: x[1], reverse=True):
            pct = (cnt / total_ep) * 100
            lines.append(f"| `{crop}` | {cnt} | {pct:.1f}% |")

        lines.extend(
            [
                "",
                "## 2. Top Hall of Fame Performers (Scores >= $30,000)",
                "",
                "| Rank | Team / Agent | Final Cash | Primary Crop | Max Hands | Land Expansions | Goose / Animals |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )

        performers = payload.get("top_performers", [])
        for idx, p in enumerate(performers, 1):
            land_str = f"{len(p.get('land_expansions', []))} expansions"
            animals = p.get("livestock_purchases", [])
            anim_str = f"{len(animals)} bought"
            lines.append(
                f"| #{idx} | `{p.get('team_name', 'Unknown')}` | **${p.get('final_cash', 0):,.2f}** | "
                f"`{p.get('primary_opening_crop', '-')}` | {p.get('max_hands', 1)} | {land_str} | {anim_str} |"
            )

        lines.append("")

        with open(INTEL_MD, "w") as f:
            f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Leaderboard Intelligence Crawler")
    parser.add_argument("--limit", type=int, default=10, help="Max episodes to process")
    parser.add_argument("--top-teams", type=int, default=5, help="Top N teams to scan")
    parser.add_argument("--interval", type=int, default=0, help="Continuous loop interval in seconds (0 = single run)")
    parser.add_argument("--import-hall-of-fame", action="store_true", help="Import top replays to EliteRecorder")

    args = parser.parse_args()

    crawler = LeaderboardCrawler()

    if args.interval > 0:
        print(f"[LeaderboardCrawler] Running continuously every {args.interval} seconds...")
        try:
            while True:
                crawler.run_scan(
                    limit_episodes=args.limit,
                    top_teams=args.top_teams,
                    import_hall_of_fame=args.import_hall_of_fame,
                )
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("[LeaderboardCrawler] Stopped by user.")
    else:
        crawler.run_scan(
            limit_episodes=args.limit,
            top_teams=args.top_teams,
            import_hall_of_fame=args.import_hall_of_fame,
        )


if __name__ == "__main__":
    main()
