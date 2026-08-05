#!/usr/bin/env python3
"""Elite Trajectory Recorder for Kaggriculture.

Records, ranks, filters, and manages high-scoring match trajectories
(full replays, per-turn decision logs, performance metadata, and metrics).

Also provides dataset export functionality to convert elite match histories
into state-action tuples suitable for Offline RL, Behavior Cloning, or
In-Context Learning exemplars.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from typing import Any


class EliteRecorder:
    """Manages an elite trajectory buffer of top-performing matches."""

    def __init__(
        self,
        elite_dir: str = "logs/elite_trajectories",
        top_k: int = 20,
        min_cash: float = 20000.0,
    ) -> None:
        self.elite_dir = os.path.abspath(elite_dir)
        self.top_k = top_k
        self.min_cash = min_cash

        self.replays_dir = os.path.join(self.elite_dir, "replays")
        self.decisions_dir = os.path.join(self.elite_dir, "decisions")
        self.manifest_path = os.path.join(self.elite_dir, "manifest.json")
        self.readme_path = os.path.join(self.elite_dir, "manifest.md")

        os.makedirs(self.replays_dir, exist_ok=True)
        os.makedirs(self.decisions_dir, exist_ok=True)

        self.entries: list[dict[str, Any]] = self._load_manifest()

    def _load_manifest(self) -> list[dict[str, Any]]:
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return sorted(data, key=lambda x: x.get("me_cash", 0), reverse=True)
            except Exception as e:
                print(f"[EliteRecorder] Warning loading manifest: {e}", file=sys.stderr)
        return []

    def save_manifest(self) -> None:
        self.entries.sort(key=lambda x: x.get("me_cash", 0), reverse=True)

        # Write manifest.json
        with open(self.manifest_path, "w") as f:
            json.dump(self.entries, f, indent=2)

        # Write manifest.md
        lines = [
            "# Elite Trajectory Hall of Fame",
            "",
            f"Top {len(self.entries)} match trajectories recorded (Min Cash Threshold: ${self.min_cash:,.2f}).",
            "",
            "| Rank | Cash | Opponent | Win | Seed | Replay File | Decision Log | Recorded At |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for idx, entry in enumerate(self.entries, 1):
            replay_rel = os.path.basename(entry.get("replay_file", ""))
            dec_rel = os.path.basename(entry.get("decision_log_file", ""))
            win_str = "W" if entry.get("win") else ("T" if entry.get("tie") else "L")
            recorded_at = entry.get("timestamp", "")
            lines.append(
                f"| #{idx} | **${entry.get('me_cash', 0):,.2f}** | `{entry.get('opponent', 'unknown')}` | "
                f"{win_str} | {entry.get('seed', '-')} | `{replay_rel}` | `{dec_rel}` | {recorded_at} |"
            )
        lines.append("")

        with open(self.readme_path, "w") as f:
            f.write("\n".join(lines))

    def qualifies(self, me_cash: float) -> bool:
        if me_cash < self.min_cash:
            return False
        if len(self.entries) < self.top_k:
            return True
        # Qualifies if better than the current lowest cash in top_k
        return me_cash > self.entries[-1].get("me_cash", 0)

    def add_candidate(
        self,
        run_result: dict[str, Any],
        agent_path: str,
        opponent_name: str,
        replay_src: str | None = None,
        decision_log_src: str | None = None,
    ) -> bool:
        me_cash = run_result.get("me_cash", 0.0)
        if not self.qualifies(me_cash):
            return False

        seed = run_result.get("seed", 0)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        run_id = f"{int(me_cash):06d}_seed{seed}_{int(time.time())}"

        dst_replay = ""
        if replay_src and os.path.exists(replay_src):
            dst_replay = os.path.join(self.replays_dir, f"elite_{run_id}.json")
            shutil.copyfile(replay_src, dst_replay)

        dst_decisions = ""
        summary_stats = {}
        if decision_log_src and os.path.exists(decision_log_src):
            dst_decisions = os.path.join(self.decisions_dir, f"elite_{run_id}.jsonl")
            shutil.copyfile(decision_log_src, dst_decisions)
            summary_stats = self._analyze_decision_log(dst_decisions)

        entry = {
            "run_id": run_id,
            "me_cash": me_cash,
            "opp_cash": run_result.get("opp_cash", 0.0),
            "win": run_result.get("win", 0),
            "tie": run_result.get("tie", 0),
            "seed": seed,
            "agent": agent_path,
            "opponent": opponent_name,
            "swap": run_result.get("swap", False),
            "replay_file": dst_replay,
            "decision_log_file": dst_decisions,
            "timestamp": timestamp,
            "turn_p95": run_result.get("turn_p95", 0.0),
            "actions_total": run_result.get("actions_total", 0),
            "actions_noop": run_result.get("actions_noop", 0),
            "summary_stats": summary_stats,
        }

        self.entries.append(entry)
        self.entries.sort(key=lambda x: x.get("me_cash", 0), reverse=True)

        # Prune if over capacity
        if len(self.entries) > self.top_k:
            to_remove = self.entries[self.top_k :]
            self.entries = self.entries[: self.top_k]
            for item in to_remove:
                self._delete_files(item)

        self.save_manifest()
        return True

    def _delete_files(self, entry: dict[str, Any]) -> None:
        rep = entry.get("replay_file")
        if rep and os.path.exists(rep):
            try:
                os.remove(rep)
            except OSError:
                pass
        dec = entry.get("decision_log_file")
        if dec and os.path.exists(dec):
            try:
                os.remove(dec)
            except OSError:
                pass

    def _analyze_decision_log(self, path: str) -> dict[str, Any]:
        max_units = 0
        total_steps = 0
        total_idle = 0
        max_money = 0.0
        try:
            with open(path) as f:
                for line in f:
                    rec = json.loads(line)
                    total_steps += 1
                    units = rec.get("n_units", 0)
                    idle = rec.get("idle_units", 0)
                    money = rec.get("money", 0.0)
                    if units > max_units:
                        max_units = units
                    if money > max_money:
                        max_money = money
                    total_idle += idle
        except Exception:
            pass
        return {
            "total_steps": total_steps,
            "max_units": max_units,
            "peak_cash": max_money,
            "avg_idle_per_step": round(total_idle / total_steps, 2) if total_steps else 0.0,
        }

    def export_dataset(self, output_path: str) -> int:
        """Export state-action sequence decision tuples from elite runs into a JSONL dataset."""
        exported_count = 0
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as out:
            for entry in self.entries:
                dec_path = entry.get("decision_log_file")
                if not dec_path or not os.path.exists(dec_path):
                    continue
                run_meta = {
                    "run_id": entry["run_id"],
                    "me_cash": entry["me_cash"],
                    "seed": entry["seed"],
                    "opponent": entry["opponent"],
                }
                with open(dec_path) as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            record["meta"] = run_meta
                            out.write(json.dumps(record) + "\n")
                            exported_count += 1
                        except json.JSONDecodeError:
                            continue
        return exported_count

    def import_kaggle_episode(self, url_or_id: str | int, player_index: int = 0) -> bool:
        """Download a live Kaggle competition episode replay by URL or ID and add to elite buffer."""
        import re

        s = str(url_or_id)
        m = re.search(r"episodeId=(\d+)", s)
        if m:
            ep_id = int(m.group(1))
        else:
            m2 = re.search(r"(\d+)", s)
            if not m2:
                raise ValueError(f"Could not parse episode ID from {url_or_id}")
            ep_id = int(m2.group(1))

        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import kaggle_credentials as creds

            token = getattr(creds, "KAGGLE_API_TOKEN", "")
            if token:
                os.environ["KAGGLE_API_TOKEN"] = token
        except Exception:
            pass

        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
        except Exception as exc:
            print(f"[EliteRecorder] Failed Kaggle API auth: {exc}", file=sys.stderr)
            return False

        tmp_dir = os.path.join(self.elite_dir, "_tmp_import")
        os.makedirs(tmp_dir, exist_ok=True)
        try:
            print(f"[EliteRecorder] Fetching Kaggle live replay for episode {ep_id}...")
            api.competition_episode_replay(ep_id, path=tmp_dir)
            expected_filename = f"episode-{ep_id}-replay.json"
            rep_path = os.path.join(tmp_dir, expected_filename)
            if not os.path.exists(rep_path):
                files = [
                    os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.endswith(".json")
                ]
                if not files:
                    print(
                        f"[EliteRecorder] Replay file for episode {ep_id} not found after download.",
                        file=sys.stderr,
                    )
                    return False
                rep_path = files[0]

            with open(rep_path) as f:
                data = json.load(f)

            steps = data.get("steps", [])
            if not steps:
                print("[EliteRecorder] Replay file has no steps.", file=sys.stderr)
                return False

            final_step = steps[-1]
            rewards = [s.get("reward", 0.0) or 0.0 for s in final_step]
            me_cash = float(rewards[player_index]) if player_index < len(rewards) else 0.0
            opp_cash = (
                float(rewards[1 - player_index]) if (1 - player_index) < len(rewards) else 0.0
            )

            info = data.get("info", {})
            agents = info.get("Agents", [{}, {}])
            opp_name = (
                agents[1 - player_index].get("Name", "unknown")
                if (1 - player_index) < len(agents)
                else "unknown"
            )

            dec_path = os.path.join(tmp_dir, f"decisions_ep_{ep_id}.jsonl")
            with open(dec_path, "w") as out:
                for s_idx, step_states in enumerate(steps):
                    p_state = (
                        step_states[player_index]
                        if (
                            isinstance(step_states, (list, tuple))
                            and player_index < len(step_states)
                        )
                        else {}
                    )
                    act = p_state.get("action") if isinstance(p_state, dict) else None
                    obs = (p_state.get("observation") if isinstance(p_state, dict) else {}) or {}
                    farms = (obs.get("farms") if isinstance(obs, dict) else []) or []
                    farm = (
                        farms[player_index]
                        if (isinstance(farms, (list, tuple)) and player_index < len(farms))
                        else {}
                    )
                    hands_raw = farm.get("hands") if isinstance(farm, dict) else None
                    hands_cnt = len(hands_raw) if isinstance(hands_raw, (list, tuple)) else 0
                    rec = {
                        "player": player_index,
                        "step": s_idx,
                        "action": act,
                        "day": s_idx // 24,
                        "hour": s_idx % 24,
                        "money": float(farm.get("money", 0.0)) if isinstance(farm, dict) else 0.0,
                        "n_units": hands_cnt + 1,
                        "source": f"kaggle_ep_{ep_id}",
                    }
                    out.write(json.dumps(rec) + "\n")

            run_result = {
                "me_cash": me_cash,
                "opp_cash": opp_cash,
                "win": 1 if me_cash > opp_cash else 0,
                "tie": 1 if me_cash == opp_cash else 0,
                "seed": info.get("seed", 0),
                "swap": bool(player_index),
                "turn_p95": 0.0,
                "actions_total": len(steps),
                "actions_noop": 0,
            }

            added = self.add_candidate(
                run_result=run_result,
                agent_path=f"kaggle://episodes/{ep_id}",
                opponent_name=opp_name,
                replay_src=rep_path,
                decision_log_src=dec_path,
            )
            return added
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            # Kaggle API client sometimes saves raw replay to cwd
            cwd_replay = f"episode-{ep_id}-replay.json"
            if os.path.exists(cwd_replay):
                try:
                    os.remove(cwd_replay)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# CLI Command Interface
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Elite Trajectory Recorder CLI")
    parser.add_argument(
        "--dir", default="logs/elite_trajectories", help="Path to elite trajectories directory"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    subparsers.add_parser("list", help="List recorded elite trajectories")

    # summary
    subparsers.add_parser("summary", help="Show summary statistics across recorded elite runs")

    # export
    export_parser = subparsers.add_parser(
        "export", help="Export state-action dataset from elite runs"
    )
    export_parser.add_argument(
        "--output", default="logs/elite_dataset.jsonl", help="Output JSONL dataset file path"
    )

    # import-kaggle
    import_parser = subparsers.add_parser(
        "import-kaggle", help="Import live Kaggle episode replay by URL or Episode ID"
    )
    import_parser.add_argument("target", help="Kaggle URL or episode ID (e.g. 89987566)")
    import_parser.add_argument(
        "--player", type=int, default=0, help="Player index to treat as main agent (0 or 1)"
    )

    # prune
    prune_parser = subparsers.add_parser("prune", help="Prune stored trajectories down to top K")
    prune_parser.add_argument(
        "--keep", type=int, default=10, help="Number of top trajectories to keep"
    )

    args = parser.parse_args()

    recorder = EliteRecorder(elite_dir=args.dir)

    if args.command == "list":
        if not recorder.entries:
            print(f"No elite trajectories found in {args.dir}")
            return
        print(f"\n--- ELITE TRAJECTORY LEADERBOARD ({len(recorder.entries)} runs) ---")
        print(f"{'Rank':<5} {'Cash':<12} {'Opponent':<15} {'Win':<5} {'Seed':<8} {'Run ID'}")
        print("-" * 65)
        for idx, e in enumerate(recorder.entries, 1):
            win_str = "W" if e.get("win") else ("T" if e.get("tie") else "L")
            print(
                f"#{idx:<4} ${e.get('me_cash', 0):<11,.2f} {e.get('opponent', 'unknown'):<15} "
                f"{win_str:<5} {e.get('seed', '-'):<8} {e.get('run_id')}"
            )

    elif args.command == "summary":
        if not recorder.entries:
            print(f"No elite trajectories found in {args.dir}")
            return
        cashes = [e.get("me_cash", 0.0) for e in recorder.entries]
        print("\n--- ELITE BUFFER SUMMARY ---")
        print(f"Total Elite Runs:     {len(recorder.entries)}")
        print(f"Max Cash Achieved:    ${max(cashes):,.2f}")
        print(f"Min Elite Cash:       ${min(cashes):,.2f}")
        print(f"Mean Elite Cash:      ${sum(cashes) / len(cashes):,.2f}")

    elif args.command == "export":
        count = recorder.export_dataset(args.output)
        print(
            f"Exported {count} decision records from {len(recorder.entries)} elite trajectories to {args.output}"
        )

    elif args.command == "import-kaggle":
        success = recorder.import_kaggle_episode(args.target, player_index=args.player)
        if success:
            print(f"Successfully imported Kaggle episode '{args.target}' into elite buffer!")
        else:
            print(
                f"Failed to import Kaggle episode '{args.target}'. Check cash threshold or log output."
            )

    elif args.command == "prune":
        initial_len = len(recorder.entries)
        recorder.top_k = args.keep
        if len(recorder.entries) > args.keep:
            to_remove = recorder.entries[args.keep :]
            recorder.entries = recorder.entries[: args.keep]
            for item in to_remove:
                recorder._delete_files(item)
            recorder.save_manifest()
            print(f"Pruned buffer from {initial_len} down to {len(recorder.entries)} runs.")
        else:
            print(f"Buffer has {initial_len} runs, no pruning needed (keep={args.keep}).")


if __name__ == "__main__":
    main()
