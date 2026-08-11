"""Leaderboard Replay Sparring Opponent.

Replays turn-by-turn recorded actions of a high-scoring player from a Kaggle
leaderboard JSON replay file. Allows local arena & parameter tuning sweeps
to spar directly against live #1 leaderboard opponents.

Usage:
    .venv/bin/python local_arena.py --agent main.py --opponent opponents/leaderboard_replay.py --episodes 10

Path override via environment variable:
    KAGGRICULTURE_REPLAY_PATH="logs/leaderboard_replays/episode-90163724-replay.json"
    KAGGRICULTURE_REPLAY_PLAYER=0
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(os.path.dirname(HERE), "logs")
REPLAY_DIR = os.path.join(LOG_DIR, "leaderboard_replays")

NO_OP = {"farmer": ["PASS"], "hands": [], "market": []}

_REPLAY_CACHE: dict = {}


def _get_replay_data():
    path = os.environ.get("KAGGRICULTURE_REPLAY_PATH", "")
    player_idx = int(os.environ.get("KAGGRICULTURE_REPLAY_PLAYER", "0"))

    if not path or not os.path.exists(path):
        # Default: newest JSON replay in REPLAY_DIR
        candidates = []
        if os.path.exists(REPLAY_DIR):
            candidates.extend(
                [
                    os.path.join(REPLAY_DIR, f)
                    for f in os.listdir(REPLAY_DIR)
                    if f.endswith(".json") and not f.startswith("_")
                ]
            )

        if not candidates:
            return [], player_idx
        # Pick the newest candidate
        candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        path = candidates[0]

    cache_key = (path, player_idx)
    if cache_key in _REPLAY_CACHE:
        return _REPLAY_CACHE[cache_key]

    try:
        with open(path) as f:
            data = json.load(f)
        steps = data.get("steps", [])
        _REPLAY_CACHE[cache_key] = (steps, player_idx)
        return steps, player_idx
    except Exception as exc:
        print(f"[LeaderboardReplayOpponent] Error loading replay {path}: {exc}")
        return [], player_idx


def agent(obs, config=None):
    """Replay agent entrypoint."""
    step = int(obs.get("step", 0) or 0)
    steps, player_idx = _get_replay_data()

    if not steps or step >= len(steps):
        return NO_OP

    step_state = steps[step]
    target_idx = player_idx
    if target_idx >= len(step_state):
        target_idx = 0

    p_data = step_state[target_idx] if target_idx < len(step_state) else {}
    action = p_data.get("action")

    if not action or not isinstance(action, dict):
        return NO_OP

    # Format return dict
    farmer = action.get("farmer") or ["PASS"]
    hands = action.get("hands") or []
    market = action.get("market") or []

    return {"farmer": farmer, "hands": hands, "market": market}
