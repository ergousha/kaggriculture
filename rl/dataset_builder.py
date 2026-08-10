"""Dataset Builder for Kaggriculture Full Offline RL.

Parses Kaggle JSON replays and converts them into structured ML-ready tensors
(Observations and Atomic Actions) suitable for PyTorch Behavior Cloning.
Implements filtering for elite matches (e.g., >$25k final cash).
"""

import glob
import json
import multiprocessing
import os

import numpy as np

from rl.action_space import encode_market_actions, encode_unit_action


def get_final_cash(step_data: list) -> tuple[float, float]:
    """Returns final cash for (player_0, player_1)."""
    p0 = step_data[0].get("reward", 0.0) or 0.0
    p1 = step_data[1].get("reward", 0.0) or 0.0
    return float(p0), float(p1)


def parse_replay(
    replay_path: str, min_cash: float = 25000.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Parse a single JSON replay into feature planes, globals, and full actions."""
    try:
        with open(replay_path) as f:
            data = json.load(f)
    except Exception:
        return None

    steps = data.get("steps", [])
    if not steps or len(steps) < 100:
        return None

    # Determine winner and if they meet the elite threshold
    p0_cash, p1_cash = get_final_cash(steps[-1])

    if p0_cash >= min_cash and p0_cash >= p1_cash:
        target_p = 0
    elif p1_cash >= min_cash and p1_cash > p0_cash:
        target_p = 1
    else:
        return None  # Neither player qualifies as elite for this match

    feature_planes = []
    global_vecs = []

    farmer_acts = []
    hands_acts = []
    market_acts = []

    # Crop encoding map
    crop_map = {"WHEAT": 1, "STRAWBERRY": 2, "MELON": 3}

    for _s_idx, step_states in enumerate(steps[:-1]):  # Last step has no actions
        if target_p >= len(step_states):
            continue

        p_state = step_states[target_p]
        obs = p_state.get("observation", {})
        act = p_state.get("action", {}) or {}

        farms = obs.get("farms", [])
        if target_p >= len(farms):
            continue

        farm = farms[target_p]

        # 1. Parse Spatial Feature Planes (10x10)
        # Channels: 0: locked, 1: empty, 2: crop_type, 3: yield, 4: watered
        grid_planes = np.zeros((5, 10, 10), dtype=np.float32)
        tiles = farm.get("tiles", [])
        for i, row in enumerate(tiles):
            for j, tile in enumerate(row):
                if tile == "LOCKED":
                    grid_planes[0, i, j] = 1.0
                elif tile == "EMPTY":
                    grid_planes[1, i, j] = 1.0
                elif isinstance(tile, dict) and tile.get("kind") == "PLANT":
                    ctype = crop_map.get(tile.get("crop", ""), 0)
                    grid_planes[2, i, j] = ctype
                    grid_planes[3, i, j] = tile.get("yield_units", 0)
                    grid_planes[4, i, j] = 1.0 if tile.get("watered_today") else 0.0

        # 2. Parse Global Vectors
        day = float(obs.get("day", 0)) / 30.0  # Normalize 0-1
        hour = float(obs.get("hour", 0)) / 24.0
        money = float(farm.get("money", 0.0)) / 10000.0  # Scale
        hands_list = farm.get("hands", []) or []
        hands_count = len(hands_list) / 24.0

        g_vec = np.array([day, hour, money, hands_count], dtype=np.float32)

        # 3. Parse Full RL Actions

        # Farmer Action
        f_act = encode_unit_action(act.get("farmer", []), is_farmer=True)

        # Hands Actions (Spatial Grid of length 10x10 containing action classes)
        # We need to map the hand actions to the grid locations of the hands.
        h_act_grid = np.zeros((10, 10), dtype=np.int64)

        hand_actions = act.get("hands", [])
        # In the environment, hands are executed in order.
        # Observation gives hand locations.
        for h_idx, hand_loc in enumerate(hands_list):
            if h_idx < len(hand_actions):
                h_act = hand_actions[h_idx]
            else:
                h_act = ["PASS"]

            x, y = hand_loc
            if 0 <= x < 10 and 0 <= y < 10:
                h_act_grid[y, x] = encode_unit_action(h_act, is_farmer=False)

        # Market Actions
        market_orders = act.get("market", [])
        m_act = encode_market_actions(market_orders)

        feature_planes.append(grid_planes)
        global_vecs.append(g_vec)
        farmer_acts.append(f_act)
        hands_acts.append(h_act_grid)
        market_acts.append(m_act)

    if not feature_planes:
        return None

    return (
        np.stack(feature_planes),
        np.stack(global_vecs),
        np.array(farmer_acts, dtype=np.int64),
        np.stack(hands_acts),
        np.stack(market_acts),
    )


def parse_online_replay(replay_path: str, target_p: int = 0) -> tuple[float, float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None:
    """Parse a single replay for Online RL, returning (me_cash, opp_cash, arrays)."""
    try:
        with open(replay_path) as f:
            data = json.load(f)
    except Exception:
        return None

    steps = data.get("steps", [])
    if not steps or len(steps) < 10:
        return None

    # Determine final cash
    p0_cash, p1_cash = get_final_cash(steps[-1])
    me_cash = p0_cash if target_p == 0 else p1_cash
    opp_cash = p1_cash if target_p == 0 else p0_cash

    feature_planes = []
    global_vecs = []
    farmer_acts = []
    hands_acts = []
    market_acts = []
    crop_map = {"WHEAT": 1, "STRAWBERRY": 2, "MELON": 3}

    for _s_idx, step_states in enumerate(steps[:-1]):
        if target_p >= len(step_states):
            continue

        p_state = step_states[target_p]
        obs = p_state.get("observation", {})
        act = p_state.get("action", {}) or {}

        farms = obs.get("farms", [])
        if target_p >= len(farms):
            continue
        farm = farms[target_p]

        # 1. Parse Spatial Feature Planes (10x10)
        grid_planes = np.zeros((5, 10, 10), dtype=np.float32)
        tiles = farm.get("tiles", [])
        for i, row in enumerate(tiles):
            for j, tile in enumerate(row):
                if tile == "LOCKED":
                    grid_planes[0, i, j] = 1.0
                elif tile == "EMPTY":
                    grid_planes[1, i, j] = 1.0
                elif isinstance(tile, dict) and tile.get("kind") == "PLANT":
                    ctype = crop_map.get(tile.get("crop", ""), 0)
                    grid_planes[2, i, j] = ctype
                    grid_planes[3, i, j] = tile.get("yield_units", 0)
                    grid_planes[4, i, j] = 1.0 if tile.get("watered_today") else 0.0

        # 2. Parse Global Vectors
        day = float(obs.get("day", 0)) / 30.0
        hour = float(obs.get("hour", 0)) / 24.0
        money = float(farm.get("money", 0.0)) / 10000.0
        hands_list = farm.get("hands", []) or []
        hands_count = len(hands_list) / 24.0
        g_vec = np.array([day, hour, money, hands_count], dtype=np.float32)

        # 3. Parse Full RL Actions
        f_act = encode_unit_action(act.get("farmer", []), is_farmer=True)
        h_act_grid = np.zeros((10, 10), dtype=np.int64)
        hand_actions = act.get("hands", [])
        for h_idx, hand_loc in enumerate(hands_list):
            h_act = hand_actions[h_idx] if h_idx < len(hand_actions) else ["PASS"]
            x, y = hand_loc
            if 0 <= x < 10 and 0 <= y < 10:
                h_act_grid[y, x] = encode_unit_action(h_act, is_farmer=False)

        market_orders = act.get("market", [])
        m_act = encode_market_actions(market_orders)

        feature_planes.append(grid_planes)
        global_vecs.append(g_vec)
        farmer_acts.append(f_act)
        hands_acts.append(h_act_grid)
        market_acts.append(m_act)

    if not feature_planes:
        return None

    arrays = (
        np.stack(feature_planes),
        np.stack(global_vecs),
        np.array(farmer_acts, dtype=np.int64),
        np.stack(hands_acts),
        np.stack(market_acts),
    )
    return me_cash, opp_cash, arrays
def build_dataset_worker(args):
    return parse_replay(*args)


def build_dataset(replay_dir: str, output_path: str, min_cash: float = 25000.0):
    """Builds a complete dataset from all downloaded elite replays."""
    replay_files = glob.glob(os.path.join(replay_dir, "*.json"))
    print(f"Found {len(replay_files)} total replay files.")

    args_list = [(f, min_cash) for f in replay_files]

    all_planes = []
    all_globals = []
    all_f_acts = []
    all_h_acts = []
    all_m_acts = []

    parsed_count = 0
    with multiprocessing.Pool() as pool:
        for result in pool.imap_unordered(build_dataset_worker, args_list, chunksize=10):
            if result is not None:
                planes, globals_, f_acts, h_acts, m_acts = result
                all_planes.append(planes)
                all_globals.append(globals_)
                all_f_acts.append(f_acts)
                all_h_acts.append(h_acts)
                all_m_acts.append(m_acts)
                parsed_count += 1
                if parsed_count % 100 == 0:
                    print(f"Parsed {parsed_count} elite trajectories...")

    if not all_planes:
        print("No replays met the elite threshold!")
        return

    print(f"Concatenating {parsed_count} elite trajectories into master dataset...")
    X_planes = np.concatenate(all_planes, axis=0)
    X_globals = np.concatenate(all_globals, axis=0)
    Y_f_acts = np.concatenate(all_f_acts, axis=0)
    Y_h_acts = np.concatenate(all_h_acts, axis=0)
    Y_m_acts = np.concatenate(all_m_acts, axis=0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        planes=X_planes,
        globals=X_globals,
        farmer_acts=Y_f_acts,
        hands_acts=Y_h_acts,
        market_acts=Y_m_acts,
    )
    print(f"SUCCESS: Built offline RL dataset with {len(X_planes)} samples.")
    print(f"Saved to: {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replay-dir", default="replays", help="Directory containing Kaggle JSON replays"
    )
    parser.add_argument("--output", default="logs/offline_rl_dataset.npz", help="Output .npz file")
    parser.add_argument(
        "--min-cash", type=float, default=25000.0, help="Minimum final cash to consider elite"
    )
    args = parser.parse_args()

    build_dataset(args.replay_dir, args.output, args.min_cash)
