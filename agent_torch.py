"""Full RL Agent for Kaggriculture.

Bypasses all heuristic logic and directly outputs atomic actions
predicted by the PyTorch Behavior Cloning model.
"""

import os
import sys

import numpy as np
import torch

AGENT_VERSION = "0.0.8"

# Ensure local imports work correctly for Kaggle environment
try:
    from rl.action_space import decode_market_actions, decode_unit_action
    from rl.architecture import KaggriculturePolicyFullRL
except ImportError:
    import sys

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from rl.action_space import decode_market_actions, decode_unit_action
    from rl.architecture import KaggriculturePolicyFullRL


_MODEL_PATH = os.path.join(os.path.dirname(__file__), "logs", "bc_full_model.pt")

_POLICY = None
_DEVICE = None


def _load_model():
    global _POLICY, _DEVICE
    if _POLICY is not None:
        return

    _DEVICE = torch.device("cpu")  # Inference on CPU is fine for Kaggle submissions usually
    _POLICY = KaggriculturePolicyFullRL()
    if os.path.exists(_MODEL_PATH):
        try:
            _POLICY.load_state_dict(torch.load(_MODEL_PATH, map_location=_DEVICE))
            _POLICY.eval()
        except Exception as e:
            print(f"Failed to load weights: {e}", file=sys.stderr)
    else:
        print(
            f"WARNING: No trained weights found at {_MODEL_PATH}. Using random init.",
            file=sys.stderr,
        )


def _extract_features(obs, player):
    """Extract spatial and vector features for the current step."""
    farms = obs.get("farms", [])
    if player >= len(farms):
        return None, None

    farm = farms[player]

    grid_planes = np.zeros((5, 10, 10), dtype=np.float32)
    crop_map = {"WHEAT": 1, "STRAWBERRY": 2, "MELON": 3}

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

    day = float(obs.get("day", 0)) / 30.0
    hour = float(obs.get("hour", 0)) / 24.0
    money = float(farm.get("money", 0.0)) / 10000.0

    hands_list = farm.get("hands", []) or []
    hands_count = len(hands_list) / 24.0

    g_vec = np.array([day, hour, money, hands_count], dtype=np.float32)

    return grid_planes, g_vec


def agent(obs, config=None):
    """Kaggriculture entrypoint for Full RL Agent."""
    try:
        step = int(obs.get("step", 0) or 0)
        if step == 0:
            print(f"Kaggriculture Agent v{AGENT_VERSION}")

        _load_model()
        player = int(obs.get("player", 0) or 0)

        spatial_feat, vector_feat = _extract_features(obs, player)
        if spatial_feat is None:
            return {"farmer": ["PASS"], "hands": [], "market": []}

        t_spatial = torch.tensor(spatial_feat).unsqueeze(0).to(_DEVICE)
        t_vector = torch.tensor(vector_feat).unsqueeze(0).to(_DEVICE)

        with torch.no_grad():
            f_logits, h_logits, m_preds = _POLICY(t_vector, t_spatial)  # type: ignore

        f_idx = torch.argmax(f_logits[0]).item()
        farmer_action = decode_unit_action(f_idx, is_farmer=True)

        h_preds = torch.argmax(h_logits[0], dim=0).cpu().numpy()  # shape (10, 10)

        hands_list = obs.get("farms", [])[player].get("hands", [])
        hands_actions = []

        for x, y in hands_list:
            if 0 <= x < 10 and 0 <= y < 10:
                h_idx = h_preds[y, x]
                act = decode_unit_action(h_idx, is_farmer=False)
                hands_actions.append(act)
            else:
                hands_actions.append(["PASS"])

        m_tensor = m_preds[0].cpu().numpy()  # shape (12, 3)
        market_orders = decode_market_actions(m_tensor)

        return {
            "farmer": farmer_action,
            "hands": hands_actions,
            "market": market_orders[:10],  # Hard limit of 10 orders per turn
        }

    except Exception as e:
        print(f"Exception in agent: {e}", file=sys.stderr)
        return {"farmer": ["PASS"], "hands": [], "market": []}
