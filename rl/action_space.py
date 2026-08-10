"""Full RL Action Space Encodings."""

import numpy as np

# Farmer / Hand Actions
BASE_ACTIONS = ["PASS", "N", "S", "E", "W", "WATER", "HARVEST", "DEMOLISH"]
PLACE_ITEMS = [
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "GOOSE",
    "COW",
    "SHEEP",
    "COOP",
    "PASTURE",
]
FARMER_ONLY = ["HIRE", "BUY_LAND"]

ALL_UNIT_ACTIONS = BASE_ACTIONS + [f"PLACE_{item}" for item in PLACE_ITEMS]
ALL_FARMER_ACTIONS = ALL_UNIT_ACTIONS + FARMER_ONLY

UNIT_ACTION_MAP = {act: i for i, act in enumerate(ALL_UNIT_ACTIONS)}
FARMER_ACTION_MAP = {act: i for i, act in enumerate(ALL_FARMER_ACTIONS)}

NUM_UNIT_ACTIONS = len(ALL_UNIT_ACTIONS)
NUM_FARMER_ACTIONS = len(ALL_FARMER_ACTIONS)

# Market Actions
MARKET_ITEMS = [
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
    "GOOSE",
    "COW",
    "SHEEP",
]
MARKET_ITEM_MAP = {item: i for i, item in enumerate(MARKET_ITEMS)}
NUM_MARKET_ITEMS = len(MARKET_ITEMS)


def encode_unit_action(action_list, is_farmer=False):
    """Encode a unit action list e.g. ['PLACE', 'WHEAT'] into an integer class."""
    if not action_list:
        return 0  # PASS

    act_str = action_list[0]
    if act_str == "PLACE" and len(action_list) > 1:
        act_str = f"PLACE_{action_list[1]}"

    m = FARMER_ACTION_MAP if is_farmer else UNIT_ACTION_MAP
    return m.get(act_str, 0)  # Default to PASS if unknown


def decode_unit_action(class_idx, is_farmer=False):
    """Decode integer class back to action list e.g. ['PLACE', 'WHEAT']."""
    actions = ALL_FARMER_ACTIONS if is_farmer else ALL_UNIT_ACTIONS
    if class_idx < 0 or class_idx >= len(actions):
        return ["PASS"]

    act_str = actions[class_idx]
    if act_str.startswith("PLACE_"):
        return ["PLACE", act_str.split("_")[1]]
    return [act_str]


def encode_market_actions(market_orders):
    """
    Encode market orders into a fixed tensor of shape (NUM_MARKET_ITEMS, 3).
    Channels: [buy_flag, sell_flag, log_qty]
    """
    tensor = np.zeros((NUM_MARKET_ITEMS, 3), dtype=np.float32)
    for order in market_orders:
        if not order or len(order) < 3:
            continue
        op, item, qty = order
        if item not in MARKET_ITEM_MAP:
            continue
        idx = MARKET_ITEM_MAP[item]

        if op == "BUY_PRODUCT":
            tensor[idx, 0] = 1.0
        elif op == "SELL":
            tensor[idx, 1] = 1.0

        tensor[idx, 2] = np.log1p(qty)
    return tensor


def decode_market_actions(tensor):
    """
    Decode market tensor (NUM_MARKET_ITEMS, 3) back to list of orders.
    Threshold for buy/sell is > 0.5. If both > 0.5, take the max.
    """
    orders = []
    for idx, item in enumerate(MARKET_ITEMS):
        buy_logit = tensor[idx, 0]
        sell_logit = tensor[idx, 1]
        log_qty = tensor[idx, 2]

        # Check if the raw logit is > 0 (equivalent to probability > 0.5)
        if buy_logit < 0.0 and sell_logit < 0.0:
            continue

        qty = int(np.expm1(log_qty))
        if qty <= 0:
            continue

        if buy_logit > sell_logit:
            orders.append(["BUY_PRODUCT", item, qty])
        else:
            orders.append(["SELL", item, qty])

    return orders
