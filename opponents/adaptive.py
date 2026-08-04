"""`adaptive` sparring partner for local_arena.py.

HONEST PROVENANCE: this is NOT a port of the public
"adaptive-farming-strategy-for-kaggriculture" notebook. That notebook's source
could not be retrieved (kaggle.com renders competition/notebook pages in JS and
returns no readable body to a plain fetch, and the Kaggle API needs credentials
we do not have yet). This is an independently written staged-progression agent
built to the same *idea* — a multi-crop plan that shifts composition as the
season advances, plus hired labour, land expansion and livestock. It exists to
give `main.py` a harder target than the env's built-in `starter`, which farms a
single carrot tile with one farmer and is trivially beaten.

Once credentials are configured, `submit.py --dry-run` prints the command to pull
the real notebook so this can be replaced with a faithful port.
"""

import math

CROPS = {
    "WHEAT":      {"seed": 10, "first_yield_day": 2, "max_yield_day": 4, "max_yield": 6, "ongoing": False},
    "CARROT":     {"seed": 20, "first_yield_day": 2, "max_yield_day": 3, "max_yield": 4, "ongoing": False},
    "TOMATO":     {"seed": 50, "first_yield_day": 8, "max_yield_day": 8, "max_yield": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "max_yield": 4, "ongoing": True},
    "MELON":      {"seed": 80, "first_yield_day": 10, "max_yield_day": 12, "max_yield": 6, "ongoing": False},
}
ANIMALS = {
    "GOOSE": {"cost": 300, "structure": "COOP", "first_yield_day": 4, "max_held": 4, "product": "EGG"},
    "COW":   {"cost": 400, "structure": "PASTURE", "first_yield_day": 8, "max_held": 6, "product": "MILK"},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "first_yield_day": 6, "max_held": 6, "product": "WOOL"},
}
SELLABLE = ["EGG", "MELON", "WOOL", "MILK", "STRAWBERRY", "TOMATO", "CARROT", "WHEAT"]
MOVES = {(0, -1): "NORTH", (0, 1): "SOUTH", (1, 0): "EAST", (-1, 0): "WEST"}

TARGET_HANDS = 8
MAX_COOPS = 14


def _access(board):
    h = board // 2
    return [(h - 1, h - 1), (h, h - 1), (h - 1, h), (h, h)]


def _adjacent_shed(pos, board):
    return (pos[0], pos[1]) in set(_access(board))


def _step_toward(src, dst, board):
    dx, dy = dst[0] - src[0], dst[1] - src[1]
    order = [(1 if dx > 0 else -1, 0), (0, 1 if dy > 0 else -1)]
    if abs(dy) > abs(dx):
        order.reverse()
    for ddx, ddy in order:
        if ddx == 0 and ddy == 0:
            continue
        nx, ny = src[0] + ddx, src[1] + ddy
        if 0 <= nx < board and 0 <= ny < board:
            return MOVES[(ddx, ddy)]
    return None


def _stage_crop(day):
    """Composition shifts as the season advances: fast cash first, then the
    long-dated high-value crop while it can still reach harvest."""
    if day <= 3:
        return "CARROT"
    if day <= 16:
        return "MELON"
    if day <= 25:
        return "CARROT"
    return "WHEAT"


def _decide(obs, config):
    player = int(obs.get("player", 0) or 0)
    farms = obs.get("farms", []) or []
    if not farms or player >= len(farms):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    farm = farms[player]
    private = obs.get("private", {}) or {}
    tiles = farm.get("tiles", []) or []
    board = len(tiles)
    if not board:
        return {"farmer": ["PASS"], "hands": [], "market": []}

    day = int(obs.get("day", 0) or 0)
    step = int(obs.get("step", 0) or 0)
    tpd = 24
    total_days = 30
    if config is not None:
        try:
            tpd = int(config.get("turnsPerDay", 24)) if isinstance(config, dict) else int(getattr(config, "turnsPerDay", 24))
            eps = int(config.get("episodeSteps", 720)) if isinstance(config, dict) else int(getattr(config, "episodeSteps", 720))
            total_days = max(1, eps // max(1, tpd))
        except Exception:
            pass
    days_left = total_days - day

    shed = dict(private.get("shed", {}) or {})
    seeds = dict(private.get("seeds", {}) or {})
    cash = float(farm.get("money", 0.0))
    crop = _stage_crop(day)

    n_animals = sum(
        1 for row in tiles for t in row if isinstance(t, dict) and t.get("animal")
    )
    coops = sum(1 for row in tiles for t in row if isinstance(t, dict) and t.get("kind") == "COOP")

    # ---- market orders ----
    orders = []
    if cash >= 1700 and len(farm.get("unlocked_quadrants", ["NW"])) < 3 and day <= 20:
        orders.append(["BUY_LAND"])
        cash -= 1000
    have_hands = int(farm.get("hires_today", 0) or 0)
    if have_hands < TARGET_HANDS and days_left > 1:
        for _ in range(min(4, TARGET_HANDS - have_hands)):
            orders.append(["HIRE"])
    for item in SELLABLE:
        n = int(shed.get(item, 0) or 0)
        if item == "WHEAT":
            n = max(0, n - n_animals * 2)
        if n > 0:
            orders.append(["SELL", item, n])
    if n_animals > 0 and int(shed.get("WHEAT", 0) or 0) < n_animals * 2:
        orders.append(["BUY_PRODUCT", "WHEAT", n_animals * 2])
    vacant = sum(
        1 for row in tiles
        for t in row if isinstance(t, dict) and t.get("kind") == "COOP" and not t.get("animal")
    )
    if vacant > int(shed.get("GOOSE", 0) or 0) and cash >= 300 and days_left >= 8:
        orders.append(["BUY_ANIMAL", "GOOSE", min(2, vacant)])
        cash -= 300
    if days_left > CROPS[crop]["max_yield_day"] and int(seeds.get(crop, 0) or 0) < 6:
        n = int(min(6, cash // CROPS[crop]["seed"]))
        if n > 0:
            orders.append(["BUY_SEED", crop, n])
    orders = orders[:10]

    # ---- unit tasks ----
    tasks = []
    for y in range(board):
        for x in range(board):
            t = tiles[y][x]
            if t == "LOCKED" or (x, y) in _access(board):
                continue
            if t is None:
                if coops < MAX_COOPS and days_left >= 8 and n_animals + 2 >= coops:
                    tasks.append((520, x, y, ["BUILD_COOP"], None))
                elif int(seeds.get(crop, 0) or 0) > 0 and day + CROPS[crop]["max_yield_day"] <= total_days - 1:
                    tasks.append((500, x, y, ["PLANT", crop], None))
                continue
            if not isinstance(t, dict):
                continue
            kind = t.get("kind")
            if kind == "WEED":
                tasks.append((300, x, y, ["DIG"], None))
            elif kind == "PLANT":
                cd = CROPS.get(t.get("crop"), None)
                if cd is None:
                    continue
                age = day - int(t.get("planted_day", day) or 0)
                units = int(t.get("yield_units", 0) or 0)
                if units > 0 and age >= cd["first_yield_day"]:
                    if cd["ongoing"] or age >= cd["max_yield_day"] or days_left <= 1:
                        tasks.append((850, x, y, ["HARVEST"], None))
                if not t.get("watered_today"):
                    if int(t.get("consecutive_unwatered", 0) or 0) >= 1:
                        tasks.append((950, x, y, ["WATER"], None))
                    else:
                        ws = (cd["max_yield_day"] + 1) // 2
                        if not cd["ongoing"] and ws <= age <= cd["max_yield_day"] and units < cd["max_yield"]:
                            tasks.append((800, x, y, ["WATER"], None))
            elif t.get("animal"):
                if int(t.get("yield_units", 0) or 0) > 0:
                    tasks.append((700, x, y, ["HARVEST"], None))
                if not t.get("fed_today") and days_left > 1:
                    tasks.append((1000 if int(t.get("consecutive_unfed", 0) or 0) >= 1 else 690, x, y, ["FEED"], "WHEAT"))
                if not t.get("cared_today") and days_left > 1:
                    tasks.append((650, x, y, ["CARE"], None))
            elif kind in ("COOP", "PASTURE") and int(shed.get("GOOSE", 0) or 0) + _carry(private, "GOOSE") > 0:
                if kind == "COOP":
                    tasks.append((880, x, y, ["PLACE", "GOOSE"], "GOOSE"))
    tasks.sort(key=lambda a: -a[0])

    units = [(0, tuple(farm.get("farmer", [0, 0])))]
    for i, p in enumerate(farm.get("hands", []) or []):
        units.append((i + 1, tuple(p)))

    assigned, used_u, used_t = {}, set(), set()
    for _prio, tx, ty, op, req in tasks:
        if (tx, ty) in used_t:
            continue
        best, bd = None, None
        for i, (ux, uy) in units:
            if i in used_u:
                continue
            if req is not None and int(_inv(private, i).get(req, 0) or 0) <= 0:
                continue
            d = abs(ux - tx) + abs(uy - ty)
            if bd is None or d < bd:
                best, bd = i, d
        if best is None:
            continue
        ux, uy = dict(units)[best]
        if (ux, uy) == (tx, ty):
            assigned[best] = op
        else:
            s = _step_toward((ux, uy), (tx, ty), board)
            if s is None:
                continue
            assigned[best] = [s]
        used_u.add(best)
        used_t.add((tx, ty))

    for i, pos in units:
        if i in assigned:
            continue
        inv = _inv(private, i)
        if _adjacent_shed(pos, board):
            dropped = False
            for item in SELLABLE:
                n = int(inv.get(item, 0) or 0)
                if item == "WHEAT" and n <= 4:
                    continue
                if n > 0:
                    assigned[i] = ["PLACE", item, n]
                    dropped = True
                    break
            if dropped:
                continue
            if int(shed.get("GOOSE", 0) or 0) > 0:
                assigned[i] = ["PICKUP", "GOOSE", 1]
            elif n_animals > 0 and int(inv.get("WHEAT", 0) or 0) < 4 and int(shed.get("WHEAT", 0) or 0) > 0:
                assigned[i] = ["PICKUP", "WHEAT", 4]
            else:
                assigned[i] = ["PASS"]
            continue
        carrying = sum(int(inv.get(k, 0) or 0) for k in SELLABLE if k != "WHEAT")
        if carrying > 0 or (n_animals > 0 and int(inv.get("WHEAT", 0) or 0) <= 0):
            tgt = min(
                (a for a in _access(board) if tiles[a[1]][a[0]] != "LOCKED"),
                key=lambda a: abs(pos[0] - a[0]) + abs(pos[1] - a[1]),
                default=None,
            )
            if tgt:
                s = _step_toward(pos, tgt, board)
                assigned[i] = [s] if s else ["PASS"]
                continue
        assigned[i] = ["PASS"]

    return {
        "farmer": assigned.get(0, ["PASS"]),
        "hands": [assigned.get(i + 1, ["PASS"]) for i in range(len(units) - 1)],
        "market": orders,
    }


def _inv(private, idx):
    invs = private.get("inventories", []) or []
    return invs[idx] if idx < len(invs) and isinstance(invs[idx], dict) else {}


def _carry(private, item):
    return sum(int(i.get(item, 0) or 0) for i in (private.get("inventories", []) or []) if isinstance(i, dict))


# NOTE: must be the LAST callable in the file — kaggle-environments picks the
# last callable in namespace order, not the one named `agent`.
def agent(obs, config=None):
    try:
        return _decide(obs, config)
    except Exception:
        return {"farmer": ["PASS"], "hands": [], "market": []}
