"""Kaggriculture agent — self-contained submission entrypoint.

Submitted as `main.py` (Kaggle requires a `main.py` at the archive root exposing
`agent`). Standard library only; no sibling imports.

Verified against kaggle-environments 1.32.3 (kaggriculture spec version 0.1.0).
Every mechanic below was read out of
`kaggle_environments/envs/kaggriculture/kaggriculture.py` and re-verified by
direct-call experiments; see README.md "Ground truth" for the measurements.

Strategy in one paragraph
------------------------
Two engines, in sequence. First a melon opening: 12 days and ~$1,400 of seed turn
into roughly $21k, because melon's scarcity side barely moves while the first
~100 units sell near base price -- nothing else in the game returns 15x that
fast. Then an egg engine for the back half, funded by that windfall: EGG is the
only product whose price does not collapse under volume (its glut curve is `log`
at target 0.20, so the 1,600th egg still fetches $37, cumulative $62k), whereas
melon/wool/milk/strawberry hit the $1 floor within 50-200 units. So melon is a
one-shot pot, wool and milk are small capped pots we race the opponent for, and
geese are the only unbounded income. Wheat is feed infrastructure, not a crop:
a wheat tile yields ~1/day and a goose eats exactly 1/day, and growing beats
buying because the buy side ramps on `sqrt` ($26 -> $67 as supply drains).
Capital is ring-fenced for the egg engine before land or seed can claim it, and
everything harvested is sold on sight -- withholding to protect a price measured
strictly worse, and unsold stock scores nothing.

WARNING: the paragraph above describes what this file does, and it is built on a
falsified model of the market. Mining 69 real Kaggle episodes (138 player-seats,
`scripts/mine_daily.py`) shows the market starts at 10,000 inventory per product
and town demand drains it, so prices RISE rather than collapse -- median MILK
160 -> 329, STRAWBERRY 120 -> 294 -- and melon is the ONLY product that ends
below its peak. Top-decile revenue is MILK 32.5%, STRAWBERRY 18.7%, WOOL 16.2%,
and EGG 0.6%. The cumulative-pot table those claims came from was evaluated at a
starting inventory the game never visits. Concretely wrong here: the egg engine
is a rounding error yet `engine_claim` gives it the first claim on capital; wool
and milk are the two largest revenue lines, not small capped pots; buying wheat
beats growing it (the frontier buys ~592/episode on ~1 owned tile); and
FERTILIZE is never emitted even though fertilizer sells for 13.2% of our revenue
and the frontier applies it to strawberry instead. The behaviour is unchanged
pending a measured rework -- see README.md "What the leaderboard data says".

Reading order: FLAGS and the tunable block record what is measured vs assumed;
StrategicPlanner is the macro layer; SpatialScheduler the per-turn one.

Why there is no neural network in here
--------------------------------------
v0.0.8 replaced this file with a behaviour-cloned policy emitting atomic actions.
It scored $2,910 against `baseline` and lost 0/4; this file scores ~$75k on the
same seeds. The failure was structural, not a training-budget problem: the
learned action space had no FEED, CARE or PICKUP token, so the egg engine was
literally inexpressible. Layer B here is an assignment problem with an exact
polynomial-time solution, so learning it can only lose. Learning belongs in the
~20-number macro vector (see search/cem.py), not in the atomic actions.
"""

import json
import math
import os
import time
import traceback

AGENT_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Environment constants (mirrored from kaggriculture.py — do NOT guess these)
# ---------------------------------------------------------------------------

CROPS = {
    "WHEAT": {
        "seed": 10,
        "first_yield_day": 2,
        "max_yield_day": 4,
        "interval": 0,
        "max_yield": 6,
        "ongoing": False,
    },
    "CARROT": {
        "seed": 20,
        "first_yield_day": 2,
        "max_yield_day": 3,
        "interval": 0,
        "max_yield": 4,
        "ongoing": False,
    },
    "TOMATO": {
        "seed": 50,
        "first_yield_day": 8,
        "max_yield_day": 8,
        "interval": 1,
        "max_yield": 4,
        "ongoing": True,
    },
    "STRAWBERRY": {
        "seed": 100,
        "first_yield_day": 10,
        "max_yield_day": 10,
        "interval": 2,
        "max_yield": 4,
        "ongoing": True,
    },
    "MELON": {
        "seed": 80,
        "first_yield_day": 10,
        "max_yield_day": 12,
        "interval": 0,
        "max_yield": 6,
        "ongoing": False,
    },
}

ANIMALS = {
    "GOOSE": {
        "cost": 300,
        "structure": "COOP",
        "first_yield_day": 4,
        "interval": 1,
        "max_held": 4,
        "product": "EGG",
    },
    "COW": {
        "cost": 400,
        "structure": "PASTURE",
        "first_yield_day": 8,
        "interval": 2,
        "max_held": 6,
        "product": "MILK",
    },
    "SHEEP": {
        "cost": 500,
        "structure": "PASTURE",
        "first_yield_day": 6,
        "interval": 3,
        "max_held": 6,
        "product": "WOOL",
    },
}

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]

MARKET_I0 = 10000
PRICE_FLOOR = 1

MARKET_PARAMS = {
    "WHEAT": {
        "base": 25,
        "I0": MARKET_I0,
        "T": 400,
        "below_func": "sqrt",
        "below_target": 0.80,
        "above_func": "log",
        "above_target": 0.20,
    },
    "CARROT": {
        "base": 35,
        "I0": MARKET_I0,
        "T": 450,
        "below_func": "log",
        "below_target": 0.20,
        "above_func": "sqrt",
        "above_target": 0.70,
    },
    "TOMATO": {
        "base": 60,
        "I0": MARKET_I0,
        "T": 200,
        "below_func": "linear",
        "below_target": 0.40,
        "above_func": "sqrt",
        "above_target": 0.60,
    },
    "STRAWBERRY": {
        "base": 120,
        "I0": MARKET_I0,
        "T": 100,
        "below_func": "sqrt",
        "below_target": 0.70,
        "above_func": "linear",
        "above_target": 1.60,
    },
    "MELON": {
        "base": 250,
        "I0": MARKET_I0,
        "T": 300,
        "below_func": "log",
        "below_target": 0.20,
        "above_func": "sq",
        "above_target": 3.60,
    },
    "EGG": {
        "base": 50,
        "I0": MARKET_I0,
        "T": 332,
        "below_func": "linear",
        "below_target": 0.40,
        "above_func": "log",
        "above_target": 0.20,
    },
    "MILK": {
        "base": 160,
        "I0": MARKET_I0,
        "T": 122,
        "below_func": "sqrt",
        "below_target": 0.60,
        "above_func": "linear",
        "above_target": 1.60,
    },
    "WOOL": {
        "base": 200,
        "I0": MARKET_I0,
        "T": 105,
        "below_func": "log",
        "below_target": 0.20,
        "above_func": "sq",
        "above_target": 3.20,
    },
    "FERTILIZER": {
        "base": 100,
        "I0": MARKET_I0,
        "T": 200,
        "below_func": "linear",
        "below_target": 0.40,
        "above_func": "linear",
        "above_target": 0.40,
    },
}

LAND_ORDER = ["NE", "SW", "SE"]
LAND_PRICES = [1000, 2000, 4000]

MOVE_OF_DELTA = {(0, -1): "NORTH", (0, 1): "SOUTH", (1, 0): "EAST", (-1, 0): "WEST"}

# ---------------------------------------------------------------------------
# Tunable constants.
# ---------------------------------------------------------------------------

FLAGS = {
    "HIRE_HANDS": True,
    "EXPAND_LAND": True,
    "PREMIUM_LIVESTOCK": True,
    "ANIMAL_CARE": True,
    # OFF on measurement, v0.1.0. Mined from 150 sampled episodes of the
    # 2026-08-10 daily dump (300 seats): the entire field, every cohort, runs
    # ZERO coops and ZERO geese, and egg is 0.0% of top-decile revenue. Our own
    # 10 live v0.0.9 seats ended with 14.7 coops holding 0.3 geese -- 14.4 of 50
    # tiles were EMPTY coops, 29% of the farm doing nothing, plus the
    # `BUILD_COOP` ops to raise them. A goose is $300 for ~25 eggs at $50-62
    # against a cow at $400 for ~11-22 milk at $169-222, so it loses 3-5x per
    # structure AND occupies a tile that could carry a crop. Switchable so the
    # result stays reproducible; see docs/experiments.md.
    "EGG_ENGINE": False,
    # OFF on measurement, v0.1.0: $13,760 (0W-30L) as a flat price floor and
    # $45,144 vs $58,635 (5W-25L) as a per-order slippage cap. See the
    # SELL_MAX_SLIPPAGE block for why the premise was wrong.
    "SELL_RATE_LIMIT": False,
    # OFF on measurement: -31.3% vs baseline and -40.4% vs adaptive over 30
    # paired seeds (p~0.0, better on 3/30 both times). See
    # SpatialScheduler._assign_optimal for why an exactly optimal matching
    # loses to greedy here. Kept switchable so the result stays reproducible.
    "HUNGARIAN_ASSIGN": False,
}

# Labour: 13 hands matches top #1 leaderboard agents ($187k score insight), with 14.5% cash gate.
MAX_HANDS = 12
HIRES_PER_TURN = 7
HIRE_CASH_FRACTION = 0.0513

# Animal logistics.
FERRY_MAX_UNITS = 6
ANIMAL_BACKLOG_CAP = 8

# Land. $1k/$2k/$4k.
# TUNED. Lowering the buffer to 400 and LAND_LAST_DAY to 22 was tried in v0.1.0 on
# the reasoning that the field ends on 75 owned tiles to our 50, and it is wrong:
# it bought the whole 100-tile board (96.9 owned) for ~$7k and left us without the
# labour to work it. In the field data `corr(cash, owned_tiles)` is -0.139 and
# `corr(cash, land_buys)` is -0.139 -- land is mildly NEGATIVE, not a target. The
# 75-tile figure is what cash-sorted cohorts happen to hold, not what earns.
LAND_CASH_BUFFER = 1961.9271
LAND_EXPAND_SLACK = 2
LAND_RICH_BUFFER = 3000
LAND_LAST_DAY = 16

# Livestock ROI guards & targets (v0.0.2 replay insight: scale up Pasture livestock).
GOOSE_MIN_DAYS_LEFT = 18
SHEEP_MIN_DAYS_LEFT = 10
COW_MIN_DAYS_LEFT = 11
# MAX_SHEEP stays at 6. Cutting to 4 was tried on the reasoning that the field
# runs 4.1 sheep and WOOL's above-target curve is `sq`, and it loses where it
# counts: 30 paired seeds against v0.0.9 give 15W-15L at MAX_SHEEP=4 against
# 23W-7L at 6. The mean prefers 4 ($56,531 vs $54,383) and the WIN RATE prefers 6,
# which is the objective -- the competition ranks pairwise, so a $1 win and a
# $50,000 win score the same. `local_arena --sweep` reports its "best" by mean;
# do not read that line as the answer.
MAX_SHEEP = 6
MAX_COWS = 9
PREMIUM_MIN_TILES = 12

# Wheat feed.
WHEAT_TILES_PER_ANIMAL = 0.4778
WHEAT_FEED_DAYS_RESERVE = 2
WHEAT_MAX_BUY_PRICE = 70
FEED_GATE_MAX = 50
WHEAT_CARRY_PER_UNIT = 6

# Strawberry & Melon targets.
# MELON stays at 9. Cutting it to 3 was tried in v0.1.0 on the reasoning that it
# is 1.3% of top-decile revenue and ends at $25, and it measured 0W-30L against
# v0.0.9 (mean $43,059 vs $58,635). The field's low melon *share* is not evidence
# that melon is unprofitable: the field sells 30.6 melon for $1,410, so the share
# is low because everything else is bigger, not because melon loses money. Our 9
# tiles were earning real cash and the freed land went to strawberry, which needs
# 16 days from planting before it returns anything.
STRAWBERRY_TILE_TARGET = 41
STRAWBERRY_LAND_FRACTION = 0.4058
MELON_TILE_TARGET = 9
MELON_LAND_FRACTION = 0.4866
MELON_LAST_PLANT_DAY = 15

# Sale-rate limiting, v0.1.0. THE finding from the 2026-08-10 field: the top
# decile and the bottom quartile sell almost identical strawberry volume (229.1 vs
# 224.5 units) for 2.9x the revenue ($42,042 vs $14,381) -- $184/unit against
# $64/unit. Volume is not what separates them; realised price is.
#
# The mechanism is the curve. STRAWBERRY is `sqrt` below I0 (amp 8.4) and `linear`
# above it, so $184/unit means selling at inventory ~9942, i.e. into scarcity,
# while $64/unit means pushing ~29 units past I0. Four of the eight shop types buy
# strawberry, so the town drains it faster than anything else; selling at or under
# that drain rate keeps the price above base indefinitely, and dumping outruns it.
#
# Sale-rate limiting. OFF on measurement -- see FLAGS["SELL_RATE_LIMIT"]. Kept
# because the reasoning that motivated it is a trap worth documenting.
#
# The 2026-08-10 field shows the top decile realising $192/unit on strawberry
# against the bottom quartile's $62 at *identical* volume (229.1 vs 224.5 units),
# which reads as an obvious call to stop dumping. Two implementations were tried:
#   * flat floor at 1.0 x base: price only reaches base at inventory <= I0, so it
#     permitted ~1 unit/turn, lost 305 units to shed overflow, scored $13,760
#     (0W-30L vs v0.0.9);
#   * cap each order where its marginal price slipped 10% below its first unit:
#     $45,144 vs $58,635 (5W-25L).
#
# The premise was wrong. Within a single episode the two seats' strawberry $/unit
# differ by a mean of $7.20; between episodes the stdev is $56.80 (range $25-$227),
# and the spread is explained by how many strawberry-buying shops the RNG happened
# to unlock (1 shop -> $37/unit, 6 shops -> $190/unit). Shops are drawn with
# replacement every `townShopUnlockInterval` days, so realised price is mostly an
# episode-level dice roll shared by both players. There was no restraint to learn.
SELL_MAX_SLIPPAGE = 0.10
SELL_LIQUIDATE_DAYS = 2
SHED_PRESSURE = 70

# Layer-B assignment. Only the top-K tasks by priority can ever be reached by
# <= MAX_HANDS+1 units in one turn, so the matrix is truncated to keep the
# O(n*m^2) solver inside the per-turn budget.
HUNGARIAN_MAX_TASKS = 48
# Sentinels for the tiered assignment matrix: any real cost is a Manhattan
# distance on a 10x10 board, so both are unreachable by construction.
_ASSIGN_SLACK = 1.0e6
_ASSIGN_INFEASIBLE = 1.0e9

# Safety guard.
TURN_TIME_BUDGET = 1.0
TIME_GUARD_FRACTION = 0.70
MAX_MARKET_ORDERS = 10

# Task priorities.
PRIO_FEED_URGENT = 1085
PRIO_WATER_URGENT = 913
PRIO_HARVEST_ANIMAL_FULL = 964
PRIO_PLACE_ANIMAL = 880
PRIO_HARVEST_CROP = 986
PRIO_WATER_BONUS = 800
PRIO_CARE = 705
PRIO_FEED = 807
PRIO_COLLECT_FERTILIZER = 650
PRIO_HARVEST_ANIMAL = 600
PRIO_DIG_WEED = 570
PRIO_BUILD = 560
PRIO_PLANT = 550
# A weed on a role tile blocks that tile until dug, so clearing outranks the
# planting it unblocks. At priority 300 weeds accumulated to 20 dead tiles.
PRIO_DIG_WEED = 570

NO_OP = {"farmer": ["PASS"], "hands": [], "market": []}

DECISION_LOG_ENV = "KAGGRICULTURE_DECISION_LOG"


# ---------------------------------------------------------------------------
# Market model
# ---------------------------------------------------------------------------


def _shape(func, x):
    x = max(0.0, x)
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return math.sqrt(x)
    if func == "log":
        return math.log(1.0 + x)
    if func == "log10":
        return math.log10(1.0 + x)
    return x


class MarketAnalyzer:
    """Exact replica of the env price curve, plus the derived quantities the
    planner needs: marginal revenue, and how many units can be sold before the
    price drops through a floor."""

    def __init__(self, market):
        self.inventory = dict(market.get("inventory", {}) or {})
        self.prices = dict(market.get("prices", {}) or {})

    def price_at(self, item, inventory):
        p = MARKET_PARAMS.get(item)
        if p is None:
            return float(PRICE_FLOOR)
        base, i0, t = float(p["base"]), float(p["I0"]), float(p["T"])
        inv = float(inventory)
        if inv < i0:
            f = p["below_func"]
            amp = float(p["below_target"]) * base / _shape(f, t)
            price = base + amp * _shape(f, i0 - inv)
        else:
            f = p["above_func"]
            amp = float(p["above_target"]) * base / _shape(f, t)
            price = base - amp * _shape(f, inv - i0)
        return float(max(PRICE_FLOOR, round(price)))

    def units_sellable_above(self, item, floor, cap=200):
        """How many units clear at >= `floor`, walking the curve the way SELL does.

        `_commit_unit` sells one unit at a time and re-quotes after each, raising
        market inventory by one whenever the unit cleared above the $1 floor. So
        the count is a walk, not a division: the Nth unit of an order is priced at
        `inventory + N - 1`.
        """
        p = MARKET_PARAMS.get(item)
        if p is None:
            return 0
        try:
            inv = int(self.inventory.get(item, p["I0"]))
        except (ValueError, TypeError):
            return 0
        n = 0
        while n < cap:
            if self.price_at(item, inv + n) < floor:
                break
            n += 1
        return n

    def slippage_floor(self, item, slippage):
        """The price at which an order has moved the market `slippage` against us."""
        return self.price(item) * (1.0 - float(slippage))

    def price(self, item):
        """Current sell price (what SELL pays for the next unit)."""
        if item in self.inventory:
            try:
                inv = int(self.inventory[item])
                return self.price_at(item, inv)
            except (ValueError, TypeError):
                pass
        val = self.prices.get(item, PRICE_FLOOR)
        try:
            return float(val)
        except (ValueError, TypeError):
            return float(PRICE_FLOOR)

    def buy_price(self, item):
        """BUY_PRODUCT quotes at post-buy inventory (inv - 1)."""
        try:
            inv = int(self.inventory.get(item, MARKET_I0)) - 1
            return self.price_at(item, inv)
        except (ValueError, TypeError):
            return float(PRICE_FLOOR)


# ---------------------------------------------------------------------------
# Opponent tracking (their farm IS public — only shed/seeds/inventories are not)
# ---------------------------------------------------------------------------


class OpponentTracker:
    """Reads the opponent's public farm: composition, cash, labour, and a profile
    label. `farms` exposes money, tiles, farmer, hands, unlocked_quadrants and
    hires_today for BOTH players, so none of this is guesswork; only their shed,
    seeds and carried inventories are hidden.

    HONEST SCOPE: this currently feeds the decision log only -- it does NOT change
    what the agent does. That is a measured decision, not an oversight. Three ways
    of acting on this data were implemented and A/B'd, and all three lost:

      * backing melon land off when they contest melon: -26.9% (p~0.0, 0/30 seeds)
      * dumping stock ahead of their forecast harvest: exactly 0 delta, 80 seeds
      * withholding stock to defend a price floor:      -1.8% (p~0.00014)

    The pattern is consistent: the shared pots go to whoever produces into them
    first, so tempo dominates and reacting to the opponent costs more than it
    earns. Kept because the profile label is what makes a replay legible when
    asking "why did that episode go badly" -- see local_arena --log-decisions.
    """

    def __init__(self):
        self.profile = "UNKNOWN"
        self.peak_animals = 0
        self.peak_hands = 0

    def update(self, opp_farm, day):
        tiles = opp_farm.get("tiles", []) or []
        self.money = float(opp_farm.get("money", 0.0))
        self.hands = len(opp_farm.get("hands", []) or [])
        self.peak_hands = max(self.peak_hands, self.hands)
        self.animals = {}
        self.crops = {}
        for row in tiles:
            for tile in row:
                if not isinstance(tile, dict):
                    continue
                kind = tile.get("kind")
                if kind == "PLANT":
                    crop = tile.get("crop")
                    self.crops[crop] = self.crops.get(crop, 0) + 1
                elif "animal" in tile and tile.get("animal"):
                    animal = tile["animal"]
                    self.animals[animal] = self.animals.get(animal, 0) + 1
        self.peak_animals = max(self.peak_animals, sum(self.animals.values()))
        self._classify(day)

    def _classify(self, day):
        n_animals = sum(self.animals.values())
        n_melon = self.crops.get("MELON", 0)
        n_crops = sum(self.crops.values())
        if n_animals >= 8 and n_animals > n_crops:
            self.profile = "ANIMAL_LONGTERM"
        elif n_melon >= 8:
            self.profile = "MELON_MAXXER"
        elif day <= 6 and n_crops >= 10:
            self.profile = "EARLY_RUSH"
        elif n_crops or n_animals:
            self.profile = "BALANCED"
        else:
            self.profile = "UNKNOWN"

    def market_pressure(self, item):
        """Estimate price saturation impact based on opponent's active crops/livestock."""
        if item == "MELON":
            n = self.crops.get("MELON", 0)
            if n >= 12:
                return 0.45  # Opponent is heavy Melon maxxer: market will crash hard
            elif n >= 6:
                return 0.70
        elif item == "STRAWBERRY":
            n = self.crops.get("STRAWBERRY", 0)
            if n >= 20:
                return 0.65
        elif item == "GOOSE":
            n = self.animals.get("GOOSE", 0)
            if n >= 12:
                return 0.85
        return 1.0


# ---------------------------------------------------------------------------
# Strategic planner (macro: money, land, labour, tile roles)
# ---------------------------------------------------------------------------


def _load_meta_intelligence():
    """Attempt to load local leaderboard intelligence JSON if available."""
    try:
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "logs", "leaderboard_intelligence.json"
        )
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
                return data.get("top_opening_crops", {})
    except Exception:
        pass
    return {}


class StrategicPlanner:
    """Decides what the farm should *be*: which tile plays which role, how many
    hands to hire, when to buy land and livestock, and what to sell."""

    def __init__(self):
        self.roles = {}  # (x, y) -> "COOP" | "PASTURE_SHEEP" | ... | "MELON" | "WHEAT"
        self.last_role_day = -1

    # -- tile roles ---------------------------------------------------------

    def plan_roles(self, farm, day, days_left, market, cash, opp=None):
        """Assign a role to every unlocked, non-shed-access tile using dynamic ROI auction.

        Ordering matters: animals are placed nearest the shed because they need
        wheat carried out from it every day, melons/strawberries next, wheat
        on the remainder.
        """
        board = len(farm.get("tiles", []) or [])
        if not board:
            return self.roles
        access = _shed_access_tiles(board)
        tiles = farm["tiles"]

        usable = []
        for y in range(board):
            for x in range(board):
                if tiles[y][x] == "LOCKED":
                    continue
                if (x, y) in access:
                    continue  # keep the logistics hub clear
                usable.append((x, y))
        usable.sort(key=lambda p: (_shed_distance(p, board), p[1], p[0]))

        # Animal roles are pinned to tiles that ALREADY carry a structure, before
        # the nearest-shed order is consulted.
        #
        # Without this the allocation churns under land expansion. Roles are
        # recomputed every turn from `usable` sorted by shed distance, and
        # unlocking a quadrant inserts tiles that sort *ahead* of ones already
        # holding a pasture. The first `n_sheep + n_cows` slots then shift onto new
        # ground: the old pasture keeps standing (and keeps its animal), the new
        # tile reads as a vacant animal role, and `BUILD_PASTURE` raises another
        # one. Measured on v0.1.0 with the land buffer lowered: 23.8 pastures built
        # for 13 animals, i.e. ~11 empty pastures plus their build ops -- the same
        # failure the empty coops were, relocated. Cheap to prevent, invisible
        # without a fingerprint of the finished farm.
        def _has_structure(p):
            tile = tiles[p[1]][p[0]]
            return isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE")

        usable.sort(key=lambda p: 0 if _has_structure(p) else 1)

        n_animals_existing = _count_animals(farm)

        # Dynamic ROI valuation, Opponent market pressure, and Leaderboard Meta Intelligence
        melon_press = (
            opp.market_pressure("MELON") if (opp and hasattr(opp, "market_pressure")) else 1.0
        )
        straw_press = (
            opp.market_pressure("STRAWBERRY") if (opp and hasattr(opp, "market_pressure")) else 1.0
        )

        p_melon = market.price("MELON") * melon_press
        p_straw = market.price("STRAWBERRY") * straw_press

        meta_crops = _load_meta_intelligence()
        straw_meta_dominant = meta_crops.get("STRAWBERRY", 0) > meta_crops.get("MELON", 0)

        # Dynamic tile allocation targets
        if (p_straw >= 100 or straw_meta_dominant) and (p_straw > p_melon or melon_press < 0.6):
            strawberry_target = STRAWBERRY_TILE_TARGET if days_left >= 10 else 0
            melon_target = (
                min(6, MELON_TILE_TARGET) if days_left > (29 - MELON_LAST_PLANT_DAY) else 0
            )
        else:
            strawberry_target = STRAWBERRY_TILE_TARGET if days_left >= 10 else 0
            melon_target = MELON_TILE_TARGET if days_left > (29 - MELON_LAST_PLANT_DAY) else 0

        premium_ok = FLAGS["PREMIUM_LIVESTOCK"] and len(usable) >= PREMIUM_MIN_TILES
        n_sheep = MAX_SHEEP if premium_ok else 0
        n_cows = MAX_COWS if premium_ok else 0

        roles = {}
        idx = 0
        # Pasture livestock (high value Wool, Milk, Fertilizer).
        for _ in range(n_sheep):
            if idx < len(usable):
                roles[usable[idx]] = "PASTURE_SHEEP"
                idx += 1
        for _ in range(n_cows):
            if idx < len(usable):
                roles[usable[idx]] = "PASTURE_COW"
                idx += 1

        remaining = len(usable) - idx
        straw_n = min(strawberry_target, int(remaining * STRAWBERRY_LAND_FRACTION))
        rem_2 = max(0, remaining - straw_n)
        melon_n = min(melon_target, int(rem_2 * MELON_LAND_FRACTION))
        rest = max(0, rem_2 - melon_n)

        # Coops: just-in-time coops for geese if pasture is satisfied.
        if FLAGS["EGG_ENGINE"]:
            affordable = int(cash) // ANIMALS["GOOSE"]["cost"]
            coop_cap = n_animals_existing + ANIMAL_BACKLOG_CAP + affordable
            coop_n = round(rest / (1.0 + WHEAT_TILES_PER_ANIMAL))
            coop_n = max(0, min(rest, coop_n, coop_cap))
        else:
            # This line was claiming ~68% of every tile left after strawberry and
            # melon (`rest / 1.478`), which is how 29% of the farm ended up as
            # empty coops. With the egg engine off, `rest` falls through to wheat
            # and then to the strawberry/wheat tail below.
            coop_n = 0

        wheat_n = max(3, math.ceil((n_sheep + n_cows + coop_n) * WHEAT_TILES_PER_ANIMAL))
        wheat_n = min(rest, wheat_n)

        for _ in range(straw_n):
            if idx < len(usable):
                roles[usable[idx]] = "STRAWBERRY"
                idx += 1
        for _ in range(melon_n):
            if idx < len(usable):
                roles[usable[idx]] = "MELON"
                idx += 1
        for _ in range(coop_n):
            if idx < len(usable):
                roles[usable[idx]] = "COOP"
                idx += 1
        for _ in range(wheat_n):
            if idx < len(usable):
                roles[usable[idx]] = "WHEAT"
                idx += 1

        while idx < len(usable):
            if days_left >= 10:
                roles[usable[idx]] = "STRAWBERRY"
            else:
                roles[usable[idx]] = "WHEAT"
            idx += 1

        self.roles = roles
        self.n_animals_existing = n_animals_existing
        return roles

    # -- market orders ------------------------------------------------------

    def market_orders(self, obs, farm, private, market, opp, day, days_left, hour, log):
        """Build the (max 10) market orders for this turn, in priority order."""
        orders = []
        cash = float(farm.get("money", 0.0))
        shed = dict(private.get("shed", {}) or {})
        seeds = dict(private.get("seeds", {}) or {})
        n_animals = _count_animals(farm)

        # Capital priority ladder.
        #
        # A goose costs $300 and, once mature on day 4, returns ~2 eggs/day at
        # ~$40 — roughly $1,000/day across a $3,000 opening flock, a three-day
        # payback. Nothing else in the game comes close, so cash owed to the egg
        # engine is ring-fenced BEFORE land or cash-crop seeds can claim it.
        # Without this, day 0 spent $1,000 on land and ~$800 on carrot seed, the
        # flock never got built, and cash sat at ~$0 from day 5 to day 22.
        # v0.1.0: with FLAGS["EGG_ENGINE"] off there is no goose claim, so land and
        # seed draw on full cash. The ring-fence was giving first call on capital to
        # the game's weakest revenue line (0.0% of top-decile revenue) and is what
        # starved land expansion down to 1.0 plot against the field's 2.0.
        engine_claim = 0.0
        if FLAGS["EGG_ENGINE"] and days_left >= GOOSE_MIN_DAYS_LEFT:
            vacant_coops = _vacant_structures(farm, "COOP", self.roles, "GOOSE")
            pending = int(shed.get("GOOSE", 0) or 0) + _carried_total(private, "GOOSE")
            claim_units = max(0, min(vacant_coops - pending, ANIMAL_BACKLOG_CAP))
            engine_claim = claim_units * ANIMALS["GOOSE"]["cost"]
        # Feed already-owned birds before anything else: an unfed animal escapes
        # after two days and the $300 is gone outright.
        engine_claim += n_animals * WHEAT_FEED_DAYS_RESERVE * market.buy_price("WHEAT")
        discretionary = max(0.0, cash - engine_claim)

        # 1. Land, from discretionary cash only, keeping $500 reserve so Day 0 seed/hire bank is safe.
        if FLAGS["EXPAND_LAND"] and day <= LAND_LAST_DAY:
            n_extra = len(farm.get("unlocked_quadrants", ["NW"])) - 1
            if 0 <= n_extra < len(LAND_PRICES):
                price = LAND_PRICES[n_extra]
                # Expand only once the land we already hold is saturated. Land is
                # not the early constraint -- cash is. Buying NE on day 0 spent
                # a third of the opening bank on tiles we had no birds for.
                # Counts unstocked animal roles of any kind. This used to look at
                # "COOP" alone, which with the egg engine off is always 0 and would
                # have made the saturation gate vacuous.
                unstocked = sum(
                    1
                    for (rx, ry), r in self.roles.items()
                    if r in ("COOP", "PASTURE_SHEEP", "PASTURE_COW")
                    and not (
                        isinstance(farm["tiles"][ry][rx], dict)
                        and farm["tiles"][ry][rx].get("animal")
                    )
                )
                rich = discretionary >= price + LAND_RICH_BUFFER
                if (unstocked <= LAND_EXPAND_SLACK or rich) and (
                    discretionary - 500
                ) >= price + LAND_CASH_BUFFER:
                    orders.append(["BUY_LAND"])
                    cash -= price
                    discretionary -= price
                    log["buy_land"] = LAND_ORDER[n_extra]

        # 2. Hands. Computed here but APPENDED LAST: only 10 market orders are
        #    processed per turn and extras are silently dropped, so emitting
        #    hires early starved the SELL orders and produce rotted in the shed.
        #    Hires can slip a turn at no cost; unsold produce cannot.
        hire_orders = []
        if FLAGS["HIRE_HANDS"]:
            want = self.target_hands(farm, days_left)
            have = int(farm.get("hires_today", 0) or 0)
            budget = cash * HIRE_CASH_FRACTION
            n_hire = 0
            spend = 0.0
            while have + n_hire < want and n_hire < HIRES_PER_TURN:
                c = _hire_cost(have + n_hire)
                if spend + c > budget:
                    break
                spend += c
                n_hire += 1
            for _ in range(n_hire):
                hire_orders.append(["HIRE"])
            if n_hire:
                cash -= spend
                log["hire"] = n_hire

        # 3. Sells. Glut-proof goods unconditionally; capped goods only down to
        #    Sell everything, every turn.
        #
        #    Holding capped goods back to protect their price was measured strictly
        #    worse (-1.8%, p~0.00014, better on only 6/30 self-play seeds): melon
        #    price never recovers enough for the withheld stock to clear, and stock
        #    left in the shed at the end scores nothing. Selling on sight also keeps
        #    the shed under its 100-item cap, past which harvests are discarded.
        # Rate-limited by the curve: see SELL_FLOOR_FRACTION. `liquidate` and
        # `shed_pressure` are the two escapes -- unsold stock scores $0, and a shed
        # at 100 discards every further deposit, including the harvest that would
        # have been the next sale.
        liquidate = days_left <= SELL_LIQUIDATE_DAYS
        shed_total = sum(int(v or 0) for v in shed.values())
        shed_pressure = shed_total >= SHED_PRESSURE
        sells = []
        for item in (
            "EGG",
            "MELON",
            "WOOL",
            "MILK",
            "STRAWBERRY",
            "FERTILIZER",
            "TOMATO",
            "CARROT",
        ):
            # Shed only. `_commit_unit` fills a SELL exclusively from
            # `private["shed"]` and aborts the order the instant it runs dry --
            # carried inventory is never reachable by a market order, whatever tile
            # the unit is standing on. Sizing orders as shed + carried made 73.2%
            # of our ordered SELL volume unfillable (the field's figure is 28.6%),
            # and against `maxMarketOrdersPerTurn = 10` a slot spent on volume that
            # cannot clear is a sale that did not happen.
            held = int(shed.get(item, 0) or 0)
            if held <= 0:
                continue
            n = held
            if FLAGS["SELL_RATE_LIMIT"] and not (liquidate or shed_pressure):
                allowed = market.units_sellable_above(
                    item, market.slippage_floor(item, SELL_MAX_SLIPPAGE)
                )
                n = min(held, max(1, allowed))
            if n <= 0:
                continue
            sells.append((market.price(item) * n, item, n))

        # Highest-value sales first so the 10-order cap never starves the big one.
        sells.sort(reverse=True)
        for _value, item, n in sells:
            orders.append(["SELL", item, n])

        # 4. Wheat: feed first, sell only genuine surplus.
        feed_reserve = n_animals * WHEAT_FEED_DAYS_RESERVE
        wheat_held = int(shed.get("WHEAT", 0) or 0)
        carried = _carried_total(private, "WHEAT")
        if days_left <= 1:
            # Nothing left to feed for: unsold wheat scores $0, so liquidate it.
            if wheat_held > 0:
                orders.append(["SELL", "WHEAT", wheat_held])
        elif wheat_held + carried < feed_reserve:
            need = feed_reserve - wheat_held - carried
            bp = market.buy_price("WHEAT")
            if bp <= WHEAT_MAX_BUY_PRICE and cash >= bp:
                n = max(1, min(need, int(cash * 0.5 // max(1, bp))))
                orders.append(["BUY_PRODUCT", "WHEAT", n])
                cash -= n * bp
                log["buy_wheat"] = [n, bp]
        elif wheat_held > feed_reserve + n_animals * 2:
            orders.append(["SELL", "WHEAT", wheat_held - feed_reserve])

        # 5. Livestock for empty structures we have already built.
        for animal, cap, min_days in (
            ("SHEEP", MAX_SHEEP, SHEEP_MIN_DAYS_LEFT),
            ("COW", MAX_COWS, COW_MIN_DAYS_LEFT),
            ("GOOSE", None, GOOSE_MIN_DAYS_LEFT),
        ):
            if days_left < min_days:
                continue
            if animal == "GOOSE" and not FLAGS["EGG_ENGINE"]:
                continue
            if animal != "GOOSE" and not FLAGS["PREMIUM_LIVESTOCK"]:
                continue
            vacant = _vacant_structures(farm, ANIMALS[animal]["structure"], self.roles, animal)
            owned = _count_animals(farm, animal) + int(shed.get(animal, 0) or 0)
            if cap is not None:
                vacant = min(vacant, max(0, cap - owned))
            in_stock = int(shed.get(animal, 0) or 0) + _carried_total(private, animal)
            want = max(0, vacant - in_stock)
            want = min(want, max(0, ANIMAL_BACKLOG_CAP - in_stock))
            if want <= 0:
                continue
            cost = ANIMALS[animal]["cost"]
            feed_on_hand = int(shed.get("WHEAT", 0) or 0) + _carried_total(private, "WHEAT")
            feed_needed = min((n_animals + 1) * WHEAT_FEED_DAYS_RESERVE, FEED_GATE_MAX)
            if feed_on_hand < feed_needed:
                continue
            budget = cash
            n = int(min(want, budget // cost))
            if n > 0:
                orders.append(["BUY_ANIMAL", animal, n])
                cash -= n * cost
                discretionary -= n * cost
                log.setdefault("buy_animal", {})[animal] = n

        # 6. Seeds for the tiles that need them.
        for crop, last_day in (("STRAWBERRY", 20), ("MELON", MELON_LAST_PLANT_DAY), ("WHEAT", 29)):
            if day > last_day:
                continue
            need = _empty_role_tiles(farm, self.roles, crop)
            have = int(seeds.get(crop, 0) or 0)
            want = max(0, need - have)
            budget = cash if crop == "WHEAT" else discretionary
            if want <= 0:
                continue
            cost = CROPS[crop]["seed"]
            n = int(min(want, budget // cost))
            if n > 0:
                orders.append(["BUY_SEED", crop, n])
                cash -= n * cost
                if crop != "WHEAT":
                    discretionary -= n * cost

        # Hires fill whatever slots are left over.
        orders.extend(hire_orders)
        if len(orders) > MAX_MARKET_ORDERS:
            log["orders_dropped"] = len(orders) - MAX_MARKET_ORDERS
        return orders[:MAX_MARKET_ORDERS]

    def target_hands(self, farm, days_left):
        """Hands worth hiring, bounded by BOTH marginal cost and actual work.

        Hire cost is fib(n) for the n-th hire of a day, which is cheap and then
        suddenly is not: hands 1-12 total $376, but hand 14 alone costs $377,
        hand 16 costs $987. Each hand supplies `turnsPerDay` action-slots, so the
        n-th hand is only worth it while fib(n) / 24 stays under what a slot can
        earn. Hiring to a flat 16 every day cost a measured $22,009 over one
        season — more than the agent's entire final bank — because the last three
        hands cost $41 a slot against actions worth $15-30.

        The second bound is real work: a hand with nothing to do earns nothing at
        any price.
        """
        if days_left <= 0:
            return 0

        if days_left <= 1:
            # Last day: only enough to harvest and sell what already exists.
            n_animals = _count_animals(farm)
            n_plants = _count_plants(farm)
            return max(1, min(MAX_HANDS, math.ceil((n_animals + n_plants) / 6.0)))

        # Deliberately NOT scaling to measured steady-state workload. Two attempts
        # at that both failed badly: a load formula counting current tiles gave 2-3
        # hands, and the farm then built out too slowly to create the work that
        # would justify more (measured: $3,287 final, geese starving because 4 units
        # could not feed 28 birds). Throughput early is worth more than precision --
        # a melon planted on day 0 gets two harvest cycles, one planted on day 5
        # gets one. So hire to the cap and let the fib curve (via cost_cap) and
        # HIRE_CASH_FRACTION do the throttling.
        return MAX_HANDS


# ---------------------------------------------------------------------------
# Optimal assignment (Jonker-Volgenant shortest augmenting path)
# ---------------------------------------------------------------------------


def _hungarian_min_cost(cost, n_rows, n_cols):
    """Minimum-cost perfect assignment of `n_rows` rows into `n_cols` columns.

    Requires n_rows <= n_cols. Returns a list of length n_rows giving each row's
    column, or -1. O(n_rows * n_cols^2) with tiny constants; at n_rows <= 17 and
    n_cols <= HUNGARIAN_MAX_TASKS + 17 this is single-digit milliseconds against
    a 1s turn budget.
    """
    if n_rows <= 0 or n_cols < n_rows:
        return [-1] * max(0, n_rows)

    inf = float("inf")
    u = [0.0] * (n_rows + 1)
    v = [0.0] * (n_cols + 1)
    p = [0] * (n_cols + 1)
    way = [0] * (n_cols + 1)

    for i in range(1, n_rows + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (n_cols + 1)
        used = [False] * (n_cols + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = -1
            row = cost[i0 - 1]
            ui = u[i0]
            for j in range(1, n_cols + 1):
                if used[j]:
                    continue
                cur = row[j - 1] - ui - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            if j1 < 0:
                break
            for j in range(n_cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    out = [-1] * n_rows
    for j in range(1, n_cols + 1):
        if p[j]:
            out[p[j] - 1] = j - 1
    return out


# ---------------------------------------------------------------------------
# Spatial scheduler (micro: who does what, this turn)
# ---------------------------------------------------------------------------


class SpatialScheduler:
    """Optimal (unit, task) assignment with one-unit-per-tile reservation.

    Layer B carries no strategic content: it is a weighted bipartite matching,
    and matching has an exact polynomial-time solution. Greedy-by-priority was a
    1/2-approximation that routinely burned the only unit near task B on task A,
    so `assign` now solves the whole matrix with `_hungarian_min_cost`.

    Still deliberately NOT multi-agent pathfinding: units may share tiles (the
    env allows it), so there are no collisions to resolve.
    """

    def build_tasks(self, farm, private, roles, day, step, days_left, market):
        """Enumerate every useful (tile, op) for this turn with a priority."""
        tiles = farm.get("tiles", []) or []
        board = len(tiles)
        seeds = dict(private.get("seeds", {}) or {})
        shed = dict(private.get("shed", {}) or {})
        tasks = []

        seed_budget = dict(seeds)
        animal_budget = {a: int(shed.get(a, 0) or 0) + _carried_total(private, a) for a in ANIMALS}

        for y in range(board):
            for x in range(board):
                tile = tiles[y][x]
                if tile == "LOCKED":
                    continue
                role = roles.get((x, y))

                if tile is None:
                    if role == "COOP":
                        tasks.append((PRIO_BUILD, x, y, ["BUILD_COOP"], None))
                    elif role in ("PASTURE_SHEEP", "PASTURE_COW"):
                        tasks.append((PRIO_BUILD, x, y, ["BUILD_PASTURE"], None))
                    elif role in ("STRAWBERRY", "MELON", "WHEAT"):
                        crop = role
                        last = (
                            20
                            if crop == "STRAWBERRY"
                            else (MELON_LAST_PLANT_DAY if crop == "MELON" else 29)
                        )
                        # Only plant if it can still reach harvest before the end.
                        if day <= last and seed_budget.get(crop, 0) > 0:
                            if day + CROPS[crop]["first_yield_day"] <= 29:
                                seed_budget[crop] -= 1
                                tasks.append((PRIO_PLANT, x, y, ["PLANT", crop], None))
                    continue

                if not isinstance(tile, dict):
                    continue
                kind = tile.get("kind")

                if kind == "WEED":
                    if role is not None:
                        tasks.append((PRIO_DIG_WEED, x, y, ["DIG"], None))
                    continue

                if kind == "PLANT":
                    tasks.extend(self._plant_tasks(tile, x, y, day, days_left))
                    continue

                if "animal" in tile and tile.get("animal"):
                    tasks.extend(self._animal_tasks(tile, x, y, day, days_left, market))
                    continue

                # Empty coop / pasture: place an animal if we are carrying one.
                want = _animal_for_structure(kind, role)
                if want and animal_budget.get(want, 0) > 0:
                    animal_budget[want] -= 1
                    tasks.append((PRIO_PLACE_ANIMAL, x, y, ["PLACE", want], want))

        tasks.sort(key=lambda t: -t[0])
        return tasks

    def _plant_tasks(self, tile, x, y, day, days_left):
        crop = tile.get("crop")
        cd = CROPS.get(crop)
        if cd is None:
            return []
        out = []
        age = day - int(tile.get("planted_day", day) or 0)
        units = int(tile.get("yield_units", 0) or 0)
        watered = bool(tile.get("watered_today"))
        unwatered = int(tile.get("consecutive_unwatered", 0) or 0)

        # Harvest when mature. One-time crops: at max_yield_day (full yield), or
        # immediately if the season is ending, or if decay has started.
        if units > 0 and age >= cd["first_yield_day"]:
            if cd["ongoing"]:
                out.append((PRIO_HARVEST_CROP, x, y, ["HARVEST"], None))
            else:
                decaying = age > cd["max_yield_day"]
                ending = days_left <= 1
                if age >= cd["max_yield_day"] or decaying or ending:
                    out.append((PRIO_HARVEST_CROP, x, y, ["HARVEST"], None))

        if not watered:
            # Survival: unwatered>=1 means the end-of-day refresh kills it.
            if unwatered >= 1:
                # No point saving a plant that will never be harvested in time.
                if cd["ongoing"] or day + max(0, cd["max_yield_day"] - age) <= 29:
                    out.append((PRIO_WATER_URGENT, x, y, ["WATER"], None))
            # Yield: watering inside the bonus window adds a unit per day.
            window_start = (cd["max_yield_day"] + 1) // 2
            if not cd["ongoing"] and window_start <= age <= cd["max_yield_day"]:
                if units < cd["max_yield"]:
                    out.append((PRIO_WATER_BONUS, x, y, ["WATER"], None))
            elif cd["ongoing"] and tile.get("fertilized_until_day", -1) >= day:
                out.append((PRIO_WATER_BONUS, x, y, ["WATER"], None))
        return out

    def _animal_tasks(self, tile, x, y, day, days_left, market):
        animal = tile["animal"]
        a = ANIMALS[animal]
        out = []
        units = int(tile.get("yield_units", 0) or 0)
        fed = bool(tile.get("fed_today"))
        unfed = int(tile.get("consecutive_unfed", 0) or 0)
        cared = bool(tile.get("cared_today"))

        # Harvest: urgent at max_held (further production would be discarded).
        if units > 0:
            prio = PRIO_HARVEST_ANIMAL_FULL if units >= a["max_held"] else PRIO_HARVEST_ANIMAL
            if days_left <= 1:
                prio = PRIO_HARVEST_ANIMAL_FULL
            out.append((prio, x, y, ["HARVEST"], None))

        # Feed. Only worth wheat if the animal still has production left to give.
        worth_keeping = days_left > 1
        if not fed and worth_keeping:
            prio = PRIO_FEED_URGENT if unfed >= 1 else PRIO_FEED
            out.append((prio, x, y, ["FEED"], "WHEAT"))

        # CARE banks +1 yield paid out on the next production tick.
        if FLAGS["ANIMAL_CARE"] and not cared and worth_keeping:
            if units < a["max_held"]:
                out.append((PRIO_CARE, x, y, ["CARE"], None))

        # Collect fertilizer when available on animal tiles (sells for $100/unit base).
        if tile.get("fertilizer_available"):
            out.append((PRIO_COLLECT_FERTILIZER, x, y, ["COLLECT_FERTILIZER"], None))

        return out

    def assign(self, tasks, units, farm, private, board, n_animals, roles, log):
        """Assign units to tasks, then give whatever is left over to logistics."""
        actions = {}
        used_units = set()
        used_tiles = set()

        local_shed = dict(private.get("shed", {}) or {})

        self._ferry_animals(
            actions, used_units, used_tiles, units, farm, private, board, roles, log, local_shed
        )
        self._provision_feed(actions, used_units, units, farm, private, board, log, local_shed)

        free = [(i, p) for i, p in units if i not in used_units]
        if FLAGS["HUNGARIAN_ASSIGN"]:
            self._assign_optimal(
                actions, used_units, used_tiles, free, tasks, farm, private, board, log
            )
        else:
            self._assign_greedy(actions, used_units, used_tiles, free, tasks, farm, private, board)

        # Idle units: run logistics — fetch feed, ferry animals, drop produce.
        idle = [(i, p) for i, p in units if i not in used_units]
        for i, pos in idle:
            actions[i] = self._logistics(i, pos, farm, private, board, n_animals, local_shed)
        log["idle_units"] = len(idle)
        log["assigned_units"] = len(used_units)
        return actions

    @staticmethod
    def _tiers(tasks):
        """Split the priority-sorted task list into equal-priority tiers.

        Tiles are NOT deduplicated across tiers. A tile usually carries several
        tasks (an animal has FEED, HARVEST and CARE), and keeping only the
        top-priority one silently drops the fallback: when nobody is carrying
        wheat the FEED is unassignable, and the tile then has to stay eligible
        for its HARVEST. Collapsing to one task per tile cost -44.6%.
        """
        tiers = []
        for task in tasks:
            if tiers and tiers[-1][0][0] == task[0]:
                tiers[-1].append(task)
            else:
                tiers.append([task])
        return tiers

    def _assign_optimal(
        self, actions, used_units, used_tiles, free, tasks, farm, private, board, log
    ):
        """Priority-tiered optimal assignment: exact matching *inside* each tier.

        A flat max-value matching over `priority - k * travel` loses badly
        (-38.0% on 8 paired seeds), and the reason is that priorities are not
        utilities, they are deadline classes: an unwatered plant dies tonight,
        so trading 50 priority for 6 tiles of travel is never the deal it looks
        like on paper. So the objective is lexicographic instead — serve the top
        tier first, and only inside a tier is total travel minimised.

        That is exactly greedy's semantics with greedy's one real defect fixed:
        greedy hands out a tier's tasks in list order, so the first task can
        take the only unit that was near the second. Within a tier this solves
        the whole matrix at once.
        """
        remaining = list(free)
        if not remaining or not tasks:
            return

        n_matched = 0
        for tier in self._tiers(tasks):
            if not remaining:
                break
            cols = []
            seen = set()
            for t in tier:
                key = (t[1], t[2])
                if key in used_tiles or key in seen:
                    continue
                seen.add(key)
                cols.append(t)
                if len(cols) >= HUNGARIAN_MAX_TASKS:
                    break
            if not cols:
                continue
            n, m = len(remaining), len(cols)
            n_cols = m + n  # trailing slack columns == "this unit stays free"
            cost = []
            for i, (ux, uy) in remaining:
                inv = _inv_of(private, i)
                row = [_ASSIGN_SLACK] * n_cols
                for c, (_prio, tx, ty, _op, req) in enumerate(cols):
                    if req is not None and int(inv.get(req, 0) or 0) <= 0:
                        row[c] = _ASSIGN_INFEASIBLE
                    else:
                        row[c] = float(abs(ux - tx) + abs(uy - ty))
                cost.append(row)

            still_free = []
            for r, c in enumerate(_hungarian_min_cost(cost, n, n_cols)):
                i, (ux, uy) = remaining[r]
                if c < 0 or c >= m or cost[r][c] >= _ASSIGN_SLACK:
                    still_free.append(remaining[r])
                    continue
                _prio, tx, ty, op, _req = cols[c]
                if (ux, uy) == (tx, ty):
                    actions[i] = op
                else:
                    step_op = _step_toward((ux, uy), (tx, ty), farm, board)
                    if step_op is None:
                        still_free.append(remaining[r])
                        continue
                    actions[i] = [step_op]
                used_units.add(i)
                used_tiles.add((tx, ty))
                n_matched += 1
            remaining = still_free
        log["hungarian"] = [len(free), len(tasks), n_matched]

    @staticmethod
    def _assign_greedy(actions, used_units, used_tiles, free, tasks, farm, private, board):
        """Pre-v0.0.9 behaviour, kept so `--ablate HUNGARIAN_ASSIGN` is a real A/B."""
        positions = dict(free)
        for _prio, tx, ty, op, req in tasks:
            if (tx, ty) in used_tiles:
                continue
            best = None
            best_d = None
            for i, (ux, uy) in free:
                if i in used_units:
                    continue
                if req is not None and _inv_count(private, i, req) <= 0:
                    continue
                d = abs(ux - tx) + abs(uy - ty)
                if best_d is None or d < best_d:
                    best, best_d = i, d
            if best is None:
                continue
            ux, uy = positions[best]
            if (ux, uy) == (tx, ty):
                actions[best] = op
            else:
                step_op = _step_toward((ux, uy), (tx, ty), farm, board)
                if step_op is None:
                    continue
                actions[best] = [step_op]
            used_units.add(best)
            used_tiles.add((tx, ty))

    def _ferry_animals(
        self, actions, used_units, used_tiles, units, farm, private, board, roles, log, local_shed
    ):
        """Dedicate units to the two-step chain that stocks a structure.

        Placing one goose is worth roughly $1.5k over a season — far more than
        any watering — but it takes two coupled actions on different tiles:
        PICKUP at the shed, then PLACE on the structure. The generic task loop
        cannot express that coupling, because a PLACE task is only assignable to
        a unit that ALREADY carries the animal. With every unit busy watering,
        nobody ever became idle at the shed, so purchased geese piled up unplaced
        (measured: 40 geese in the shed, 0 on tiles, by day 21). This pass runs
        before the generic loop and reserves units for the whole chain.
        """
        tiles = farm.get("tiles", []) or []

        vacancies = []
        for y, row in enumerate(tiles):
            for x, tile in enumerate(row):
                if not isinstance(tile, dict) or tile.get("animal"):
                    continue
                if tile.get("kind") not in ("COOP", "PASTURE"):
                    continue
                want = _animal_for_structure(tile.get("kind"), roles.get((x, y)))
                if want:
                    vacancies.append((x, y, want))
        if not vacancies:
            return

        # Step 2 first: anyone already carrying an animal delivers it.
        for i, (ux, uy) in units:
            if i in used_units:
                continue
            inv = _inv_of(private, i)
            carried = [a for a in ("GOOSE", "SHEEP", "COW") if int(inv.get(a, 0) or 0) > 0]
            if not carried:
                continue
            animal = carried[0]
            targets = [v for v in vacancies if v[2] == animal and (v[0], v[1]) not in used_tiles]
            if not targets:
                continue
            tx, ty, _ = min(targets, key=lambda v: abs(ux - v[0]) + abs(uy - v[1]))
            if (ux, uy) == (tx, ty):
                actions[i] = ["PLACE", animal]
            else:
                step_op = _step_toward((ux, uy), (tx, ty), farm, board)
                if step_op is None:
                    continue
                actions[i] = [step_op]
            used_units.add(i)
            used_tiles.add((tx, ty))

        # Step 1: send units to the shed to collect stock for the rest.
        open_vac = [v for v in vacancies if (v[0], v[1]) not in used_tiles]
        fetchable = []
        for _x, _y, animal in open_vac:
            if int(local_shed.get(animal, 0) or 0) > sum(1 for a in fetchable if a == animal):
                fetchable.append(animal)
        if not fetchable:
            return
        n_fetch = min(len(fetchable), FERRY_MAX_UNITS)
        free = [(i, p) for i, p in units if i not in used_units]
        # Nearest to the shed goes, so the round trip is shortest.
        free.sort(key=lambda u: _shed_distance(u[1], board))
        for k in range(min(n_fetch, len(free))):
            i, pos = free[k]
            animal = fetchable[k]
            if _is_shed_adjacent(pos, board):
                actions[i] = ["PICKUP", animal, 1]
            else:
                target = _nearest_shed_access(pos, farm, board)
                if target is None:
                    continue
                step_op = _step_toward(pos, target, farm, board)
                if step_op is None:
                    continue
                actions[i] = [step_op]
            used_units.add(i)
            local_shed[animal] = local_shed.get(animal, 0) - 1
        log["ferry"] = len(used_units)

    def _provision_feed(self, actions, used_units, units, farm, private, board, log, local_shed):
        """Make sure somebody is carrying wheat before animals need feeding.

        FEED consumes wheat from the acting unit's own inventory, so a FEED task
        is unassignable unless a unit already carries wheat. That only happened
        by accident, when a unit fell through to idle logistics — so on busy
        turns animals went unfed, and two consecutive unfed days makes them
        escape permanently. This reserves carriers up front.
        """
        tiles = farm.get("tiles", []) or []
        unfed = 0
        for row in tiles:
            for tile in row:
                if isinstance(tile, dict) and tile.get("animal") and not tile.get("fed_today"):
                    unfed += 1
        if unfed <= 0:
            return
        shed_wheat = int(local_shed.get("WHEAT", 0) or 0)
        if shed_wheat <= 0:
            return

        free = [(i, p) for i, p in units if i not in used_units]
        carried = sum(_inv_count(private, i, "WHEAT") for i, _ in free)
        if carried >= unfed:
            return
        short = unfed - carried
        need_carriers = math.ceil(short / float(max(1, WHEAT_CARRY_PER_UNIT)))
        need_carriers = min(need_carriers, FERRY_MAX_UNITS)

        free.sort(key=lambda u: _shed_distance(u[1], board))
        sent = 0
        for i, pos in free:
            if sent >= need_carriers:
                break
            if _inv_count(private, i, "WHEAT") > 0:
                continue
            take = min(WHEAT_CARRY_PER_UNIT, shed_wheat)
            if take <= 0:
                break
            if _is_shed_adjacent(pos, board):
                actions[i] = ["PICKUP", "WHEAT", take]
            else:
                target = _nearest_shed_access(pos, farm, board)
                if target is None:
                    continue
                step_op = _step_toward(pos, target, farm, board)
                if step_op is None:
                    continue
                actions[i] = [step_op]
            used_units.add(i)
            sent += 1
            shed_wheat -= take
            local_shed["WHEAT"] = local_shed.get("WHEAT", 0) - take
        if sent:
            log["provision_feed"] = sent

    def _logistics(self, idx, pos, farm, private, board, n_animals, local_shed):
        """What a unit with no field task should do: keep the supply chain fed."""
        shed = local_shed
        inv = _inv_of(private, idx)
        adjacent = _is_shed_adjacent(pos, board)

        if adjacent:
            # Drop produce selectively (PLACE keeps our carried wheat; DROP would
            # dump it too). Only produce we can actually sell.
            for item in ("EGG", "MELON", "WOOL", "MILK", "STRAWBERRY", "TOMATO", "CARROT"):
                n = int(inv.get(item, 0) or 0)
                if n > 0:
                    return ["PLACE", item, n]
            # Ferry an animal out to a vacant structure.
            for animal in ("GOOSE", "SHEEP", "COW"):
                if int(shed.get(animal, 0) or 0) > 0:
                    shed[animal] = shed.get(animal, 0) - 1
                    return ["PICKUP", animal, 1]
            # Stock up on feed.
            if n_animals > 0 and int(inv.get("WHEAT", 0) or 0) < WHEAT_CARRY_PER_UNIT:
                if int(shed.get("WHEAT", 0) or 0) > 0:
                    take = min(WHEAT_CARRY_PER_UNIT, shed.get("WHEAT", 0))
                    shed["WHEAT"] = shed.get("WHEAT", 0) - take
                    return ["PICKUP", "WHEAT", take]
            # Surplus wheat we grew is in inventory, not the shed: bank it.
            if int(inv.get("WHEAT", 0) or 0) > WHEAT_CARRY_PER_UNIT * 2:
                return ["PLACE", "WHEAT", int(inv["WHEAT"]) - WHEAT_CARRY_PER_UNIT]
            return ["PASS"]

        # Not at the shed: go there if we have a reason to.
        carrying = sum(
            int(inv.get(i, 0) or 0)
            for i in ("EGG", "MELON", "WOOL", "MILK", "STRAWBERRY", "TOMATO", "CARROT")
        )
        need_feed = n_animals > 0 and int(inv.get("WHEAT", 0) or 0) <= 0
        goose_in_shed = int(shed.get("GOOSE", 0) or 0) > 0
        wheat_in_shed = int(shed.get("WHEAT", 0) or 0) > 0

        if carrying > 0 or need_feed or goose_in_shed:
            if carrying <= 0:
                if goose_in_shed:
                    shed["GOOSE"] -= 1
                elif need_feed and wheat_in_shed:
                    take = min(WHEAT_CARRY_PER_UNIT, shed["WHEAT"])
                    shed["WHEAT"] -= take
            target = _nearest_shed_access(pos, farm, board)
            if target:
                step_op = _step_toward(pos, target, farm, board)
                if step_op:
                    return [step_op]
        return ["PASS"]


# ---------------------------------------------------------------------------
# Geometry / farm helpers
# ---------------------------------------------------------------------------


def _shed_access_tiles(board):
    half = board // 2
    return [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]


def _is_shed_adjacent(pos, board):
    return (pos[0], pos[1]) in set(_shed_access_tiles(board))


def _shed_distance(pos, board):
    return min(abs(pos[0] - ax) + abs(pos[1] - ay) for ax, ay in _shed_access_tiles(board))


def _nearest_shed_access(pos, farm, board):
    """Nearest shed-access tile that is not locked (locked tiles block ops but
    not movement; we still want a usable one)."""
    tiles = farm.get("tiles", []) or []
    best, best_d = None, None
    for ax, ay in _shed_access_tiles(board):
        if tiles[ay][ax] == "LOCKED":
            continue
        d = abs(pos[0] - ax) + abs(pos[1] - ay)
        if best_d is None or d < best_d:
            best, best_d = (ax, ay), d
    return best


def _step_toward(src, dst, farm, board):
    """One orthogonal step reducing Manhattan distance. Movement onto LOCKED
    tiles is legal in this env, so no obstacle avoidance is needed."""
    dx = dst[0] - src[0]
    dy = dst[1] - src[1]
    # Move along the longer axis first: fewer turns spent boxed against an edge.
    order = []
    if abs(dx) >= abs(dy):
        order = [(1 if dx > 0 else -1, 0), (0, 1 if dy > 0 else -1)]
    else:
        order = [(0, 1 if dy > 0 else -1), (1 if dx > 0 else -1, 0)]
    for ddx, ddy in order:
        if ddx == 0 and ddy == 0:
            continue
        nx, ny = src[0] + ddx, src[1] + ddy
        if 0 <= nx < board and 0 <= ny < board:
            return MOVE_OF_DELTA[(ddx, ddy)]
    return None


def _count_animals(farm, animal=None):
    n = 0
    for row in farm.get("tiles", []) or []:
        for tile in row:
            if isinstance(tile, dict) and tile.get("animal"):
                if animal is None or tile["animal"] == animal:
                    n += 1
    return n


def _count_plants(farm):
    n = 0
    for row in farm.get("tiles", []) or []:
        for tile in row:
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                n += 1
    return n


def _count_structures(farm):
    """Coops/pastures still waiting for an animal."""
    n = 0
    for row in farm.get("tiles", []) or []:
        for tile in row:
            if isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE"):
                if not tile.get("animal"):
                    n += 1
    return n


def _animal_for_structure(kind, role):
    if kind == "COOP":
        return "GOOSE"
    if kind == "PASTURE":
        if role == "PASTURE_COW":
            return "COW"
        return "SHEEP"
    return None


def _vacant_structures(farm, structure, roles, animal):
    n = 0
    tiles = farm.get("tiles", []) or []
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            if not isinstance(tile, dict) or tile.get("kind") != structure:
                continue
            if tile.get("animal"):
                continue
            if structure == "PASTURE":
                if _animal_for_structure(structure, roles.get((x, y))) != animal:
                    continue
            n += 1
    return n


def _empty_role_tiles(farm, roles, role):
    """Tiles assigned `role` that are currently empty (need a seed)."""
    n = 0
    tiles = farm.get("tiles", []) or []
    for (x, y), r in roles.items():
        if r != role:
            continue
        if y < len(tiles) and x < len(tiles[y]) and tiles[y][x] is None:
            n += 1
    return n


def _inv_of(private, idx):
    invs = private.get("inventories", []) or []
    if idx < len(invs) and isinstance(invs[idx], dict):
        return invs[idx]
    return {}


def _inv_count(private, idx, item):
    return int(_inv_of(private, idx).get(item, 0) or 0)


def _carried_total(private, item):
    total = 0
    for inv in private.get("inventories", []) or []:
        if isinstance(inv, dict):
            total += int(inv.get(item, 0) or 0)
    return total


def _carried_adjacent_to_shed(farm, private, item):
    board = len(farm.get("tiles", []) or [])
    if not board:
        return 0
    total = 0
    pos = tuple(farm.get("farmer", [0, 0]))
    if _is_shed_adjacent(pos, board):
        total += _inv_count(private, 0, item)
    for i, p in enumerate(farm.get("hands", []) or []):
        if _is_shed_adjacent(tuple(p), board):
            total += _inv_count(private, i + 1, item)
    return total


def _fib(n):
    """_fib(0)=1, _fib(1)=1, _fib(2)=2, ... matching the env's hire cost."""
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _hire_cost(n_already_today):
    return _fib(n_already_today)


# ---------------------------------------------------------------------------
# Per-player persistent state
#
# Keyed by player index rather than held in a single global, so that running the
# same imported function as BOTH players (self-play in local_arena) cannot let
# one player's plan leak into the other's.
# ---------------------------------------------------------------------------

_STATE: dict = {}


class _PlayerState:
    def __init__(self):
        self.planner = StrategicPlanner()
        self.scheduler = SpatialScheduler()
        self.opponent = OpponentTracker()
        self.turns = 0
        self.errors = 0
        self.last_step = -1


def _state_for(player, step):
    st = _STATE.get(player)
    # A step that goes backwards means a new episode in the same process.
    if st is None or step < st.last_step:
        st = _PlayerState()
        _STATE[player] = st
    st.last_step = step
    return st


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def _decide(obs, config, st, log):
    player = int(obs.get("player", 0) or 0)
    farms = obs.get("farms", []) or []
    if not farms or player >= len(farms):
        return NO_OP
    farm = farms[player]
    private = obs.get("private", {}) or {}
    tiles = farm.get("tiles", []) or []
    board = len(tiles)
    if not board:
        return NO_OP

    turns_per_day = int(_cfg(config, "turnsPerDay", 24))
    episode_steps = int(_cfg(config, "episodeSteps", 720))
    step = int(obs.get("step", 0) or 0)
    day = int(obs.get("day", step // max(1, turns_per_day)) or 0)
    hour = int(obs.get("hour", step % max(1, turns_per_day)) or 0)
    total_days = max(1, episode_steps // max(1, turns_per_day))
    days_left = total_days - day

    market = MarketAnalyzer(obs.get("market", {}) or {})
    if len(farms) > 1:
        st.opponent.update(farms[1 - player], day)
    log["day"] = day
    log["hour"] = hour
    log["money"] = round(float(farm.get("money", 0.0)), 1)
    log["opp_profile"] = st.opponent.profile

    # --- Layer A: strategy -------------------------------------------------
    roles = st.planner.plan_roles(
        farm, day, days_left, market, float(farm.get("money", 0.0)), opp=st.opponent
    )
    orders = st.planner.market_orders(
        obs, farm, private, market, st.opponent, day, days_left, hour, log
    )

    # --- Layer B: scheduling ----------------------------------------------
    n_animals = _count_animals(farm)
    units = [(0, tuple(farm.get("farmer", [0, 0])))]
    for i, pos in enumerate(farm.get("hands", []) or []):
        units.append((i + 1, tuple(pos)))

    tasks = st.scheduler.build_tasks(farm, private, roles, day, step, days_left, market)
    log["n_tasks"] = len(tasks)
    log["n_units"] = len(units)
    assigned = st.scheduler.assign(tasks, units, farm, private, board, n_animals, roles, log)

    farmer_action = assigned.get(0, ["PASS"])
    hands_actions = [assigned.get(i + 1, ["PASS"]) for i in range(len(units) - 1)]

    return {
        "farmer": _validate_unit_op(farmer_action),
        "hands": [_validate_unit_op(a) for a in hands_actions],
        "market": orders,
    }


UNIT_OPS = {
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "PASS",
    "PICKUP",
    "PLACE",
    "DROP",
    "PLANT",
    "WATER",
    "HARVEST",
    "FERTILIZE",
    "BUILD_COOP",
    "BUILD_PASTURE",
    "DIG",
    "FEED",
    "COLLECT_FERTILIZER",
    "CARE",
}


def _validate_unit_op(action):
    """Last line of defence: emit only ops the interpreter recognises. Invalid
    actions are silent no-ops in this env, but dropping them here keeps the
    decision log honest about what we actually intended."""
    if not isinstance(action, list) or not action:
        return ["PASS"]
    if action[0] not in UNIT_OPS:
        return ["PASS"]
    return action


def _cfg(config, key, default):
    if config is None:
        return default
    try:
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)
    except Exception:
        return default


def _write_log(player, step, log, action):
    """Append one JSON line per turn when the arena asks for it.

    Enabled only via the KAGGRICULTURE_DECISION_LOG env var, so a graded episode
    on Kaggle does no file I/O at all.
    """
    dest = os.environ.get(DECISION_LOG_ENV)
    if not dest:
        return
    import json

    rec = {"player": player, "step": step, "action": action}
    rec.update(log)
    with open(dest, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


# NOTE: `agent` MUST be the last callable defined in this file.
# kaggle-environments loads a file-based agent with
# `[v for v in env.values() if callable(v)][-1]` (agent.py:get_last_callable),
# i.e. it takes the LAST callable in module namespace order, not the one named
# `agent`. Defining any helper below this point silently makes THAT the agent
# and every turn of the episode errors out. submit.py pre-flight asserts this.
def agent(obs, config=None):
    """Kaggriculture entrypoint.

    Wrapped per the safety-guard requirement: a hard time check against
    actTimeout, and a blanket try/except that degrades to a legal no-op rather
    than ever erroring out of the match.
    """
    t0 = time.perf_counter()
    log = {}
    player = 0
    try:
        player = int(obs.get("player", 0) or 0)
        step = int(obs.get("step", 0) or 0)
        st = _state_for(player, step)
        st.turns += 1

        action = _decide(obs, config, st, log)

        elapsed = time.perf_counter() - t0
        budget = float(_cfg(config, "actTimeout", TURN_TIME_BUDGET) or TURN_TIME_BUDGET)
        if elapsed > budget * TIME_GUARD_FRACTION:
            # Over the guard: keep this turn's action but record it. The action is
            # already computed, so returning it costs nothing extra; the log lets
            # the arena flag the turn as a near-timeout.
            log["slow_turn"] = round(elapsed, 4)
        log["elapsed"] = round(elapsed, 5)
        _write_log(player, step, log, action)
        return action
    except Exception:
        st = _STATE.get(player)
        if st is not None:
            st.errors += 1
        log["exception"] = traceback.format_exc()
        try:
            _write_log(player, int(obs.get("step", -1) or -1), log, NO_OP)
        except Exception:
            pass
        return NO_OP
