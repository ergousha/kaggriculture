"""Offline shortest-path re-planning over a fixed route (issue #29).

Why this exists. A unit-op census of the shipped route says half of all labour
is walking: 3,484 of 6,999 unit-turns are `NORTH`/`SOUTH`/`EAST`/`WEST`, against
1,646 productive tile ops. Issue #29's premise is that this is the one part of
the game with an exact algorithm — the route is fixed, the board is
deterministic, and between two consecutive tile ops a unit's path is free, so
any walk that arrives by the turn the next op is scheduled is equivalent.

What this module does, and only this:

  * `simulate_positions` replays a route's *movement* stream against the env's
    own position rules and returns every unit's tile at every step. It is exact,
    not approximate — `tests/test_board_paths.py` pins it against a live
    `kaggle_environments` episode, all 719 steps, all 13 unit slots.
  * `segments` cuts each unit's day into maximal runs of `MOVE`/`PASS` bracketed
    by *anchors* (any op whose effect depends on where the unit stands, which is
    every op that is not a move and not a `PASS`).
  * `repath` rewrites each segment as the Manhattan-shortest walk between its
    fixed endpoints, front-loaded, and banks the difference as `PASS`.
  * `verify_schedule` is the gate: every anchor still fires on its original step
    with its unit on its original tile. #29 calls this the no-op gate, and it is
    what makes the re-path trustworthy before anything downstream spends the
    turns it recovers.

Three facts about this env make the problem this small (all verified against
`kaggle_environments/envs/kaggriculture/kaggriculture.py`):

  * **Movement is unobstructed.** A move is applied iff the destination is on the
    board; `LOCKED` tiles do not block it ("blocking movement would strand it
    there forever") and units do not collide. So the shortest path between two
    tiles is any monotone staircase of length `manhattan(a, b)`, and there is no
    graph search to do — a BFS over the board would return the same number.
  * **Positions reset every day.** `_end_of_day` teleports the farmer to the
    default spawn and dismisses every hand, so a movement run can never cross a
    day boundary and each day is an independent problem.
  * **Hands spawn deterministically.** `_do_hire` appends a hand at the first
    free shed-access tile in NWSE order, ties broken by occupancy, and it
    resolves *after* the turn's unit actions. Hire cost is `fib(n)` dollars, so
    on this route it never fails for want of money and the spawn order is fixed.

The honest finding this module produced is recorded in `docs/experiments.md`:
the incumbent's interior walking is already 98.3% Manhattan-optimal (3,073 of
3,125 moves are Manhattan-required), so the 50% movement share is a property of
the *task assignment* — which tiles each unit is sent to, in which order — not of
slack in the pathing. What is recoverable is mostly end-of-day drift: 359 of the
411 slack turns are moves in segments the day reset makes pointless.
"""

from __future__ import annotations

from dataclasses import dataclass

# Board geometry and clock. The env defaults, which every route in this repo is
# recorded and replayed at (`kaggriculture.json`: boardSize 10, turnsPerDay 24,
# maxMarketOrdersPerTurn 10).
BOARD_SIZE = 10
TURNS_PER_DAY = 24
MAX_MARKET_ORDERS = 10

# (dx, dy), y growing downward — the env's `FARMER_MOVES`.
MOVE_DELTAS = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}
MOVE_OPS = frozenset(MOVE_DELTAS)

# `PASS` is the env's explicit no-op: it neither reads nor writes the tile the
# unit stands on, so it is position-free and can be moved inside a segment (or
# created by one) without changing anything.
IDLE_OP = "PASS"


# ---------------------------------------------------------------------------
# Board geometry (mirrors kaggriculture.py)
# ---------------------------------------------------------------------------


def shed_access_tiles(board_size: int = BOARD_SIZE) -> list[tuple[int, int]]:
    """The four inner-corner tiles around the shed, in NWSE order."""
    half = board_size // 2
    return [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]


def default_spawn(board_size: int = BOARD_SIZE) -> tuple[int, int]:
    """Where the farmer stands at the start of every day: first NW shed-access
    tile. Only NW is unlocked at reset, so this is the only unlocked one."""
    half = board_size // 2
    for x, y in shed_access_tiles(board_size):
        if x < half and y < half:
            return (x, y)
    return (0, 0)


def spawn_hand(
    farmer: tuple[int, int],
    hands: list[tuple[int, int]],
    board_size: int = BOARD_SIZE,
) -> tuple[int, int]:
    """Where the next hire lands: least-occupied shed-access tile, NWSE order."""
    tiles = shed_access_tiles(board_size)
    occupancy = dict.fromkeys(tiles, 0)
    for pos in (farmer, *hands):
        if pos in occupancy:
            occupancy[pos] += 1
    return min(tiles, key=lambda tile: (occupancy[tile], tiles.index(tile)))


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def shortest_path_ops(origin: tuple[int, int], destination: tuple[int, int]) -> list[str]:
    """A Manhattan-shortest walk from `origin` to `destination`.

    Longer axis first, the convention the retired planner's `_step_toward` used.
    The choice is free — the walk's length is `manhattan(origin, destination)`
    whatever order the axes come in, and no intermediate tile can leave the board
    because every one of them lies in the bounding box of two on-board tiles.
    That last property is why this never needs the edge clamp the env applies.
    """
    dx = destination[0] - origin[0]
    dy = destination[1] - origin[1]
    horizontal = ["EAST" if dx > 0 else "WEST"] * abs(dx)
    vertical = ["SOUTH" if dy > 0 else "NORTH"] * abs(dy)
    return horizontal + vertical if abs(dx) >= abs(dy) else vertical + horizontal


# ---------------------------------------------------------------------------
# Route reading
# ---------------------------------------------------------------------------


def unit_ops(action: dict) -> list[list]:
    """`[farmer, *hands]` for one step, as the env reads them."""
    farmer = list(action.get("farmer") or [IDLE_OP])
    return [farmer, *[list(h or [IDLE_OP]) for h in (action.get("hands") or [])]]


def op_at(route: list[dict], step: int, slot: int) -> str:
    """The op slot `slot` performs at `step`, `PASS` if the route is short there.

    A route recorded with N hands is replayed on a turn with M live hands, and
    `main._align_hands` pads the short side with `PASS` — so a slot the route
    does not mention is idle, not absent.
    """
    units = unit_ops(route[step])
    if slot >= len(units):
        return IDLE_OP
    unit = units[slot]
    return unit[0] if unit else IDLE_OP


def write_unit(action: dict, slot: int, value: list) -> None:
    """Set one unit's op, growing the hands list with `PASS` as needed."""
    if slot == 0:
        action["farmer"] = list(value)
        return
    hands = action.setdefault("hands", [])
    while len(hands) < slot:
        hands.append([IDLE_OP])
    hands[slot - 1] = list(value)


def op_grid(route: list[dict], n_slots: int) -> list[list[str]]:
    """`grid[step][slot]` = that unit's op name, `PASS` where the route is short.

    Everything below reads the same op tens of thousands of times (`segments`
    scans every slot across every step, and `repath`'s fallback re-verifies once
    per candidate segment). Building the names once, instead of rebuilding a
    step's unit lists on every lookup, is what keeps a whole-route re-path in
    seconds rather than minutes.
    """
    grid = []
    for action in route:
        ops = [op if (op := (action.get("farmer") or [IDLE_OP])[0]) else IDLE_OP]
        for hand in action.get("hands") or []:
            ops.append(hand[0] if hand else IDLE_OP)
        if len(ops) < n_slots:
            ops.extend([IDLE_OP] * (n_slots - len(ops)))
        grid.append(ops)
    return grid


def is_anchor(op: str) -> bool:
    """Does this op's effect depend on the tile the unit is standing on?

    Every op in the game does, except the four moves (whose only effect *is* the
    position) and `PASS` (which has none).
    """
    return op not in MOVE_OPS and op != IDLE_OP


# ---------------------------------------------------------------------------
# The position simulator
# ---------------------------------------------------------------------------


def simulate_positions(
    route: list[dict],
    board_size: int = BOARD_SIZE,
    turns_per_day: int = TURNS_PER_DAY,
) -> list[tuple[tuple[int, int], ...]]:
    """Every unit's tile at the *start* of every step, exactly as the env has it.

    Only three things move a unit, and all three are in the route: a move op, the
    end-of-day reset, and a `HIRE` order's spawn. Nothing else in the game reads
    or writes a position, so this is a complete model rather than a heuristic —
    see the env-equivalence test in `tests/test_board_paths.py`.

    The returned tuple's length is the number of units that are *live* on that
    step, which is not the same as the number of hand entries the route carries:
    the route can name a hand that has not been hired yet (its op is dropped) and
    the game can have a hand the route never mentions (it idles).
    """
    spawn = default_spawn(board_size)
    farmer = spawn
    hands: list[tuple[int, int]] = []
    out: list[tuple[tuple[int, int], ...]] = []

    for step, action in enumerate(route):
        # `_end_of_day` runs after the last turn of a day, so the reset is
        # visible at the start of every step that is a multiple of the day
        # length — including step 0, which is the initial state.
        if step % turns_per_day == 0:
            farmer, hands = spawn, []
        out.append((farmer, *hands))

        for slot, unit in enumerate(unit_ops(action)):
            if slot > len(hands):
                continue  # the route names a hand that has not been hired
            op = unit[0] if unit else IDLE_OP
            if op not in MOVE_OPS:
                continue
            x, y = farmer if slot == 0 else hands[slot - 1]
            dx, dy = MOVE_DELTAS[op]
            if not (0 <= x + dx < board_size and 0 <= y + dy < board_size):
                continue  # the env drops an off-board move; the unit stays put
            if slot == 0:
                farmer = (x + dx, y + dy)
            else:
                hands[slot - 1] = (x + dx, y + dy)

        # `_process_market` resolves after the turn's unit actions, so a hand
        # hired on this step first acts on the next one.
        for order in (action.get("market") or [])[:MAX_MARKET_ORDERS]:
            if order and order[0] == "HIRE":
                hands.append(spawn_hand(farmer, hands, board_size))

    return out


# ---------------------------------------------------------------------------
# Segments: the free stretches between two position-dependent ops
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    """One unit's stretch of steps in which only movement and idling happen.

    `start`..`end` inclusive, never crossing a day boundary. `origin` is where
    the unit stands entering it and `destination` where it must stand leaving it,
    so `required` moves are load-bearing and `moves - required` are slack.

    `terminal` marks a segment with no anchor after it in its day: the day reset
    discards the unit's position, so *every* move in it is dead labour. Those are
    only rewritten when `repath(drop_terminal=True)`, because banking them means
    the route no longer reproduces the incumbent's final position and only the
    empirical panel gate can say whether that matters.
    """

    slot: int
    start: int
    end: int
    moves: int
    required: int
    origin: tuple[int, int]
    destination: tuple[int, int]
    terminal: bool

    @property
    def slack(self) -> int:
        return self.moves - self.required


def anchor_steps(route: list[dict], slot: int) -> set[int]:
    """Steps where this unit does something that depends on where it stands."""
    grid = op_grid(route, slot + 1)
    return {step for step in range(len(route)) if is_anchor(grid[step][slot])}


def segments(
    route: list[dict],
    positions: list[tuple[tuple[int, int], ...]] | None = None,
    board_size: int = BOARD_SIZE,
    turns_per_day: int = TURNS_PER_DAY,
) -> list[Segment]:
    """Every movement segment in the route, in (slot, step) order.

    Segments with no moves in them are dropped: they are already-idle stretches
    with nothing to re-plan, and #29 does not touch idle units.
    """
    if positions is None:
        positions = simulate_positions(route, board_size, turns_per_day)
    n_slots = max((len(p) for p in positions), default=0)
    grid = op_grid(route, n_slots)
    found: list[Segment] = []

    for slot in range(n_slots):
        anchors = {step for step in range(len(route)) if is_anchor(grid[step][slot])}
        for day_start in range(0, len(route), turns_per_day):
            day_end = min(day_start + turns_per_day, len(route))
            cut = sorted(s for s in anchors if day_start <= s < day_end)
            # The day splits into the stretches between consecutive anchors.
            bounds = [(day_start, cut[0] - 1) if cut else (day_start, day_end - 1)]
            for i, anchor in enumerate(cut):
                nxt = cut[i + 1] if i + 1 < len(cut) else day_end
                bounds.append((anchor + 1, nxt - 1))
            for start, end in bounds:
                # A hand hired on this step first acts on the next one, so the
                # stretch from a day's start to its first anchor can begin before
                # the unit exists. Its ops there are dropped by the env, so the
                # segment really starts when the unit does. Liveness is monotone
                # within a day (hands are only appended until the day reset), so
                # walking the start forward once is enough.
                while start <= end and slot >= len(positions[start]):
                    start += 1
                if start > end:
                    continue
                moves = sum(1 for step in range(start, end + 1) if grid[step][slot] in MOVE_OPS)
                if not moves:
                    continue
                origin = positions[start][slot]
                after = end + 1
                if after < day_end and slot < len(positions[after]):
                    destination = positions[after][slot]
                    terminal = False
                else:
                    destination = _replay(grid, positions, slot, start, end, board_size)
                    terminal = True
                found.append(
                    Segment(
                        slot=slot,
                        start=start,
                        end=end,
                        moves=moves,
                        required=manhattan(origin, destination),
                        origin=origin,
                        destination=destination,
                        terminal=terminal,
                    )
                )
    return found


def _replay(
    grid: list[list[str]],
    positions: list[tuple[tuple[int, int], ...]],
    slot: int,
    start: int,
    end: int,
    board_size: int,
) -> tuple[int, int]:
    """Where the unit ends up after `start`..`end`, for a segment whose successor
    step is past the day boundary and so has no recorded position."""
    x, y = positions[start][slot]
    for step in range(start, end + 1):
        op = grid[step][slot]
        if op not in MOVE_OPS:
            continue
        dx, dy = MOVE_DELTAS[op]
        if 0 <= x + dx < board_size and 0 <= y + dy < board_size:
            x, y = x + dx, y + dy
    return (x, y)


# ---------------------------------------------------------------------------
# The re-path itself
# ---------------------------------------------------------------------------


def _rewrite_segment(route: list[dict], seg: Segment) -> list[str] | None:
    """Overwrite one segment in place with its shortest walk, front-loaded.

    Moves come first and the recovered turns land at the *end* of the segment,
    with the unit already standing where its next op needs it. That ordering is
    free nominally — nothing reads a position mid-segment — and it is strictly
    more robust live: `main._repair_weeds` replays a unit's trace one step behind
    for up to eight steps after a `DIG`, and a trailing `PASS` absorbs that lag
    where a trailing move would have made the unit arrive late and fire its next
    op on the wrong tile.

    Returns the walk it wrote, or `None` if the segment was already shortest (in
    which case nothing is written and front-loading alone would change nothing).
    """
    walk = [] if seg.terminal else shortest_path_ops(seg.origin, seg.destination)
    span = seg.end - seg.start + 1
    if len(walk) == seg.moves or len(walk) > span:
        # Second case cannot happen for a segment read off `segments()` (the
        # original walk fits and is no shorter), but a caller-supplied `only`
        # list could be stale. Skip rather than clobber the following anchor.
        return None
    for offset in range(span):
        op = [walk[offset]] if offset < len(walk) else [IDLE_OP]
        write_unit(route[seg.start + offset], seg.slot, op)
    return walk


def repath(
    route: list[dict],
    only: list[Segment] | None = None,
    drop_terminal: bool = False,
    board_size: int = BOARD_SIZE,
    turns_per_day: int = TURNS_PER_DAY,
) -> tuple[list[dict], dict]:
    """Rewrite movement segments as shortest walks; bank the rest as `PASS`.

    `only` restricts the rewrite to a subset of `segments(route)` (the search
    loop uses it to mutate one site at a time). Returns the new route and a stats
    dict; the route comes back unchanged when there is no slack to bank.

    Segments are **not** independent, which is the one surprise in this problem.
    `_do_hire` spawns a hand on the least-occupied shed-access tile, so where a
    unit idles on the turn a hire resolves decides where the *next* hand starts
    its day — and re-pathing one unit can therefore relocate another. So every
    rewrite is checked: the whole batch is applied and verified first (it is
    clean in the ordinary case, one verification), and on a violation the batch
    is rebuilt segment by segment, keeping only the rewrites the schedule still
    survives. `segments_rejected` counts what that discards.
    """
    import copy

    positions = simulate_positions(route, board_size, turns_per_day)
    targets = only if only is not None else segments(route, positions, board_size, turns_per_day)
    targets = [s for s in targets if drop_terminal or not s.terminal]

    def _stats() -> dict:
        return {
            "segments": len(targets),
            "segments_rewritten": 0,
            "segments_rejected": 0,
            "moves_before": 0,
            "moves_after": 0,
            "turns_recovered": 0,
            "terminal_segments_dropped": 0,
        }

    def _bank(stats: dict, seg: Segment, walk: list[str]) -> None:
        stats["segments_rewritten"] += 1
        stats["moves_before"] += seg.moves
        stats["moves_after"] += len(walk)
        stats["turns_recovered"] += seg.moves - len(walk)
        if seg.terminal:
            stats["terminal_segments_dropped"] += 1

    # Fast path: rewrite everything, then verify once.
    batch, batch_stats = copy.deepcopy(route), _stats()
    for seg in targets:
        walk = _rewrite_segment(batch, seg)
        if walk is not None:
            _bank(batch_stats, seg, walk)
    if not batch_stats["segments_rewritten"]:
        return batch, batch_stats
    if not verify_schedule(route, batch, board_size, turns_per_day):
        return batch, batch_stats

    # Slow path: a rewrite moved somebody else's op off its tile. Rebuild the
    # batch one segment at a time and keep only what verifies. Each trial writes
    # in place and is rolled back on rejection from the saved span, because
    # deep-copying a 719-step route per candidate segment costs an order of
    # magnitude more than the verification it guards.
    new, stats = copy.deepcopy(route), _stats()
    for seg in targets:
        span = range(seg.start, seg.end + 1)
        saved = [list(unit_ops(new[step])[seg.slot]) for step in span]
        walk = _rewrite_segment(new, seg)
        if walk is None:
            continue
        if verify_schedule(route, new, board_size, turns_per_day):
            for step, unit in zip(span, saved, strict=True):
                write_unit(new[step], seg.slot, unit)
            stats["segments_rejected"] += 1
            continue
        _bank(stats, seg, walk)
    return new, stats


def verify_schedule(
    original: list[dict],
    mutated: list[dict],
    board_size: int = BOARD_SIZE,
    turns_per_day: int = TURNS_PER_DAY,
) -> list[str]:
    """#29's no-op gate. Returns a list of violations; empty means clean.

    Three things must hold for a re-path to be a no-op: the market stream is
    untouched, every non-movement unit op still fires on its original step, and
    the unit fires it from the tile it fired it from before. The third is the one
    that needs the simulator — the first two are only bookkeeping.
    """
    violations: list[str] = []
    if len(original) != len(mutated):
        return [f"step count changed: {len(original)} -> {len(mutated)}"]

    for step, (before, after) in enumerate(zip(original, mutated, strict=True)):
        b_market = [list(o) for o in (before.get("market") or [])]
        a_market = [list(o) for o in (after.get("market") or [])]
        if b_market != a_market:
            violations.append(f"step {step}: market stream changed")

    before_pos = simulate_positions(original, board_size, turns_per_day)
    after_pos = simulate_positions(mutated, board_size, turns_per_day)

    n_slots = max(
        max((len(p) for p in before_pos), default=0),
        max((len(p) for p in after_pos), default=0),
    )
    before_grid = op_grid(original, n_slots)
    after_grid = op_grid(mutated, n_slots)
    for step in range(len(original)):
        b_row, a_row = before_grid[step], after_grid[step]
        b_here, a_here = before_pos[step], after_pos[step]
        for slot in range(n_slots):
            b_op, a_op = b_row[slot], a_row[slot]
            b_anchor, a_anchor = is_anchor(b_op), is_anchor(a_op)
            if b_anchor or a_anchor:
                # An anchor must be identical down to its arguments, not just its
                # op name: a `PICKUP WHEAT 4` that became a `PICKUP WHEAT 1`
                # would pass a name-only check.
                b_units, a_units = unit_ops(original[step]), unit_ops(mutated[step])
                b_full = b_units[slot] if slot < len(b_units) else [IDLE_OP]
                a_full = a_units[slot] if slot < len(a_units) else [IDLE_OP]
                if b_full != a_full:
                    violations.append(f"step {step} slot {slot}: op {b_full} -> {a_full}")
                    continue
            if not b_anchor:
                continue
            b_tile = b_here[slot] if slot < len(b_here) else None
            a_tile = a_here[slot] if slot < len(a_here) else None
            if b_tile != a_tile:
                violations.append(
                    f"step {step} slot {slot}: {b_op} fires at {a_tile}, was {b_tile}"
                )
    return violations


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

PRODUCTIVE_OPS = ("WATER", "HARVEST", "PLANT", "FERTILIZE")


def census(route: list[dict]) -> dict:
    """The unit-op census #29 opens with: movement share, productive share, idle."""
    counts: dict[str, int] = {}
    total = 0
    for action in route:
        for unit in unit_ops(action):
            op = unit[0] if unit else IDLE_OP
            counts[op] = counts.get(op, 0) + 1
            total += 1
    movement = sum(counts.get(op, 0) for op in MOVE_OPS)
    productive = sum(counts.get(op, 0) for op in PRODUCTIVE_OPS)
    return {
        "unit_turns": total,
        "counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "movement": movement,
        "productive": productive,
        "idle": counts.get(IDLE_OP, 0),
        "movement_share": movement / total if total else 0.0,
        "productive_share": productive / total if total else 0.0,
        "idle_share": counts.get(IDLE_OP, 0) / total if total else 0.0,
        "moves_per_productive_op": movement / productive if productive else 0.0,
    }


def slack_report(
    route: list[dict],
    board_size: int = BOARD_SIZE,
    turns_per_day: int = TURNS_PER_DAY,
) -> dict:
    """How much of the route's walking is slack, split by segment kind."""
    positions = simulate_positions(route, board_size, turns_per_day)
    segs = segments(route, positions, board_size, turns_per_day)
    interior = [s for s in segs if not s.terminal]
    terminal = [s for s in segs if s.terminal]
    moves = sum(s.moves for s in segs)
    required = sum(s.required for s in interior)  # terminal destinations are free
    return {
        "segments": len(segs),
        "interior_segments": len(interior),
        "terminal_segments": len(terminal),
        "moves": moves,
        "interior_moves": sum(s.moves for s in interior),
        "interior_required": required,
        "interior_slack": sum(s.slack for s in interior),
        "terminal_moves": sum(s.moves for s in terminal),
        "wasteful_segments": sum(1 for s in interior if s.slack),
        "path_optimality": (required / sum(s.moves for s in interior)) if interior else 1.0,
        "segments_detail": segs,
    }
