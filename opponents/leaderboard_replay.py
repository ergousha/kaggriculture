"""Leaderboard Replay Sparring Opponent.

Replays a recorded Kaggle episode turn-by-turn so a real leaderboard match can be
used as a sparring partner in `local_arena.py` and in the macro searches. This is
the only opponent in the repo that plays at the tempo the leaderboard actually
plays at; `starter` and `adaptive` barely produce, so beating them measures
survival rather than throughput.

**This file never worked before, for three independent reasons.** All three failed
silently as "the opponent finished on exactly its $3,000 starting money", which
reads like a weak opponent rather than a broken one:

1. It referenced `__file__` at module level. The env loads an agent with
   `exec(compile(src, path), {})` (`agent.py: get_last_callable`), and that `{}`
   has no `__file__`, so importing the module raised `NameError` and the env
   rejected the agent as `InvalidArgument` before a single turn ran. The path is
   recovered from the calling frame's `co_filename` instead, which `compile()`
   does set, with a walk-up from the cwd as a fallback.
2. It indexed the replay with `obs.get("step", 0)`. `step` is in the **shared**
   part of the observation, so only the seat at index 0 receives it — and
   `local_arena` alternates seats, so on half of every run the replay read step 0
   on all 720 turns. `_step_index` derives it from the per-seat `day`/`hour`.
3. It picked the newest `.json` by mtime with no schema check.
   `logs/leaderboard_replays/` accumulated 116 *Halite 4* replays alongside the 70
   kaggriculture ones, so the newest file was usually a Halite episode: every step
   index missed and every turn returned `PASS`. Candidates are now sniffed for
   `"name": "kaggriculture"` before they are eligible, selection is announced on
   stderr, and having no valid candidate is an explicit warning.

Any A/B or search run that used `--opponent leaderboard` before this was measured
against nothing.

The default pick is the episode with the **highest final cash** — the strongest
available opponent — and the default seat is that episode's winner.

Both the schema check and the score come from a 64 KB head of the file: `name`
sits at ~byte 860 and `rewards` at ~byte 880, while `steps` starts at ~byte 6300
and runs for megabytes. So choosing between 186 candidates costs 186 small reads
instead of 186 full JSON parses, which matters because selection happens on the
first turn inside a 1s `actTimeout`.

Usage:
    uv run python local_arena.py --agent main.py --opponent leaderboard --episodes 30
    uv run python local_arena.py --agent main.py \
        --opponent logs/leaderboard_replays/episode-90158870-replay.json --episodes 30

Environment overrides:
    KAGGRICULTURE_REPLAY_DIR     directory to search (default logs/leaderboard_replays)
    KAGGRICULTURE_REPLAY_PATH    exact replay to use; still schema-checked
    KAGGRICULTURE_REPLAY_PLAYER  seat to replay: "auto" (default, the winner) or 0/1

Caveat: a replay is an **open-loop** opponent. It re-issues the moves it made in
its own episode against its own seed, so its actions do not respond to what the
agent under test does. That makes it a strong tempo and pot-contention benchmark,
not a strategic one — it will keep watering a tile it no longer benefits from.
"""

import json
import os
import re
import sys

NO_OP = {"farmer": ["PASS"], "hands": [], "market": []}

REPLAY_SUBDIR = os.path.join("logs", "leaderboard_replays")

# `name` and `rewards` both live in the first kilobyte; `steps` starts at ~6 KB.
HEAD_BYTES = 65536

_NAME_RE = re.compile(rb'"name"\s*:\s*"kaggriculture"')
_REWARDS_RE = re.compile(rb'"rewards"\s*:\s*\[([^\]]*)\]')

_REPLAY_CACHE: dict = {}
_WARNED: set = set()


def _warn(message):
    """Warn once per distinct message; an agent process is re-created per episode."""
    if message in _WARNED:
        return
    _WARNED.add(message)
    print(f"[LeaderboardReplayOpponent] {message}", file=sys.stderr)


def _this_dir():
    """Directory of this file, without relying on `__file__`.

    The env execs an agent as `exec(compile(src, path), {})`, so `__file__` is
    absent — but `compile()` records `path` as the code object's filename, and the
    frame executing this function still carries it.
    """
    path = globals().get("__file__") or ""
    if not path:
        try:
            path = sys._getframe().f_code.co_filename
        except (AttributeError, ValueError):
            path = ""
    if path and not path.startswith("<"):
        return os.path.dirname(os.path.abspath(path))
    return os.getcwd()


def _default_replay_dir():
    """First existing `logs/leaderboard_replays`, searching upward from this file."""
    override = os.environ.get("KAGGRICULTURE_REPLAY_DIR", "").strip()
    if override:
        return override

    roots = []
    here = _this_dir()
    roots.append(os.path.dirname(here))  # opponents/ -> project root
    roots.append(here)
    cwd = os.path.abspath(os.getcwd())
    for _ in range(4):
        roots.append(cwd)
        parent = os.path.dirname(cwd)
        if parent == cwd:
            break
        cwd = parent

    for root in roots:
        candidate = os.path.join(root, REPLAY_SUBDIR)
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(os.path.dirname(here), REPLAY_SUBDIR)


def _sniff(path):
    """Read a 64 KB head and return (is_kaggriculture, rewards) without parsing steps."""
    try:
        with open(path, "rb") as f:
            head = f.read(HEAD_BYTES)
    except OSError:
        return False, []

    if not _NAME_RE.search(head):
        return False, []

    rewards = []
    m = _REWARDS_RE.search(head)
    if m:
        for part in m.group(1).split(b","):
            part = part.strip()
            if not part or part == b"null":
                rewards.append(0.0)
                continue
            try:
                rewards.append(float(part))
            except ValueError:
                rewards.append(0.0)
    return True, rewards


def _candidates(replay_dir):
    """Every schema-valid kaggriculture replay in `replay_dir`, best score first."""
    if not os.path.isdir(replay_dir):
        return []

    found = []
    skipped = 0
    for name in sorted(os.listdir(replay_dir)):
        if not name.endswith(".json") or name.startswith("_"):
            continue
        path = os.path.join(replay_dir, name)
        ok, rewards = _sniff(path)
        if not ok:
            skipped += 1
            continue
        found.append((max(rewards) if rewards else 0.0, path, rewards))

    if skipped:
        _warn(f"ignored {skipped} non-kaggriculture JSON file(s) in {replay_dir}")
    found.sort(reverse=True)
    return found


def _resolve_seat(rewards, requested):
    """Seat to replay: the winner for "auto", otherwise the requested index."""
    if requested.strip().lower() in ("", "auto", "best", "winner"):
        if not rewards:
            return 0
        return max(range(len(rewards)), key=lambda i: rewards[i])
    try:
        return max(0, int(requested))
    except ValueError:
        return 0


def _select():
    """Pick (path, seat), preferring an explicit path and otherwise the best episode."""
    requested_seat = os.environ.get("KAGGRICULTURE_REPLAY_PLAYER", "auto")
    explicit = os.environ.get("KAGGRICULTURE_REPLAY_PATH", "").strip()

    if explicit:
        if not os.path.exists(explicit):
            _warn(f"KAGGRICULTURE_REPLAY_PATH does not exist: {explicit}")
            return None, 0
        ok, rewards = _sniff(explicit)
        if not ok:
            _warn(
                f"KAGGRICULTURE_REPLAY_PATH is not a kaggriculture replay: {explicit} "
                "(a Halite replay here would make this opponent a PASS bot)"
            )
            return None, 0
        return explicit, _resolve_seat(rewards, requested_seat)

    replay_dir = _default_replay_dir()
    found = _candidates(replay_dir)
    if not found:
        _warn(
            f"no valid kaggriculture replay found in {replay_dir}; "
            "this opponent will PASS every turn and is not a real sparring partner"
        )
        return None, 0

    best_reward, path, rewards = found[0]
    seat = _resolve_seat(rewards, requested_seat)
    _warn(
        f"sparring against {os.path.basename(path)} seat {seat} "
        f"(final cash ${best_reward:,.0f}; {len(found)} valid replay(s) available)"
    )
    return path, seat


def _get_replay_data():
    """Load and cache (steps, seat) for the selected replay."""
    key = (
        os.environ.get("KAGGRICULTURE_REPLAY_PATH", ""),
        os.environ.get("KAGGRICULTURE_REPLAY_DIR", ""),
        os.environ.get("KAGGRICULTURE_REPLAY_PLAYER", "auto"),
    )
    if key in _REPLAY_CACHE:
        return _REPLAY_CACHE[key]

    path, seat = _select()
    if not path:
        _REPLAY_CACHE[key] = ([], 0)
        return [], 0

    try:
        with open(path) as f:
            data = json.load(f)
        steps = data.get("steps") or []
        if not steps:
            _warn(f"replay has no steps: {path}")
    except (OSError, ValueError) as exc:
        _warn(f"error loading replay {path}: {exc}")
        steps = []

    _REPLAY_CACHE[key] = (steps, seat)
    return steps, seat


def _step_index(obs, config):
    """Turn index, derived from `day`/`hour` because `step` is seat-0 only.

    This is the bug that made this opponent inert for its entire existence, and it
    is invisible from the outside: the env puts `step` in the **shared** part of
    the observation, so only the player at index 0 receives it. Whenever the
    replay sat at index 1 -- which `local_arena` guarantees for half of every run,
    since seats alternate -- `obs.get("step", 0)` returned 0 on every one of the
    720 turns. It replayed step 0 forever, never hired, never sold, and finished
    on exactly the $3,000 starting money.

    `day` and `hour` are per-seat and always present, and the env's own indexing is
    `step == day * turnsPerDay + hour`, verified across a full 720-step replay.
    """
    step = obs.get("step")
    if step is not None:
        return int(step)

    turns_per_day = 24
    if isinstance(config, dict):
        turns_per_day = int(config.get("turnsPerDay") or 24)
    return int(obs.get("day", 0) or 0) * turns_per_day + int(obs.get("hour", 0) or 0)


def agent(obs, config=None):
    """Re-issue the selected seat's recorded action for this step."""
    steps, seat = _get_replay_data()
    if not steps:
        return NO_OP

    # `steps[t]["action"]` is the action that PRODUCED `steps[t]`, not the one taken
    # from it -- `steps[0]`'s action is empty for every seat, and a step's money
    # already includes that step's own sales. So the action to emit while looking at
    # the state recorded as `observation[s]` is the one stored at index s + 1.
    step = _step_index(obs, config) + 1
    if step >= len(steps):
        _warn(f"replay exhausted at step {step} of {len(steps)}; PASSing from here on")
        return NO_OP

    step_state = steps[step] or []
    if seat >= len(step_state):
        seat = 0
    if seat >= len(step_state):
        return NO_OP

    action = (step_state[seat] or {}).get("action")
    if not isinstance(action, dict):
        return NO_OP

    return {
        "farmer": action.get("farmer") or ["PASS"],
        "hands": action.get("hands") or [],
        "market": action.get("market") or [],
    }
