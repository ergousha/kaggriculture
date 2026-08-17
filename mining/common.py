"""Shared helpers for the robust-route mining pipeline (Phases 1-4).

Everything here is deliberately small and dependency-free so the four pipeline
scripts agree on the three things that must not drift between them:

  * the route encoding (Phase 1 stores it, Phase 2 bakes it, Phase 4 ships it) --
    a single implementation, delegating the chunk width and the agent template to
    `scripts/build_route_agent.py` so the emitted v0.2.5 agent is byte-compatible
    with the v0.2.4 one;
  * the CVaR definition -- CVaR_a is the mean of the worst `a` fraction of
    outcomes, which is stricter than the a-th percentile itself;
  * the common-random-number seed sets -- fixed once, here, so every candidate in
    every stage is scored on the identical seeds.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import statistics
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# `orjson` is optional. Measured on this corpus, stdlib json parses a full 32 MB
# replay in ~0.11 s and the pipeline is bound by episode simulation (~2.4 s) and
# by 100 GB of disk reads, so orjson buys nothing measurable -- but if it is
# installed we use it.
try:  # pragma: no cover - environment dependent
    import orjson

    def json_loads(raw: bytes):
        return orjson.loads(raw)

    JSON_BACKEND = "orjson"
except ImportError:  # pragma: no cover - environment dependent

    def json_loads(raw: bytes):
        return json.loads(raw)

    JSON_BACKEND = "json"


ROUTE_STEPS = 719
CHUNK_WIDTH = 92  # must match scripts/build_route_agent.py's chunking
DEFAULT_STEPS = 720


# ---------------------------------------------------------------------------
# Cheap head scan
#
# Both `rewards` and `info.seed` live in the first kilobyte of a replay while
# `steps` starts at ~6.6 KB and runs for 32 MB, so screening 3,400 files costs
# 3,400 small reads instead of 100 GB of parsing.
# ---------------------------------------------------------------------------

HEAD_BYTES = 65536


def _find_object(head: bytes, key: bytes) -> dict | None:
    """Brace-match the JSON object stored at `key` inside a byte prefix.

    A naive `"seed"` regex is wrong here: `specification` carries a `"seed": 10`
    field for every crop, so the first match in the file is a decoy. We locate
    the `info` object and parse only that.
    """
    marker = b'"' + key + b'":'
    start = head.find(marker)
    if start < 0:
        return None
    open_at = head.find(b"{", start)
    if open_at < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(open_at, len(head)):
        ch = head[i : i + 1]
        if in_string:
            if escaped:
                escaped = False
            elif ch == b"\\":
                escaped = True
            elif ch == b'"':
                in_string = False
            continue
        if ch == b'"':
            in_string = True
        elif ch == b"{":
            depth += 1
        elif ch == b"}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(head[open_at : i + 1])
                except ValueError:
                    return None
    return None


def _find_rewards(head: bytes) -> list[float] | None:
    marker = b'"rewards":'
    start = head.find(marker)
    if start < 0:
        return None
    open_at = head.find(b"[", start)
    close_at = head.find(b"]", open_at)
    if open_at < 0 or close_at < 0:
        return None
    try:
        raw = json.loads(head[open_at : close_at + 1])
    except ValueError:
        return None
    return [0.0 if v is None else float(v) for v in raw]


def head_scan(path: str) -> dict | None:
    """Return {rewards, seed, is_kaggriculture} from a 64 KB prefix.

    Returns None when the prefix is not a recognisable kaggriculture replay. A
    prefix that parses but lacks rewards/seed returns those as None so the caller
    can decide to fall back to a full parse rather than silently dropping a file.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(HEAD_BYTES)
    except OSError:
        return None
    if b'"kaggriculture"' not in head:
        return None
    info = _find_object(head, b"info") or {}
    seed = info.get("seed")
    return {
        "rewards": _find_rewards(head),
        "seed": int(seed) if isinstance(seed, int) else None,
        "episode": info.get("EpisodeId"),
        "teams": [str(t).strip('"') for t in (info.get("TeamNames") or [])],
    }


# ---------------------------------------------------------------------------
# Route normalisation, hashing and encoding
# ---------------------------------------------------------------------------


def normalize_route(route: list) -> list[dict]:
    """Canonical per-step encoding: explicit PASS, no nulls, fixed key order.

    Mirrors `scripts/build_route_agent.extract_route` and `main._copy_action`, so
    a normalized trace is exactly what the deployed agent replays.
    """
    out = []
    for entry in route:
        action = entry if isinstance(entry, dict) else {}
        out.append(
            {
                "farmer": list(action.get("farmer") or ["PASS"]),
                "hands": [list(h or ["PASS"]) for h in (action.get("hands") or [])],
                "market": [list(o) for o in (action.get("market") or []) if o],
            }
        )
    return out


def canonical_json(route: list[dict]) -> str:
    return json.dumps(route, separators=(",", ":"), sort_keys=True)


def route_hash(route: list[dict]) -> str:
    """Hash of the *canonicalized* trace, so encoding differences cannot hide
    duplicates."""
    return hashlib.blake2b(canonical_json(route).encode("utf-8"), digest_size=16).hexdigest()


def encode_route_b85(route: list[dict]) -> str:
    """zlib -> base85, exactly as scripts/build_route_agent.emit_agent does."""
    payload = json.dumps(route, separators=(",", ":")).encode("utf-8")
    return base64.b85encode(zlib.compress(payload, 9)).decode("ascii")


def decode_route_b85(blob: str) -> list[dict]:
    return json.loads(zlib.decompress(base64.b85decode(blob)))


def chunk_b85(blob: str, width: int = CHUNK_WIDTH) -> list[str]:
    """Split into fixed-width chunks. `ruff format` rewraps one multi-kilobyte
    literal on every run, which is why main.py stores a list (see main.py:37)."""
    return [blob[i : i + width] for i in range(0, len(blob), width)]


def route_parts_source(blob: str, width: int = CHUNK_WIDTH) -> str:
    """The `_ROUTE_B85_PARTS` body. The b85 alphabet contains no quote or
    backslash, so chunks embed verbatim."""
    return "".join(f'    "{chunk}",\n' for chunk in chunk_b85(blob, width))


def write_route_agent(
    route: list[dict],
    out_path: str,
    provenance: dict,
    version: str = "0.2.5",
) -> int:
    """Write a deployable route-replay agent, reusing build_route_agent's template.

    Phase 2 evaluates candidates through this exact artifact rather than replaying
    the bare trace, because what ships is the route *plus* main.py's three runtime
    layers (WEED repair, SELL-slot ordering, hands alignment). Scoring the bare
    trace would measure something we do not deploy.
    """
    from build_route_agent import AGENT_TEMPLATE

    blob = encode_route_b85(route)
    block = "".join(f"\n  {k}: {v}" for k, v in provenance.items())
    team = provenance.get("team", "?")
    episode = provenance.get("episode", "?")
    seat = provenance.get("seat", "?")
    source = AGENT_TEMPLATE.format(
        provenance_line=f" -- {team} episode {episode} seat {seat}.",
        provenance_block=block,
        n_steps=len(route),
        route_parts=route_parts_source(blob),
        version=version,
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(source)
    return len(source)


# ---------------------------------------------------------------------------
# CVaR
# ---------------------------------------------------------------------------


def cvar(values: list[float], alpha: float = 0.05) -> float:
    """Mean of the worst `alpha` fraction of outcomes (lower tail).

    This is the metric the pipeline ranks on. It is deliberately *not* the
    alpha-th percentile: the percentile tells you the boundary, CVaR tells you how
    bad things are past it, which is what distinguishes "bad seeds earn $80k" from
    "bad seeds earn $20k".

    `k = max(1, floor(alpha * n))`. For n=500, alpha=0.05 that is the mean of the
    25 worst scores, which coincides with "mean of all scores at or below the 5th
    percentile".
    """
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = max(1, int(len(ordered) * alpha))
    return statistics.fmean(ordered[:k])


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile, q in [0, 1]."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "cvar5": round(cvar(values, 0.05), 1),
        "mean": round(statistics.fmean(values), 1),
        "median": round(statistics.median(values), 1),
        "p5": round(percentile(values, 0.05), 1),
        "min": round(min(values), 1),
        "max": round(max(values), 1),
        "stdev": round(statistics.stdev(values), 1) if len(values) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Common random numbers
#
# The seed sets are nested supersets (SCREEN subset of MID subset of FINAL) so a
# stage can reuse the episodes an earlier stage already paid for, and HOLDOUT is
# disjoint from all of them so Phase 3's winner's-curse guard is honest.
# ---------------------------------------------------------------------------

# Seed counts are smaller than the single-opponent run because each stage now
# multiplies seeds by the opponent panel: a candidate's sample size is
# seeds x |panel|, not seeds. At the final stage 100 seeds x 6 opponents = 600
# games per candidate, so the win-rate standard error is ~2%.
SEED_BASE = 1_000_000
N_SCREEN = 12
N_MID = 30
N_FINAL = 100
N_HOLDOUT = 100


def seed_set(n: int, base: int = SEED_BASE, mode: str = "sequential") -> list[int]:
    """Fixed seed list. `sequential` is base..base+n (the spec's default).

    `random31` draws from a fixed stream in the same 31-bit range the engine's own
    fallback uses (`resolve_episode_seed`), which is closer to how Kaggle seeds
    real episodes; see the caveat rank_cvar.py prints.
    """
    if mode == "random31":
        rng = random.Random(base)
        return [rng.randrange(2**31) for _ in range(n)]
    return [base + i for i in range(n)]


def seed_sets(mode: str = "sequential", n_holdout: int = N_HOLDOUT) -> dict[str, list[int]]:
    """`n_holdout` shrinks only the holdout set, and it stays based at
    SEED_BASE + N_FINAL so any smaller holdout is a prefix of the full one and
    remains disjoint from every evaluation seed."""
    evaluation = seed_set(N_FINAL, SEED_BASE, mode)
    holdout = seed_set(n_holdout, SEED_BASE + N_FINAL, mode)
    return {
        "screen": evaluation[:N_SCREEN],
        "mid": evaluation[:N_MID],
        "final": evaluation,
        "holdout": holdout,
    }


# ---------------------------------------------------------------------------
# Win-rate metrics
#
# The competition scores a skill rating driven by wins, not by cash, and cash is
# overwhelmingly common-mode: measured live, corr(our cash, opponent cash) = +0.86,
# because both seats draw from one shared market. Own-cash CVaR therefore mostly
# measures "was this a good seed", which affects both players and cancels in the
# head-to-head that actually sets the rating. These operate on the margin instead.
# ---------------------------------------------------------------------------


def win_rate(pairs: list[tuple[float, float]]) -> float:
    """Fraction of (me, opp) pairs we win outright. Ties count as losses, matching
    the arena's `win` field and the conservative reading of a tied rating update."""
    if not pairs:
        return float("nan")
    return sum(1 for me, opp in pairs if me > opp) / len(pairs)


def margins(pairs: list[tuple[float, float]]) -> list[float]:
    return [me - opp for me, opp in pairs]


def effective_labels(candidate_hash: str, labels: list[str]) -> list[str]:
    """Panel labels excluding the candidate itself.

    A panel member played against itself is a mirror it can at best draw, which
    would dock every panel member ~1/N relative to non-members and bias selection
    against exactly the strong routes the panel is built from. Self-matches are
    dropped from a candidate's own aggregate; the per-opponent breakdown makes the
    resulting difference in opponent count visible.
    """
    return [lab for lab in labels if not candidate_hash.startswith(lab)]


def panel_scores(per_opponent: dict[str, list[tuple[float, float]]]) -> dict:
    """Aggregate a candidate's results across the opponent panel.

    `mean_win` is the primary ranking metric; `worst_win` is the robustness
    tiebreak — the win rate against whichever panel member counters it best. A
    route that beats five opponents and is crushed by the sixth is exactly the
    single-opponent overfit this panel exists to catch.
    """
    per = {k: v for k, v in per_opponent.items() if v}
    if not per:
        return {"n": 0}
    rates = {k: win_rate(v) for k, v in per.items()}
    allpairs = [p for v in per.values() for p in v]
    allmargins = margins(allpairs)
    worst_label = min(rates, key=lambda k: rates[k])
    return {
        "n": len(allpairs),
        "opponents": len(per),
        "mean_win": statistics.fmean(rates.values()),
        "worst_win": rates[worst_label],
        "worst_opponent": worst_label,
        "per_opponent": {k: round(v, 4) for k, v in sorted(rates.items())},
        "pooled_win": win_rate(allpairs),
        "mean_margin": statistics.fmean(allmargins),
        "margin_cvar5": cvar(allmargins, 0.05),
        "cash_cvar5": cvar([me for me, _ in allpairs], 0.05),
        "cash_mean": statistics.fmean([me for me, _ in allpairs]),
    }


def panel_sort_key(s: dict) -> tuple:
    """Primary mean win rate, worst-case win rate as the robustness tiebreak."""
    return (s.get("mean_win", 0.0), s.get("worst_win", 0.0))


# ---------------------------------------------------------------------------
# Ladder ranks
#
# `scripts/fetch_team_ranks.py` writes the map; Phase 1 stamps it onto every
# candidate and Phase 2 draws its opponent panel from it. Kept here so the two
# phases cannot disagree about the file's shape.
# ---------------------------------------------------------------------------

TEAM_RANKS_PATH = os.path.join(PROJECT_ROOT, "logs", "team_ranks.json")


def load_team_ranks(path: str | None = None) -> tuple[dict[str, int], str]:
    """`{team name: rank}` plus the snapshot it came from.

    Returns an empty map when the file is absent rather than raising: Phase 1 must
    still run on a machine that has never talked to the Kaggle API, and it records
    `team_rank: null` for every candidate in that case. Phase 2 is the phase that
    refuses to proceed without ranks, and only when asked for a leaderboard panel.
    """
    path = path or TEAM_RANKS_PATH
    if not os.path.exists(path):
        return {}, ""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    teams = payload.get("teams") or {}
    return (
        {name: int(meta["rank"]) for name, meta in teams.items()},
        payload.get("snapshot") or "",
    )


# ---------------------------------------------------------------------------
# JSONL io
# ---------------------------------------------------------------------------


def write_jsonl(path: str, rows) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
