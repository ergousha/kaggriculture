#!/usr/bin/env python3
"""Phase 1 -- mine a pool of candidate routes from historical leaderboard replays.

    uv run python mine_replays.py --replays replays --out candidates.jsonl
    uv run python mine_replays.py --limit 200 --workers 12        # quick pass
    uv run python mine_replays.py --no-fidelity                   # skip the gate (debug only)

Three passes, cheapest first:

  1. HEAD SCAN (I/O bound, threads). Read a 64 KB prefix of every replay and pull
     `rewards` and `info.seed` out of it. `rewards` sits at ~byte 880 and
     `info` earlier still, while `steps` starts at ~6.6 KB and runs for 32 MB, so
     this screens the corpus for ~1/4000th of the read cost of parsing it. Only
     files with a seat above the threshold survive to pass 2.

  2. EXTRACT (CPU bound, processes). Full-parse the survivors, extract *both*
     seats' 719-step traces, normalize, validate length and hash the canonical
     form. Both seats are needed even when only one qualifies, because the
     fidelity check has to reconstruct the whole episode.

  3. FIDELITY GATE (simulation bound, processes). Re-run each surviving episode
     closed-loop -- both seats replaying their own extracted traces verbatim on the
     episode's recovered seed -- and require the reproduced rewards to equal the
     recorded ones. A mismatch means the extraction or the format conversion is
     wrong, so the trace is logged and excluded rather than allowed to poison every
     downstream phase.

On the threshold: $85k is deliberately modest. A route that banked $110k+ did so
partly on a lucky seed, and filtering at that level selects for luck-dependent
routes -- the opposite of the goal. This corpus proves the point directly: one
719-step trace appears twice, having banked $96,946 on one seed and $131,597 on
another. Phase 3's CVaR ranking does the real selection.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from mining import common
from mining.common import (
    PROJECT_ROOT,
    ROUTE_STEPS,
    head_scan,
    json_loads,
    normalize_route,
    route_hash,
)

VERBATIM_AGENT = os.path.join(PROJECT_ROOT, "mining", "verbatim_agent.py")
DEFAULT_THRESHOLD = 85_000.0


# ---------------------------------------------------------------------------
# Pass 1: head scan
# ---------------------------------------------------------------------------


def scan_one(path: str) -> dict | None:
    info = head_scan(path)
    if info is None:
        return None
    info["path"] = path
    return info


def pass_head_scan(files: list[str], threshold: float, workers: int) -> tuple[list[dict], dict]:
    stats = {"files": len(files), "not_replay": 0, "no_rewards": 0, "no_seed": 0, "below": 0}
    survivors = []
    with ThreadPoolExecutor(max(4, workers * 2)) as ex:
        for info in ex.map(scan_one, files, chunksize=16):
            if info is None:
                stats["not_replay"] += 1
                continue
            rewards = info.get("rewards")
            if not rewards:
                stats["no_rewards"] += 1
                continue
            seats = [i for i, v in enumerate(rewards) if v > threshold]
            if not seats:
                stats["below"] += 1
                continue
            if info.get("seed") is None:
                # Without the seed the fidelity gate cannot run; do not admit.
                stats["no_seed"] += 1
                continue
            info["qualifying_seats"] = seats
            survivors.append(info)
    return survivors, stats


# ---------------------------------------------------------------------------
# Pass 2: extract + normalize + hash
# ---------------------------------------------------------------------------


def extract_one(task: dict) -> dict:
    """Parse one replay and return both seats' normalized traces."""
    path = task["path"]
    out = {"path": path, "seed": task["seed"], "episode": task.get("episode")}
    try:
        with open(path, "rb") as f:
            data = json_loads(f.read())
    except (OSError, ValueError) as exc:
        out["error"] = f"parse: {type(exc).__name__}"
        return out

    steps = data.get("steps") or []
    if len(steps) < 2:
        out["error"] = f"steps={len(steps)}"
        return out

    rewards = [0.0 if v is None else float(v) for v in (data.get("rewards") or [])]
    teams = [str(t).strip('"') for t in ((data.get("info") or {}).get("TeamNames") or [])]

    # `steps[t]["action"]` is the action that PRODUCED `steps[t]`, and steps[0]'s
    # action is a framework placeholder, so the action taken while observing the
    # state at index t is stored at index t + 1.
    traces = {}
    for seat in range(len(rewards)):
        raw = []
        for t in range(1, len(steps)):
            entry = steps[t][seat] if seat < len(steps[t]) else {}
            raw.append((entry or {}).get("action"))
        traces[seat] = normalize_route(raw)

    out["rewards"] = rewards
    out["teams"] = teams
    out["traces"] = {str(s): common.encode_route_b85(r) for s, r in traces.items()}
    out["hashes"] = {str(s): route_hash(r) for s, r in traces.items()}
    out["lengths"] = {str(s): len(r) for s, r in traces.items()}
    out["final_money"] = _final_money(steps)
    return out


def _final_money(steps) -> list[float]:
    """Score-read sanity: rewards[seat] should equal the last observation's money."""
    try:
        farms = steps[-1][0]["observation"]["farms"]
        return [float(f.get("money", 0.0)) for f in farms]
    except (IndexError, KeyError, TypeError, ValueError):
        return []


# ---------------------------------------------------------------------------
# Pass 3: fidelity gate
# ---------------------------------------------------------------------------


def fidelity_one(task: dict) -> dict:
    """Reconstruct one episode closed-loop on its recovered seed.

    Both seats replay their own extracted traces verbatim. The environment is
    deterministic given the seed, so this must reproduce the recorded rewards
    exactly; `tolerance` exists only to absorb float round-tripping.
    """
    import json as _json
    import tempfile

    from mining.common import decode_route_b85

    seed = task["seed"]
    tolerance = task["tolerance"]
    tmpdir = tempfile.mkdtemp(prefix="kaggri_fid_")
    try:
        for seat, blob in task["traces"].items():
            with open(os.path.join(tmpdir, f"t{seat}.json"), "w") as f:
                _json.dump(decode_route_b85(blob), f)
            os.environ[f"KAGGRI_TRACE_{seat}"] = os.path.join(tmpdir, f"t{seat}.json")

        from kaggle_environments import make

        env = make(
            "kaggriculture",
            configuration={"episodeSteps": task["steps"], "seed": seed},
            debug=False,
        )
        n_seats = len(task["traces"])
        env.run([VERBATIM_AGENT] * n_seats)
        final = env.steps[-1]
        got = [float(s.reward) if s.reward is not None else 0.0 for s in final]
        statuses = [s.status for s in final]
    except Exception as exc:  # a harness failure is itself a gate failure
        return {"path": task["path"], "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)

    want = task["rewards"]
    deltas = [abs(a - b) for a, b in zip(got, want, strict=False)]
    ok = len(got) == len(want) and all(d <= tolerance for d in deltas)
    return {
        "path": task["path"],
        "ok": ok,
        "reproduced": got,
        "recorded": want,
        "max_delta": max(deltas) if deltas else float("inf"),
        "statuses": statuses,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase 1: mine candidate routes from replays")
    ap.add_argument("--replays", default="replays", help="replay root (searched recursively)")
    ap.add_argument("--out", default="candidates.jsonl")
    ap.add_argument("--rejects", default="logs/mine_rejects.jsonl")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--steps", type=int, default=common.DEFAULT_STEPS)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 3))
    ap.add_argument("--limit", type=int, default=0, help="cap files scanned (smoke tests)")
    ap.add_argument(
        "--tolerance",
        type=float,
        default=1.0,
        help="max |reproduced - recorded| cash allowed by the fidelity gate",
    )
    ap.add_argument(
        "--no-fidelity",
        action="store_true",
        help="skip the fidelity gate (debugging only; pool is then unvalidated)",
    )
    ap.add_argument(
        "--team-ranks",
        default=common.TEAM_RANKS_PATH,
        help="team -> ladder rank map from scripts/fetch_team_ranks.py",
    )
    args = ap.parse_args(argv)

    root = args.replays if os.path.isabs(args.replays) else os.path.join(PROJECT_ROOT, args.replays)
    files = sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no replay JSON found under {root}")

    print(f"Phase 1: mining {len(files):,} replays under {root}")
    print(
        f"  threshold ${args.threshold:,.0f}   workers {args.workers}   json backend {common.JSON_BACKEND}"
    )

    team_ranks, rank_snapshot = common.load_team_ranks(args.team_ranks)
    if team_ranks:
        print(f"  team ranks: {len(team_ranks):,} teams from {rank_snapshot}")
    else:
        # Not fatal here, but Phase 2's leaderboard panel needs these, so make the
        # omission loud rather than letting it surface 7 hours later.
        print(
            f"  !! no team ranks at {args.team_ranks}; every candidate gets team_rank=null "
            "and `--panel-source leaderboard-top` will refuse to run. "
            "Run scripts/fetch_team_ranks.py --refresh first."
        )

    t0 = time.time()
    survivors, stats = pass_head_scan(files, args.threshold, args.workers)
    t_scan = time.time() - t0
    print(
        f"\n  [1/3] head scan      {t_scan:6.1f}s  "
        f"{len(survivors):,} of {stats['files']:,} files have a seat > ${args.threshold:,.0f}"
    )
    print(
        f"        skipped: not-a-replay {stats['not_replay']}  no-rewards {stats['no_rewards']}  "
        f"below-threshold {stats['below']}  no-seed {stats['no_seed']}"
    )
    if not survivors:
        raise SystemExit("nothing above threshold")

    t0 = time.time()
    extracted = []
    errors = []
    with ProcessPoolExecutor(args.workers) as ex:
        for row in ex.map(extract_one, survivors, chunksize=2):
            if row.get("error"):
                errors.append(row)
            else:
                extracted.append(row)
    t_ext = time.time() - t0
    print(
        f"\n  [2/3] extract        {t_ext:6.1f}s  {len(extracted):,} parsed, {len(errors)} failed"
    )

    # Build the candidate list: one per (file, qualifying seat), deduplicated by
    # canonical trace hash. Dedup happens BEFORE the fidelity gate so we never pay
    # for the same trace twice.
    by_hash: dict[str, dict] = {}
    rejects = list(errors)
    length_bad = 0
    money_mismatch = 0
    lookup = {row["path"]: row for row in survivors}

    for row in extracted:
        seats = lookup[row["path"]]["qualifying_seats"]
        for seat in seats:
            key = str(seat)
            if row["lengths"].get(key) != ROUTE_STEPS:
                length_bad += 1
                rejects.append(
                    {
                        "path": row["path"],
                        "seat": seat,
                        "error": f"trace length {row['lengths'].get(key)} != {ROUTE_STEPS}",
                    }
                )
                continue
            cash = row["rewards"][seat]
            money = row.get("final_money") or []
            if seat < len(money) and abs(money[seat] - cash) > 1.0:
                money_mismatch += 1
                rejects.append(
                    {
                        "path": row["path"],
                        "seat": seat,
                        "error": f"rewards {cash} != final money {money[seat]}",
                    }
                )
                continue
            h = row["hashes"][key]
            if h in by_hash:
                by_hash[h]["duplicates"] += 1
                continue
            team = row["teams"][seat] if seat < len(row["teams"]) else "?"
            by_hash[h] = {
                "hash": h,
                "route_b85": row["traces"][key],
                "steps": ROUTE_STEPS,
                "recorded_cash": cash,
                "seed": row["seed"],
                "seat": seat,
                "episode": row.get("episode"),
                "team": team,
                # Ladder rank of the team that played this route, as of the snapshot
                # named above. None for a team no longer on the board (or renamed);
                # Phase 2 treats unknown as "not a top team" rather than guessing.
                "team_rank": team_ranks.get(team),
                "source_path": os.path.relpath(row["path"], PROJECT_ROOT),
                "duplicates": 0,
            }

    print(
        f"        unique traces  {len(by_hash):,}   "
        f"rejected: length {length_bad}  score-read {money_mismatch}"
    )

    admitted = list(by_hash.values())

    if args.no_fidelity:
        print("\n  [3/3] fidelity gate  SKIPPED (--no-fidelity)")
        for row in admitted:
            row["fidelity"] = "skipped"
    else:
        # One episode validates every qualifying seat of a given source file at
        # once, so group the work by file rather than by candidate.
        groups: dict[str, list[dict]] = {}
        for row in admitted:
            groups.setdefault(row["source_path"], []).append(row)
        tasks = []
        by_path = {os.path.relpath(r["path"], PROJECT_ROOT): r for r in extracted}
        for rel in groups:
            src = by_path[rel]
            tasks.append(
                {
                    "path": rel,
                    "seed": src["seed"],
                    "steps": args.steps,
                    "rewards": src["rewards"],
                    "traces": src["traces"],
                    "tolerance": args.tolerance,
                }
            )

        t0 = time.time()
        passed_paths = set()
        n_done = 0
        with ProcessPoolExecutor(args.workers) as ex:
            for res in ex.map(fidelity_one, tasks, chunksize=1):
                n_done += 1
                if res.get("ok"):
                    passed_paths.add(res["path"])
                else:
                    rejects.append({**res, "error": res.get("error") or "fidelity mismatch"})
                if n_done % 250 == 0:
                    rate = n_done / (time.time() - t0)
                    print(
                        f"        ... {n_done:,}/{len(tasks):,} episodes  "
                        f"{rate:.1f}/s  {len(passed_paths):,} passed",
                        flush=True,
                    )
        t_fid = time.time() - t0
        kept = []
        for row in admitted:
            if row["source_path"] in passed_paths:
                row["fidelity"] = "exact"
                kept.append(row)
        failed = len(admitted) - len(kept)
        print(
            f"\n  [3/3] fidelity gate  {t_fid:6.1f}s  {len(tasks):,} episodes  "
            f"{len(kept):,} admitted, {failed:,} excluded"
        )
        admitted = kept

    admitted.sort(key=lambda r: -r["recorded_cash"])
    out_path = args.out if os.path.isabs(args.out) else os.path.join(PROJECT_ROOT, args.out)
    n = common.write_jsonl(out_path, admitted)
    rej_path = (
        args.rejects if os.path.isabs(args.rejects) else os.path.join(PROJECT_ROOT, args.rejects)
    )
    common.write_jsonl(rej_path, rejects)

    print(f"\n  wrote {n:,} candidates -> {out_path}")
    print(f"  wrote {len(rejects):,} rejects  -> {rej_path}")
    if admitted:
        cash = [r["recorded_cash"] for r in admitted]
        print(
            f"  recorded cash of pool: min ${min(cash):,.0f}  "
            f"median ${sorted(cash)[len(cash) // 2]:,.0f}  max ${max(cash):,.0f}"
        )
        print(
            "  NOTE recorded cash is seed-luck contaminated and is metadata only; "
            "Phase 3 selects on CVaR."
        )
        _report_rank_coverage(admitted, rank_snapshot)
    return 0


def _report_rank_coverage(admitted: list[dict], snapshot: str) -> None:
    """How much of the pool the ladder join actually reached.

    Phase 2's leaderboard panel can only draw from the matched part, so a thin
    match at the top of the ladder is the failure mode to catch here -- not the
    overall match rate.
    """
    ranked = [r for r in admitted if r.get("team_rank") is not None]
    print(
        f"  ladder join ({snapshot or 'no snapshot'}): {len(ranked):,} of {len(admitted):,} "
        f"candidates matched a team on the board, "
        f"{len({r['team'] for r in ranked}):,} distinct teams"
    )
    for n in (10, 30, 100):
        band = [r for r in ranked if r["team_rank"] <= n]
        print(
            f"    top {n:>3}: {len(band):>5,} candidates from "
            f"{len({r['team'] for r in band}):>3} teams"
        )


if __name__ == "__main__":
    sys.exit(main())
