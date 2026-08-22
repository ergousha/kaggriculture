"""Movement census and shortest-path slack report for a route (issue #29).

Issue #29 opens with a unit-op census showing half the shipped route's labour is
walking, and asks how much of that walking is recoverable by re-planning each
movement run as a shortest path between its two fixed endpoints. This script is
the measurement, kept separate from `search/route_search.py` so the finding is
reproducible without spending a panel evaluation:

    uv run python scripts/analyse_movement.py                 # the incumbent
    uv run python scripts/analyse_movement.py --top 20        # worst segments
    uv run python scripts/analyse_movement.py --verify        # run the no-op gate
    uv run python scripts/analyse_movement.py --emit out.py   # bake the re-path

The answer for the v0.3.1 incumbent is in `docs/experiments.md`: the walking is
already 98.3% Manhattan-optimal, so the 50% movement share is a property of the
task assignment and not of slack in the pathing. What is recoverable is mostly
end-of-day drift -- 359 of the 411 slack turns are moves whose destination the
day reset discards, which is what `--drop-terminal` banks.
"""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mining import common  # noqa: E402
from search import board_paths  # noqa: E402


def _load_route(args) -> tuple[list[dict], str]:
    """The shipped incumbent by default; any pooled route by hash prefix."""
    if args.candidate:
        pool = args.candidates
        if not os.path.exists(pool):
            raise SystemExit(f"candidate pool missing at {pool}")
        for cand in common.read_jsonl(pool):
            if cand.get("hash", "").startswith(args.candidate):
                route = common.decode_route_b85(cand["route_b85"])
                return common.normalize_route(route), f"{cand['hash'][:10]} ({cand.get('team')})"
        raise SystemExit(f"no candidate in {pool} with hash prefix {args.candidate}")
    import main as main_module

    return common.normalize_route(main_module._ROUTE), "main.py._ROUTE"


def print_census(route: list[dict]) -> None:
    c = board_paths.census(route)
    print(f"\n  Unit-op census: {len(route)} steps, {c['unit_turns']} unit-turns")
    print(f"  {'op':<20} {'count':>7} {'share':>8}")
    for op, n in c["counts"].items():
        print(f"  {op:<20} {n:>7} {n / c['unit_turns']:>7.1%}")
    print(
        f"  {'movement total':<20} {c['movement']:>7} {c['movement_share']:>7.1%}\n"
        f"  {'productive total':<20} {c['productive']:>7} {c['productive_share']:>7.1%}\n"
        f"  {'idle (PASS)':<20} {c['idle']:>7} {c['idle_share']:>7.1%}\n"
        f"  {c['moves_per_productive_op']:.2f} moves per productive tile op"
    )


def print_slack(route: list[dict], top: int) -> dict:
    r = board_paths.slack_report(route)
    print(
        f"\n  Movement segments (a unit's stretch between two position-dependent ops,\n"
        f"  never crossing a day boundary):\n"
        f"    segments with movement in them      {r['segments']:>6}\n"
        f"      interior (an op follows)          {r['interior_segments']:>6}\n"
        f"      terminal (day ends first)         {r['terminal_segments']:>6}\n"
        f"    moves in interior segments          {r['interior_moves']:>6}\n"
        f"      load-bearing (Manhattan-required) {r['interior_required']:>6}\n"
        f"      slack (recoverable)               {r['interior_slack']:>6}"
        f"   in {r['wasteful_segments']} segment(s)\n"
        f"    moves in terminal segments          {r['terminal_moves']:>6}"
        f"   (all dead: the day reset discards where they lead)\n"
        f"    interior path optimality            {r['path_optimality']:>6.1%}"
    )
    if top:
        worst = sorted(
            (s for s in r["segments_detail"] if s.slack or (s.terminal and s.moves)),
            key=lambda s: -(s.moves if s.terminal else s.slack),
        )[:top]
        print(f"\n  Worst {len(worst)} segment(s):")
        print(f"  {'slot':>4} {'steps':>12} {'moves':>6} {'need':>5} {'slack':>6}  route")
        for s in worst:
            need = 0 if s.terminal else s.required
            slack = s.moves if s.terminal else s.slack
            kind = " (terminal)" if s.terminal else ""
            print(
                f"  {s.slot:>4} {f'{s.start}..{s.end}':>12} {s.moves:>6} {need:>5} {slack:>6}"
                f"  {s.origin} -> {s.destination}{kind}"
            )
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Movement census and re-path slack (issue #29)")
    ap.add_argument("--candidate", help="hash prefix of a pooled route (default: main.py._ROUTE)")
    ap.add_argument("--candidates", default=os.path.join(_ROOT, "candidates.jsonl"))
    ap.add_argument("--top", type=int, default=10, help="how many worst segments to list")
    ap.add_argument(
        "--drop-terminal",
        action="store_true",
        help="also bank moves in segments the day reset makes pointless",
    )
    ap.add_argument("--verify", action="store_true", help="run #29's no-op gate on the re-path")
    ap.add_argument(
        "--emit", help="bake the re-pathed route into a route-replay agent at this path"
    )
    args = ap.parse_args(argv)

    route, source = _load_route(args)
    print(f"  route: {source}, hash {common.route_hash(route)[:10]}")
    print_census(route)
    print_slack(route, args.top)

    mutated, stats = board_paths.repath(route, drop_terminal=args.drop_terminal)
    print(
        f"\n  Re-path (drop_terminal={args.drop_terminal}):\n"
        f"    segments rewritten {stats['segments_rewritten']}, "
        f"{stats['moves_before']} -> {stats['moves_after']} moves, "
        f"{stats['turns_recovered']} turn(s) banked as PASS\n"
        f"    new hash {common.route_hash(common.normalize_route(mutated))[:10]}"
    )
    after = board_paths.census(mutated)
    print(
        f"    movement {after['movement']} ({after['movement_share']:.1%}), "
        f"idle {after['idle']} ({after['idle_share']:.1%})"
    )

    if args.verify:
        violations = board_paths.verify_schedule(route, mutated)
        if violations:
            print(f"\n  GATE FAILED: {len(violations)} violation(s)")
            for v in violations[:20]:
                print(f"    {v}")
            return 1
        print(
            "\n  GATE OK: every non-movement op fires on its original step from its\n"
            "  original tile, and the market stream is untouched."
        )

    if args.emit:
        normalized = common.normalize_route(mutated)
        h = common.route_hash(normalized)
        common.write_route_agent(
            normalized,
            args.emit,
            provenance={
                "team": "route-search",
                "episode": "repath",
                "seat": 0,
                "hash": h,
                "steps": len(normalized),
            },
            version=f"repath-{h[:8]}",
        )
        print(f"  wrote {args.emit} (hash {h})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
