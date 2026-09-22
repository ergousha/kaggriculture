#!/usr/bin/env python3
"""v0.5.2: paired A/B of the v0.4.3 routing-table revert against REAL leaderboard agents.

    uv run python scripts/v052_replay_ab.py --workers 12
    uv run python scripts/v052_replay_ab.py --report

WHY NOT SELF-PLAY. The v0.4.3 table was promoted on in-sample self-play cash against
this repo's own weaker snapshots, and live play against 2000+ opposition says it costs
money (scripts/audit_opponent_bands.py). Re-running self-play therefore cannot arbitrate
its revert: it is the criterion under suspicion, and it will keep preferring the table it
was used to fit. `local_arena.py --opponent opponents/v0_5_1.py` duly reports 0W-7T-5L
for the revert -- which is evidence about self-play, not about the leaderboard.

DESIGN. The strongest opponents available offline are the recorded action streams of the
rank 1-12 teams in logs/september_spy/. A replay is only coherent on the board it was
recorded against, so each episode is run at the replay's OWN seed, recovered from
`info.seed`; that also pins the day-6 shop draw, which is the only thing this change
touches. Both seats are played for every pairing, because the engine commits market
fills in player order.

    24 of the 49 spied episodes carry one of the 27 draws whose route assignment
    differs between v0.5.1 and v0.5.2. The other 25 are the ANCHOR: on those draws
    the two runtimes emit identical actions and every episode must tie exactly. A
    non-tie there is a seat-asymmetric seed, and it is dropped from the scored set
    rather than voiding the run (the lesson of scripts/endgame_rescore.py).

Total 49 * 2 seats * 2 arms = 196 episodes.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
SPY = HERE / "logs" / "september_spy"
OUT = HERE / "logs" / "v052_replay_ab.json"
ARMS = {"v0.5.2": "main.py", "v0.5.1": "opponents/v0_5_1.py"}


def changed_cells() -> set[tuple[str, str]]:
    """The draws whose route assignment differs between the two arms."""
    import ast
    import re

    src = (HERE / "opponents" / "v0_5_1.py").read_text()

    def grab(name: str) -> dict:
        match = re.search(rf"^{name}=(\{{.*?\}})$", src, re.M)
        assert match is not None, name
        return ast.literal_eval(match.group(1))

    # v0.5.2's router is DUAL-expert: `_EXP_SHOP_ROUTES` for the non-YARN pairs and a
    # separate inline V39 table for any pair containing YARN_STORE. Comparing the
    # shipped table against `_EXP_SHOP_ROUTES` alone therefore marks 14 YARN draws as
    # "changed" when both arms in fact route them identically -- which is exactly what
    # the 11 exact-$0 deltas in the first run of this script were.
    table, graft = grab("_V043_TABLE"), grab("_EXP_SHOP_ROUTES")
    v39 = grab_inline_v39()

    def reverted(pair: tuple[str, str]) -> int | None:
        return v39.get(pair) if "YARN_STORE" in pair else graft.get(pair)

    return {k for k in table if table[k] != reverted(k)}


def grab_inline_v39() -> dict:
    """The V39 branch's literal, as it appears inside v0.4.1's `_router`."""
    import ast
    import re

    src = (HERE / "opponents" / "v0_4_1.py").read_text()
    match = re.search(r"state\['expert'\]='V39'\n\s*state\['route'\]=(\{.*?\})\.get\(", src, re.S)
    assert match is not None, "V39 inline table not found in opponents/v0_4_1.py"
    return ast.literal_eval(match.group(1))


def catalogue() -> list[dict]:
    """Every spied replay with a recoverable seed, tagged affected / anchor."""
    sys.path.insert(0, str(HERE / "scripts"))
    import mine_daily as md

    changed = changed_cells()
    rows = []
    for path in sorted(glob.glob(str(SPY / "episode-*.json"))):
        data = md.load_episode(path)
        if not data:
            continue
        seed = (data.get("info") or {}).get("seed") or (data.get("configuration") or {}).get("seed")
        steps = data.get("steps") or []
        if seed is None or len(steps) < 145:
            continue
        town = (steps[144][0].get("observation") or {}).get("town") or {}
        pair = tuple((town.get("unlocked_shops") or [])[:2])
        if len(pair) != 2:
            continue
        rows.append(
            {"replay": path, "seed": int(seed), "pair": list(pair), "affected": pair in changed}
        )
    return rows


def run(workers: int) -> dict:
    import local_arena as la

    rows = catalogue()
    print(
        f"{len(rows)} spied replays with a seed; "
        f"{sum(r['affected'] for r in rows)} carry a changed draw, "
        f"{sum(not r['affected'] for r in rows)} are anchors"
    )
    results: dict[str, dict[str, list]] = {}
    for arm, agent in ARMS.items():
        print(f"\n== arm {arm} ({agent}) ==")
        per: dict[str, list] = {}
        for i, row in enumerate(rows, 1):
            os.environ["KAGGRICULTURE_REPLAY_PATH"] = row["replay"]
            opp = str(HERE / "opponents" / "leaderboard_replay.py")
            res, _ = la.run_set(
                agent,
                opp,
                [row["seed"], row["seed"]],
                la.DEFAULT_STEPS,
                workers,
                False,
                f"v052_{arm.replace('.', '')}_{i}",
            )
            per[row["replay"]] = [
                {"me": r.get("me_cash"), "opp": r.get("opp_cash"), "err": r.get("harness_error")}
                for r in res
            ]
            done = per[row["replay"]]
            print(
                f"  {i:>2}/{len(rows)} seed {row['seed']:<11} {'AFFECTED' if row['affected'] else 'anchor  '} "
                f"{tuple(row['pair'])} -> "
                + ", ".join(f"{d['me']}" if d["me"] is not None else "ERR" for d in done)
            )
        results[arm] = per
    payload = {"rows": rows, "results": results}
    OUT.write_text(json.dumps(payload, indent=1))
    return payload


def report(payload: dict) -> None:
    rows = {r["replay"]: r for r in payload["rows"]}
    a, b = payload["results"]["v0.5.2"], payload["results"]["v0.5.1"]
    anchor_nontie = []
    scored = []
    for replay, meta in rows.items():
        ea, eb = a.get(replay) or [], b.get(replay) or []
        if len(ea) != 2 or len(eb) != 2 or any(x["me"] is None for x in ea + eb):
            continue
        diffs = [x["me"] - x["opp"] for x in ea], [x["me"] - x["opp"] for x in eb]
        if not meta["affected"]:
            if any(abs(p - q) > 1e-6 for p, q in zip(*diffs, strict=True)):
                anchor_nontie.append(replay)
            continue
        scored.append((replay, meta, st.fmean(diffs[0]), st.fmean(diffs[1])))
    print(
        f"\nanchors: {sum(1 for r in rows.values() if not r['affected'])}, "
        f"non-ties (seat-asymmetric, dropped from the read): {len(anchor_nontie)}"
    )
    if not scored:
        print("nothing scored")
        return
    wins = sum(1 for _, _, x, y in scored if x > y)
    print(f"\nAFFECTED DRAWS vs rank 1-12 replays, both seats, paired (n={len(scored)} draws)")
    print(f"  {'draw':34s} {'v0.5.2':>11s} {'v0.5.1':>11s} {'delta':>11s}")
    for _, meta, x, y in sorted(scored, key=lambda s: s[2] - s[3]):
        print(f"  {str(tuple(meta['pair'])):34s} {x:>+11,.0f} {y:>+11,.0f} {x - y:>+11,.0f}")
    deltas = [x - y for _, _, x, y in scored]
    print(f"\n  v0.5.2 better on {wins}/{len(scored)} draws")
    print(
        f"  mean margin   v0.5.2 ${st.fmean(x for _, _, x, _ in scored):+,.0f}"
        f"   v0.5.1 ${st.fmean(y for _, _, _, y in scored):+,.0f}"
    )
    print(f"  paired delta  ${st.fmean(deltas):+,.0f}  (median ${st.median(deltas):+,.0f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--report", action="store_true", help="re-print from logs/v052_replay_ab.json")
    args = ap.parse_args()
    payload = json.loads(OUT.read_text()) if args.report else run(args.workers)
    report(payload)


if __name__ == "__main__":
    main()
