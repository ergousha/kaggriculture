#!/usr/bin/env python3
"""v0.5.1: properly powered A/B of the one v0.4.3 assignment whose evidence did not hold.

    uv run python scripts/v051_pair_ab.py --workers 12
    uv run python scripts/v051_pair_ab.py --report

THE DECISION. v0.4.3 shipped 20 conservative per-draw assignments. Nineteen replicated
out of sample; one did not:

    BRUNCH_SPOT/ICE_CREAM_SHOP  105 -> 110   in-sample +$1,388 (10/12)
                                             held-out  -$1,105 ( 6/12)

The same route change on the REVERSED draw order was REJECTED by the conservative
filter, and its held-out evidence points the other way:

    ICE_CREAM_SHOP/BRUNCH_SPOT  105 -> 110   in-sample   +$521 ( 7/12)
                                             held-out    +$584 ( 8/12)

So route 110 is promoted on one ordering and rejected on the other, on 12 seeds each,
with the two held-out readings disagreeing in sign. Twelve seeds cannot settle this, and
reverting on a single 12-seed negative would be chasing noise -- 1 negative in 20 is
exactly what chance produces. This measures it instead.

DESIGN. The chassis draw enumeration has 73 seeds carrying this shop pair (38 forward,
35 reversed), against 12 used in-sample and 12 held out. Three arms, all built from the
v0.4.2 chassis with a full table so they ship as they measure:

    B  shipped table, the forward pair REVERTED to 105  -> plays 105 on all 73 seeds
    A  shipped table as-is                              -> plays 110 on the 38 forward
    C  shipped table + reversed pair also 110           -> plays 110 on the 35 reversed

B IS ALSO THE ANCHOR. On these 73 seeds B's routing is identical to the peer's
(opponents/v0_4_2.py), so every B episode must tie exactly. Any non-tie is a
seat-asymmetric seed -- the engine commits market fills per-unit in player order, so
identical agents do not tie on ~4-5% of seeds -- and that seed is dropped from every
arm, not used to void the run (the lesson of scripts/endgame_rescore.py).

A is identical to B on the reversed seeds and C is identical to B on the forward seeds,
so each arm is only run where it differs: 73*2 + 38*2 + 35*2 = 292 episodes rather
than 438.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import v043_common as V  # noqa: E402

PAIR_FWD = ("BRUNCH_SPOT", "ICE_CREAM_SHOP")
PAIR_REV = ("ICE_CREAM_SHOP", "BRUNCH_SPOT")
REVERT_ROUTE = 105
CANDIDATE_ROUTE = 110
OUT = os.path.join(V.PROJECT_ROOT, "logs", "v051_pair_ab.jsonl")
VARIANT_DIR = os.path.join(V.PROJECT_ROOT, "logs", "_v051_variants")


def shipped_table() -> dict[tuple[str, str], int]:
    """The table v0.4.3 actually ships: conservative rows take the new route."""
    table: dict[tuple[str, str], int] = {}
    path = os.path.join(V.PROJECT_ROOT, "logs", "v043_draw_table.csv")
    with open(path) as f:
        for row in csv.DictReader(f):
            pair = (row["shop1"], row["shop2"])
            use_new = row.get("conservative") == "1"
            route = row["route"] if use_new else row["incumbent_route"]
            table[pair] = int(route)
    return table


def build_variants() -> dict[str, str]:
    base = shipped_table()
    assert base[PAIR_FWD] == CANDIDATE_ROUTE, base[PAIR_FWD]
    assert base[PAIR_REV] == REVERT_ROUTE, base[PAIR_REV]

    b = dict(base)
    b[PAIR_FWD] = REVERT_ROUTE  # revert the shipped assignment -> equals incumbent here
    c = dict(base)
    c[PAIR_REV] = CANDIDATE_ROUTE  # also promote the reversed ordering

    os.makedirs(VARIANT_DIR, exist_ok=True)
    return {
        "B_revert": V.make_table_variant(
            b, os.path.join(VARIANT_DIR, "B_revert.py"), version_tag="0.5.1-B"
        ),
        "A_shipped": V.make_table_variant(
            base, os.path.join(VARIANT_DIR, "A_shipped.py"), version_tag="0.5.1-A"
        ),
        "C_both": V.make_table_variant(
            c, os.path.join(VARIANT_DIR, "C_both.py"), version_tag="0.5.1-C"
        ),
    }


def target_seeds() -> tuple[list[int], list[int]]:
    draws = V.load_chassis_draws()
    fwd = sorted(s for s, p in draws.items() if p == PAIR_FWD)
    rev = sorted(s for s, p in draws.items() if p == PAIR_REV)
    return fwd, rev


def _run(job: dict) -> dict:
    import local_arena as la

    res = la.run_episode(job)
    res["arm"] = job["arm"]
    res["order"] = job["order"]
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    fwd, rev = target_seeds()
    print(
        f"seeds carrying {PAIR_FWD[0]}/{PAIR_FWD[1]}: {len(fwd)} forward, "
        f"{len(rev)} reversed  (v0.4.3 decided this on 12 + 12)"
    )

    if not args.report:
        variants = build_variants()
        jobs = []
        plan = [
            ("B_revert", fwd, "fwd"),
            ("B_revert", rev, "rev"),
            ("A_shipped", fwd, "fwd"),
            ("C_both", rev, "rev"),
        ]
        for arm, seeds, order in plan:
            for seed in seeds:
                for swap in (False, True):
                    jobs.append(
                        {
                            "agent": variants[arm],
                            "opponent": V.INCUMBENT,
                            "seed": seed,
                            "steps": 720,
                            "swap": swap,
                            "decision_log": None,
                            "replay": None,
                            "arm": arm,
                            "order": order,
                        }
                    )
        done = set()
        if os.path.exists(OUT):
            with open(OUT) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        done.add((r["arm"], r["seed"], r["swap"]))
                    except Exception:
                        continue
        jobs = [j for j in jobs if (j["arm"], j["seed"], j["swap"]) not in done]
        print(f"{len(jobs)} episodes to run ({len(done)} on file)")
        if jobs:
            t0 = time.time()
            ctx = mp.get_context("spawn")
            with open(OUT, "a") as sink, ctx.Pool(args.workers) as pool:
                for i, res in enumerate(pool.imap_unordered(_run, jobs, chunksize=2)):
                    sink.write(json.dumps(res) + "\n")
                    sink.flush()
                    if (i + 1) % 50 == 0:
                        el = (time.time() - t0) / 60
                        print(f"  {i + 1}/{len(jobs)}  ({el:.0f}m)", flush=True)

    rows = [json.loads(line) for line in open(OUT) if line.strip()]
    # 1. the anchor identifies seat-asymmetric seeds
    bad = {r["seed"] for r in rows if r["arm"] == "B_revert" and r["me_cash"] != r["opp_cash"]}
    anchor_n = sum(1 for r in rows if r["arm"] == "B_revert")
    print(
        f"\nanchor (B_revert) episodes: {anchor_n}, "
        f"non-tying seeds: {len(bad)} -> dropped from every arm"
    )
    for s in sorted(bad):
        d = [
            r["me_cash"] - r["opp_cash"] for r in rows if r["arm"] == "B_revert" and r["seed"] == s
        ]
        print(f"    seed {s}: anchor diff " + ", ".join(f"${x:+,.0f}" for x in d))

    clean = [r for r in rows if r["seed"] not in bad]
    print(
        f"\n{'arm':<11}{'order':<6}{'n':>4} {'W':>3} {'T':>3} {'L':>3} "
        f"{'mean margin':>13}  {'route played':<13}"
    )
    stats = {}
    for arm, order, route in (
        ("B_revert", "fwd", 105),
        ("A_shipped", "fwd", 110),
        ("B_revert", "rev", 105),
        ("C_both", "rev", 110),
    ):
        rs = [r for r in clean if r["arm"] == arm and r["order"] == order]
        if not rs:
            continue
        d = [r["me_cash"] - r["opp_cash"] for r in rs]
        w = sum(1 for x in d if x > 0)
        t = sum(1 for x in d if x == 0)
        stats[(order, route)] = d
        print(
            f"{arm:<11}{order:<6}{len(rs):>4} {w:>3} {t:>3} {len(d) - w - t:>3} "
            f"{statistics.fmean(d):>+13,.0f}  route {route}"
        )

    print("\nVERDICT")
    print("  Selection is on WIN RATE, not cash. rank_cvar.py: the competition scores a")
    print("  skill rating driven by wins, and cash is overwhelmingly common-mode")
    print("  (corr(our cash, opponent cash) = +0.86 on live episodes), so a cash margin")
    print("  mostly measures whether the seed was generous to both seats.")
    for order, label in (
        ("fwd", f"{PAIR_FWD[0]}/{PAIR_FWD[1]}"),
        ("rev", f"{PAIR_REV[0]}/{PAIR_REV[1]}"),
    ):
        a = stats.get((order, CANDIDATE_ROUTE))
        b = stats.get((order, REVERT_ROUTE))
        if not a or not b:
            continue
        w = sum(1 for x in a if x > 0)
        n = len(a)
        wr = w / n * 100
        # one-sided normal approximation to the binomial against p=0.5
        z = (w - n / 2) / (n**0.5 / 2)
        cash = statistics.fmean(a) - statistics.fmean(b)
        shipped = "SHIPPED in v0.4.3" if order == "fwd" else "rejected by v0.4.3"
        print(f"\n  {label}  ({shipped})")
        print(
            f"    route {CANDIDATE_ROUTE} win rate {w}/{n} = {wr:.1f}%   z={z:+.2f}   "
            f"cash ${cash:+,.0f}/game"
        )
        if z >= 1.64:
            print(
                f"    -> PROMOTE route {CANDIDATE_ROUTE}: win-rate edge significant "
                f"at one-sided 0.05"
            )
        elif z <= -1.64:
            print(f"    -> REVERT to route {REVERT_ROUTE}: win-rate deficit significant")
        else:
            print(
                f"    -> NO CHANGE: {wr:.1f}% is inside the noise band (|z| < 1.64); "
                f"changing it either way would be chasing noise"
            )


if __name__ == "__main__":
    main()
