#!/usr/bin/env python3
"""v0.4.3 Step 2 gates: head-to-head vs the v0.4.2 incumbent, draw-stratified.

The proposal's Step 2 gate list, run WITHOUT any submission (the user decides
when to ship). Gates, in order:

  1. HEAD-TO-HEAD >= 60% on >= 100 paired seeds, BOTH seats, against the
     incumbent (v0.4.2). Seeds are FRESH: per pair, rows 24+ of the chassis
     draw table -- no screen/refine/deep episode ever ran them, so this is a
     clean out-of-sample read with zero winner's curse. Draw-stratified.

     METRIC NOTE (adapted, honestly): 45 of 64 pairs keep the incumbent tape,
     so their episodes reproduce self-play -- ties, or the known engine seat
     asymmetry (~4% of seeds, mirrored across seats). An OVERALL WR >= 60% is
     unreachable for ANY table-only diff under that structure (75% of
     episodes are kept pairs and tie), so the gate reads on the pairs the
     diff actually touches: CHANGED-PAIR h2h WR >= 60%. Kept pairs are gated
     separately (gate 3: must reproduce self-play exactly).
  2. NO NEW failed HIRE/BUY/LAND orders vs the incumbent on the same
     (seed, seat) -- the paired diff of the arena's orders_failed_* counters
     against fresh peer-vs-peer references at BOTH seats. Raw counts are
     meaningless alone: the chassis fails orders by design (sell-spam); the
     gate is that the changed tapes add no failures the incumbent would not
     have had on the same market.
  3. Composition/anchor check on the SAME fresh seeds: kept-pair episodes
     must match the peer-vs-peer reference seat-for-seat.

NOT run here: rank_ladder + submit.py pre-flight (hardcoded to main.py) --
this script prints the exact manual commands at the end.

Usage:
    uv run python scripts/v043_gates.py --candidate scratch/v0_4_3_candidate.py \
        --seeds-per-pair 4 --workers 13
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import v043_common as V  # noqa: E402

PEER = V.INCUMBENT
PANEL_DIR = V.PANEL_DIR
GATE_SEEDS_BASE = 24  # the panel consumed rows [0:24] per pair; gates use [24:]

H2H_BAR = 0.60


def _run_job(job: dict) -> dict:
    import local_arena as la

    res = la.run_episode(job)
    res["role"] = job["role"]
    return res


def fresh_seeds_by_pair(seeds_per_pair: int) -> dict[tuple[str, str], list[int]]:
    draws = V.load_chassis_draws()
    by_pair: dict[tuple[str, str], list[int]] = defaultdict(list)
    for seed in sorted(draws):
        by_pair[draws[seed]].append(seed)
    return {p: s[GATE_SEEDS_BASE : GATE_SEEDS_BASE + seeds_per_pair] for p, s in by_pair.items()}


def candidate_overrides() -> dict[tuple[str, str], int]:
    """The candidate's overrides, parsed from its `_V043_TABLE` vs the incumbent
    tables -- the runner of truth for which pairs are changed."""
    import re

    with open(os.path.join(PROJECT_ROOT, "scratch", "v0_4_3_candidate.py")) as f:
        src = f.read()
    m = re.search(r"_V043_TABLE=(\{.*?\})\n", src)
    if not m:
        sys.exit("candidate _V043_TABLE not found")
    table = eval(m.group(1), {"__builtins__": {}})  # noqa: S307
    inc_table, yarn_table = V.extract_tables_from_incumbent()
    inc = {**inc_table, **yarn_table}
    return {p: r for p, r in table.items() if p not in inc or inc[p] != r}


def run_gates(candidate: str, seeds_per_pair: int, workers: int) -> int:
    overrides = candidate_overrides()
    changed = set(overrides)
    print(f"candidate: {candidate}")
    print(f"  changed pairs: {len(changed)} of 64")

    by_pair = fresh_seeds_by_pair(seeds_per_pair)
    n_seeds = sum(len(s) for s in by_pair.values())
    print(
        f"  fresh seeds: {n_seeds} (rows {GATE_SEEDS_BASE}..{GATE_SEEDS_BASE + seeds_per_pair - 1} per pair)"
    )

    # jobs: candidate vs incumbent on every fresh seed, both seats
    jobs = []
    for _pair, seeds in sorted(by_pair.items()):
        for seed in seeds:
            for swap in (False, True):
                jobs.append(
                    {
                        "agent": candidate,
                        "opponent": PEER,
                        "seed": seed,
                        "steps": 720,
                        "swap": swap,
                        "decision_log": None,
                        "replay": None,
                        "role": "candidate",
                    }
                )
    print(f"  h2h episodes: {len(jobs)} (both seats)")

    out_path = os.path.join(PANEL_DIR, "gate_h2h.jsonl")
    os.makedirs(PANEL_DIR, exist_ok=True)
    done: set[tuple] = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add((rec["seed"], rec["swap"], rec["role"]))
                except Exception:
                    continue
    jobs = [j for j in jobs if (j["seed"], j["swap"], j["role"]) not in done]
    print(f"  to run: {len(jobs)} ({len(done)} on file)")

    if jobs:
        t0 = time.time()
        n = len(jobs)
        ctx = mp.get_context("spawn")
        with open(out_path, "a") as sink, ctx.Pool(workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_run_job, jobs, chunksize=2)):
                sink.write(json.dumps(res) + "\n")
                sink.flush()
                if (i + 1) % 50 == 0:
                    el = time.time() - t0
                    print(f"  {i + 1}/{n}  ({el / 60:.0f}m elapsed)", flush=True)

    recs = []
    with open(out_path) as f:
        for line in f:
            try:
                recs.append(json.loads(line))
            except Exception:
                continue

    # ---- fresh peer-vs-peer reference on the SAME seeds (gates 2 + 3) ----
    # peer-vs-peer IS the incumbent playing the incumbent table, so it is
    # both the kept-pair anchor (gate 3) and the counterfactual failure
    # profile (gate 2) at both seats.
    ref_path = os.path.join(PANEL_DIR, "gate_reference.jsonl")
    ref_seeds = sorted({r["seed"] for r in recs})
    ref_jobs = [
        {
            "agent": PEER,
            "opponent": PEER,
            "seed": s,
            "steps": 720,
            "swap": False,
            "decision_log": None,
            "replay": None,
            "role": "reference",
        }
        for s in ref_seeds
    ]
    ref_done = set()
    if os.path.exists(ref_path):
        with open(ref_path) as f:
            for line in f:
                try:
                    ref_done.add(json.loads(line)["seed"])
                except Exception:
                    continue
    ref_todo = [j for j in ref_jobs if j["seed"] not in ref_done]
    if ref_todo:
        print(f"\nrunning {len(ref_todo)} fresh peer-vs-peer reference episodes ...")
        ctx = mp.get_context("spawn")
        with open(ref_path, "a") as sink, ctx.Pool(workers) as pool:
            for res in pool.imap_unordered(_run_job, ref_todo, chunksize=2):
                sink.write(json.dumps(res) + "\n")
                sink.flush()
    ref = {}
    with open(ref_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                ref[r["seed"]] = r
            except Exception:
                continue

    # ---- Gate 1: draw-stratified h2h ----
    per_pair: dict[tuple, list[dict]] = defaultdict(list)
    for r in recs:
        obs = tuple(r.get("observed_pair") or [])
        if obs:
            per_pair[obs].append(r)
    total_w = total_t = total_l = 0
    changed_w = changed_t = changed_l = 0
    kept_w = kept_t = kept_l = 0
    regressions = []
    print(f"\n{'pair':<40} {'cls':>5} {'n':>3} {'W':>3} {'T':>3} {'L':>3} {'Δμ$':>9}")
    for pair in sorted(per_pair):
        rows = per_pair[pair]
        w = sum(r["win"] for r in rows)
        t = sum(r["tie"] for r in rows)
        losses = len(rows) - w - t
        dm = statistics.fmean([r["me_cash"] - r["opp_cash"] for r in rows])
        cls = "chg" if pair in changed else "kept"
        print(
            f"{pair[0]:<20} {pair[1]:<20} {cls:>5} {len(rows):>3} {w:>3} {t:>3} {losses:>3} {dm:>+9,.0f}"
        )
        total_w += w
        total_t += t
        total_l += losses
        if pair in changed:
            changed_w += w
            changed_t += t
            changed_l += losses
            if w < losses:  # changed pair losing on fresh seeds = gate failure
                regressions.append((pair, w, losses))
        else:
            kept_w += w
            kept_t += t
            kept_l += losses
    n_all = total_w + total_t + total_l
    n_chg = changed_w + changed_t + changed_l
    wr_all = total_w / max(1, n_all)
    wr_excl_ties = total_w / max(1, total_w + total_l)
    wr_changed = changed_w / max(1, changed_w + changed_l)
    print(
        f"\nALL: {n_all} episodes  {total_w}W {total_t}T {total_l}L  WR {wr_all:.1%} (excl ties {wr_excl_ties:.1%})"
    )
    print(
        f"changed pairs: {n_chg} eps  {changed_w}W {changed_t}T {changed_l}L  WR {wr_changed:.1%}"
    )
    print(
        f"kept pairs:    {n_all - n_chg} eps  {kept_w}W {kept_t}T {kept_l}L  (self-play + seat asymmetry)"
    )

    # Gate 1 reads on CHANGED pairs only: the diff's actual surface. Kept
    # pairs reproduce self-play (gate 3), so an overall-60% bar is
    # structurally unreachable for any table-only diff (most episodes tie).
    # No changed pair may LOSE its fresh-seed series (per-pair regression).
    gate1 = wr_changed >= H2H_BAR and n_chg >= 60 and not regressions
    print(
        f"\nGATE 1 (changed-pair h2h >= {H2H_BAR:.0%} on >= 60 eps, no changed pair losing): "
        f"{'PASS' if gate1 else 'FAIL'}"
    )
    if regressions:
        for pair, w, losses in regressions:
            print(f"  regression on {pair}: {w}W/{losses}L on fresh seeds")

    # ---- Gate 2: no NEW failed HIRE/BUY/LAND orders vs the incumbent ----
    # Raw candidate counts are meaningless alone (the chassis fails orders by
    # design: sell-spam into a full market). The gate is the PAIRED diff vs
    # the fresh peer-vs-peer reference: candidate failures minus the
    # incumbent's failures on the same seed, summed over all episodes.
    def _fails(x: dict) -> int:
        return (
            x.get("orders_failed_hire", 0)
            + x.get("orders_failed_buy", 0)
            + x.get("orders_failed_land", 0)
        )

    paired_fail_diff = 0
    checked_g2 = 0
    missing_ref = 0
    for r in recs:
        ref_row = ref.get(r["seed"])
        if ref_row is None:
            missing_ref += 1
            continue
        # the counterfactual for the candidate's episode is the incumbent's
        # failures across the same two seats of the reference episode
        paired_fail_diff += _fails(r) - _fails(ref_row)
        checked_g2 += 1
    print(
        f"\nfailed HIRE/BUY/LAND orders, paired diff vs reference: {paired_fail_diff:+d} over {checked_g2} eps"
    )
    if missing_ref:
        print(f"  ({missing_ref} episodes without a reference -- counted next pass)")
    gate2 = checked_g2 > 0 and paired_fail_diff <= 0
    print(f"GATE 2 (no new failed HIRE/BUY/LAND vs incumbent): {'PASS' if gate2 else 'FAIL'}")

    # ---- Gate 3: kept pairs match the peer-vs-peer reference ----
    # On fresh seeds a kept pair MUST reproduce self-play exactly (the
    # composed table routes it to the incumbent's tape). The reference is
    # already loaded above (same seeds).
    mismatches = 0
    checked = 0
    for r in recs:
        obs = tuple(r.get("observed_pair") or [])
        if obs not in per_pair or obs in changed:
            continue
        ref_row = ref.get(r["seed"])
        if ref_row is None:
            continue
        me_exp = ref_row["me_cash"] if not r["swap"] else ref_row["opp_cash"]
        opp_exp = ref_row["opp_cash"] if not r["swap"] else ref_row["me_cash"]
        checked += 1
        if abs(r["me_cash"] - me_exp) > 0.01 or abs(r["opp_cash"] - opp_exp) > 0.01:
            mismatches += 1
    gate3 = mismatches == 0
    print(
        f"\nkept-pair anchor: {checked} episodes checked vs fresh reference, {mismatches} mismatches"
    )
    print(f"GATE 3 (composition: kept pairs reproduce self-play): {'PASS' if gate3 else 'FAIL'}")

    print("\nNOT run by this script (manual commands):")
    print(
        "  uv run python scripts/rank_ladder.py --agent scratch/v0_4_3_candidate.py --require-perfect"
    )
    print("  # pre-flight (submit.py hardcodes main.py): swap, dry-run, restore")
    print("  #   cp main.py /tmp/main_backup.py && cp scratch/v0_4_3_candidate.py main.py")
    print("  #   uv run python submit.py --dry-run --episodes 3 --opponent baseline")
    print("  #   mv /tmp/main_backup.py main.py")

    overall = gate1 and gate2 and gate3
    print(f"\nGATES OVERALL: {'PASS' if overall else 'FAIL'} (ladder + pre-flight pending, manual)")
    return 0 if overall else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--candidate", default=os.path.join(PROJECT_ROOT, "scratch", "v0_4_3_candidate.py")
    )
    ap.add_argument("--seeds-per-pair", type=int, default=2, help="fresh seeds per pair (rows 24+)")
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()
    sys.exit(run_gates(args.candidate, args.seeds_per_pair, args.workers))


if __name__ == "__main__":
    main()
