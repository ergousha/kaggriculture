#!/usr/bin/env python3
"""v0.4.3 Step 1: the per-draw measurement panel (docs/proposal-2026-09-16-v043.md).

Measures, for each day-6 shop pair, which of the 42-tape portfolio's unique
windows beats the incumbent's static table choice against the chassis peer
(v0.4.2 self-play) on paired seeds.

DESIGN (all decisions measured or inherited from repo lessons):

  * GROUND TRUTH: day-6 pairs come from the chassis-game enumeration
    (logs/seed_draws_chassis.csv), NOT the pass-pass table -- measured 2026-09-16,
    pass-pass pairs do not transfer to chassis games (0/12 matched). Every
    episode's observed pair is additionally read from the episode itself
    (local_arena records `observed_pair`), and cells aggregate by OBSERVED pair.

  * ENGINE SEAT ASYMMETRY (measured 2026-09-16, the hard way): identical agents
    do NOT tie on every seed -- e.g. peer-vs-peer on seed 2000138 gives seat0
    $83,399 vs seat1 $82,539, deterministically, and the cash follows the SEAT
    regardless of which agent file occupies it. Consequences, all implemented:
      - the anchor check is REFERENCE-MATCHING, not tie-matching: a
        forced-incumbent variant must reproduce the peer-vs-peer reference
        episode exactly, seat for seat (`--stage validate`);
      - every tape's evidence is the PAIRED DIFF vs the incumbent cell on the
        same (seed, seat-assignment) keys -- the asymmetry cancels inside a seed;
      - the incumbent cell is deepened alongside the top-3 in refine (it is the
        counterfactual baseline every candidate is differenced against).

  * PAIRED SEEDS, BOTH SEATS: every (pair, tape) cell runs the same seed list,
    seat alternating by seed index (repo's measured role-asymmetry lesson).

  * STAGES (proposal Step 1's pruning), all resumable (append-only JSONL):
      screen    every unique tape x every pair, seeds[0:4]   (2432 cells)
      refine    top-3 tapes + the incumbent cell per pair, seeds[0:12]
      validate  peer-vs-peer reference on the same seeds + anchor matching
      deep      (a) composition check: the composed table variant must reproduce
                the candidate cell episodes EXACTLY on refine's (seed, swap)
                keys; (b) HELD-OUT read: reference + composed variant on seeds
                [12:24] -- fresh seeds screen/refine never touched. The decision
                gate (>= +$1,500 EV) is applied to THIS number, because the
                in-sample greedy EV is winner's-curse inflated (max-of-4
                selection on the same episodes).
      report    evidence table + greedy table + weighted EV (no episodes)
      table     writes logs/v043_draw_table.csv (no episodes)

Usage:
    uv run python scripts/v043_panel.py --stage screen   --workers 12
    uv run python scripts/v043_panel.py --stage refine   --workers 12
    uv run python scripts/v043_panel.py --stage validate --workers 12
    uv run python scripts/v043_panel.py --stage deep     --workers 12
    uv run python scripts/v043_panel.py --stage report
    uv run python scripts/v043_panel.py --stage table
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
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import v043_common as V  # noqa: E402

PEER = V.INCUMBENT  # v0.4.2 self-play is the field-class proxy
PANEL_DIR = V.PANEL_DIR

SCREEN_SEEDS = 4
REFINE_SEEDS = 12
DEEP_SEEDS = 24  # [0:12] = composition keys, [12:24] = held-out read
# A flip needs paired evidence at refine depth on BOTH the candidate and the
# incumbent cell (the counterfactual). Pairs short of seeds keep the incumbent.
FLIP_MIN_N = 10
# The proposal's decision gate: the composed table must clear +$1,500 mean
# cash per game (frequency-weighted) or the premise dies.
EV_GATE = 1_500.0


# ---------------------------------------------------------------------------
# Episode running
# ---------------------------------------------------------------------------


def _job_to_result(job: dict) -> dict:
    import local_arena as la

    res = la.run_episode(job)
    res["pair"] = job["pair"]
    res["tape_id"] = job["tape_id"]
    return res


def _ref_job_to_result(job: dict) -> dict:
    import local_arena as la

    return la.run_episode(job)


def _dedup_key(rec: dict) -> tuple:
    pair = tuple(rec["pair"]) if rec.get("pair") else None
    return (pair, rec["tape_id"], rec["seed"], rec["swap"])


def run_cells(cells: list[dict], workers: int, stage: str) -> int:
    """cells: [{pair, tape_id, seeds, variant_path, peer}] -> episodes run."""
    jobs = []
    for cell in cells:
        for i, seed in enumerate(cell["seeds"]):
            jobs.append(
                {
                    "agent": cell["variant_path"],
                    "opponent": cell["peer"],
                    "seed": seed,
                    "steps": 720,
                    "swap": bool(i % 2),
                    "decision_log": None,
                    "replay": None,
                    "pair": list(cell["pair"]),
                    "tape_id": cell["tape_id"],
                }
            )
    out_path = os.path.join(PANEL_DIR, f"{stage}.jsonl")
    os.makedirs(PANEL_DIR, exist_ok=True)
    # refine/deep reuse screen's episodes for their overlapping cells (same
    # forced-tape variants, same paired seeds) -- do not re-run them.
    done_files = [out_path]
    if stage in ("refine", "deep"):
        done_files.append(os.path.join(PANEL_DIR, "screen.jsonl"))
    if stage == "deep":
        done_files.append(os.path.join(PANEL_DIR, "refine.jsonl"))
    done: set[tuple] = set()
    for path in done_files:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add(_dedup_key(rec))
                except Exception:
                    continue
    jobs = [j for j in jobs if _dedup_key(j) not in done]
    print(f"[{stage}] {len(cells)} cells, {len(jobs)} episodes to run ({len(done)} on file)")
    if not jobs:
        return 0

    t0 = time.time()
    n = len(jobs)
    wrote = 0
    ctx = mp.get_context("spawn")
    with open(out_path, "a") as sink, ctx.Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_job_to_result, jobs, chunksize=2)):
            sink.write(json.dumps(res) + "\n")
            sink.flush()
            wrote += 1
            if (i + 1) % 100 == 0:
                el = time.time() - t0
                eta = (n - i - 1) * el / (i + 1) / 60
                print(f"  {i + 1}/{n}  ({el / 60:.0f}m elapsed, ~{eta:.0f}m left)", flush=True)
    print(f"  wrote {wrote} episodes to {out_path}")
    return wrote


def run_reference(seeds: list[int], workers: int) -> int:
    """Peer-vs-peer on the given seeds: the seat-mapped self-play baseline the
    anchors are validated against (cash follows the SEAT, not the file)."""
    out_path = os.path.join(PANEL_DIR, "reference.jsonl")
    os.makedirs(PANEL_DIR, exist_ok=True)
    done: set[int] = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["seed"])
                except Exception:
                    continue
    todo = [s for s in seeds if s not in done]
    print(f"[reference] {len(todo)} peer-vs-peer episodes to run ({len(done)} on file)")
    if not todo:
        return 0
    jobs = [
        {
            "agent": PEER,
            "opponent": PEER,
            "seed": s,
            "steps": 720,
            "swap": False,
            "decision_log": None,
            "replay": None,
        }
        for s in todo
    ]
    t0 = time.time()
    n = len(jobs)
    ctx = mp.get_context("spawn")
    with open(out_path, "a") as sink, ctx.Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_ref_job_to_result, jobs, chunksize=2)):
            sink.write(json.dumps(res) + "\n")
            sink.flush()
            if (i + 1) % 100 == 0:
                el = time.time() - t0
                eta = (n - i - 1) * el / (i + 1) / 60
                print(f"  {i + 1}/{n}  ({el / 60:.0f}m elapsed, ~{eta:.0f}m left)", flush=True)
    print(f"  wrote {n} reference episodes to {out_path}")
    return n


def load_stage(stage: str) -> list[dict]:
    path = os.path.join(PANEL_DIR, f"{stage}.jsonl")
    out = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    return out


def load_reference() -> dict[int, dict]:
    path = os.path.join(PANEL_DIR, "reference.jsonl")
    ref: dict[int, dict] = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    ref[rec["seed"]] = rec
                except Exception:
                    continue
    return ref


# ---------------------------------------------------------------------------
# Aggregation (observed-pair attribution, dedup, paired indexes)
# ---------------------------------------------------------------------------


def _row_pair(r: dict) -> tuple | None:
    obs = r.get("observed_pair")
    if obs:
        return tuple(obs)
    if r.get("pair") and tuple(r["pair"]) != ("*", "*"):
        return tuple(r["pair"])
    return None


def aggregate(recs: list[dict]) -> dict[tuple, dict]:
    """{(observed_pair, tape_id): stats} with (seed, swap) dedup."""
    cells: dict[tuple, list[dict]] = defaultdict(list)
    seen: set[tuple] = set()
    for r in recs:
        pair = _row_pair(r)
        if pair is None:
            continue
        key = (pair, r["tape_id"], r["seed"], r["swap"])
        if key in seen:
            continue
        seen.add(key)
        cells[(pair, r["tape_id"])].append(r)
    agg = {}
    for key, rows in cells.items():
        n = len(rows)
        deltas = [r["me_cash"] - r["opp_cash"] for r in rows]
        agg[key] = {
            "n": n,
            "mean_delta": statistics.fmean(deltas) if deltas else 0.0,
            "wins": sum(r["win"] for r in rows),
            "ties": sum(r["tie"] for r in rows),
            "losses": n - sum(r["win"] for r in rows) - sum(r["tie"] for r in rows),
            "anomalies": sum(
                1
                for r in rows
                if r["crashes"] or r["timeouts"] or r["invalid"] or r["harness_error"]
            ),
        }
    return agg


def cells_index(recs: list[dict]) -> dict[tuple, dict[tuple, dict]]:
    """{(observed_pair, tape_id): {(seed, swap): row}} with dedup -- the paired
    index every difference is computed from."""
    idx: dict[tuple, dict[tuple, dict]] = defaultdict(dict)
    seen: set[tuple] = set()
    for r in recs:
        pair = _row_pair(r)
        if pair is None:
            continue
        key = (pair, r["tape_id"], r["seed"], r["swap"])
        if key in seen:
            continue
        seen.add(key)
        idx[(pair, r["tape_id"])][(r["seed"], r["swap"])] = r
    return idx


def incumbent_choice(pair: tuple[str, str], inc_table, yarn_table) -> int:
    if pair.count("YARN_STORE") <= 0:
        return inc_table.get(pair, 100)
    return yarn_table.get(pair, 0)


def rep_of(rid: int, uniq: dict[int, list[int]]) -> int:
    for rep, members in uniq.items():
        if rid in members:
            return rep
    return rid


def paired_diffs(pair: tuple, idx: dict, uniq, inc_table, yarn_table) -> dict[int, list[float]]:
    """{tape_id: [candidate_delta - incumbent_delta, ...]} over the (seed, swap)
    keys common to the candidate cell and the incumbent cell. The engine's seat
    asymmetry is identical on both sides of each difference and cancels."""
    inc_rep = rep_of(incumbent_choice(pair, inc_table, yarn_table), uniq)
    inc_rows = idx.get((pair, inc_rep))
    if not inc_rows:
        return {}
    out: dict[int, list[float]] = {}
    for (p, tape), rows in idx.items():
        if p != pair or tape == inc_rep:
            continue
        diffs = []
        for key, cand_row in rows.items():
            inc_row = inc_rows.get(key)
            if inc_row is None:
                continue
            diffs.append(
                (cand_row["me_cash"] - cand_row["opp_cash"])
                - (inc_row["me_cash"] - inc_row["opp_cash"])
            )
        if diffs:
            out[tape] = diffs
    return out


# ---------------------------------------------------------------------------
# Anchor validation (reference matching)
# ---------------------------------------------------------------------------


def check_anchors(recs: list[dict], uniq, inc_table, yarn_table, ref: dict[int, dict]) -> list[str]:
    """Every forced-incumbent episode must reproduce the peer-vs-peer reference
    EXACTLY, seat for seat (swap maps which reference seat the variant held).
    Any mismatch means the variant generation or harness is broken."""
    problems = []
    seen: set[tuple] = set()
    for r in recs:
        pair = _row_pair(r)
        if pair is None:
            continue
        if r["tape_id"] != rep_of(incumbent_choice(pair, inc_table, yarn_table), uniq):
            continue
        key = (pair, r["tape_id"], r["seed"], r["swap"])
        if key in seen:
            continue
        seen.add(key)
        ref_row = ref.get(r["seed"])
        if ref_row is None:
            problems.append(f"no reference episode for seed {r['seed']}")
            continue
        me_expected = ref_row["me_cash"] if not r["swap"] else ref_row["opp_cash"]
        opp_expected = ref_row["opp_cash"] if not r["swap"] else ref_row["me_cash"]
        if abs(r["me_cash"] - me_expected) > 0.01 or abs(r["opp_cash"] - opp_expected) > 0.01:
            problems.append(
                f"ANCHOR MISMATCH seed={r['seed']} swap={r['swap']} pair={pair} "
                f"tape={r['tape_id']}: got ({r['me_cash']:.0f},{r['opp_cash']:.0f}) "
                f"expected ({me_expected:.0f},{opp_expected:.0f})"
            )
        ref_obs = tuple(ref_row.get("observed_pair") or ())
        if ref_obs and tuple(r.get("observed_pair") or ()) != ref_obs:
            problems.append(
                f"OBSERVED PAIR MISMATCH seed={r['seed']}: anchor {r.get('observed_pair')} "
                f"vs reference {ref_obs}"
            )
    return problems


# ---------------------------------------------------------------------------
# Greedy table + EV
# ---------------------------------------------------------------------------


def pair_frequencies() -> dict[tuple[str, str], float]:
    draws = V.load_chassis_draws()
    total = max(1, len(draws))
    freq: dict[tuple[str, str], int] = defaultdict(int)
    for pair in draws.values():
        freq[tuple(pair)] += 1
    return {p: c / total for p, c in freq.items()}


def greedy_table(recs, uniq, inc_table, yarn_table, min_n: int = FLIP_MIN_N):
    """Greedy per-pair argmax over PAIRED evidence: a pair keeps the incumbent's
    assignment unless another tape's mean paired diff (candidate minus
    incumbent cell, same seeds/seats) is strictly positive on >= min_n common
    episodes. Returns (table, flips, flip_evidence, weighted_ev)."""
    idx = cells_index(recs)
    freq = pair_frequencies()
    pairs = sorted({k[0] for k in idx})
    table: dict[tuple[str, str], int] = {}
    flip_evidence: dict[tuple, dict] = {}
    flips = []
    for pair in pairs:
        incumbent_id = incumbent_choice(pair, inc_table, yarn_table)
        diffs = paired_diffs(pair, idx, uniq, inc_table, yarn_table)
        best_id, best_mean = incumbent_id, 0.0
        for tape, ds in diffs.items():
            if len(ds) < min_n:
                continue
            mean = statistics.fmean(ds)
            if mean > best_mean + 1e-9:
                best_id, best_mean = tape, mean
        if best_id != incumbent_id:
            ds = diffs[best_id]
            flips.append((pair, incumbent_id, best_id))
            flip_evidence[pair] = {
                "mean_diff": best_mean,
                "n": len(ds),
                "wins": sum(1 for d in ds if d > 0),
                "losses": sum(1 for d in ds if d < 0),
            }
        table[pair] = best_id
    for pair, rid in {**inc_table, **yarn_table}.items():
        table.setdefault(pair, rid)
    ev = sum(freq.get(pair, 0.0) * evd["mean_diff"] for pair, evd in flip_evidence.items())
    return table, flips, flip_evidence, ev


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def build_cells(stage, by_pair, uniq, inc_table, yarn_table, table=None) -> list[dict]:
    cells = []
    if stage == "screen":
        for pair in sorted(by_pair):
            seeds = by_pair[pair][:SCREEN_SEEDS]
            for rep in sorted(uniq):
                cells.append(
                    {
                        "pair": pair,
                        "tape_id": rep,
                        "seeds": seeds,
                        "variant_path": V.make_forced_variant(rep),
                        "peer": PEER,
                    }
                )
    elif stage == "refine":
        agg = aggregate(load_stage("screen"))
        for pair in sorted(by_pair):
            scored = []
            for rep in uniq:
                st = agg.get((pair, rep))
                if st:
                    scored.append((st["mean_delta"], rep))
            scored.sort(reverse=True)
            chosen = [rep for _, rep in scored[:3]]
            inc_rep = rep_of(incumbent_choice(pair, inc_table, yarn_table), uniq)
            if inc_rep not in chosen:
                chosen.append(inc_rep)  # the counterfactual baseline must reach depth
            for rep in chosen:
                cells.append(
                    {
                        "pair": pair,
                        "tape_id": rep,
                        "seeds": by_pair[pair][:REFINE_SEEDS],
                        "variant_path": V.make_forced_variant(rep),
                        "peer": PEER,
                    }
                )
    elif stage == "deep":
        out = V.make_table_variant(table, os.path.join(PANEL_DIR, "v043_table_candidate.py"))
        seeds = []
        for pair in sorted(set(table) | set(by_pair)):
            seeds.extend(by_pair.get(pair, [])[:DEEP_SEEDS])
        cells = [
            {
                "pair": ("*", "*"),
                "tape_id": -1,
                "seeds": sorted(set(seeds)),
                "variant_path": out,
                "peer": PEER,
            }
        ]
    return cells


def report(uniq, inc_table, yarn_table, ref) -> None:
    recs = load_stage("screen") + load_stage("refine")
    problems = check_anchors(recs, uniq, inc_table, yarn_table, ref)
    print(f"panel records: {len(recs)}  (reference episodes: {len(ref)})")
    if problems:
        print(f"\n!! {len(problems)} ANCHOR MISMATCHES -- panel is VOID:")
        for p in problems[:20]:
            print(f"  {p}")
        return
    print("anchors: all forced-incumbent episodes match the peer-vs-peer reference exactly")
    table, flips, evd, ev = greedy_table(recs, uniq, inc_table, yarn_table)
    print(f"\n{'pair':<40} {'inc':>4} {'best':>5} {'Δμ$':>9} {'n':>3} {'+/-':>7}")
    for pair in sorted({k[0] for k in cells_index(recs)}):
        incumbent = incumbent_choice(pair, inc_table, yarn_table)
        e = evd.get(pair)
        if e and table[pair] != incumbent:
            print(
                f"{pair[0]:<20} {pair[1]:<20} {incumbent:>4} {table[pair]:>5} "
                f"{e['mean_diff']:>+9,.0f} {e['n']:>3} {e['wins']}/{e['losses']}  <-- CHANGE"
            )
    print(f"\n{len(flips)} of 64 pairs would change tape (paired, n>={FLIP_MIN_N}, mean diff > 0)")
    print(
        f"frequency-weighted EV of the greedy table vs incumbent: {ev:+,.0f}$ "
        f"mean cash per game (gate bar: >= +{EV_GATE:,.0f}$)"
    )


def write_draw_table(uniq, inc_table, yarn_table, ref) -> None:
    recs = load_stage("screen") + load_stage("refine")
    problems = check_anchors(recs, uniq, inc_table, yarn_table, ref)
    if problems:
        print("\n".join(problems))
        sys.exit("panel is void -- no table written")
    table, flips, evd, ev = greedy_table(recs, uniq, inc_table, yarn_table)
    flip_pairs = {f[0] for f in flips}
    with open(V.DRAW_TABLE_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "shop1",
                "shop2",
                "route",
                "incumbent_route",
                "changed",
                "paired_mean_delta",
                "n_paired",
                "paired_wins",
                "paired_losses",
            ]
        )
        for pair in sorted(table):
            rid = table[pair]
            incumbent = incumbent_choice(pair, inc_table, yarn_table)
            e = evd.get(pair, {"mean_diff": 0.0, "n": 0, "wins": 0, "losses": 0})
            w.writerow(
                [
                    pair[0],
                    pair[1],
                    rid,
                    incumbent,
                    int(pair in flip_pairs),
                    round(e["mean_diff"], 1),
                    e["n"],
                    e["wins"],
                    e["losses"],
                ]
            )
    print(f"wrote {V.DRAW_TABLE_CSV}  ({len(flip_pairs)} changed assignments)")
    print(
        f"frequency-weighted EV vs incumbent: {ev:+,.0f}$ mean cash per game "
        f"(gate bar: >= +{EV_GATE:,.0f}$)"
    )


def deep_summary(uniq, inc_table, yarn_table, table, ref) -> None:
    """Deep = (a) composition validation on refine's keys, (b) HELD-OUT read.

    (a) The composed table variant must reproduce, EXACTLY, the episodes its
        chosen tapes already produced on refine's (seed, swap) keys -- kept
        pairs match the peer-vs-peer reference, changed pairs match their
        candidate cells. This proves the table composes without interaction.
    (b) Held-out: reference + composed variant on seeds[12:24] per pair, fresh
        vs everything screen/refine measured. The decision gate (EV >= +$1,500)
        is applied to the held-out paired evidence -- in-sample EV is
        winner's-curse inflated and cannot gate anything.
    """
    deep = load_stage("deep")
    panel = load_stage("screen") + load_stage("refine")
    idx = cells_index(panel)
    by_pair = V.seeds_by_pair(V.load_chassis_draws(), min_per_pair=24)
    problems = []

    # ---- (a) composition validation on refine keys ----
    kept_mismatch = 0
    changed_match_fail = 0
    checked = 0
    for r in deep:
        pair = _row_pair(r)
        if pair is None:
            continue
        incumbent = incumbent_choice(pair, inc_table, yarn_table)
        sel = table.get(pair)
        if sel is None:
            continue
        if sel == incumbent:
            ref_row = ref.get(r["seed"])
            if ref_row is None:
                continue
            me_exp = ref_row["me_cash"] if not r["swap"] else ref_row["opp_cash"]
            opp_exp = ref_row["opp_cash"] if not r["swap"] else ref_row["me_cash"]
            if abs(r["me_cash"] - me_exp) > 0.01 or abs(r["opp_cash"] - opp_exp) > 0.01:
                kept_mismatch += 1
            checked += 1
        else:
            cand = idx.get((pair, sel), {}).get((r["seed"], r["swap"]))
            inc_rep = rep_of(incumbent, uniq)
            inc_row = idx.get((pair, inc_rep), {}).get((r["seed"], r["swap"]))
            if cand is None or inc_row is None:
                continue
            if (
                abs(r["me_cash"] - cand["me_cash"]) > 0.01
                or abs(r["opp_cash"] - cand["opp_cash"]) > 0.01
            ):
                changed_match_fail += 1
            checked += 1
    if kept_mismatch:
        problems.append(f"{kept_mismatch} kept-pair episodes diverge from the self-play reference")
    if changed_match_fail:
        problems.append(
            f"{changed_match_fail} changed-pair episodes diverge from their candidate cells"
        )
    for p in problems:
        print(f"!! {p}")
    if problems:
        sys.exit("deep composition validation FAILED")
    print(
        f"deep composition validation: {checked} episodes reproduce reference/candidate cells exactly"
    )

    # ---- (b) held-out read ----
    # the deep stage ran the composed variant on seeds[0:24]; held-out keys are
    # those in seeds[12:24] (screen used [0:4], refine [0:12]). The counterfactual
    # is the REFERENCE episode on the same seed (the peer IS the incumbent
    # table), seat-mapped -- no extra incumbent-cell runs needed.
    freq = pair_frequencies()
    per_pair: dict[tuple, list[float]] = defaultdict(list)
    for r in deep:
        pair = _row_pair(r)
        if pair is None:
            continue
        seeds = by_pair.get(pair, [])
        # held-out window: element [12] up to the LAST seed (the table may hold
        # exactly 24 per pair, so index 24 does not exist; the window is open)
        if len(seeds) <= 12 or not (seeds[12] <= r["seed"] <= seeds[-1]):
            continue
        incumbent = incumbent_choice(pair, inc_table, yarn_table)
        sel = table.get(pair)
        if sel is None or sel == incumbent:
            continue
        ref_row = ref.get(r["seed"])
        if ref_row is None:
            continue
        inc_me = ref_row["opp_cash"] if r["swap"] else ref_row["me_cash"]
        inc_opp = ref_row["me_cash"] if r["swap"] else ref_row["opp_cash"]
        per_pair[pair].append((r["me_cash"] - r["opp_cash"]) - (inc_me - inc_opp))
    if not per_pair:
        print("\nno held-out evidence yet (need deep episodes on seeds[12:24])")
        return
    weighted_sum = 0.0
    total_pairs = 0
    print(f"\n{'pair':<40} {'new':>4} {'Δμ$':>9} {'+/-':>7}")
    for pair in sorted(per_pair):
        ds = per_pair[pair]
        m = statistics.fmean(ds)
        w = sum(1 for d in ds if d > 0)
        losses = sum(1 for d in ds if d < 0)
        weighted_sum += freq.get(pair, 0.0) * m
        total_pairs += 1
        print(f"{pair[0]:<20} {pair[1]:<20} {len(ds):>4} {m:>+9,.0f} {w}/{losses}")
    print(
        f"\nHELD-OUT: {total_pairs} changed pairs, frequency-weighted EV "
        f"{weighted_sum:+,.0f}$ mean cash per game (gate bar: >= +{EV_GATE:,.0f}$)"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage",
        required=True,
        choices=["screen", "refine", "validate", "deep", "report", "table"],
    )
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--min-per-pair", type=int, default=12)
    args = ap.parse_args()

    inc_table, yarn_table = V.extract_tables_from_incumbent()
    tapes = V.load_tapes()
    uniq = V.unique_tapes(tapes)
    print(
        f"portfolio: {len(tapes)} route ids, {len(uniq)} unique windows "
        f"(measured window steps {V.WINDOW[0]}..{V.WINDOW[1] - 1})"
    )

    draws = V.load_chassis_draws()
    by_pair = V.seeds_by_pair(draws, min_per_pair=args.min_per_pair)
    print(
        f"draw table: {len(draws)} chassis seeds, "
        f"{len(by_pair)} pairs with >= {args.min_per_pair} seeds"
    )

    if args.stage == "report":
        report(uniq, inc_table, yarn_table, load_reference())
        return
    if args.stage == "table":
        write_draw_table(uniq, inc_table, yarn_table, load_reference())
        return

    if args.stage == "validate":
        seeds = []
        for pair in by_pair:
            seeds.extend(by_pair[pair][:DEEP_SEEDS])
        run_reference(sorted(set(seeds)), args.workers)
        ref = load_reference()
        recs = load_stage("screen") + load_stage("refine")
        problems = check_anchors(recs, uniq, inc_table, yarn_table, ref)
        if problems:
            print(f"\n!! {len(problems)} ANCHOR MISMATCHES:")
            for p in problems[:20]:
                print(f"  {p}")
            sys.exit(1)
        print(
            f"validate: {len(recs)} panel records checked against "
            f"{len(ref)} reference episodes -- all anchors match exactly"
        )
        return

    if args.stage == "deep":
        recs = load_stage("screen") + load_stage("refine")
        table, _flips, _evd, _ev = greedy_table(recs, uniq, inc_table, yarn_table)
        cells = build_cells("deep", by_pair, uniq, inc_table, yarn_table, table=table)
    else:
        cells = build_cells(args.stage, by_pair, uniq, inc_table, yarn_table)
    if not cells:
        print("no cells to run")
        return
    run_cells(cells, args.workers, args.stage)
    if args.stage in ("screen", "refine"):
        report(uniq, inc_table, yarn_table, load_reference())
    else:
        deep_summary(uniq, inc_table, yarn_table, table, load_reference())


if __name__ == "__main__":
    main()
