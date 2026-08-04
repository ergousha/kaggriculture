#!/usr/bin/env python3
"""Local match runner, metrics harness and A/B rig for the Kaggriculture agent.

    # basic match set
    python local_arena.py --agent main.py --opponent baseline --episodes 20 --seed 42

    # head to head between two files, with per-turn decision logs
    python local_arena.py --agent main.py --opponent opponents/adaptive.py \
        --episodes 50 --log-decisions

    # paired-seed A/B of one strategy flag (this is how a PLACEHOLDER graduates)
    python local_arena.py --agent main.py --opponent baseline --episodes 30 \
        --ablate EXPAND_LAND

    # sweep a numeric constant over candidate values, paired seeds
    python local_arena.py --agent main.py --opponent baseline --episodes 30 \
        --sweep MAX_HANDS=8,16,24

    # inspect a saved replay
    python local_arena.py --replay logs/match_0007.json

Opponents may be `baseline` (env built-in `starter`), `adaptive`, `random`,
`pass`, `mirror` (self-play against a frozen snapshot of --agent), or any path
to a .py file.

Determinism: every episode is run with an explicit `seed` in the environment
configuration, so `--seed S --episodes N` always reproduces the same N matches.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import shutil
import statistics
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")

BUILTIN = {"baseline": "starter", "random": "random", "pass": "pass"}
DEFAULT_STEPS = 720


# ---------------------------------------------------------------------------
# Agent-variant generation (for --ablate / --sweep)
#
# Variants are produced by rewriting a COPY of the agent file. This keeps all
# tuning scaffolding out of main.py, which has to stay submission-clean.
# ---------------------------------------------------------------------------


def make_variant(src_path: str, dest_path: str, flags: dict | None = None, consts: dict | None = None) -> str:
    with open(src_path) as f:
        src = f.read()
    for name, value in (flags or {}).items():
        pattern = rf'("{re.escape(name)}"\s*:\s*)(True|False)'
        new, n = re.subn(pattern, lambda m: m.group(1) + repr(bool(value)), src)
        if n == 0:
            raise SystemExit(f"--ablate: flag {name!r} not found in {src_path}")
        src = new
    for name, value in (consts or {}).items():
        pattern = rf'(?m)^({re.escape(name)}\s*=\s*)([^\n#]+)'
        new, n = re.subn(pattern, lambda m: m.group(1) + repr(value), src)
        if n == 0:
            raise SystemExit(f"--sweep: constant {name!r} not found in {src_path}")
        src = new
    with open(dest_path, "w") as f:
        f.write(src)
    return dest_path


def resolve_opponent(name: str, agent_path: str, workdir: str) -> str:
    if name in BUILTIN:
        return BUILTIN[name]
    if name == "adaptive":
        p = os.path.join(HERE, "opponents", "adaptive.py")
        if not os.path.exists(p):
            raise SystemExit(f"adaptive opponent missing at {p}")
        return p
    if name == "mirror":
        snap = os.path.join(workdir, "mirror_snapshot.py")
        shutil.copyfile(agent_path, snap)
        return snap
    if os.path.exists(name):
        return name
    raise SystemExit(f"unknown opponent {name!r} (not a builtin and not a file)")


# ---------------------------------------------------------------------------
# Instrumentation: shed overflow + invalid unit actions
#
# Both are invisible in the observation, so we wrap the interpreter's own
# helpers inside the worker process and count what they discard/ignore.
# ---------------------------------------------------------------------------

COUNTERS: dict[str, int] = {}


def _install_instrumentation(me_index: int = 0):
    """Count what the observation cannot show us: shed overflow and no-op actions.

    Both counters must be attributed to ONE player or they are meaningless. The
    interpreter processes players in index order every turn -- `interpreter()`
    loops `for i, s in enumerate(state)` for unit actions, and `_end_of_day()`
    loops `for player_id, farm in enumerate(obs0.farms)` for the shed drop -- so
    the first distinct farm/private object seen each turn belongs to player 0.
    We map object identity to player index on that basis.
    """
    from kaggle_environments.envs.kaggriculture import kaggriculture as K

    if getattr(K, "_arena_instrumented", False):
        K._arena_me_index = me_index
        return K
    K._arena_me_index = me_index
    orig_drop = K._drop_inventories_to_shed
    orig_apply = K._apply_unit_action

    seen_order: dict[int, int] = {}

    def player_of(obj) -> int:
        key = id(obj)
        if key not in seen_order:
            seen_order[key] = len(seen_order) % 2
        return seen_order[key]

    def drop(private, capacity):
        if player_of(private) != K._arena_me_index:
            return orig_drop(private, capacity)
        before_inv = sum(
            sum(v for v in inv.values() if isinstance(v, int))
            for inv in private["inventories"]
        )
        before_shed = sum(private["shed"].values())
        orig_drop(private, capacity)
        after_shed = sum(private["shed"].values())
        moved = after_shed - before_shed
        lost = max(0, before_inv - moved)
        COUNTERS["shed_overflow_lost"] = COUNTERS.get("shed_overflow_lost", 0) + lost
        return None

    # Ops whose effect we can cheaply verify by snapshotting the tile + inventory.
    CHECKED = {
        "PLANT", "WATER", "HARVEST", "FERTILIZE", "BUILD_COOP", "BUILD_PASTURE",
        "DIG", "PLACE", "FEED", "COLLECT_FERTILIZER", "CARE", "PICKUP", "DROP",
    }

    def apply(farm, private, idx, action, board_size, day, turns_per_day, shed_capacity=100):
        if player_of(farm) != K._arena_me_index:
            return orig_apply(farm, private, idx, action, board_size, day, turns_per_day, shed_capacity)
        op = action[0] if isinstance(action, list) and action else None
        if op not in CHECKED:
            COUNTERS["actions_total"] = COUNTERS.get("actions_total", 0) + 1
            if op == "PASS":
                COUNTERS["actions_pass"] = COUNTERS.get("actions_pass", 0) + 1
            return orig_apply(farm, private, idx, action, board_size, day, turns_per_day, shed_capacity)
        pos = K._farmer_position(farm, idx)
        snap_tile = repr(farm["tiles"][pos[1]][pos[0]]) if pos else None
        snap_inv = repr(sorted(K._farmer_inventory(private, idx).items())) if pos else None
        snap_shed = repr(sorted(private["shed"].items()))
        res = orig_apply(farm, private, idx, action, board_size, day, turns_per_day, shed_capacity)
        COUNTERS["actions_total"] = COUNTERS.get("actions_total", 0) + 1
        if pos:
            same = (
                repr(farm["tiles"][pos[1]][pos[0]]) == snap_tile
                and repr(sorted(K._farmer_inventory(private, idx).items())) == snap_inv
                and repr(sorted(private["shed"].items())) == snap_shed
            )
            if same:
                COUNTERS["actions_noop"] = COUNTERS.get("actions_noop", 0) + 1
                COUNTERS[f"noop_{op}"] = COUNTERS.get(f"noop_{op}", 0) + 1
        return res

    K._drop_inventories_to_shed = drop
    K._apply_unit_action = apply
    # The interpreter captured references at def time only for these two names,
    # both of which it looks up on the module at call time, so patching sticks.
    K._arena_instrumented = True
    return K


# ---------------------------------------------------------------------------
# One episode
# ---------------------------------------------------------------------------


def run_episode(job: dict) -> dict:
    agent_path = job["agent"]
    opp = job["opponent"]
    seed = job["seed"]
    steps = job["steps"]
    swap = job["swap"]
    decision_log = job.get("decision_log")
    replay_path = job.get("replay")

    COUNTERS.clear()
    if decision_log:
        os.environ["KAGGRICULTURE_DECISION_LOG"] = decision_log
    else:
        os.environ.pop("KAGGRICULTURE_DECISION_LOG", None)

    _install_instrumentation(me_index=1 if swap else 0)
    from kaggle_environments import make

    env = make(
        "kaggriculture",
        configuration={"episodeSteps": steps, "seed": seed},
        debug=False,
    )
    players = [agent_path, opp]
    me_index = 0
    if swap:
        players = [opp, agent_path]
        me_index = 1

    t0 = time.perf_counter()
    err = None
    try:
        env.run(players)
    except Exception as exc:  # a harness-level failure, not an agent exception
        err = f"{type(exc).__name__}: {exc}"
    wall = time.perf_counter() - t0

    final = env.steps[-1]
    rewards = [s.reward for s in final]
    me = rewards[me_index] if rewards[me_index] is not None else 0.0
    them = rewards[1 - me_index] if rewards[1 - me_index] is not None else 0.0

    # Statuses across the whole episode: TIMEOUT / ERROR / INVALID are fatal.
    crashes = timeouts = invalid = 0
    for step_states in env.steps:
        st = step_states[me_index].status
        if st == "TIMEOUT":
            timeouts += 1
        elif st == "ERROR":
            crashes += 1
        elif st == "INVALID":
            invalid += 1

    durations = []
    stderr_lines = []
    for step_logs in env.logs or []:
        if me_index < len(step_logs):
            entry = step_logs[me_index] or {}
            d = entry.get("duration")
            if d is not None:
                durations.append(float(d))
            e = (entry.get("stderr") or "").strip()
            if e:
                stderr_lines.append(e)

    if replay_path:
        with open(replay_path, "w") as f:
            json.dump(env.toJSON(), f)

    durations.sort()

    def pct(p):
        if not durations:
            return 0.0
        k = min(len(durations) - 1, int(round((len(durations) - 1) * p)))
        return durations[k]

    return {
        "seed": seed,
        "swap": swap,
        "me_cash": float(me),
        "opp_cash": float(them),
        "win": 1 if me > them else 0,
        "tie": 1 if me == them else 0,
        "crashes": crashes,
        "timeouts": timeouts,
        "invalid": invalid,
        "harness_error": err,
        "wall_seconds": round(wall, 2),
        "turn_p50": round(pct(0.50), 5),
        "turn_p95": round(pct(0.95), 5),
        "turn_max": round(max(durations), 5) if durations else 0.0,
        "shed_overflow_lost": COUNTERS.get("shed_overflow_lost", 0),
        "actions_total": COUNTERS.get("actions_total", 0),
        "actions_noop": COUNTERS.get("actions_noop", 0),
        "actions_pass": COUNTERS.get("actions_pass", 0),
        "noop_breakdown": {k[5:]: v for k, v in COUNTERS.items() if k.startswith("noop_")},
        "stderr_sample": stderr_lines[:3],
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(results: list[dict], decision_logs: list[str] | None = None) -> dict:
    cash = [r["me_cash"] for r in results]
    opp = [r["opp_cash"] for r in results]
    n = len(results)
    agg = {
        "episodes": n,
        "mean_cash": round(statistics.fmean(cash), 1) if cash else 0.0,
        "median_cash": round(statistics.median(cash), 1) if cash else 0.0,
        "min_cash": round(min(cash), 1) if cash else 0.0,
        "max_cash": round(max(cash), 1) if cash else 0.0,
        "stdev_cash": round(statistics.stdev(cash), 1) if n > 1 else 0.0,
        "opp_mean_cash": round(statistics.fmean(opp), 1) if opp else 0.0,
        "wins": sum(r["win"] for r in results),
        "ties": sum(r["tie"] for r in results),
        "win_rate": round(sum(r["win"] for r in results) / n, 4) if n else 0.0,
        "crashes": sum(r["crashes"] for r in results),
        "timeouts": sum(r["timeouts"] for r in results),
        "invalid": sum(r["invalid"] for r in results),
        "harness_errors": [r["harness_error"] for r in results if r["harness_error"]],
        "turn_p50": round(statistics.fmean([r["turn_p50"] for r in results]), 5) if n else 0.0,
        "turn_p95": round(max(r["turn_p95"] for r in results), 5) if n else 0.0,
        "turn_max": round(max(r["turn_max"] for r in results), 5) if n else 0.0,
        "shed_overflow_lost": sum(r["shed_overflow_lost"] for r in results),
        "actions_total": sum(r["actions_total"] for r in results),
        "actions_noop": sum(r["actions_noop"] for r in results),
        "actions_pass": sum(r["actions_pass"] for r in results),
        "wall_seconds": round(sum(r["wall_seconds"] for r in results), 1),
    }
    if agg["actions_total"]:
        agg["noop_rate"] = round(agg["actions_noop"] / agg["actions_total"], 4)
        agg["pass_rate"] = round(agg["actions_pass"] / agg["actions_total"], 4)
    noop_all: dict[str, int] = {}
    for r in results:
        for k, v in (r.get("noop_breakdown") or {}).items():
            noop_all[k] = noop_all.get(k, 0) + v
    agg["noop_breakdown"] = dict(sorted(noop_all.items(), key=lambda kv: -kv[1])[:8])
    if decision_logs:
        agg.update(summarize_decision_logs(decision_logs))
    return agg


def summarize_decision_logs(paths: list[str]) -> dict:
    """Worker-idle rate and heuristic firing counts, straight from the agent's
    own per-turn decision records."""
    idle = units = turns = slow = exc = dropped = 0
    fired: dict[str, int] = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                turns += 1
                idle += int(rec.get("idle_units", 0) or 0)
                units += int(rec.get("n_units", 0) or 0)
                if rec.get("slow_turn"):
                    slow += 1
                if rec.get("exception"):
                    exc += 1
                dropped += int(rec.get("orders_dropped", 0) or 0)
                for key in ("buy_land", "hire", "buy_wheat", "buy_animal", "predatory"):
                    if rec.get(key):
                        fired[key] = fired.get(key, 0) + 1
    out = {
        "log_turns": turns,
        "unit_turns": units,
        "idle_unit_turns": idle,
        "slow_turns": slow,
        "logged_exceptions": exc,
        "market_orders_dropped": dropped,
        "heuristics_fired": fired,
    }
    if units:
        out["idle_rate"] = round(idle / units, 4)
    return out


def print_report(title: str, agg: dict) -> None:
    print(f"\n=== {title} ===")
    print(f"  episodes         {agg['episodes']}   wall {agg['wall_seconds']}s")
    print(f"  final cash       mean {agg['mean_cash']:>12,.0f}   median {agg['median_cash']:>12,.0f}")
    print(f"                   min  {agg['min_cash']:>12,.0f}   max    {agg['max_cash']:>12,.0f}   sd {agg['stdev_cash']:,.0f}")
    print(f"  opponent cash    mean {agg['opp_mean_cash']:>12,.0f}")
    print(f"  win rate         {agg['win_rate']:.1%}  ({agg['wins']}W {agg['ties']}T {agg['episodes'] - agg['wins'] - agg['ties']}L)")
    print(f"  crashes {agg['crashes']}   timeouts {agg['timeouts']}   invalid {agg['invalid']}")
    print(f"  turn compute     p50 {agg['turn_p50'] * 1000:.2f}ms  p95 {agg['turn_p95'] * 1000:.2f}ms  max {agg['turn_max'] * 1000:.2f}ms")
    if "noop_rate" in agg:
        print(f"  action no-ops    {agg['actions_noop']:,}/{agg['actions_total']:,} ({agg['noop_rate']:.2%})   PASS {agg['pass_rate']:.2%}")
    if agg.get("noop_breakdown"):
        print(f"  no-op ops        {agg['noop_breakdown']}")
    print(f"  shed overflow    {agg['shed_overflow_lost']:,} items lost")
    if "idle_rate" in agg:
        print(f"  worker idle rate {agg['idle_rate']:.2%}  ({agg['idle_unit_turns']:,}/{agg['unit_turns']:,} unit-turns)")
    if agg.get("market_orders_dropped"):
        print(f"  market orders dropped (>10/turn): {agg['market_orders_dropped']}")
    if agg.get("heuristics_fired"):
        print(f"  heuristics fired {agg['heuristics_fired']}")
    if agg.get("slow_turns"):
        print(f"  slow turns (>70% budget): {agg['slow_turns']}")
    if agg.get("logged_exceptions"):
        print(f"  !! agent exceptions logged: {agg['logged_exceptions']}")
    if agg.get("harness_errors"):
        print(f"  !! harness errors: {agg['harness_errors'][:2]}")


def paired_significance(a: list[float], b: list[float]) -> dict:
    """Paired t-test on the per-seed differences, plus a sign-test count.

    Paired (same seeds, same opponent) so the seed-to-seed variance that
    dominates this environment cancels out.
    """
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    if n < 2:
        return {"n": n, "mean_diff": diffs[0] if diffs else 0.0, "t": 0.0, "p_approx": 1.0, "a_better": 0}
    mean = statistics.fmean(diffs)
    sd = statistics.stdev(diffs)
    se = sd / (n ** 0.5) if sd > 0 else 0.0
    t = mean / se if se > 0 else (float("inf") if mean else 0.0)
    # Two-sided normal approximation; n>=30 paired seeds makes this adequate.
    p = 2.0 * (1.0 - _norm_cdf(abs(t))) if se > 0 else (0.0 if mean else 1.0)
    return {
        "n": n,
        "mean_diff": round(mean, 1),
        "sd_diff": round(sd, 1),
        "t": round(t, 3) if t not in (float("inf"),) else "inf",
        "p_approx": round(p, 5),
        "a_better": sum(1 for d in diffs if d > 0),
    }


def _norm_cdf(x: float) -> float:
    import math

    return 0.5 * (1.0 + math.erf(x / (2 ** 0.5)))


# ---------------------------------------------------------------------------
# Match-set driver
# ---------------------------------------------------------------------------


def run_set(agent_path, opp, seeds, steps, workers, log_decisions, tag, save_replays=0):
    jobs = []
    logs = []
    for i, seed in enumerate(seeds):
        dl = None
        if log_decisions:
            dl = os.path.join(LOG_DIR, f"decisions_{tag}_{seed}.jsonl")
            if os.path.exists(dl):
                os.remove(dl)
            logs.append(dl)
        replay = None
        if i < save_replays:
            replay = os.path.join(LOG_DIR, f"match_{tag}_{seed:04d}.json")
        jobs.append(
            {
                "agent": agent_path,
                "opponent": opp,
                "seed": seed,
                "steps": steps,
                # Alternate seats so neither player-index advantage nor the
                # market's player-order tie-breaking biases the result.
                "swap": bool(i % 2),
                "decision_log": dl,
                "replay": replay,
            }
        )
    if workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            results = pool.map(run_episode, jobs)
    else:
        results = [run_episode(j) for j in jobs]
    return results, logs


def main(argv=None):
    ap = argparse.ArgumentParser(description="Kaggriculture local arena")
    ap.add_argument("--agent", default="main.py")
    ap.add_argument("--opponent", default="baseline")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--log-decisions", action="store_true")
    ap.add_argument("--save-replays", type=int, default=0, metavar="N",
                    help="write replay JSON for the first N episodes")
    ap.add_argument("--ablate", metavar="FLAG", help="paired A/B of one FLAGS entry, ON vs OFF")
    ap.add_argument("--sweep", metavar="NAME=v1,v2,...", help="paired sweep of a numeric constant")
    ap.add_argument("--replay", metavar="PATH", help="summarize a saved replay and exit")
    ap.add_argument("--json", metavar="PATH", help="also write the aggregate report as JSON")
    args = ap.parse_args(argv)

    os.makedirs(LOG_DIR, exist_ok=True)

    if args.replay:
        return summarize_replay(args.replay)

    agent_path = args.agent if os.path.isabs(args.agent) else os.path.join(HERE, args.agent)
    if not os.path.exists(agent_path):
        raise SystemExit(f"agent not found: {agent_path}")

    workdir = tempfile.mkdtemp(prefix="kaggri_arena_")
    try:
        opp = resolve_opponent(args.opponent, agent_path, workdir)
        seeds = [args.seed + i for i in range(args.episodes)]

        if args.ablate:
            on = make_variant(agent_path, os.path.join(workdir, "flag_on.py"), flags={args.ablate: True})
            off = make_variant(agent_path, os.path.join(workdir, "flag_off.py"), flags={args.ablate: False})
            r_on, l_on = run_set(on, opp, seeds, args.steps, args.workers, args.log_decisions, f"{args.ablate}_on")
            r_off, l_off = run_set(off, opp, seeds, args.steps, args.workers, args.log_decisions, f"{args.ablate}_off")
            a_on, a_off = aggregate(r_on, l_on), aggregate(r_off, l_off)
            print_report(f"{args.ablate} = ON  vs {args.opponent}", a_on)
            print_report(f"{args.ablate} = OFF vs {args.opponent}", a_off)
            sig = paired_significance([r["me_cash"] for r in r_on], [r["me_cash"] for r in r_off])
            print(f"\n--- A/B: {args.ablate} ON minus OFF, {sig['n']} paired seeds ---")
            print(f"  mean delta   {sig['mean_diff']:+,.0f}  (sd {sig['sd_diff']:,.0f})")
            print(f"  ON better on {sig['a_better']}/{sig['n']} seeds")
            print(f"  t = {sig['t']}   p ~ {sig['p_approx']}")
            rel = (a_on["mean_cash"] / a_off["mean_cash"] - 1.0) if a_off["mean_cash"] else float("nan")
            print(f"  relative     {rel:+.1%}")
            verdict = "KEEP ON" if sig["p_approx"] < 0.05 and sig["mean_diff"] > 0 else (
                "KEEP OFF" if sig["p_approx"] < 0.05 and sig["mean_diff"] < 0 else "INCONCLUSIVE"
            )
            print(f"  verdict      {verdict}")
            if args.json:
                _dump_json(args.json, {"ablate": args.ablate, "on": a_on, "off": a_off, "significance": sig,
                                       "relative": rel, "verdict": verdict})
            return 0

        if args.sweep:
            name, _, raw = args.sweep.partition("=")
            values = [_num(v) for v in raw.split(",") if v.strip()]
            if len(values) < 2:
                raise SystemExit("--sweep needs at least two values")
            table = []
            for v in values:
                path = make_variant(agent_path, os.path.join(workdir, f"sweep_{name}_{v}.py"), consts={name: v})
                res, logs = run_set(path, opp, seeds, args.steps, args.workers, args.log_decisions, f"{name}_{v}")
                agg = aggregate(res, logs)
                print_report(f"{name} = {v} vs {args.opponent}", agg)
                table.append((v, agg, [r["me_cash"] for r in res]))
            best = max(table, key=lambda row: row[1]["mean_cash"])
            print(f"\n--- sweep {name}: best = {best[0]} (mean {best[1]['mean_cash']:,.0f}) ---")
            for v, agg, cash in table:
                if v == best[0]:
                    continue
                sig = paired_significance(best[2], cash)
                print(f"  {name}={best[0]} vs {name}={v}: delta {sig['mean_diff']:+,.0f}  p~{sig['p_approx']}  "
                      f"better on {sig['a_better']}/{sig['n']}")
            if args.json:
                _dump_json(args.json, {"sweep": name, "results": [(v, a) for v, a, _ in table], "best": best[0]})
            return 0

        results, logs = run_set(
            agent_path, opp, seeds, args.steps, args.workers, args.log_decisions,
            "run", save_replays=args.save_replays,
        )
        agg = aggregate(results, logs)
        print_report(f"{os.path.basename(args.agent)} vs {args.opponent}", agg)
        _print_acceptance(agg, args)
        if args.json:
            _dump_json(args.json, agg)
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _print_acceptance(agg, args):
    """The spec's submission gate, evaluated explicitly rather than by eyeball."""
    budget = 1.0  # actTimeout
    checks = [
        ("zero crashes", agg["crashes"] == 0, f"{agg['crashes']}"),
        ("zero timeouts", agg["timeouts"] == 0, f"{agg['timeouts']}"),
        ("zero invalid statuses", agg["invalid"] == 0, f"{agg['invalid']}"),
        ("p95 turn < 50% of budget", agg["turn_p95"] < budget * 0.5,
         f"{agg['turn_p95'] * 1000:.1f}ms vs {budget * 500:.0f}ms"),
        ("beats opponent mean cash", agg["mean_cash"] > agg["opp_mean_cash"],
         f"{agg['mean_cash']:,.0f} vs {agg['opp_mean_cash']:,.0f}"),
        ("win rate > 50%", agg["win_rate"] > 0.5, f"{agg['win_rate']:.1%}"),
    ]
    print("\n  acceptance criteria")
    for name, ok, detail in checks:
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if agg["episodes"] < 30:
        print(f"    [WARN] only {agg['episodes']} episodes; the spec gate wants >= 30")


def _num(s):
    s = s.strip()
    try:
        return int(s)
    except ValueError:
        return float(s)


def _dump_json(path, payload):
    p = path if os.path.isabs(path) else os.path.join(HERE, path)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n  wrote {p}")


def summarize_replay(path):
    p = path if os.path.isabs(path) else os.path.join(HERE, path)
    if not os.path.exists(p):
        raise SystemExit(f"replay not found: {p}")
    with open(p) as f:
        data = json.load(f)
    steps = data.get("steps", [])
    print(f"replay {p}")
    print(f"  env {data.get('name')}  steps {len(steps)}  seed {(data.get('info') or {}).get('seed')}")
    if not steps:
        return 0
    final = steps[-1]
    for i, s in enumerate(final):
        print(f"  player {i}: reward={s.get('reward')} status={s.get('status')}")
    print("\n  cash by day (player0 / player1):")
    tpd = int((data.get("configuration") or {}).get("turnsPerDay", 24))
    for idx in range(0, len(steps), tpd):
        obs = steps[idx][0].get("observation", {})
        farms = obs.get("farms") or []
        if len(farms) < 2:
            continue
        day = idx // tpd
        m0, m1 = farms[0].get("money", 0), farms[1].get("money", 0)
        tiles0 = farms[0].get("tiles") or []
        animals = sum(1 for row in tiles0 for t in row if isinstance(t, dict) and t.get("animal"))
        plants = sum(1 for row in tiles0 for t in row if isinstance(t, dict) and t.get("kind") == "PLANT")
        print(f"    day {day:2d}  ${m0:>12,.0f} / ${m1:>12,.0f}   p0 animals={animals} plants={plants}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
