#!/usr/bin/env python3
"""Pre-flight, submit and track a Kaggriculture submission.

    python submit.py --dry-run                 # all checks, no submission
    python submit.py                           # check, submit, poll status
    python submit.py -m "custom message"       # override the auto message
    python submit.py --episodes 10             # deeper smoke test

Credentials come from `kaggle_credentials.py` (see kaggle_credentials.example.py)
and are exported to the environment BEFORE the `kaggle` package is imported, so
no ~/.kaggle/kaggle.json is needed. The key is never printed or logged.

Pre-flight hard-fails on any of:
  1. main.py imports something outside the allowed set
  2. `agent` is missing, has the wrong arity, or is NOT the last callable defined
     (kaggle-environments loads the LAST callable in the file, not the one named
     `agent` — getting this wrong errors out every turn of every episode)
  3. the smoke test records any crash, timeout or invalid status
"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SUBMISSION = os.path.join(HERE, "main.py")
COMPETITION = "kaggriculture"
LOG_DIR = os.path.join(HERE, "logs")
HISTORY = os.path.join(LOG_DIR, "submission_history.md")

# main.py must be self-contained: stdlib only. numpy is permitted because the
# Kaggle image provides it, but main.py currently does not need it.
ALLOWED_IMPORTS = {
    "math",
    "os",
    "time",
    "traceback",
    "json",
    "random",
    "collections",
    "itertools",
    "functools",
    "heapq",
    "bisect",
    "statistics",
    "copy",
    "sys",
    "numpy",
}


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def load_credentials() -> str:
    """Export credentials to the environment. Returns the method used."""
    env_token = (
        os.environ.get("KAGGLE_API_TOKEN", "") or os.environ.get("KAGGLE_TOKEN", "")
    ).strip()
    if env_token:
        os.environ["KAGGLE_API_TOKEN"] = env_token
        return "access token (environment)"

    try:
        sys.path.insert(0, HERE)
        import kaggle_credentials as creds  # noqa: WPS433
    except ImportError as exc:
        print(
            "ERROR: kaggle_credentials.py not found.\n"
            "\n"
            "  cp kaggle_credentials.example.py kaggle_credentials.py\n"
            "  # then edit it and paste in your credentials from\n"
            "  # https://www.kaggle.com/settings/api\n"
            "\n"
            "It is already in .gitignore. Do not commit it and do not paste it\n"
            "into a published notebook: it grants full API access to your account.",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc

    token = (getattr(creds, "KAGGLE_API_TOKEN", "") or "").strip()
    user = (getattr(creds, "KAGGLE_USERNAME", "") or "").strip()
    key = (getattr(creds, "KAGGLE_KEY", "") or "").strip()

    if token:
        os.environ["KAGGLE_API_TOKEN"] = token
        return "access token"
    if user and key and user != "your_username" and key != "your_api_key":
        os.environ["KAGGLE_USERNAME"] = user
        os.environ["KAGGLE_KEY"] = key
        return f"legacy API key (user {user})"

    print(
        "ERROR: kaggle_credentials.py has no usable credentials.\n"
        "Set KAGGLE_API_TOKEN, or both KAGGLE_USERNAME and KAGGLE_KEY.\n"
        "Get them at https://www.kaggle.com/settings/api.",
        file=sys.stderr,
    )
    raise SystemExit(2)


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------


def check_static() -> list[str]:
    """Parse main.py and verify imports + entrypoint. Returns failure strings."""
    fails = []
    with open(SUBMISSION) as f:
        src = f.read()
    tree = ast.parse(src, filename=SUBMISSION)

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imported.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                fails.append(
                    f"relative import (level {node.level}) — submission must be self-contained"
                )
            elif node.module:
                imported.add(node.module.split(".")[0])
    bad = sorted(imported - ALLOWED_IMPORTS)
    if bad:
        fails.append(f"disallowed imports: {bad} (allowed: {sorted(ALLOWED_IMPORTS)})")

    # Entrypoint: name, arity, and — critically — position.
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    agent_def = next((n for n in funcs if n.name == "agent"), None)
    if agent_def is None:
        fails.append("no top-level `agent` function defined")
    else:
        nargs = len(agent_def.args.args)
        if nargs not in (1, 2):
            fails.append(
                f"`agent` takes {nargs} positional args; the env passes (obs) or (obs, config)"
            )
        # kaggle-environments: get_last_callable() returns
        # [v for v in env.values() if callable(v)][-1] — the LAST callable bound
        # in module namespace order. A helper defined after `agent` becomes the
        # agent and every turn raises.
        last_callable = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                last_callable = node.name
        if last_callable != "agent":
            fails.append(
                f"`agent` is not the last callable defined (last is `{last_callable}`). "
                "kaggle-environments loads the LAST callable in the file, so this would "
                "error on every turn."
            )
    if not classes:
        fails.append("expected the planner/scheduler classes to be inline (self-contained)")
    return fails


def check_loadable() -> list[str]:
    """Load main.py exactly the way kaggle-environments does."""
    fails = []
    try:
        from kaggle_environments.agent import get_last_callable

        with open(SUBMISSION) as f:
            fn = get_last_callable(f.read(), path=SUBMISSION)
        if getattr(fn, "__name__", None) != "agent":
            fails.append(
                f"env would load `{getattr(fn, '__name__', fn)}` as the agent, not `agent`"
            )
    except Exception as exc:
        fails.append(f"main.py failed to load as an agent: {type(exc).__name__}: {exc}")
    return fails


def smoke_test(episodes: int, opponent: str) -> tuple[list[str], dict]:
    """Run real episodes and require zero crashes/timeouts/invalid statuses."""
    sys.path.insert(0, HERE)
    import local_arena

    seeds = [1000 + i for i in range(episodes)]
    results, logs = local_arena.run_set(
        SUBMISSION,
        local_arena.resolve_opponent(opponent, SUBMISSION, LOG_DIR),
        seeds,
        720,
        min(episodes, max(1, (os.cpu_count() or 2) - 1)),
        False,
        "preflight",
    )
    agg = local_arena.aggregate(results, logs)
    fails = []
    if agg["crashes"]:
        fails.append(f"{agg['crashes']} ERROR statuses in smoke test")
    if agg["timeouts"]:
        fails.append(f"{agg['timeouts']} TIMEOUT statuses in smoke test")
    if agg["invalid"]:
        fails.append(f"{agg['invalid']} INVALID statuses in smoke test")
    if agg["harness_errors"]:
        fails.append(f"harness errors: {agg['harness_errors'][:1]}")
    budget = 1.0  # actTimeout
    if agg["turn_p95"] >= budget * 0.5:
        fails.append(f"p95 turn {agg['turn_p95'] * 1000:.0f}ms exceeds 50% of the {budget}s budget")
    return fails, agg


# ---------------------------------------------------------------------------
# Message + history
# ---------------------------------------------------------------------------


def agent_version() -> str:
    with open(SUBMISSION) as f:
        m = re.search(r'^AGENT_VERSION\s*=\s*["\']([^"\']+)["\']', f.read(), re.M)
    return m.group(1) if m else "0.0.0"


def build_message(agg: dict, opponent: str) -> str:
    return (
        f"v{agent_version()} | {agg['win_rate']:.0%} WR vs {opponent}, {agg['episodes']} eps"
        f" | mean ${agg['mean_cash']:,.0f} (opp ${agg['opp_mean_cash']:,.0f})"
        f" | p95 {agg['turn_p95'] * 1000:.1f}ms"
    )


def append_history(message: str, agg: dict, opponent: str, submitted: bool) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        diff = subprocess.run(
            ["git", "diff", "--stat", "HEAD", "--", "main.py"],
            cwd=HERE,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except Exception:
        diff = ""
    new = not os.path.exists(HISTORY)
    with open(HISTORY, "a") as f:
        if new:
            f.write("# Submission history\n\n")
        f.write(f"## {stamp} — v{agent_version()}{'' if submitted else ' (DRY RUN)'}\n\n")
        f.write(f"- message: `{message}`\n")
        f.write(f"- opponent: {opponent}, episodes: {agg['episodes']}\n")
        f.write(
            f"- cash: mean ${agg['mean_cash']:,.0f}, median ${agg['median_cash']:,.0f}, "
            f"min ${agg['min_cash']:,.0f}\n"
        )
        f.write(f"- win rate: {agg['win_rate']:.1%} ({agg['wins']}W {agg['ties']}T)\n")
        f.write(
            f"- reliability: {agg['crashes']} crashes, {agg['timeouts']} timeouts, "
            f"{agg['invalid']} invalid\n"
        )
        f.write(
            f"- turn compute: p50 {agg['turn_p50'] * 1000:.2f}ms, "
            f"p95 {agg['turn_p95'] * 1000:.2f}ms, max {agg['turn_max'] * 1000:.2f}ms\n"
        )
        f.write(f"- shed overflow lost: {agg['shed_overflow_lost']}\n")
        if diff:
            f.write(f"- main.py diff vs HEAD:\n```\n{diff}\n```\n")
        f.write("\n")
    print(f"  appended {HISTORY}")


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------


def do_submit(message: str) -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print(f"  submitting {os.path.basename(SUBMISSION)} to {COMPETITION} ...")
    api.competition_submit(file_name=SUBMISSION, message=message, competition=COMPETITION)
    print("  submitted. polling status ...")
    for attempt in range(20):
        time.sleep(6)
        try:
            subs = api.competition_submissions(COMPETITION)
        except Exception as exc:
            print(f"  (status poll failed: {type(exc).__name__}: {exc})")
            return
        if not subs:
            continue
        latest = subs[0]
        status = getattr(latest, "status", None)
        desc = getattr(latest, "description", "")
        score = getattr(latest, "publicScore", None) or getattr(latest, "public_score", None)
        print(f"  [{attempt + 1}] status={status} score={score} desc={desc!r}")
        # `status` is a SubmissionStatus enum, so str() gives
        # "SubmissionStatus.COMPLETE" — compare on the trailing name, not the
        # whole string, or this loop never terminates early.
        state = str(status).rsplit(".", 1)[-1].lower()
        if state in ("complete", "error", "failed", "cancelled"):
            print(f"  final: {state}" + (f", score {score}" if score else ""))
            return
    print("  still pending; check with: kaggle competitions submissions kaggriculture")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Submit the Kaggriculture agent")
    ap.add_argument("--dry-run", action="store_true", help="run every check, do not submit")
    ap.add_argument("-m", "--message", help="submission message (default: auto from arena stats)")
    ap.add_argument("--episodes", type=int, default=3, help="smoke-test episodes (spec minimum 3)")
    ap.add_argument("--opponent", default="baseline")
    ap.add_argument(
        "--skip-smoke", action="store_true", help="skip the smoke test (NOT for real submits)"
    )
    args = ap.parse_args(argv)

    print(f"Kaggriculture submit — v{agent_version()}")
    print(f"  file: {SUBMISSION}")

    print("\n[1/4] static checks")
    fails = check_static()
    for f in fails:
        print(f"  FAIL {f}")
    if not fails:
        print("  ok — imports allowed, `agent` present with correct arity and defined last")

    print("\n[2/4] agent loads the way kaggle-environments loads it")
    lf = check_loadable()
    for f in lf:
        print(f"  FAIL {f}")
    if not lf:
        print("  ok")
    fails += lf

    agg: dict = {}
    if args.skip_smoke:
        print("\n[3/4] smoke test SKIPPED (--skip-smoke)")
        if not args.dry_run:
            print("  refusing to submit without a smoke test", file=sys.stderr)
            return 2
    else:
        print(f"\n[3/4] smoke test — {args.episodes} episodes vs {args.opponent}")
        sf, agg = smoke_test(args.episodes, args.opponent)
        for f in sf:
            print(f"  FAIL {f}")
        if not sf:
            print(
                f"  ok — 0 crashes / 0 timeouts / 0 invalid, "
                f"mean ${agg['mean_cash']:,.0f} vs ${agg['opp_mean_cash']:,.0f}, "
                f"win {agg['win_rate']:.0%}, p95 {agg['turn_p95'] * 1000:.1f}ms"
            )
        fails += sf

    if fails:
        print(f"\nPRE-FLIGHT FAILED ({len(fails)} problem(s)). Not submitting.", file=sys.stderr)
        return 1

    message = args.message or (build_message(agg, args.opponent) if agg else f"v{agent_version()}")
    print(f"\n[4/4] {'dry run' if args.dry_run else 'submitting'}")
    print(f"  message: {message}")

    if args.dry_run:
        print("  --dry-run: no submission made")
        print("\n  To fetch the reference notebooks for comparison (needs credentials):")
        print("    kaggle kernels pull bovard/kaggriculture-getting-started -p reference/")
        print(
            "    kaggle kernels pull tetsutani/adaptive-farming-strategy-for-kaggriculture -p reference/"
        )
        if agg:
            append_history(message, agg, args.opponent, submitted=False)
        return 0

    method = load_credentials()
    print(f"  authenticated via {method}")
    do_submit(message)
    if agg:
        append_history(message, agg, args.opponent, submitted=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
