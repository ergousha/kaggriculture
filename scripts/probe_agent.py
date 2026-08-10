"""Schema probe: dumps the raw observation + configuration to logs/probe_schema.json.

Build-order step 1. This is the empirical source of truth for every field name used
in main.py. Run:  python probe_agent.py
"""

import json
import os
import sys
import time

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
PROBE_TURNS = 5

_records: list = []


def _plain(obj, depth=0):
    """Recursively convert kaggle-environments Struct/dict/list into plain JSON types."""
    if depth > 12:
        return "<max-depth>"
    if isinstance(obj, dict):
        return {str(k): _plain(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_plain(v, depth + 1) for v in obj]
    if isinstance(obj, str | int | float | bool) or obj is None:
        return obj
    return repr(obj)


def probe_agent(obs, config):
    """Records the full obs/config, then plays a legal no-op."""
    if len(_records) < PROBE_TURNS:
        plain_obs = _plain(obs)
        obs_keys = sorted(plain_obs.keys()) if isinstance(plain_obs, dict) else []
        _records.append(
            {
                "arity_seen": 2,
                "obs_keys": obs_keys,
                "config": _plain(config),
                "obs": plain_obs,
            }
        )
    return {"farmer": ["PASS"], "hands": [], "market": []}


def main():
    from kaggle_environments import make

    os.makedirs(LOG_DIR, exist_ok=True)

    # Short episode: we only need the first few turns of schema.
    env = make("kaggriculture", configuration={"episodeSteps": 30, "seed": 42}, debug=True)
    t0 = time.perf_counter()
    env.run([probe_agent, "starter"])
    elapsed = time.perf_counter() - t0

    out = {
        "kaggle_environments_version": _kenv_version(),
        "spec_configuration": _plain(env.configuration),
        "episode_wall_seconds": round(elapsed, 3),
        "final_rewards": [s.reward for s in env.steps[-1]],
        "final_statuses": [s.status for s in env.steps[-1]],
        "turns": _records,
    }
    dest = os.path.join(LOG_DIR, "probe_schema.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2, sort_keys=False)

    print(f"wrote {dest}")
    print("config:", json.dumps(out["spec_configuration"], sort_keys=True))
    print("obs keys:", _records[0]["obs_keys"] if _records else "NONE")
    print("final rewards:", out["final_rewards"], "statuses:", out["final_statuses"])

    # Print the shape of each top-level obs field so field names are visible at a glance.
    if _records:
        o = _records[0]["obs"]
        for k in _records[0]["obs_keys"]:
            v = o[k]
            print(f"  obs[{k!r}]: {type(v).__name__} = {json.dumps(v)[:220]}")


def _kenv_version():
    try:
        import kaggle_environments

        return getattr(kaggle_environments, "__version__", None) or _pkg_version()
    except Exception:
        return None


def _pkg_version():
    try:
        from importlib.metadata import version

        return version("kaggle-environments")
    except Exception:
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
