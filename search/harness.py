"""Paired-seed match harness.

Runs one 720-step match between a parameterised copy of the agent and an
opponent, and reports the two final cash figures. That is the entire interface
the macro searches need.

This was a `gymnasium.Env` with `reset`/`step`/`observation_space`. The wrapper
was fiction: an episode is a single 720-step rollout of a *parameter vector*, so
`step` always terminated immediately and the observation was never read by
anything. Nothing here is a reinforcement-learning environment, and pretending
otherwise cost a real dependency and a lot of dead code.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any

from kaggle_environments import make as make_kaggle_env

from search.space import StrategySpace

BUILTIN_OPPONENTS = {
    "baseline": "starter",
    "starter": "starter",
    "random": "random",
    "pass": "pass",
}


class MatchHarness:
    """Evaluates a macro-strategy vector over full matches."""

    def __init__(
        self,
        agent_base_path: str = "main.py",
        opponent_name: str = "baseline",
        max_steps: int = 720,
        strategy_space: StrategySpace | None = None,
    ) -> None:
        self.agent_base_path = os.path.abspath(agent_base_path)
        self.opponent_name = opponent_name
        self.max_steps = max_steps
        self.strategy_space = strategy_space or StrategySpace()
        self._workdir = tempfile.mkdtemp(prefix="kaggriculture_search_")

    def set_opponent(self, opponent_name: str) -> None:
        self.opponent_name = opponent_name

    def resolve_opponent(self, name: str) -> str:
        if name in BUILTIN_OPPONENTS:
            return BUILTIN_OPPONENTS[name]
        if name == "adaptive":
            p = os.path.join(os.path.dirname(__file__), "..", "opponents", "adaptive.py")
            if os.path.exists(p):
                return os.path.abspath(p)
        if os.path.exists(name):
            return os.path.abspath(name)
        return "starter"

    def evaluate_strategy(
        self,
        strategy_vector: list[float],
        seed: int = 42,
        swap_seats: bool = False,
    ) -> dict[str, Any]:
        """Run one match for a strategy vector and report both players' cash."""
        var_path = os.path.join(self._workdir, f"variant_seed_{seed}_{int(swap_seats)}.py")
        self.strategy_space.apply_to_file(self.agent_base_path, var_path, strategy_vector)

        opp_spec = self.resolve_opponent(self.opponent_name)
        agents = [opp_spec, var_path] if swap_seats else [var_path, opp_spec]
        me_idx = 1 if swap_seats else 0

        env = make_kaggle_env(
            "kaggriculture",
            configuration={"seed": seed, "actTimeout": 2.0},
            debug=False,
        )

        try:
            env.run(agents)  # pyrefly: ignore [bad-argument-type]
            final_step = env.steps[-1]
            rewards = [s.get("reward", 0.0) or 0.0 for s in final_step]
            statuses = [s.get("status", "DONE") for s in final_step]

            me_cash = float(rewards[me_idx]) if me_idx < len(rewards) else 0.0
            opp_cash = float(rewards[1 - me_idx]) if (1 - me_idx) < len(rewards) else 0.0
            me_status = statuses[me_idx] if me_idx < len(statuses) else "DONE"

            return {
                "me_cash": me_cash,
                "opp_cash": opp_cash,
                "win": 1 if me_cash > opp_cash else 0,
                "tie": 1 if me_cash == opp_cash else 0,
                "status": me_status,
                "seed": seed,
                "swap": bool(swap_seats),
                "opponent": self.opponent_name,
                "strategy_vector": strategy_vector,
            }
        except Exception as exc:
            return {
                "me_cash": 0.0,
                "opp_cash": 0.0,
                "win": 0,
                "tie": 0,
                "status": f"ERROR: {exc}",
                "seed": seed,
                "swap": bool(swap_seats),
                "opponent": self.opponent_name,
                "strategy_vector": strategy_vector,
            }
        finally:
            if os.path.exists(var_path):
                try:
                    os.remove(var_path)
                except OSError:
                    pass

    def close(self) -> None:
        if os.path.exists(self._workdir):
            try:
                shutil.rmtree(self._workdir)
            except OSError:
                pass
