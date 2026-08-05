"""Gymnasium-compatible Environment Wrapper for Kaggriculture.

Wraps kaggle-environments to expose standard RL training interfaces (reset, step)
and support parametric strategy evaluations against a pool of opponents.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

try:
    import gymnasium as gym
    from gymnasium import spaces

    _HAS_GYM = True
except ImportError:
    try:
        import gym  # type: ignore
        from gym import spaces  # type: ignore

        _HAS_GYM = True
    except ImportError:
        gym = None  # type: ignore
        spaces = None  # type: ignore
        _HAS_GYM = False

from kaggle_environments import make as make_kaggle_env

from rl.strategy_space import StrategySpace

_BaseEnv: type = gym.Env if (_HAS_GYM and gym is not None) else object


class KaggricultureGymEnv(_BaseEnv):  # type: ignore[misc]
    """Gymnasium environment for Kaggriculture strategy optimization."""

    metadata: dict[str, list] = {"render_modes": []}

    def __init__(
        self,
        agent_base_path: str = "main.py",
        opponent_name: str = "baseline",
        max_steps: int = 720,
        strategy_space: StrategySpace | None = None,
    ) -> None:
        if _HAS_GYM:
            super().__init__()

        self.agent_base_path = os.path.abspath(agent_base_path)
        self.opponent_name = opponent_name
        self.max_steps = max_steps
        self.strategy_space = strategy_space or StrategySpace()

        if spaces is not None:
            # Action space: continuous strategy vector in [-1, 1]^D
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.strategy_space.dim,),
                dtype=float,
            )

            # Observation space: basic game state metrics
            self.observation_space = spaces.Box(
                low=0.0,
                high=1e6,
                shape=(8,),
                dtype=float,
            )
        else:
            self.action_space = None
            self.observation_space = None

        self._curr_seed: int = 42
        self._curr_step: int = 0
        self._env: Any = None
        self._workdir: str = tempfile.mkdtemp(prefix="kaggriculture_rl_")

    def set_opponent(self, opponent_name: str) -> None:
        """Dynamically update opponent strategy."""
        self.opponent_name = opponent_name

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        if _HAS_GYM:
            super().reset(seed=seed)

        if seed is not None:
            self._curr_seed = seed
        else:
            self._curr_seed = int(options.get("seed", 42)) if options else 42

        self._curr_step = 0
        obs = [0.0, 0.0, 1000.0, 1.0, 1000.0, 1.0, 0.0, 0.0]
        info = {"seed": self._curr_seed, "opponent": self.opponent_name}
        return obs, info

    def evaluate_strategy(
        self,
        strategy_vector: list[float],
        seed: int = 42,
        swap_seats: bool = False,
    ) -> dict[str, Any]:
        """Run a complete 720-step evaluation match for a strategy vector.

        Returns game stats: me_cash, opp_cash, win, tie, steps, status.
        """
        # Create temporary variant main.py file with strategy overrides
        var_path = os.path.join(self._workdir, f"variant_seed_{seed}.py")
        self.strategy_space.apply_to_file(self.agent_base_path, var_path, strategy_vector)

        opp_spec = self._resolve_opponent(self.opponent_name)
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

            win = 1 if me_cash > opp_cash else 0
            tie = 1 if me_cash == opp_cash else 0

            return {
                "me_cash": me_cash,
                "opp_cash": opp_cash,
                "win": win,
                "tie": tie,
                "status": me_status,
                "seed": seed,
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
                "opponent": self.opponent_name,
                "strategy_vector": strategy_vector,
            }
        finally:
            if os.path.exists(var_path):
                try:
                    os.remove(var_path)
                except OSError:
                    pass

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Run episode evaluation for the strategy action vector."""
        res = self.evaluate_strategy(list(action), seed=self._curr_seed)
        obs = [
            720.0,
            30.0,
            res["me_cash"],
            12.0,
            res["opp_cash"],
            12.0,
            1.0 if res["win"] else 0.0,
            0.0,
        ]
        reward = res["me_cash"] - res["opp_cash"]
        terminated = True
        truncated = False
        return obs, reward, terminated, truncated, res

    def _resolve_opponent(self, name: str) -> str:
        builtin = {"baseline": "starter", "starter": "starter", "random": "random", "pass": "pass"}
        if name in builtin:
            return builtin[name]
        if name == "adaptive":
            p = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "opponents", "adaptive.py")
            )
            if os.path.exists(p):
                return p
        if os.path.exists(name):
            return os.path.abspath(name)
        return "starter"

    def close(self) -> None:
        if os.path.exists(self._workdir):
            import shutil

            try:
                shutil.rmtree(self._workdir)
            except OSError:
                pass
