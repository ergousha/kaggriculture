#!/usr/bin/env python3
"""Cross-Entropy Method search over the macro-strategy vector.

Where learning belongs in this project
--------------------------------------
The game decomposes into a macro allocation problem (~20 numbers: labour cap,
land timing, livestock windows, tile fractions, task priorities) and a micro
assignment problem (which unit does which task). The micro layer is a weighted
bipartite matching with an exact polynomial-time solution, so learning it can
only lose — v0.0.8 cloned atomic actions from replays and scored $2,910 against
`baseline` where the heuristic scores ~$50k. The macro layer has no closed
form, is low-dimensional, and is exactly what a distribution-based search is
good at. So that is the only thing this searches.

Two properties matter more than the optimiser choice:

  * common random numbers — every candidate in a generation is evaluated on the
    identical seed set, so differences are strategy, not luck;
  * a P(win)-with-CVaR objective (see search/objective.py), because scoring is a
    pairwise comparison and a single collapsing seed costs a whole match.

Usage:
    python -m search.cem --iterations 12 --pop 16 --episodes 8
    python -m search.cem --diagnose 40      # tail analysis of the current agent
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import time
from typing import Any

from search.harness import MatchHarness
from search.objective import score_results, worst_episodes
from search.space import StrategySpace

SIGMA_INIT = 0.30
SIGMA_FLOOR = 0.03
ELITE_FRAC = 0.25
SMOOTHING = 0.7


def _eval_worker(task: tuple[list[float], int, str, bool, str]) -> dict[str, Any]:
    strategy_vec, seed, opponent_name, swap, base_path = task
    env = MatchHarness(agent_base_path=base_path, opponent_name=opponent_name)
    try:
        return env.evaluate_strategy(strategy_vec, seed=seed, swap_seats=swap)
    finally:
        env.close()


class MacroCEM:
    """Cross-Entropy Method over StrategySpace, with common random numbers."""

    def __init__(
        self,
        agent_base_path: str = "main.py",
        opponents: list[str] | None = None,
        pop_size: int = 16,
        episodes_per_eval: int = 8,
        output_dir: str = "logs/macro_search",
        n_workers: int | None = None,
    ) -> None:
        self.agent_base_path = os.path.abspath(agent_base_path)
        self.opponents = opponents or ["baseline", "adaptive"]
        self.pop_size = pop_size
        self.episodes_per_eval = episodes_per_eval
        self.output_dir = os.path.abspath(output_dir)
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        self.space = StrategySpace()
        os.makedirs(self.output_dir, exist_ok=True)

    def _seed_set(self, base_seed: int, iteration: int) -> list[int]:
        n_pairs = max(1, self.episodes_per_eval // 2)
        return [base_seed + iteration * 1000 + i for i in range(n_pairs)]

    def _evaluate(self, vector: list[float], seeds: list[int], pool: Any) -> dict[str, Any]:
        tasks = []
        for opp in self.opponents:
            for seed in seeds:
                # Both seats on every seed: the interpreter breaks market ties
                # in player-index order, so a one-sided sample is biased.
                tasks.append((vector, seed, opp, False, self.agent_base_path))
                tasks.append((vector, seed, opp, True, self.agent_base_path))
        results = pool.map(_eval_worker, tasks)
        summary = score_results(results)
        summary["strategy_vector"] = vector
        summary["param_dict"] = self.space.vector_to_dict(vector)
        return summary

    def search(self, iterations: int = 12, base_seed: int = 7000) -> dict[str, Any]:
        dim = self.space.dim
        mu = self.space.get_file_vector(self.agent_base_path)
        sigma = [SIGMA_INIT] * dim
        n_elite = max(2, int(round(ELITE_FRAC * self.pop_size)))

        print("=== Kaggriculture macro CEM ===")
        print(f"dim {dim} | pop {self.pop_size} | elites {n_elite} | workers {self.n_workers}")
        print(f"opponents {', '.join(self.opponents)} | episodes/eval {self.episodes_per_eval}\n")

        history: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        pool = mp.Pool(processes=self.n_workers)
        try:
            for it in range(1, iterations + 1):
                t0 = time.time()
                seeds = self._seed_set(base_seed, it)

                population = [list(mu)]  # always keep the incumbent in the sample
                while len(population) < self.pop_size:
                    population.append(
                        [max(-1.0, min(1.0, random.gauss(mu[d], sigma[d]))) for d in range(dim)]
                    )

                scored = [self._evaluate(vec, seeds, pool) for vec in population]
                scored.sort(key=lambda s: s["fitness"], reverse=True)
                elites = scored[:n_elite]

                for d in range(dim):
                    vals = [e["strategy_vector"][d] for e in elites]
                    mean_d = sum(vals) / len(vals)
                    var_d = sum((v - mean_d) ** 2 for v in vals) / len(vals)
                    mu[d] = SMOOTHING * mean_d + (1.0 - SMOOTHING) * mu[d]
                    sigma[d] = max(
                        SIGMA_FLOOR, SMOOTHING * (var_d**0.5) + (1.0 - SMOOTHING) * sigma[d]
                    )

                top = elites[0]
                if best is None or top["fitness"] > best["fitness"]:
                    best = top

                print(
                    f"[iter {it:02d}/{iterations}] fitness {top['fitness']:.3f} | "
                    f"mean ${top['mean_cash']:,.0f} | median ${top['median_cash']:,.0f} | "
                    f"CVaR ${top['cvar_cash']:,.0f} | min ${top['min_cash']:,.0f} | "
                    f"win {top['win_rate'] * 100:.0f}% | {time.time() - t0:.0f}s"
                )
                history.append(
                    {
                        "iteration": it,
                        "fitness": top["fitness"],
                        "mean_cash": top["mean_cash"],
                        "cvar_cash": top["cvar_cash"],
                        "sigma_mean": sum(sigma) / dim,
                        "params": top["param_dict"],
                    }
                )
        finally:
            pool.close()
            pool.join()

        with open(os.path.join(self.output_dir, "best_macro.json"), "w") as f:
            json.dump(best or {}, f, indent=2)
        with open(os.path.join(self.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        return best or {}

    def diagnose(self, episodes: int, base_seed: int = 4242) -> dict[str, Any]:
        """Evaluate the current agent as-is and print the loss tail.

        Under a P(win) objective the interesting statistic is not the mean, it
        is which seeds collapse and why. This prints every seat/seed result so a
        bad one can be replayed with `local_arena.py --replay`.
        """
        vector = self.space.get_file_vector(self.agent_base_path)
        seeds = [base_seed + i for i in range(max(1, episodes // 2))]
        pool = mp.Pool(processes=self.n_workers)
        try:
            tasks = []
            for opp in self.opponents:
                for seed in seeds:
                    tasks.append((vector, seed, opp, False, self.agent_base_path))
                    tasks.append((vector, seed, opp, True, self.agent_base_path))
            results = pool.map(_eval_worker, tasks)
        finally:
            pool.close()
            pool.join()
        summary = score_results(results)

        print(f"episodes    {summary['episodes']}")
        print(f"mean        ${summary['mean_cash']:,.0f}")
        print(f"median      ${summary['median_cash']:,.0f}")
        print(f"CVaR@25%    ${summary['cvar_cash']:,.0f}")
        print(f"min         ${summary['min_cash']:,.0f}")
        print(f"win rate    {summary['win_rate'] * 100:.1f}%")
        print("\nworst seeds:")
        for r in worst_episodes(results, k=min(8, len(results))):
            print(
                f"  seed {r['seed']:>6}  vs {r['opponent']:<10} "
                f"me ${r['me_cash']:>10,.0f}  opp ${r['opp_cash']:>10,.0f}  {r['status']}"
            )
        return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="CEM search over the macro-strategy vector")
    ap.add_argument("--agent", default="main.py")
    ap.add_argument("--iterations", type=int, default=12)
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--opponents", default="baseline,adaptive")
    ap.add_argument("--output", default="logs/macro_search")
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument(
        "--diagnose",
        type=int,
        metavar="N",
        help="skip the search; report the loss tail of the agent as it stands",
    )
    args = ap.parse_args()

    cem = MacroCEM(
        agent_base_path=args.agent,
        opponents=[o.strip() for o in args.opponents.split(",") if o.strip()],
        pop_size=args.pop,
        episodes_per_eval=args.episodes,
        output_dir=args.output,
    )
    if args.diagnose:
        cem.diagnose(args.diagnose, base_seed=args.seed)
    else:
        cem.search(iterations=args.iterations, base_seed=args.seed)


if __name__ == "__main__":
    main()
