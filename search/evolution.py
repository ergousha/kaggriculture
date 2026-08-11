#!/usr/bin/env python3
"""Population-based search over the macro-strategy vector.

Same search space, objective and paired-seed harness as `search.cem`; the
difference is the optimiser (elitist mutation instead of a fitted Gaussian).
Kept as a second opinion when CEM's variance collapses early.

Usage:
    python -m search.evolution --generations 15 --pop-size 12 --episodes 8
    python -m search.evolution --opponents baseline,opponents/v0_0_6.py
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
from typing import Any

from search.harness import MatchHarness
from search.objective import score_results, worst_episodes
from search.space import StrategySpace


def _eval_worker(task: tuple[list[float], int, str, bool, str]) -> dict[str, Any]:
    """Worker function for executing a single strategy evaluation episode."""
    strategy_vec, seed, opponent_name, swap, base_path = task
    env = MatchHarness(agent_base_path=base_path, opponent_name=opponent_name)
    try:
        res = env.evaluate_strategy(strategy_vec, seed=seed, swap_seats=swap)
        return res
    finally:
        env.close()


class EvolutionSearch:
    """Elitist population search with paired-seed multi-processing."""

    def __init__(
        self,
        agent_base_path: str = "main.py",
        opponents: list[str] | None = None,
        pop_size: int = 12,
        elite_size: int = 3,
        episodes_per_eval: int = 8,
        output_dir: str = "logs/evolution_search",
        n_workers: int | None = None,
    ) -> None:
        self.agent_base_path = os.path.abspath(agent_base_path)
        self.opponents = opponents or ["baseline", "adaptive"]
        self.pop_size = pop_size
        self.elite_size = min(elite_size, pop_size // 2)
        self.episodes_per_eval = episodes_per_eval
        self.output_dir = os.path.abspath(output_dir)
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)

        os.makedirs(self.output_dir, exist_ok=True)
        self.space = StrategySpace()

    def initialize_population(self, seed_with_defaults: bool = True) -> list[list[float]]:
        """Initialize the population around the agent file as it currently stands."""
        population = []
        default_vec = self.space.get_file_vector(self.agent_base_path)

        if seed_with_defaults:
            population.append(default_vec)

        while len(population) < self.pop_size:
            mutated = self.space.mutate(default_vec, scale=0.20, p_mutate=0.40)
            population.append(mutated)

        return population

    def evaluate_candidate(
        self,
        candidate_vector: list[float],
        seeds: list[int],
        opponents: list[str],
        pool: Any,
    ) -> dict[str, Any]:
        """Evaluate a strategy candidate vector across paired seeds and opponents."""
        tasks = []
        for opp in opponents:
            for seed in seeds:
                # Seat 0 (regular) and Seat 1 (swapped)
                tasks.append((candidate_vector, seed, opp, False, self.agent_base_path))
                tasks.append((candidate_vector, seed, opp, True, self.agent_base_path))

        results = pool.map(_eval_worker, tasks)

        # Ranked pairwise scoring means the objective is P(win) with a tail
        # penalty, not mean cash: `mean + 0.5*net + 2000*win_rate` rewarded a
        # strategy that averaged well and collapsed on one seed in eight.
        summary = score_results(results)
        summary["strategy_vector"] = candidate_vector
        summary["param_dict"] = self.space.vector_to_dict(candidate_vector)
        summary["net_cash"] = summary["mean_cash"] - summary["mean_opp_cash"]
        summary["matches_run"] = len(results)
        summary["worst"] = [
            {"seed": r["seed"], "opponent": r["opponent"], "me_cash": r["me_cash"]}
            for r in worst_episodes(results, k=3)
        ]
        return summary

    def train(
        self,
        generations: int = 15,
        base_seed: int = 100,
    ) -> dict[str, Any]:
        """Run the elitist population search loop."""
        print("=== Kaggriculture macro evolution search ===")
        print(f"Strategy Space Dim: {self.space.dim}")
        print(f"Population Size:    {self.pop_size} | Elites: {self.elite_size}")
        print(f"Episodes/Eval:     {self.episodes_per_eval} | Workers: {self.n_workers}")
        print(f"Opponents:          {', '.join(self.opponents)}")
        print(f"Output Directory:   {self.output_dir}\n")

        population = self.initialize_population(seed_with_defaults=True)
        best_overall: dict[str, Any] | None = None
        history = []

        train_start_t = time.time()
        log_file_path = os.path.join(self.output_dir, "training.log")
        log_file = open(log_file_path, "a")

        def log_msg(msg: str) -> None:
            print(msg)
            sys.stdout.flush()
            log_file.write(msg + "\n")
            log_file.flush()

        pool = mp.Pool(processes=self.n_workers)

        try:
            for gen in range(1, generations + 1):
                start_t = time.time()

                # Generate seed set for this generation (paired seeds for fair comparison)
                seeds = [base_seed + gen * 100 + i for i in range(self.episodes_per_eval // 2)]

                log_msg(f"--- Generation {gen}/{generations} ---")
                eval_results = []
                for idx, candidate in enumerate(population, 1):
                    res = self.evaluate_candidate(candidate, seeds, self.opponents, pool)
                    eval_results.append(res)
                    log_msg(
                        f"  [Gen {gen}/{generations} | Cand {idx:02d}/{self.pop_size:02d}] "
                        f"Score: {res['fitness']:>6.3f} | "
                        f"Cash: ${res['mean_cash']:>9,.0f} | CVaR: ${res['cvar_cash']:>9,.0f} | "
                        f"Min: ${res['min_cash']:>9,.0f} | WinRate: {res['win_rate'] * 100:>5.1f}%"
                    )

                # Rank population by fitness score
                eval_results.sort(key=lambda x: x["fitness"], reverse=True)
                top_cand = eval_results[0]

                if best_overall is None or top_cand["fitness"] > best_overall["fitness"]:
                    best_overall = top_cand

                elapsed_gen = time.time() - start_t
                total_elapsed = time.time() - train_start_t
                avg_gen_t = total_elapsed / gen
                est_rem_t = avg_gen_t * (generations - gen)
                rem_str = (
                    f"{int(est_rem_t // 60)}m {int(est_rem_t % 60)}s"
                    if est_rem_t > 60
                    else f"{est_rem_t:.1f}s"
                )

                log_msg(
                    f"Gen {gen}/{generations} Complete | Leader Cash: ${top_cand['mean_cash']:,.2f} "
                    f"(CVaR ${top_cand['cvar_cash']:,.0f}, WinRate {top_cand['win_rate'] * 100:.1f}%) "
                    f"| Gen Time: {elapsed_gen:.1f}s | Est. Remaining: {rem_str}\n"
                )

                history.append(
                    {
                        "generation": gen,
                        "top_score": top_cand["fitness"],
                        "top_cash": top_cand["mean_cash"],
                        "top_cvar": top_cand["cvar_cash"],
                        "top_params": top_cand["param_dict"],
                        "elapsed_sec": round(elapsed_gen, 2),
                    }
                )

                # Produce next generation (Elitism + Mutation)
                elites = [r["strategy_vector"] for r in eval_results[: self.elite_size]]
                next_pop = list(elites)

                # Mutate from elites to fill remaining population
                mutation_scale = max(0.05, 0.25 * (1.0 - (gen / generations)))
                while len(next_pop) < self.pop_size:
                    parent = random.choice(elites)
                    mutated = self.space.mutate(parent, scale=mutation_scale, p_mutate=0.35)
                    next_pop.append(mutated)

                population = next_pop

        finally:
            try:
                pool.close()
                pool.join()
            except Exception:
                pool.terminate()
            try:
                log_file.close()
            except OSError:
                pass

        # Save best strategy checkpoint
        best_path = os.path.join(self.output_dir, "best_strategy.json")
        history_path = os.path.join(self.output_dir, "training_history.json")

        if best_overall:
            with open(best_path, "w") as f:
                json.dump(best_overall, f, indent=2)
            print(f"[EvolutionSearch] Saved overall best strategy to {best_path}")

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        return best_overall or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Population-based macro-strategy search")
    parser.add_argument("--agent", default="main.py", help="Path to base main.py agent")
    parser.add_argument(
        "--generations", type=int, default=5, help="Number of evolutionary generations"
    )
    parser.add_argument("--pop-size", type=int, default=8, help="Population size per generation")
    parser.add_argument("--episodes", type=int, default=6, help="Episodes per candidate evaluation")
    parser.add_argument(
        "--opponents", default="baseline,adaptive", help="Comma-separated opponent list"
    )
    parser.add_argument("--output", default="logs/evolution_search", help="Output directory")
    args = parser.parse_args()

    opponents = [o.strip() for o in args.opponents.split(",") if o.strip()]

    optimizer = EvolutionSearch(
        agent_base_path=args.agent,
        opponents=opponents,
        pop_size=args.pop_size,
        episodes_per_eval=args.episodes,
        output_dir=args.output,
    )
    optimizer.train(generations=args.generations)


if __name__ == "__main__":
    main()
