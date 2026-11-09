# Kaggriculture Post-Mortem: Analysis of the 1st Place Solution & Lessons Learned

This document compares the winning solution by team **M & M & P & Q** ([msdsm/kaggriculture-1st-solution](https://github.com/msdsm/kaggriculture-1st-solution) & [Writeup](https://kaggle.com/competitions/kaggriculture/writeups/1st-place-solution-our-journey-to-becoming-profe)) with the approach developed in this repository ([ergousha/kaggriculture](https://github.com/ergousha/kaggriculture)).

A single competition winner demonstrates what *can* win, not the only way to win. The goal of this retrospective is to examine where our hypotheses, diagnostics, and resource allocations worked well, where our reasoning slipped, and what concrete practices carry forward.

---

## 1. Overview & Comparison Matrix

Kaggriculture is a 720-step (30 days × 24 hours) farming and market simulation with up to 20 units per player, shared dynamic commodity markets, a 100-item shed cap, and random shop unlocks every 3 days.

Our project evolved from a two-layer heuristic planner (`v0.0.1`–`v0.0.6`), through a failed Behavior Cloning experiment (`v0.0.8`), into a mined open-loop route-replay agent (`v0.2.0`–`v0.3.1`), and finally into a 41-tape, 64-draw shop-routing chassis (`v0.4.0`–`v0.5.1`).

The 1st place team — a four-person collaboration with GPU and Kubernetes infrastructure — combined a 10.23M-parameter Transformer trained via alternating Behavior Cloning (BC) and self-play PPO on a custom Rust simulator with a C++17 heuristic planner and rule-based action patches.

| Dimension | Our Approach (`ergousha`) | 1st Place Solution (`msdsm / M & M & P & Q`) | Notes & Context |
| :--- | :--- | :--- | :--- |
| **Core Architecture** | **Open-Loop Route-Replay / Chassis:** Replaying recorded 719-step action tapes with bounded WEED repair, SELL-slot ordering, hands alignment, and a Day-6 shop-draw router across 64 pairs. | **Hybrid Neural + Heuristics + Search:** 12-block, 10.23M-parameter Transformer trained with BC ↔ PPO, paired with two distinct inference controllers (**Final A** and **Final B**). | Both final winning submissions shared the 10.23M architecture with different weights and controllers; the repository does not state which of the two scored higher on the final leaderboard. |
| **Final Agent A vs. B** | Single active submission paradigm per release (`main.py`). | **Final A:** Confidence-ordered action repair, shed rules, and a Day 29 C++17 search handoff.<br>**Final B:** Sequential unit action masks during training and inference; neural policy for all 720 steps. | Agent A relied on post-processing repairs and endgame search; Agent B internalized unit coordination into conditional action masks. |
| **Learning & Search Loop** | `v0.0.8` BC failed (\$2,910 vs. \$75,668 starter) → ML deleted. Used CEM/Evolution over 23 macro numbers (`v0.0.9`), then offline sieve selection and local tape mutations (`#26`–`#30`). | **Iterative Flywheel:** Public replay BC → frozen-actor critic warmup → self-play PPO → weakness diagnosis → synthetic C++ teacher games → mixed-cache BC → PPO. Net2Net growth (6 → 12 blocks; 24-block parallel line). | In the winning pipeline, hand-crafted heuristics and search acted as teachers and guardrails for the neural network rather than competing paradigms. |
| **Observation Space** | `v0.0.8` used 4 scalars (`[day, hour, money, hands]`). Route-replay read only step, weed tiles, shed/carried counts, and Day-6 shop unlocks. | **264 Tokens × 124 Features:** Farm cells (200), units (40), market slots (10), commodities (12), global state (1), and rule-updated opponent inventory belief (1). 2D RoPE spatial encoding. | The 264-token representation made the state approximately Markovian, tracking public farm layouts, prices, shed headroom, and estimated opponent stock. |
| **Action Space & Coordination** | Verbatim replay of mined unit and market action streams. Earlier heuristic line (`v0.0.6`) used greedy task assignment after Hungarian matching measured worse. | **500 unit logits + 1,903 market choices** (including an absolute SELL-quantity head for 9 commodities × 100 units). Coordination via confidence-ordered post-repair (A) or sequential prefix masks (B). | Explicit SELL quantities (1–100) and sequential reservation of tiles/seeds avoided the action collisions that plagued naive independent heads. |
| **Market & Shed Management** | Replayed recorded market orders verbatim. Offline attempts to meter milk sales (`#30`) hit the 100-item shed cap and displaced higher-value wool/strawberry. | Neural market head proposes orders; `shed_patch.py` predicts pre-night inventory, sells overflow above 100 (ranked by price vs. base price while reserving wheat for feed), and liquidates on steps 717–718. | Both projects independently found that the 100-item shed cap and pre-night overflow were binding constraints on revenue. |
| **Endgame (Day 29)** | Replayed the tape's recorded terminal steps verbatim. | **Final A:** Preloads a C++17 planner on Day 28 and hands off at Day 29 dawn (~6 s budget, simulated annealing with restarts) if ≥10 s overage remains.<br>**Final B:** Neural policy through step 719. | Day 29 has no further shop unlocks and a 24-step horizon, making terminal harvesting, delivery, and liquidation amenable to local search. |
| **Simulation & Compute** | Python `kaggle-environments` (~2.4 s/game). ~80,000 episodes per Phase 2 sieve (~5.8 × 10⁷ env steps per run; ~10⁸ steps total across the project) on 12 CPU workers. | Custom Rust engine (`kagg-engine`) + JAX on GPUs. **8.29M PPO games / 11.92B env steps** on the main 10M line, plus ~3.95M games / 5.68B steps on parallel 20M forks. | Roughly a 10²–10³× gap in total environment steps simulated, driven by both engine speed (Rust vs. Python) and hardware scale (multi-GPU cluster vs. single 12-core machine). |

---

## 2. Anatomy of the 1st Place Solution

The winning team structured their effort around complementary hypotheses across four contributors:
- **`@morim3`:** Neural network architecture and the 10M model line (testing whether more self-play games yield stronger agents).
- **`@msdsm`:** The 20M model line and Net2Net depth scaling (testing whether more parameters yield stronger agents).
- **`@BergBuch`:** A C++17 heuristic planner used both as a synthetic data generator and as Agent A's final-day controller.
- **`@qistripute`:** Rule-based action patches, shed overflow protection, and replay analysis to identify strategic weaknesses.

### 2.1. Network Architecture & Token Schema
Both final submitted agents use a **10,225,070-parameter** pre-LayerNorm Transformer (12 blocks, width 256, 8 attention heads, 256 → 1,024 → 256 GELU FFN, zero dropout):
- **264 Typed Tokens:** Each token type selects relevant columns from a 124-feature schema before linear projection to width 256:
  - `Farm cells` (200 tokens, 34 features): Both players' 10 × 10 boards.
  - `Units` (40 tokens, 22 features): Farmer and hired hands for both players.
  - `Commodities & animals` (12 tokens, 19 features): Market prices, inventory, town demand.
  - `Market-order slots` (10 tokens, 10 features): Sequential order slots.
  - `Global state` (1 token, 34 features): Time, money, hires, unlocked quadrants, shop counts, shed usage.
  - `Opponent-inventory memory` (1 token, 50 features): A rule-updated estimate of the opponent's hidden shed and carried stock derived from public observations.
- **Selective 2D RoPE:** Encodes relative 2D spatial offsets within farms (rotary dimension 16, base 100).
- **Zero-Sum Paired Critic:** Reads the global token and couples adjacent seats via softmax so value estimates are strictly antisymmetric across the two players, supplemented by a two-parameter money/time prior.

### 2.2. The Iterative Training Loop (BC ↔ PPO ↔ Heuristics)
Rather than treating imitation learning, reinforcement learning, and heuristic search as separate tracks, the team chained them into a closed loop across five BC stages and multiple PPO runs:

```mermaid
flowchart LR
    A["Public Replays"] --> B["Behavior Cloning (BC)"]
    B --> C["Critic Warmup\n(Actor Frozen)"]
    C --> D["Self-Play PPO\n(Rust Engine + JAX)"]
    D --> E["Replay Inspection\n(Identify Weaknesses)"]
    E --> F["Improve C++ Planner &\nGenerate Synthetic Games"]
    F --> B
```

1. **Behavior Cloning (`train_bc.py`):** Trained on valid executed teacher actions (pairing `obs[t]` with `action[t+1]` and mapping SELL labels to actual executed quantities 1–100), regularized with entropy bonuses and a KL penalty toward the initial policy.
2. **Critic Warmup (`warmup_critic.py`):** Before unfreezing the actor in PPO, the Transformer trunk and actor were held fixed while the value head was fitted on fresh self-play games using GAE (`gamma = 1.0`, `lambda = 0.97`).
3. **Self-Play PPO (`train_ppo.py`):** Terminal win/draw/loss rewards (`+1 / 0 / -1`), Huber value loss, and a fixed teacher regularizer `0.2 * KL(teacher || policy)` to prevent policy drift and collapse during self-play.
4. **Targeted Synthetic Demonstrations (`collect_search_replays.py`):** Inspecting BC/PPO policies revealed weaknesses in tomato-heavy and highly skewed shop draws. The team improved their C++ heuristic planner and generated 300 heuristic self-play games (600 seat trajectories: 100 standard shop distribution, 100 with ≥2 tomato shops, 100 with ≥3 identical shops) to mix with 849 public trajectories in Stage 5 BC.
5. **Net2Net Growth (`grow_model.py`):** Function-preserving depth expansion grew the 6-block bootstrap model into the 12-block (10M) model (and later forked 12 → 24 and 29 blocks for the 20M experiments) by inserting zero-initialized residual projections.

### 2.3. Two Inference Controllers: Final A vs. Final B
- **Final A (Post-Processing + Endgame Search):**
  - The neural network proposes greedy unit and market actions.
  - Rule patches (`unit_rules.py`, `shed_patch.py`) prioritize high-confidence unit actions, resolve tile/resource collisions, cap planting at available seed stock, reject useless fertilization/care, insert seed purchases when needed, and sell pre-night shed overflow above 100.
  - At dawn of Day 29, if at least 10 seconds of overage remain, control transfers to the compiled C++17 search planner (`terminal_search.so`, preloaded on Day 28) for ~6 seconds of simulated annealing over task assignments, routes, deliveries, and sales.
- **Final B (Sequential Action Masking, Pure Neural Inference):**
  - Continued from the base policy for 140 PPO updates with `--enable-sequential-masks`.
  - At each step, the policy selects the farmer's action first, then workers in fixed order (`actions/sequential.py`). Each unit's selection reserves its target tile and seed stock before the next unit's logits are masked.
  - Forced shed/terminal sales are excluded from the policy gradient loss while still contributing to value learning. Plays the entire 720-step game with the neural network and no Day 29 search handoff.

---

## 3. Where Our Project Worked Well and Where Reasoning Slipped

### 3.1. What Held Up Well
1. **Experimental Hygiene & Paired Evaluation:**
   - Using Common Random Numbers (paired seeds across both seats), tracking Winner's Curse shrinkage on disjoint holdout seeds, and moving from own-cash CVaR₅ to opponent-panel win rate after measuring that 86% of cash variance was common-mode market luck.
   - Pinning `kaggle-environments == 1.32.7` in CI with `--frozen` and building pre-flight checks (`submit.py`) that prevented `get_last_callable` bugs, crashes, and timeouts across all live submissions.
2. **Environment & Economic Mechanics:**
   - Identifying that `obs.private.shed` is observed *before* the current turn's unit actions resolve (`tests/test_observation_ordering.py`), which explained why capping `SELL` orders at observed shed stock truncated sales.
   - Mining daily replay dumps to falsify our early "egg engine" hypothesis (showing eggs accounted for only 0.6% of top-decile revenue) and quantifying the 2× yield multiplier of fertilizing strawberries.
3. **Pragmatic Exploits Within Compute Constraints:**
   - Given a solo developer working with a 2.4 s/game Python simulator, pivoting to mined route-replay (`v0.2.0`) and then grafting the 28-tape `EXP240` shop-draw library (`v0.4.0`–`v0.4.3`, reaching ~2632 rating) achieved strong results per unit of compute and engineering time.

### 3.2. Where Our Reasoning Slipped
1. **Treating an Invalid Pipeline Test (`v0.0.8`) as a Refutation of ML:**
   - `v0.0.8` scored \$2,910 against the starter bot (\$75,668). As documented in the README, that pipeline omitted `FEED`, `CARE`, and `PICKUP` from its action vocabulary, fed a 4-scalar non-Markovian state (`[day, hour, money, hands]`), overwrote co-located worker labels in a 10 × 10 grid, and weighted deterministic `argmax` rollouts without a critic.
   - We cannot claim that fixing those bugs under our CPU budget would have beaten route-replay — that is an untested counterfactual. The actual error was logical: `v0.0.8` tested a broken pipeline, so it provided zero evidence about whether a properly formulated policy could work. Concluding *"learning has no place here"* and deleting `rl/` turned a pipeline bug into an irreversible philosophical commitment.
2. **Inferring a Theoretical Ceiling from the Early Leaderboard:**
   - Observing that every reference ladder rung above tier 5 and the early #1 replay used fixed action plans led us to conclude: *"above ~\$56k this is not a strategy competition — every ladder rung above 5, and the current #1, is a fixed plan with a thin repair layer."*
   - This confused what was *fastest to build in the first weeks* with what was *possible*. Fixed-tape agents appear on leaderboards early because they require minimal infrastructure; hybrid RL + search systems take weeks of engine and training work before they surface.
3. **Reading Structural Fragility as an Engineering Obstacle Rather Than a Verdict:**
   - In Issue #30, we measured that 319 of the tape's 458 funded `BUY`/`HIRE` orders had zero forward slack, and shifting 5 order pairs out of 927 cost 7.2 percentage points of panel win rate.
   - Similarly, in `v0.5.2`, refitting the 64-draw routing table collapsed from +\$5,650/draw in-sample to −\$151/draw under leave-one-replay-out cross-validation because weeds altered the RNG stream whenever board occupancy diverged.
   - Both results were accurately measured, but we treated them as local hurdles inside the tape paradigm rather than evidence that open-loop recording cannot adapt to a coupled, closed-loop economy.
4. **Overstating the Latency Argument Against Closed-Loop Play:**
   - We argued that because crop production takes 8–12 days while shops unlock every 3 days, mid-game market adaptation was impossible and an open-loop plan was the closed-loop equilibrium.
   - While long-cycle capital allocation cannot pivot instantly to a single shop draw, closed-loop adaptation operates across many shorter horizons: resolving unit task conflicts, adjusting harvest/sale priorities to current prices, managing the 100-item shed cap before nightfall, and reacting to weed spawns without desynchronizing a brittle cash schedule.

---

## 4. Seven Principles for Future Competitions

Each principle pairs a general epistemic concept with a concrete practice for future competition workflows.

### 1. An experiment only refutes what it actually tested
In the philosophy of science (the Duhem–Quine problem), a failed experiment refutes a *bundle* of auxiliary assumptions alongside the main hypothesis. `v0.0.8` bundled Behavior Cloning with missing action tokens, a non-Markovian state, and a broken update rule.
- **Practice:** Before declaring a modeling paradigm dead, run pipeline validity gates:
  1. Verify that 100% of expert action types can be encoded and decoded losslessly.
  2. Confirm the model can overfit a single batch to near-zero loss.
  3. Verify that BC can clone the trivial starter bot and match its score. If it cannot, the bug is in the pipeline, not the paradigm.

### 2. Never assign probability zero, and never make a belief irreversible
By Cromwell's rule, assigning a prior probability of zero makes a belief immune to new evidence. Writing *"learning has no place here"* and running `rm -rf rl/` made a temporary negative result permanent.
- **Practice:** Archive paused tracks in a separate branch or module rather than deleting them, and write down an explicit revisit trigger (e.g., *"Re-examine closed-loop policies if open-loop draw routing plateaus or if top replays show state-dependent branching"*). Treat one-way doors as requiring a much higher evidentiary bar.

### 3. The leaderboard shows what has been built, not what is possible
Early leaderboard snapshots select for approaches with low setup cost (hand-tuned heuristics, mined replay tapes). Inferring the game's theoretical ceiling from early leaders mistakes development latency for strategic optimality.
- **Practice:** When analyzing the leaderboard mid-competition, explicitly ask: *"What approach takes three weeks of infrastructure before it wins its first game, and what will the field look like when teams finish building it?"*

### 4. Rigor protects you from noise, not from the wrong question
Paired-seed CRN, CVaR₅, and LOO cross-validation prevented us from shipping noise, which saved wasted submissions. However, scoring 60/60 against a fixed reference ladder rewarded memorizing that ladder (Goodhart's law), and precise A/B harnesses could only tell us that every local edit to a rigid tape broke it.
- **Practice:** Alongside within-distribution win rate, measure **structural sensitivity**: how gracefully the agent degrades under small perturbations (e.g., delaying an order by 1 step, injecting random no-ops, or shifting shop schedules). High fragility (such as 70% of orders having zero slack) is a signal to change representations, not to search harder over the same representation.

### 5. Match the method to how much you can know in each phase
Different phases of a sequential game have different horizon lengths, stochasticity, and error costs:
- **Long horizon, high stochasticity (Days 0–28):** Future shop draws and opponent actions are uncertain; a learned closed-loop policy amortizes risk across states.
- **Short horizon, deterministic dynamics (Day 29):** No new shops unlock and no newly planted crops can mature; combinatorial local search (C++ simulated annealing) can directly maximize terminal cash.
- **Zero-horizon invariants (Every turn):** Shed capacity (100), seed stock limits, and tile collisions have known, exact rules; hard action masks or post-processing rules prevent unforced errors with zero search cost.
- **Practice:** Map out the game's phases by horizon length, stochasticity, and error cost at the start of the project, and assign search, learned policies, and hard rules to the specific layers where each fits.

### 6. Estimate what each approach can reach with your resources
Simulator throughput bounds what algorithms are viable. At 2.4 s/game in Python on 12 cores (~3.6 × 10⁵ games/day max), billion-step PPO is unreachable without changing the simulator, whereas a 4-person team with a Rust engine and GPUs can run >10⁷ games.
- **Practice:** In Week 1, benchmark environment steps per second and multiply by available compute-days to get a hard episode budget. Make an explicit upfront decision: either invest engineering time to build a compiled simulator (Rust/C++/JAX) to unlock high-sample-complexity methods, or deliberately commit to sample-efficient methods that fit the Python budget.

### 7. Keep rival hypotheses alive in parallel
The winning team organized around competing hypotheses in parallel: `@morim3` scaled self-play games on the 10M model, `@msdsm` scaled parameters on the 20M model, `@BergBuch` built the C++ search planner, and `@qistripute` built rule-based patches and analyzed replays. Even Final A and Final B hedged against each other (post-processing + endgame search vs. sequential masks + pure neural play).
- **Practice:** Avoid putting 100% of project time into a single architectural bet. Even when working solo, reserve ~20% of time and one of the two final submission slots for the strongest rival hypothesis.
