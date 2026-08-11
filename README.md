# Kaggriculture agent

A two-tier agent for the Kaggle [Kaggriculture](https://www.kaggle.com/competitions/kaggriculture)
competition, plus a local arena for measuring it, a macro-strategy search, and a
submission pipeline.

Verified against **kaggle-environments 1.32.3** (kaggriculture spec version `0.1.0`).
Python 3.12+.

The agent itself is hand-written and standard library only. There was a neural
version; it lost to the environment's own starter bot and is documented below as the
most useful negative result in the repo.

```
main.py                       # THE submission — hand-written, stdlib only, no weights
local_arena.py                # match runner, metrics, decision logs, A/B + sweep rig
submit.py                     # pre-flight, submit, poll, history
search/space.py               # the ~23-number macro vector that is actually searched
search/objective.py           # P(win) + CVaR scoring (NOT mean cash)
search/cem.py                 # cross-entropy-method search + loss-tail diagnosis
search/evolution.py           # population-based search over the same vector
search/harness.py             # paired-seed match harness used by both searches
search/smoke_test.py          # unit tests for the search stack
opponents/adaptive.py         # sparring partner (see "Opponents" for provenance)
opponents/vX_Y_Z.py           # every submitted version, kept as a sparring partner
scripts/probe_agent.py        # schema probe; writes logs/probe_schema.json
scripts/sync_opponent.py      # pre-commit hook that versions main.py into opponents/
kaggle_credentials.example.py # copy to kaggle_credentials.py (gitignored)
logs/                         # replays, decision logs, submission history, search output
```

## Setup

```bash
uv sync
```

The submission itself needs nothing installed: `main.py` is standard library only.
`kaggle-environments` and `kaggle` are for the arena and the submit pipeline; `numpy`
is used by the tooling, not by the agent.

## Quick start

```bash
uv run python scripts/probe_agent.py                                    # confirm the schema
uv run python local_arena.py --agent main.py --opponent baseline --episodes 30
uv run python local_arena.py --agent main.py --opponent opponents/adaptive.py --episodes 30
uv run python -m search.cem --diagnose 40                                # loss-tail report
uv run python -m search.cem --iterations 12 --pop 16 --episodes 8       # macro search
uv run python submit.py --dry-run
```

---

## Implementation Notes

The following mechanics are useful to keep in mind when designing the agent:

- **16 hands cost $2,583/day** (hire cost is `farmHandCostMult * fib(n)`).
- **The opponent's whole farm is public** — money, tiles, hands, quadrants. Only shed/seeds/carried inventories are hidden.
- **Shed cap 100 applies to mid-day `PLACE` too**, so you cannot stockpile in carried inventories.
- **Compute budget:** `actTimeout: 1.0`s/turn, plus a **60s overage bank** per episode.
- **The env loads the LAST callable defined in your file**, not the one named `agent`
  (`agent.py: get_last_callable` → `[v for v in env.values() if callable(v)][-1]`).
  A helper defined below `agent` silently becomes the agent and every turn errors.
  This cost the first working build 1,436 ERROR statuses. `submit.py` asserts against it.

### The economics that decide the strategy

Measured by walking the env's own `market_price` curve (cumulative revenue for selling
N units into a virgin market):

| Product | N=50 | N=100 | N=200 | N=400 | N=1600 |
| --- | --- | --- | --- | --- | --- |
| **EGG** | $2,244 | $4,371 | $8,510 | $16,559 | **$62,421** |
| MELON | $12,098 | $21,721 | $26,527 | $26,727 | $27,927 |
| WHEAT | $1,127 | $2,193 | $4,293 | $8,313 | $31,443 |
| WOOL | $7,655 | $7,969 | $8,069 | $8,269 | $9,469 |
| MILK | $5,430 | $6,205 | $6,305 | $6,505 | $7,705 |
| CARROT | $1,482 | $2,738 | $4,832 | $7,853 | $11,438 |
| TOMATO | $2,411 | $4,318 | $7,221 | $10,453 | $12,199 |
| STRAWBERRY | $3,648 | $3,847 | $3,947 | $4,147 | $5,347 |

**Egg is the only product that scales.** Its glut curve is `log` with target 0.20, so
the 1,600th egg still fetches $37. Everything else saturates: melon is a ~$26.5k pot,
wool ~$8k, milk ~$6.5k, strawberry ~$5k. So the shape of the game is:

1. **A one-shot melon opening.** 18 melon tiles → ~108 melons ≈ **$21.7k by day 12**,
   on $1,440 of seed. Nothing else in the game returns 15× in 12 days. A second cycle
   lands around day 26 but is worth much less, because the first one already moved the price.
2. **An egg engine for the back half.** A goose costs $300, matures on day 4, then yields
   2 eggs/day with `CARE`. Eggs never crash, so this is the only unbounded income.
3. **Capped side pots** (wool, milk) raced with the opponent — 2 sheep and 2 cows
   roughly saturate them.
4. **Wheat as feed infrastructure.** A wheat tile yields ~1/day and a goose eats exactly
   1/day, so the self-sufficient ratio is 1:1. Growing beats buying: the buy side ramps
   on `sqrt` (draining 1,800 units takes wheat from $26 to $67), and surplus still sells
   at ~$20 because the sell side is `log`.

Fertilizer is deliberately ignored: it **cannot be sold**, melon already hits its
`max_yield` of 6 on watering alone, and on wheat the whole play is +2 units for two
actions — worse than a `CARE`. Collecting it anyway cost 36 of the 100 shed slots.

---

## Agent architecture

`main.py` is the whole agent: hand-written, standard library only, no learned weights.

**Layer A — `StrategicPlanner`** (macro, once per turn)
- `plan_roles` assigns every unlocked non-shed tile a role (`COOP`, `PASTURE_SHEEP`,
  `PASTURE_COW`, `MELON`, `STRAWBERRY`, `WHEAT`), nearest-the-shed first since animals
  need wheat carried out daily. Coops are built **just-in-time against cash**.
- `market_orders` walks a **capital priority ladder**. Cash owed to the egg engine
  (vacant coops + feed for birds already owned) is ring-fenced as `engine_claim`;
  land and cash-crop seed can only draw on what is left.
- `target_hands` hires to `MAX_HANDS`, throttled by `HIRE_CASH_FRACTION`. Note that
  `hires_today` resets every day, so hands are re-hired daily — pausing hiring is not
  a saving, it is a shutdown (measured: −65.9%).
- Everything sellable is sold on sight. Withholding to defend a price measured worse
  and unsold stock scores nothing.

**Layer B — `SpatialScheduler`** (micro, every turn)
- `build_tasks` enumerates every useful (tile, op) with a priority. Survival outranks
  yield: a plant at `consecutive_unwatered >= 1` dies tonight, an animal at
  `consecutive_unfed >= 1` escapes tonight and is unrecoverable.
- Two pre-passes handle work the generic loop structurally cannot:
  `_ferry_animals` (PICKUP at shed → walk → PLACE is a two-tile chain, and a `PLACE`
  task is only assignable to a unit already carrying the animal) and `_provision_feed`
  (`FEED` consumes wheat from the acting unit's own inventory).
- `assign` is **greedy nearest-capable unit**, one task per unit, one unit per tile.
  **Deliberately not MAPF** — units may legally share tiles, so there are no collisions
  to resolve. An exactly optimal assignment ships alongside it and is switched **off**;
  see "Optimal assignment loses to greedy".

**`MarketAnalyzer`** replicates the price curve exactly, and derives
`units_sellable_above(item, floor)` by walking the curve the way the interpreter does.

**`OpponentTracker`** reads their public farm to classify them
(`EARLY_RUSH` / `ANIMAL_LONGTERM` / `MELON_MAXXER` / `BALANCED`) and to forecast
imminent supply from visible on-tile yield and maturities. Their shed is hidden, so
every forecast is a **lower bound** — stated in the code. It feeds the decision log
only; four separate attempts to let it drive behaviour all measured worse.

**Safety guard** — blanket `try/except` returning a legal no-op, a time check against
`actTimeout`, and `_validate_unit_op` dropping anything the interpreter would not
recognise. Across every run reported here: **0 crashes, 0 timeouts, 0 invalid statuses.**

---

## Results

v0.0.9, seats alternated each episode, 720 steps, seeded and reproducible.

| Opponent | Eps | Mean cash | Median | Min | Win rate | Crashes | p95 turn |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline` (`starter`) | 30 | **$73,032** | $72,776 | $46,107 | **100%** (30W 0L) | 0 | 0.30 ms |
| `adaptive` | 30 | **$68,872** | $72,361 | $46,362 | **100%** (30W 0L) | 0 | 0.28 ms |
| `v0.0.6` (previous agent) | 30 | $57,172 | $57,354 | $23,258 | **70.0%** (21W 9L) | 0 | 0.29 ms |
| `mirror` (self-play) | 30 | $54,093 | $50,947 | $25,382 | 0W **27T** 3L | 0 | 0.31 ms |

All acceptance criteria pass on every non-mirror opponent: zero crashes, zero
timeouts, zero invalid statuses, p95 turn compute under 0.1% of the 1s budget (the
gate is 50%), and it beats every opponent on mean cash over 30 episodes.

Self-play resolving into 27 exact ties is the expected result for two identical
deterministic agents on a shared seed. The three losses come from the interpreter
processing atomic orders (`HIRE`, `BUY_LAND`) in player-index order. Self-play cash
is ~25% lower than either scripted matchup, because both sides race for the same
finite melon pot — **that is the more realistic guide to leaderboard scoring** than
the numbers against `baseline`/`adaptive`.

### v0.0.8 replaced this agent with a neural network and lost 26×

Worth stating up front because it is the most expensive lesson in the repo. v0.0.8
dropped `StrategicPlanner`/`SpatialScheduler` entirely and shipped a behaviour-cloned
policy emitting atomic actions, distilled from PyTorch into base64 numpy weights.

| Agent | vs `baseline`, 4 paired seeds | Record |
| --- | --- | --- |
| v0.0.8 (behaviour-cloned atomic actions) | **$2,910** | 0W 4L |
| v0.0.6 (this architecture) | **$75,668** | 4W 0L |

It lost to the environment's own one-farmer starter bot. The cause was structural,
not a training budget problem, and it is worth spelling out because it generalises:

1. **The action vocabulary had no `FEED`, `CARE` or `PICKUP` token.** `BASE_ACTIONS`
   was `PASS/N/S/E/W/WATER/HARVEST/DEMOLISH` plus `PLACE_*`, and the encoder mapped
   anything unknown to class 0 = `PASS`. So every expert `FEED` in the training data
   was labelled `PASS`, and the egg engine — the only unbounded income in the game —
   was **inexpressible** by construction.
2. **The observation was not Markov.** The global vector was
   `[day, hour, money, hands_count]`. No market prices, no glut counters, no shed,
   no animals, no maturity, no `consecutive_unfed`. A market policy cannot be learned
   from a state that does not contain prices.
3. **Labels collided.** Hand actions were written into a 10×10 grid keyed by position,
   but units may share a tile, so co-located hands overwrote each other; and ~84 of
   100 cells were `PASS`, so cross-entropy converged on "PASS everywhere".
4. **The online loop was not a policy gradient.** Rollouts were taken with `argmax`,
   so the policy was deterministic and nothing was sampled from it. `∇log π(a|s)·A`
   requires `a ~ π`; weighting `argmax` actions by an episode-level advantage over 20
   episodes measures seed luck, not policy quality — with no critic, no GAE and no
   baseline over 720 steps of credit.

The whole atomic-action path has been deleted: `agent_torch.py`,
`rl/architecture.py`, `rl/action_space.py`, `rl/dataset_builder.py`,
`rl/train_offline.py`, `rl/train_online.py`, `rl/offline_bc.py`,
`rl/export_to_numpy.py`, `rl/numpy_inference.py`, `rl/distill_to_main.py`,
`scripts/build_submission.py`, `scripts/train_pipeline.py`, the `torch`
dependency, and `opponents/v0_0_7.py` / `opponents/v0_0_8.py` — the two agent
snapshots that carried base64 network weights. The behaviour-cloning datasets,
`.pt` checkpoints and elite-trajectory store under `logs/` went with them
(~600 MB). Everything is recoverable from git history at tags `v0.0.7`/`v0.0.8`.

### Optimal assignment loses to greedy

Layer B is a weighted bipartite matching, and matching has an exact
polynomial-time solution, so "replace greedy with Hungarian" looks like free
money. It is not. `main.py` ships a complete Jonker-Volgenant solver
(`_hungarian_min_cost`, brute-force verified on 300 random matrices) and a
priority-tiered optimal assignment (`SpatialScheduler._assign_optimal`) behind
`FLAGS["HUNGARIAN_ASSIGN"]`, and the flag is **off**:

| Objective | Result |
| --- | --- |
| flat max-value, `priority / (1 + travel)` | **−37.6%** (8 seeds, 0/8) |
| flat max-value, `priority − 8·travel` | **−38.0%** (8 seeds, 0/8) |
| priority-tiered, min total travel inside each tier | **−31.3%** vs `baseline` (p~0.0, 3/30), **−40.4%** vs `adaptive` (p~0.0, 3/30) |

Three things came out of chasing this, and they are the transferable part:

1. **Priorities are deadlines, not utilities.** Any objective that trades
   priority against travel will let a cheap nearby chore outrank a plant that
   dies tonight. Only a lexicographic objective (tier first, travel only inside
   a tier) is even defensible here.
2. **Do not collapse tasks to one per tile.** An animal tile carries `FEED`,
   `HARVEST` and `CARE`. Keeping only the top-priority one drops the fallback:
   when nobody is carrying wheat the `FEED` is unassignable, and the tile must
   stay eligible for its `HARVEST`.
3. **Greedy has an emergent property the matching destroys.** Greedy leaves its
   *leftover* units clustered near the shed, and every animal and every sack of
   feed enters the farm through a shed `PICKUP`. Minimising total travel scatters
   the leftovers to the perimeter, where `_logistics` can only return `PASS`.
   Patching that (staging idle units at the shed) recovered nearly half the loss
   — −44.3% → −20.2% — which confirms the mechanism but does not close the gap.

The lesson is not "optimisation is bad". It is that the objective the greedy rule
implicitly optimises is not the one that was written down, and the written one is
worse.

### Tuning status

Every constant in `main.py` is annotated `TUNED`, `UNPROVEN`, `PLACEHOLDER` or `DERIVED`.
`TUNED` means a paired-seed A/B measured it. What graduated:

| Knob | Result | Evidence |
| --- | --- | --- |
| `HIRE_HANDS` | **+517.7%** | p~0.0, better on 20/20 seeds |
| `EXPAND_LAND` | **+199.1%** | p~0.0, better on 19/20 |
| `PREMIUM_LIVESTOCK` | **+17.8%** | p~0.020, 14/20 |
| `ANIMAL_CARE` | **+31.0%** in self-play; n.s. vs scripted opponents | p~0.021, 22/30 (mirror) |
| the 23-number macro vector | **21W–9L** head-to-head vs v0.0.6 | 30 paired seeds, p~0.021 |

The macro vector is no longer hand-swept one constant at a time; see
"Where learning belongs". The v0.0.9 values came out of a CEM search and were then
validated head-to-head against the version they were derived from.

### Every PvP heuristic tried has lost, and all four were deleted

This is the most useful negative result here, so it is worth stating plainly. The
opponent's **entire farm is public** — money, tiles, hands, quadrants — so the
adversarial features in the environment specification are all *implementable*. Four
have been implemented and A/B'd, and all four measured worse or inert:

| Heuristic | Result | Verdict |
| --- | --- | --- |
| `PRICE_FLOOR_SELLING` — withhold capped goods to defend their price | **−1.8%** in self-play (p~0.00014, better on 6/30) | deleted |
| `PREDATORY_TIMING` — dump ahead of their forecast harvest | **exactly 0 delta** over 80 paired seeds | deleted |
| `OPPONENT_ADAPTIVE` — cede melon land when they contest melon | **−26.9%** vs adaptive (p~0.0, better on **0/30**) | deleted |
| `ENDGAME_POSTURE` — lock in / gamble based on the terminal cash gap | **−10.7%** vs baseline (p~1e-05, 1/30), **−8.2%** vs adaptive (p~0.001, 2/30) | deleted |

The pattern is consistent, and the right explanation is **latency**, not "tempo" in
the hand-waving sense. Every production decision in this game pays out 8–12 days
after it is taken, and the opponent's board is only informative about decisions they
have *already* made. Information whose reaction lag exceeds the horizon over which
it is actionable has value zero, and acting on it costs the tempo spent switching.
Formally: the observation lag is roughly equal to the action's payoff lag, so the
open-loop plan **is** the closed-loop equilibrium.

That predicts exactly where opponent-awareness *could* pay: decisions with no lag.
There are two, and both are now measured. Selling is instantaneous — and
`PREDATORY_TIMING` returned exactly zero delta over 80 seeds, because the agent
already sells everything on sight and there is no timing left to optimise. The
terminal cash comparison is also lag-free in the last week — and `ENDGAME_POSTURE`,
which locked down capital spend when ahead and extended the livestock windows when
behind, lost 10.7%. So the lag argument survives its own strongest test.

Withholding stock is worst of all, because unsold inventory scores $0 and melon's
price never recovers enough for the held units to clear.

`OpponentTracker` is retained (composition, cash, labour, profile label) but it feeds
**the decision log only** — it does not change what the agent does. That is stated in
its docstring so nobody mistakes it for a live input.

### Known gaps

- **Worker PASS rate 23–33%.** Assignment leaves a quarter to a third of unit-turns
  doing nothing once the task list is exhausted mid-day. Two attempts to close it
  are already recorded as failures (see "Optimal assignment loses to greedy", and
  `IDLE_PREPOSITION` below), so this is harder than it looks. More coops would absorb
  it, but they are cash-gated.
- `IDLE_PREPOSITION` — walking otherwise-idle units toward the shed instead of
  `PASS` — looked promising at 8 seeds (+3.2%, better on 7/8) and then measured
  **−16.4%** vs baseline (p~4e-05, 4/30) and **−8.7%** vs adaptive (p~0.015, 11/30)
  at 30. Deleted. Filed here mostly as a reminder that 8 seeds is not a measurement.
- **Tail.** The $2,244 collapse reported for v0.0.5 is gone: over 24 paired
  seat/seed evaluations the minimum is $36,840 and CVaR@25% is $45,354
  (`python -m search.cem --diagnose 40`). Self-play min is $25,382, which is
  the number to watch.
- `FERRY_MAX_UNITS`, `ANIMAL_BACKLOG_CAP` and `WHEAT_CARRY_PER_UNIT` are still
  `PLACEHOLDER` — reasoned, not swept, and not in the searched vector.

---

## What game theory actually says about this game

Worth writing down, because the obvious textbook models give the right advice for
the wrong reasons, and the wrong reasons predict the wrong next experiment.

**It is not Cournot.** Cournot has firms choosing quantities each period against a
price that depends on *current* total supply, and its equilibrium involves restraint.
Here the glut counter is **cumulative and monotone** — price is a stock you deplete,
not a flow you influence. The correct model is common-pool extraction (Hotelling with
rivalry): the pot goes to whoever draws it down first, restraint is strictly
dominated because the rival simply takes what you leave, and the finite horizon adds
a *second*, independent reason not to withhold (unsold stock scores $0). That matches
`PRICE_FLOOR_SELLING` measuring −1.8%.

**It is not Chicken.** Chicken is anti-coordination: mutual aggression is the
catastrophe. Here, if both players plant melon nobody crashes — the ~$26.5k pot is
simply split. The right model is a **Tullock contest**: your share is roughly your
share of production, so over a wide region the best response to more opponent effort
is *more* effort, not less. That is why `OPPONENT_ADAPTIVE` — ceding melon land when
contested — measured −26.9% on 0/30 seeds.

**Eggs are not a "dominant strategy".** Dominance is a property of strategies, not of
products. The precise statement is stronger and more useful: because EGG's glut curve
is `log` with target 0.20, its marginal revenue is nearly independent of *total*
supply, so the two players' payoffs are **separable** in the egg dimension. The egg
sub-game has no strategic interaction at all — it is a single-agent MDP wearing a
Markov-game costume, which is exactly why it can be optimised without modelling the
opponent.

**Why playing deaf is correct** is a latency argument, not a tempo slogan: see the
PvP table above. Information whose reaction lag exceeds the horizon over which it is
actionable has value zero, so the open-loop plan *is* the closed-loop equilibrium.
The useful part of that framing is that it makes a falsifiable prediction — that
opponent-awareness can only pay on zero-lag decisions — and both zero-lag decisions
in this game (sell timing, endgame posture) have now been tested and both failed.

**Where the textbook framing was actually load-bearing** is none of the above; it is
the scoring rule. Pairwise ranking means the objective is P(win), not E[cash], and
that changed a real decision: it is why `search/objective.py` scores CVaR of the margin,
and why the parameter set that ships is the one that won a head-to-head rather than
the one with the best mean.

---

## Where learning belongs

The game decomposes cleanly, and the two halves want completely different tools:

| | Layer A (macro) | Layer B (micro) |
| --- | --- | --- |
| Decision | what to produce, when to buy, how much labour | which unit does which task |
| Size | ~23 numbers, changes on a daily timescale | 17 units × ~50 tasks, every turn |
| Strategic content | all of it | none |
| Known algorithm | none | weighted bipartite matching, exact, polynomial |

So: **learning goes where there is no closed form.** Cloning atomic actions puts a
noisy approximator on top of a problem that has an exact solution and throws away
the economics; that is v0.0.8, and it cost 26×. *AlphaStar Unplugged* works because
StarCraft's micro layer has no exact solution, the observation is complete, and the
data is ~10⁶ games. None of those three hold here.

### The objective is P(win), not E[cash]

The competition ranks agents pairwise, so a $1 win and a $50,000 win score the same.
`search/objective.py` therefore scores a smooth P(win) surrogate plus a CVaR term — the
mean of the worst 25% of seeds — so that fixing a collapsing seed is worth more than
improving an already-won one.

**What the tail is measured on turned out to matter more than the optimiser.** The
first version took CVaR of *own cash*. Against a weak scripted opponent every seed is
won by a mile, so own-cash variance is noise; the search duly bought tail safety with
production, raised the worst seed by 6.5% — and the resulting agent lost **8W–22L**
head-to-head against the very strategy it was derived from. Scoring CVaR of the
*margin* `(me − opp)` instead, and adding the frozen incumbent to the opponent pool,
produced the v0.0.9 vector, which wins that head-to-head **21W–9L**.

### Macro search

```bash
# loss-tail report for the agent exactly as it stands
uv run python -m search.cem --diagnose 40

# cross-entropy-method search; always spar against the frozen incumbent
uv run python -m search.cem --iterations 12 --pop 16 --episodes 8 \
    --opponents baseline,opponents/v0_0_6.py

# population-based alternative over the same vector and objective
uv run python -m search.evolution --generations 15 --pop-size 12 --episodes 8
```

Both searches evaluate every candidate in a generation on the **identical seed set**
and on **both seats** (common random numbers), so a difference between candidates is
strategy and not luck. `search/space.py` reads the agent's live constants, so a
search starts from the file as it actually stands rather than from range midpoints,
and writes variants by rewriting a copy — no tuning hooks inside `main.py`.

**Always validate a search result head-to-head before adopting it.** A CEM candidate
that looked better on every scripted opponent lost 8W–22L against the incumbent; the
one that shipped was checked at 30 paired seeds against `opponents/v0_0_6.py` first.

```bash
uv run python -m unittest search.smoke_test
```

---
## Local arena

```bash
# metrics + decision logs
uv run python local_arena.py --agent main.py --opponent baseline --episodes 30 --log-decisions

# graduate a PLACEHOLDER: paired-seed A/B with a significance check
uv run python local_arena.py --agent main.py --opponent baseline --episodes 30 --ablate EXPAND_LAND

# sweep a numeric constant
uv run python local_arena.py --agent main.py --opponent baseline --episodes 30 --sweep MAX_HANDS=10,12,16

# self-play, and head-to-head against a frozen previous version
uv run python local_arena.py --agent main.py --opponent mirror --episodes 30
uv run python local_arena.py --agent main.py --opponent opponents/v0_0_6.py --episodes 30

# save and inspect replays
uv run python local_arena.py --agent main.py --opponent baseline --episodes 3 --save-replays 3
uv run python local_arena.py --replay logs/match_run_0042.json
```

Reported per run: mean/median/min/max/sd final cash, win rate, crashes, timeouts,
invalid statuses, per-turn compute (p50/p95/max), action no-op rate with a per-op
breakdown, shed overflow losses, worker idle rate, market orders dropped to the
10/turn cap, and which heuristics fired.

A 30-episode A/B takes under a minute. **Run 30, not 8** — `IDLE_PREPOSITION` read
+3.2% on 8 seeds and −16.4% on 30.

Three details worth knowing:

- **Seats alternate** every episode (`swap`), so neither the player-index advantage nor
  the market's player-order tie-breaking biases a result.
- **Variants are generated by rewriting a copy of the agent file**, so `--ablate` and
  `--sweep` need no tuning hooks inside `main.py`.
- Shed overflow and no-op counts are invisible in the observation, so the arena wraps
  the interpreter's own `_drop_inventories_to_shed` / `_apply_unit_action` and
  attributes them to one player by object identity.

### Opponents

`baseline` is the env's built-in `starter` (one farmer, one carrot tile). `random` and
`pass` are also built in.

`opponents/adaptive.py` is **not** a port of the public
"adaptive-farming-strategy-for-kaggriculture" notebook. That source could not be
retrieved: kaggle.com renders competition and notebook pages in JS and returns no
usable HTML to a plain fetch. It implements the same *idea*, and it exists because
`starter` is trivially beaten.

`opponents/vX_Y_Z.py` is every previously submitted agent, written automatically by
the `sync-opponent` pre-commit hook. These are the most useful sparring partners in
the repo: they are the only ones that contest the same pots at the same tempo, and
**a change that does not beat the previous version head-to-head is not an
improvement**, whatever it does to `baseline`. `v0_0_7` and `v0_0_8` are absent on
purpose — both embedded network weights, and `v0_0_8` loses to `starter`.

`opponents/leaderboard_replay.py` replays a downloaded 720-step Kaggle episode
turn-by-turn, so a downloaded top-leaderboard match can be used as a sparring
partner:

```bash
uv run python local_arena.py --agent main.py --opponent logs/episode-89987566-replay.json --episodes 10
```

---

### Agent Failure Analysis (`scripts/examine_agent.py`)

Fetches your active Kaggle matches for a given submission description, isolates the
losses, and downloads those replays into `logs/failures_<version>/` for debugging.

```bash
uv run python scripts/examine_agent.py v0.0.9
uv run python scripts/examine_agent.py v0.0.9 --limit 5
```

---

### Versioning and Opponent Sync

When you update the `AGENT_VERSION` in `main.py` and commit, a `pre-commit` hook automatically runs `scripts/sync_opponent.py`. This copies your `main.py` into the `opponents/` directory as `vX_Y_Z.py` and stages it. This ensures that every submitted agent version remains available as a sparring partner in `local_arena.py`.

---

## Submitting

```bash
cp kaggle_credentials.example.py kaggle_credentials.py          # then edit
uv run python submit.py --dry-run                               # all checks, no submission
uv run python submit.py                                         # check, submit, poll
```

Pre-flight hard-fails on: a disallowed import in `main.py`; a missing/wrong-arity
`agent`; `agent` not being the last callable defined; the agent failing to load the way
the env loads it; or any crash, timeout or invalid status in the smoke test. Credentials
are exported to the environment before `kaggle` is imported, so no `~/.kaggle/kaggle.json`
is needed, and the key is never printed or written to the history log.

> **Security.** `kaggle_credentials.py` grants full API access to your Kaggle account —
> it can submit, download and delete on your behalf. It is in `.gitignore`; keep it out
> of version control and out of any notebook you publish.

After submitting:

```bash
kaggle competitions submissions kaggriculture
kaggle competitions episodes <SUBMISSION_ID>
kaggle competitions replay <EPISODE_ID>
kaggle competitions logs <EPISODE_ID> 0
kaggle competitions leaderboard kaggriculture -s
```

Note: you must accept the rules at
https://www.kaggle.com/competitions/kaggriculture ("Join Competition") before any
submission will be accepted.
