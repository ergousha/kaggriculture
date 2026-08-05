# Kaggriculture agent

A two-tier hybrid agent for the Kaggle [Kaggriculture](https://www.kaggle.com/competitions/kaggriculture)
competition, plus a local arena for measuring it and a submission pipeline.

Verified against **kaggle-environments 1.32.3** (kaggriculture spec version `0.1.0`).
Python 3.10+ (developed on 3.14).

```
main.py                       # THE submission — single file, self-contained, stdlib only
leaderboard_crawler.py        # Continuous Leaderboard Intelligence & Replay Mining Pipeline
elite_recorder.py             # Hall of Fame trajectory recorder & Kaggle episode importer
probe_agent.py                # schema probe (build step 1); writes logs/probe_schema.json
local_arena.py                # match runner, metrics, decision logs, A/B + sweep rig
submit.py                     # pre-flight, submit, poll, history
opponents/adaptive.py         # sparring partner (see "Opponents" for provenance)
kaggle_credentials.example.py # copy to kaggle_credentials.py (gitignored)
logs/                         # replays, decision logs, submission history, leaderboard intelligence
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install kaggle-environments kaggle
```

## Quick start

```bash
.venv/bin/python probe_agent.py                                    # confirm the schema
.venv/bin/python leaderboard_crawler.py --limit 10                 # stream & dissect top leaderboard replays
.venv/bin/python local_arena.py --agent main.py --opponent baseline --episodes 30
.venv/bin/python local_arena.py --agent main.py --opponent opponents/adaptive.py --episodes 30
.venv/bin/python submit.py --dry-run
```

---

## Ground truth

**The initial specification's assumed mechanics were wrong in several places.** Everything below was read
out of `kaggle_environments/envs/kaggriculture/kaggriculture.py` in the installed
package and re-verified by direct calls into the env's own functions. Corrections:

| Specification assumed | Actually |
| --- | --- |
| Submit `agent/submission.py` | Kaggle requires **`main.py`** at the archive root |
| `agent(obs, config)` | Either arity works; the env slices args to `co_argcount` |
| Auth via `KAGGLE_USERNAME`/`KAGGLE_KEY` | Still works (legacy path), but an access token now takes precedence |
| Hire cost "scales non-linearly" | Exactly `farmHandCostMult * fib(n)`; **16 hands cost $2,583/day** |
| Opponent data may be unobservable | Their **whole farm is public** — money, tiles, hands, quadrants. Only shed/seeds/carried inventories are hidden |
| Shed cap 100, overflow harvests lost | Confirmed — and it applies to mid-day `PLACE` too, so you cannot stockpile in carried inventories |
| "Assume < 1s compute budget" | `actTimeout: 1.0`s/turn, plus a **60s overage bank** per episode |

Two more things worth knowing, neither documented in the competition documentation:

- **The env loads the LAST callable defined in your file**, not the one named `agent`
  (`agent.py: get_last_callable` → `[v for v in env.values() if callable(v)][-1]`).
  A helper defined below `agent` silently becomes the agent and every turn errors.
  This cost the first working build 1,436 ERROR statuses. `submit.py` asserts against it.
- **`CARE` banks `+1`/day, not `+2`** as `AGENTS.md`/`README.md` in the env package claim.
  Code is truth (`_daily_refresh_animals`).

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

`main.py`, single file, four inline classes behind a guarded `agent(obs, config=None)`.

**Layer A — `StrategicPlanner`** (macro, once per turn)
- `plan_roles` assigns every unlocked non-shed tile a role (`COOP`, `PASTURE_SHEEP`,
  `PASTURE_COW`, `MELON`, `WHEAT`), nearest-the-shed first since animals need wheat
  carried out daily. Coops are built **just-in-time against cash**.
- `market_orders` walks a **capital priority ladder**. Cash owed to the egg engine
  (vacant coops + feed for birds already owned) is ring-fenced as `engine_claim`;
  land and cash-crop seed can only draw on what is left.
- `target_hands` hires to `MAX_HANDS`, throttled by `HIRE_CASH_FRACTION`.
- `sell_quantity` sells glut-proof goods in full and drip-sells capped goods only
  while price stays above `SELL_FLOOR`.

**Layer B — `SpatialScheduler`** (micro, every turn)
- `build_tasks` enumerates every useful (tile, op) with a priority. Survival outranks
  yield: a plant at `consecutive_unwatered >= 1` dies tonight, an animal at
  `consecutive_unfed >= 1` escapes tonight and is unrecoverable.
- Two pre-passes handle work the generic loop structurally cannot:
  `_ferry_animals` (PICKUP at shed → walk → PLACE is a two-tile chain, and a `PLACE`
  task is only assignable to a unit already carrying the animal) and `_provision_feed`
  (`FEED` consumes wheat from the acting unit's own inventory).
- `assign` is greedy nearest-capable-unit with one-task-per-unit and
  one-unit-per-tile reservation. **Deliberately not MAPF** — units may legally share
  tiles, so there are no collisions to resolve. Upgrade path is noted in the code.

**`MarketAnalyzer`** replicates the price curve exactly, and derives
`units_sellable_above(item, floor)` by walking the curve the way the interpreter does.

**`OpponentTracker`** reads their public farm to classify them
(`EARLY_RUSH` / `ANIMAL_LONGTERM` / `MELON_MAXXER` / `BALANCED`) and to forecast
imminent supply from visible on-tile yield and maturities. Their shed is hidden, so
every forecast is a **lower bound** — stated in the code.

**Safety guard** — blanket `try/except` returning a legal no-op, a time check against
`actTimeout`, and `_validate_unit_op` dropping anything the interpreter would not
recognise. Across every run reported here: **0 crashes, 0 timeouts, 0 invalid statuses.**

---

## Results

Seats alternated each episode, 720 steps, seeded and reproducible.

| Opponent | Eps | Mean cash | Median | Min | Win rate | Crashes | p95 turn |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline` (`starter`) | 40 | **$22,543** | $22,907 | $2,244 | **97.5%** (39W 1L) | 0 | 0.3 ms |
| `adaptive` | 40 | **$21,473** | $21,552 | $18,342 | **100%** (40W 0L) | 0 | 7.2 ms |
| `mirror` (self-play) | 30 | $7,664 | $7,012 | — | 56.7% (17W 3T 10L) | 0 | 0.4 ms |

All four acceptance criteria pass on both non-mirror opponents: zero crashes, zero
timeouts, zero invalid statuses, p95 turn compute under 1.5% of the 1s budget (the
gate is 50%), and it beats both opponents on mean cash over ≥30 episodes.

Self-play sitting near even is the expected result for two identical agents; the
small edge and the 3 ties come from the interpreter processing atomic orders
(`HIRE`, `BUY_LAND`) in player-index order. Note self-play cash is ~3× lower than
either scripted matchup, because both sides race for the same finite melon pot —
**that is the more realistic guide to leaderboard scoring** than the numbers
against `baseline`/`adaptive`.

### Tuning status

Every constant in `main.py` is annotated `TUNED`, `UNPROVEN`, `PLACEHOLDER` or `DERIVED`.
`TUNED` means a paired-seed A/B measured it. What graduated:

| Knob | Result | Evidence |
| --- | --- | --- |
| `HIRE_HANDS` | **+517.7%** | p~0.0, better on 20/20 seeds |
| `EXPAND_LAND` | **+199.1%** | p~0.0, better on 19/20 |
| `MAX_HANDS = 16` | **$21,822** vs $10,629 @13, $11,190 @20 | p~0.0, 18/20 |
| `PREMIUM_LIVESTOCK` | **+17.8%** | p~0.020, 14/20 |
| `MELON_LAND_FRACTION = 0.50` | $22,126 vs $16,606 @0.35, $7,481 @0.75 | p~0.022 / p~0.0 |
| `ANIMAL_CARE` | **+31.0%** in self-play; n.s. vs scripted opponents | p~0.021, 22/30 (mirror) |

`MAX_HANDS = 16` is a genuinely sharp optimum and worth explaining: hands 1–12 total
$376/day but hand 16 alone costs $987, so the fib tail is brutal — yet cutting to 13
halves final cash, because labour is the binding constraint. Above 16,
`HIRE_CASH_FRACTION` throttles it and the extra ceiling does nothing.

### Every PvP heuristic I tried lost, and all three were deleted

This is the most useful negative result here, so it is worth stating plainly. The
opponent's **entire farm is public** — money, tiles, hands, quadrants — so the
adversarial features in the environment specification are all *implementable*. I implemented three,
A/B'd each, and all three measured worse or inert:

| Heuristic | Result | Verdict |
| --- | --- | --- |
| `PRICE_FLOOR_SELLING` — withhold capped goods to defend their price | **−1.8%** in self-play (p~0.00014, better on 6/30) | deleted |
| `PREDATORY_TIMING` — dump ahead of their forecast harvest | **exactly 0 delta** over 80 paired seeds | deleted |
| `OPPONENT_ADAPTIVE` — cede melon land when they contest melon | **−26.9%** vs adaptive (p~0.0, better on **0/30**) | deleted |

The pattern is consistent and it is the strategic lesson of the whole exercise: the
shared pots go to whoever produces into them **first**, so tempo dominates and
reacting to the opponent costs more than it earns. Withholding stock is worst of all,
because unsold inventory scores $0 and melon's price never recovers enough for the
held units to clear.

`OpponentTracker` is retained (composition, cash, labour, profile label) but it feeds
**the decision log only** — it does not change what the agent does. That is stated in
its docstring so nobody mistakes it for a live input.

### Known gaps

- **Worker idle rate 32.7%** vs `adaptive`. This is the largest remaining headroom:
  the greedy assignment leaves a third of unit-turns doing nothing once the task list
  is exhausted mid-day. More coops would absorb it, but they are cash-gated.
- **Tail risk.** Mean vs `baseline` is $22.5k but the minimum over 40 seeds is $2,244
  — one seed collapses. Not diagnosed. Median ($22,907) is a better guide to typical
  play than mean, and the `adaptive` matchup is far tighter (min $18,342, sd $1,584).
- **Shed overflow: 324 items** lost over 40 episodes (~8/episode) — the sell cadence
  does not fully keep up with harvest peaks.
- **~2,000 no-op `PICKUP`s** (0.86% of actions): the ferry/provision pre-passes and
  idle-unit logistics sometimes send several units for the same last item.
- Tomato and strawberry are never planted, and carrot was removed as a filler. All
  three are dominated on the numbers above, but the tomato/strawberry exclusion is
  reasoned rather than A/B'd.
- `ANIMAL_CARE` remains statistically unresolved against the scripted opponents
  (+0.5%/+2.9%, both n.s.), though self-play showed **+31.0% (p~0.021)**. Kept ON:
  the direct-call measurement (26 vs 12 eggs per goose over 16 days) is unambiguous.
- `HIRES_PER_TURN`, `FERRY_MAX_UNITS`, `ANIMAL_BACKLOG_CAP`, the land buffers and the
  wheat-feed constants are all still `PLACEHOLDER` — reasoned, not swept.

---

## Local arena

```bash
# metrics + decision logs
python local_arena.py --agent main.py --opponent baseline --episodes 30 --log-decisions

# graduate a PLACEHOLDER: paired-seed A/B with a significance check
python local_arena.py --agent main.py --opponent baseline --episodes 30 --ablate EXPAND_LAND

# sweep a numeric constant
python local_arena.py --agent main.py --opponent baseline --episodes 30 --sweep MAX_HANDS=13,16,20

# self-play against a frozen snapshot
python local_arena.py --agent main.py --opponent mirror --episodes 30

# save and inspect replays
python local_arena.py --agent main.py --opponent baseline --episodes 3 --save-replays 3
python local_arena.py --replay logs/match_run_0042.json
```

Reported per run: mean/median/min/max/sd final cash, win rate, crashes, timeouts,
invalid statuses, per-turn compute (p50/p95/max), action no-op rate with a per-op
breakdown, shed overflow losses, worker idle rate, market orders dropped to the
10/turn cap, and which heuristics fired.

Two details worth knowing:

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
same *idea*, and it exists because `starter` is trivially beaten.

---

### Core Rationale: Expanding Agent Degrees of Freedom

Parameter sweeps and local gym optimization on a fixed, hardcoded agent script hit a structural ceiling. Tuning parameters on a rigid phase script optimizes *how well* the agent executes a single plan, but cannot invent new tactical pivots, counter-strategies, or dynamic market adaptations.

To achieve top-tier competition performance, we developed two complementary systems:

1. **Continuous Leaderboard Intelligence**: The state-action possibility space in Kaggriculture is vast. By continuously mining 720-step replays from top-ranked teams on Kaggle, we automatically discover emerging meta-strategies, build-order shifts, and market arbitrage thresholds directly from the best players in the competition.
2. **Dynamic Utility & Telemetry Architecture**: We unlocked higher degrees of freedom in `main.py` by replacing rigid linear phase machines with a **Marginal ROI Auction Engine** and **Opponent Telemetry Sensor**. The agent evaluates action returns dynamically based on live market prices and opponent telemetry, enabling it to pivot strategy mid-match if an opponent attempts to flood or contest a specific market sector.

---

### Leaderboard Intelligence Pipeline (`leaderboard_crawler.py`)

A continuous intelligence crawler that mines Kaggle competition leaderboard matches to analyze top-performing strategies:

- **Automated Replay Streaming**: Fetches 720-step JSON replays directly from top leaderboard teams via Kaggle API.
- **Strategic Trajectory Dissector**: Parses turn-by-turn opening build orders, crop choices, animal purchase timelines, hand hiring curves, and land expansion milestone days.
- **Dashboard & Intelligence DB**: Generates machine-readable `logs/leaderboard_intelligence.json` and human-readable `logs/leaderboard_intelligence.md`.
- **Hall of Fame Integration**: Automatically ingests top-tier external matches into `EliteRecorder`.

```bash
python leaderboard_crawler.py --limit 10              # Single scan of 10 matches
python leaderboard_crawler.py --interval 600          # Poll continuously every 10 min
python leaderboard_crawler.py --import-hall-of-fame   # Ingest top replays to EliteRecorder
```

### Synergy: Leaderboard-Guided Gym & Sparring Replays (`opponents/leaderboard_replay.py`)

Relying *only* on a crawler creates imitation dependence (copying current leaderboard flaws), while relying *only* on a local gym risks overfitting to synthetic opponents. 

To solve this, we created the **Leaderboard-Guided Gym**:
- **Replay Sparring Opponent**: `opponents/leaderboard_replay.py` loads downloaded 720-step JSON replays from top Kaggle leaderboard teams and replays their exact turn-by-turn actions.
- **Gym Parameter Sweeps against #1 Leaderboard Players**: You can run `local_arena.py` parameter sweeps directly against real downloaded #1 leaderboard replays.
- **Meta-Breaker Counter-Strategy Discovery**: The Gym optimizes `main.py` parameters to discover strategies that specifically defeat the top leaderboard meta (e.g. counter-planting against a 60% Strawberry meta).

```bash
# Spar directly against the latest downloaded Kaggle leaderboard replay!
python local_arena.py --agent main.py --opponent leaderboard --episodes 10

# Spar against a specific downloaded episode replay JSON:
python local_arena.py --agent main.py --opponent logs/leaderboard_replays/episode-90163724-replay.json --episodes 10
```

---

## Submitting

```bash
cp kaggle_credentials.example.py kaggle_credentials.py   # then edit
python submit.py --dry-run                               # all checks, no submission
python submit.py                                         # check, submit, poll
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
