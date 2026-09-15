# Proposal: the 16-day endgame (2026-09-14 → 2026-09-30)

> ## ✅ v0.4.0 LIVE RESULT — 2026-09-15 (the revised plan's step 1, verified)
>
> **v0.4.0 settled at 2632.3** — inside the predicted 2562–2800 band, +1521 over v0.3.1's 1115.
> Converged last-quartile WR 35.2% (7W 15L 5T, n=27, 105 episodes) — the field's dense band is
> now chassis-class, and our agent is a peer of it rather than a generation behind.
>
> ## v0.4.1 — the first chassis-class improvement (2026-09-15, gated, READY TO SUBMIT)
>
> **Measured field gap (Phase 1 re-spy, 2026-09-13 daily batch: 654 eps / 1308 seats, plus
> 49 fresh leader replays):** the top decile separates from the mid-band by MILK +$16.2k,
> STRAWBERRY +$12.2k, WOOL +$10.8k per seat and sells LESS wheat (−$4.2k); they keep end-of-game
> prices high (MILK $108 vs $57). Pot products (TOMATO/EGG/CARROT) are now farmed by 60–85% of
> the mid-field — the additive-pot premise of Phase 2 is DEAD as a differentiator; the field
> passed it. The remaining lever inside the chassis class is draw-specialization:
>
> - v0.4.0's day-6 router only specializes YARN-containing draws — **22% of seeds**
>   (verified by replicating the shop-draw RNG + 180 pass-vs-pass env enumerations,
>   `scripts/enumerate_draws.py`). On the other 78% it replays generic route 0.
> - The metacounter kernel carries a 28-tape EXP240 library (routes 100–128, Yusuke Hayashi
>   "Shop Router 0913" lineage, Apache-2.0) covering **all 64 draw pairs**, plus a
>   dual-expert router (EXP240 when no YARN, V39 otherwise).
> - **v0.4.1 = v0.4.0 + EXP240 graft + dual-expert router** (`scripts/build_v041.py`, built at
>   `scratch/v0_4_1_candidate.py`, now `main.py` / `opponents/v0_4_1.py`).
>
> | gate (proposal section "Offline gates") | result |
> | --- | --- |
> | head-to-head vs incumbent ≥60% on ≥100 paired seeds | **66.7% (80W 28T 12L) on 120 seeds**; excluding ties 80/92 = 87% (z=7.1, p~10⁻¹²); +$1.0k mean cash |
> | held-out draw-stratified validation (60 fresh seeds) | 37W 20T 3L, +$1,091 (`scripts/draw_stratified.py`, `logs/v041_draws.json`) |
> | zero failed HIRE/BUY vs incumbent | identical profiles: hire=0, land=0 (`scripts/analyse_market_fills.py`) |
> | adverse-draw regression | none systematic; worst single-seed loss −$5.7k (PIZZA×1 seed) |
> | `rank_ladder.py --require-perfect` | **10/10** |
> | submit.py pre-flight | GREEN (static + loadable + smoke 3/3, $151.9k mean, p95 0.9ms) |
>
> Ties (28/120) concentrate exactly on YARN draws where both sides replay identical tapes —
> expected and harmless. File size 4.0 MB (Kaggle's submission limit is 100 MB), all imports
> on the allowlist, single-worker cold-start max turn 483ms < 1s actTimeout.
>
> Next after v0.4.1 ships: (a) watch converged WR vs the 35.2% baseline for 2 days;
> (b) per-draw panel selection over the merged 41-tape portfolio with our tooling — replace
> EXP240's static pair→route table with per-draw measured best (the draw table now exists:
> `logs/seed_draws.csv`); (c) Phase-3 midgame reset stays as described, now over a portfolio
> that actually differs per draw.

> ## ⚡ REVISED PLAN — 2026-09-14, later the same day (supersedes Phases 0–3 below)
>
> Phase 1's decision gate has been **answered, and the answer kills Phases 0 and 2 as written.**
> We obtained and tested the three highest-voted public kernels (all forks of one Apache-2.0
> layered-chassis lineage, the class the September field actually runs):
>
> | measurement (local arena, `uv run python local_arena.py`) | result |
> | --- | --- |
> | metacounter vs **v0.2.7** (our live-best), 30 eps × seeds 2000000/1000000 | **60–0**, mean $117.8k vs $83.0k |
> | metacounter vs v0.2.6, 20 eps seed 3000000 | **20–0**, $108.9k vs $71.6k |
> | lynnsakurai vs metacounter, 12 eps × seeds 2000000/3000000 | **24–0**, +$21–24k mean cash |
> | lynnsakurai vs v41, 8 eps seed 2000000 | 50% W/L but +$249 mean cash edge |
> | instrumented realized revenue (metacounter vs v0.2.7, seed 2000000) | **$142,928 vs $91,946** |
>
> **Mechanism (measured, per-product realized sell revenue):** WOOL **+$35.3k** (6 sheep bought
> from day 0 vs our day 7–9 → 272 units sold vs 120), STRAWBERRY +$8.8k, TOMATO +$7.2k
> (pot product, we sell zero), FERTILIZER +$4.2k, CARROT +$2.9k (pot product), vs **−$10.7k on
> WHEAT** (they dump it early to fund animals). Their sell-spam (380 SELL orders, 88.5k total
> qty, 87 orders ≥100 units) liquidates the shed every step — the engine's `_commit_unit`
> silently drops excess, so oversized SELLs are free continuous liquidation. They hire 10–11
> hands/day from day 8 and route 41 tapes by day-6 shop pair (64-entry table). The kernel
> authors are live-ranked: **Thomas Tschinkel #8 (2976)**, Rayk Kretzschmar #144 (2795),
> Alperen Aydın #658 (2562) — this chassis class is the September meta.
>
> ### The revised plan
>
> 1. **Ship the strongest public chassis now.** lynnsakurai's fork is strongest tested
>    (24–0 over metacounter on two seed bases; pre-flight GREEN after stripping the one
>    cosmetic `from __future__ import annotations`; p95 callback 0.9–1.0 ms; stdlib-only;
>    Apache-2.0 — retain the license notice and attribute the author in the submission
>    message). Candidate staged at `scratch/public_kernels/lynnsakurai_submit_candidate.py`.
>    Expected live: their-class-vs-our-class gap is 30–0 offline → our 1115 rating agent is a
>    generation behind; a 2562–2800-class agent is the realistic ceiling **without further work**.
>    This replaces Phase 0 (there is no point measuring our obsolete route's live decay).
> 2. **Then improve on that chassis class, not on our route.** Our unique assets (4,315-route
>    corpus, paired-seed arena, panel sieve, `scripts/analyse_market_fills.py` realized-revenue
>    instrumentation — which produced the $51k decomposition above) apply to *their* engine:
>    better route portfolios per shop-draw, better day-6 routing tables, our panel-based
>    selection over their 41 routes. That is strictly higher-EV than mutating a route that
>    loses 30–0.
> 3. **Pot products are confirmed open even in the public meta** (their TOMATO +$7.2k,
>    CARROT +$2.9k per game), but the field's chassis already farms them — the additive-pot
>    module below is now *one of several* edges to graft onto their chassis, not a from-scratch
>    engine build. Keep the labour-cost math (fib(13)=$233/day) — it explains their 10–11
>    hires/day ceiling and any module must respect it.
> 4. **Gates unchanged in spirit:** any candidate must beat the *chassis incumbent*
>    head-to-head ≥60% on ≥100 paired seeds (not our obsolete route), and pass
>    `submit.py --dry-run` + `rank_ladder.py`. The live converged-WR read stays the only
>    trusted acceptance signal.
>
> The rest of this document is preserved as written this morning for the analysis trail.

Written after a full review of README, docs/experiments.md, issues #22–#31, the engine source,
and fresh live measurements taken today. Every number below is either measured today or
quoted from the repo's own records; hypotheses are labelled.

## Where we actually are (measured 2026-09-14)

| fact | value | source |
| --- | --- | --- |
| deadline | 2026-09-30 23:59 | `kaggle competitions list` |
| our rank / score | **3439 / 9040, 1115.2** | leaderboard today |
| field median / leader | ~3068 / 3224 (Majkel1337) | `logs/team_ranks.json` |
| v0.3.1 converged live win rate | **30.3%** (50W-115L, last 165 of 658 episodes) | `scripts/calculate_converged_winrate.py` |
| v0.2.7 at freeze (2 days old, Aug 20) | 2040.6; last-quartile WR 36.8% (n=38) | submissions + same script |
| v0.2.6 at freeze (2 days old, Aug 16) | 2080.7 | submissions |
| v0.3.0 at freeze (~50 min old!) | 1120.5 | submissions |
| best-ever settled score | ~2080 (v0.2.6) | submissions |

Three readings, in order of confidence:

1. **The live score did not plateau — it fell off a cliff.** v0.2.6/v0.2.7 sat at ~2040–2080
   after two days of play. v0.3.1, after **25 days**, sits at 1115 while still losing 70% of
   games. Either the v0.3.x mutations were live-harmful (they were shipped on +0.6 and
   mixed-evidence panel deltas), or the field left the whole route class behind between
   Aug 18 and today. These two hypotheses have completely different implications, and the
   repo currently assumes neither — it assumed the mutations were fine.
2. **The offline evaluation stack has near-zero live predictive validity, and this is now
   proven five times.** v0.2.5 read 97% offline / 47% live; v0.2.6 read 92.6% / plateau
   2290; v0.3.0 read 87.8% panel + 62% head-to-head / live collapse; v0.3.1 read 90.6% /
   30.3% live. Any plan whose gate is "win the panel" is planning against a measurement
   that has transferred at roughly 50% accuracy of sign.
3. **Selection over the mined corpus is exhausted by the repo's own evidence.** All 4,315
   candidates are one meta; route×seed interaction among finalists is ~1 flip in 70 seeds
   (measured today on the final-stage grid: pair `044a7741` vs `111e3cac` agrees on 68/70);
   every mutation operator measured (six of them) loses offline; the market layer is
   measured-and-exhausted (#23, #25, #30). There is no selection left to do.

## The one structural gap the field still leaves open

Re-verified today from `logs/daily_fingerprints.csv` (1,368 seats):

- **0 of 1,368 seats sell TOMATO. 0 sell EGG. 92 sell a handful of carrots (mean 14 units).**
- The town drains these products every day (center 1/day each; `PET_CAFE`,
  `PIZZA_SHOP`, `FARMERS_MARKET` instances drain more), so their market inventory ends the
  episode **below I0** — price rises all game. Engine math today: TOMATO is $60 at I0,
  **$120 at I0−500, $180 at I0−1000**; EGG $50 → $80 at I0−500.
- The contested products the whole field fights over realise **MILK $21/unit,
  STRAWBERRY $62–97, WOOL $65** — the shared pot is nearly worthless at the margin,
  which is exactly why games between meta-peers are coin flips decided by ±10% margins.
- Cash is 86% common-mode (README), so **uncontested margin converts ~1:1 into wins**
  in exactly the dense band of the field where our rating is decided.

The pot: a tomato tile is a $50 seed producing **1 unit/day from day 8 to day 29**
(~22 units, 44 if fertilized — and fertilizer is free: our own pastures generate ~420
units/episode, which the route currently *sells* at $100 instead of applying).

**Why v0.3.1's tomato attempt failed and this is different.** The #28 operators could only
*retarget wheat tiles* (starving the herd: win rate 90% → 0–43%) or plant on *dwell tiles*
(carrying-unit deadlock). They never tested the economics — they tested mutation operators
on a saturated route. The pot was never actually contested.

**The binding constraint is marginal labour, not land or cash.** The 13th hire of the day
costs `fib(13)` = **$233/day** (~$6.5k/season); the 14th $377/day. Six tomato tiles need
~12–15 turns/day of water+harvest+walk. So the engine must be tuned to the labour price:
one extra hand max, tiles adjacent to each other and near the shed, planting day 0–2, and
fertilizer (free from our own pastures) to double yield. At 6 fertilized tiles ≈ 12
tomato/day × $60–120 = **$720–1,440/day against $233/day of labour** — positive across the
whole observed price range, and pure margin because nobody else supplies the drain.

This is the only idea in the repo's "known gaps" list that *adds* production instead of
*reallocating* it. Every failed version reallocated a saturated resource; that is the
mechanism behind all five live failures.

## The plan

### Phase 0 — the control submission (days 1–3, 1 submission, runs in parallel with everything else)

Submit **v0.2.7 verbatim** (frozen at `opponents/v0_2_7.py`, route `044a7741e9`) as
`"Control: v0.2.7 re-run"` and let it play 2–3 days (~75 episodes/day for a fresh active
submission).

- If it settles **≥ ~1800**: the v0.3.x mutations were the live disaster. Revert the spine
  to v0.2.7 permanently and raise the ship gate: no version ships on a panel delta < 5
  points or a head-to-head < 60%.
- If it settles **~1100–1300**: the route *class* has decayed — the September field passed
  the August meta, and no amount of route selection can win. Only a genuinely new engine
  (Phase 2) has headroom, and expectations reset accordingly (+300–600 points is a good
  outcome, not +1000).

This costs zero build time (the file exists) and one submission slot, and it is the only
experiment that disambiguates the collapse. Note the wrinkle discovered today: **the daily
episode dumps now 403 on `dataset_download_files`, including known-good August dates** —
so the "re-download and re-mine September" runbook is broken until fixed (try the
`kaggle datasets download` CLI path, else browser, else the per-episode
`competition_list_episodes` route that `scripts/research_leaderboard.py` already uses and
that worked today for `calculate_converged_winrate.py`).

### Phase 1 — re-spy the September field (days 1–2, no submissions)

The corpus is a month old. Before building anything, answer **one** question: *what do the
3000+ agents do now that the Aug 8–14 corpus doesn't contain?*

- Extend `scripts/research_leaderboard.py` to pull the last ~10 episodes of the top ~30
  teams *today* (the per-episode API path works — verified today), and run the
  `mine_daily.py` fingerprinting on them.
- Concretely check: do the leaders sell TOMATO/CARROT/EGG now? Hand counts, land, herd mix,
  fertilizer *application* (vs selling), endgame liquidation behaviour.
- **Decision gate:** if the top agents already farm the uncontested pot, the novelty
  premise dies here — and the correct move becomes copying the best fresh open-loop plan
  (selection again, but on a September corpus with a new meta in it). If they don't, the
  pot is genuinely unclaimed and Phase 2 proceeds.
- Cheap addition while there: retry fetching the public notebook that `main.py` itself
  credits for the runtime layers (Kaito Fukami's "25-27 strict future v27 midgame meta
  reset") — with credentials the API can pull kernels; if the public meta has a v27-class
  agent, adapting it may beat building.

### Phase 2 — the additive-pot hybrid (days 3–9, 1–2 submissions)

**Architecture: recorded spine + closed-loop pot module.**

- **Spine:** v0.2.7's route (the live-proven 2040-class artifact), replayed verbatim by the
  existing runtime layers — *not* the v0.3.x mutations, pending Phase 0's verdict.
- **Module:** a small closed-loop controller owning exactly what the spine doesn't use:
  - 4–6 **tomato tiles** on land the spine never visits (verifiable with the exact
    position simulator in `search/board_paths.py` — it already re-simulates all 719 steps
    and all unit slots);
  - **one** extra hand (slot index ≥ the route's hand count; the hands-alignment layer
    pads/truncates route traces to the live count, so extra slots must be verified to
    default to PASS, never to shift the spine's slots);
  - planting days 0–2, daily WATER, harvest from day 8, continuous SELL from day 10
    (keeps the shed — which peaks at 100/100 — clear, and the price supported by the town
    drain; we are the only supplier, so our ~12 units/day do not crash a $60–180 market);
  - fertilizer applied from our own pastures' free supply, *budgeted so the spine's
    `SELL FERTILIZER` orders still fill* — the cash schedule is pinned on both sides
    (#30: 319/458 funded orders cannot slip a single step), so the module must not
    intercept revenue the spine has already promised;
  - every module order gated on the exact remaining route obligations computed by
    `search/cash_schedule.py` (it already proves "no future spine order can fail" —
    reuse it as the runtime spend gate, not just offline).
- **Offline gates, changed to reflect the five live failures:**
  1. mean panel win ≥ incumbent **+5 points** and worst-opponent ≥ incumbent's worst
     (an exploit is a reject);
  2. head-to-head vs incumbent ≥ **60%** on 100 held-out seeds (v0.3.0 shipped at 62%
     and died; 55% is no longer a bar);
  3. zero failed HIRE/BUY, zero module-attributable shed overflow (the instrumentation
     fixed in #30);
  4. **new:** no regression on an *adverse-draw* seed subset (bottom-quartile
     shop-draw seeds from the observed distribution) — the live collapse happens on
     adverse draws, and a mean over 30 seeds hides it;
  5. `rank_ladder.py --episodes 1 --require-perfect` 10/10, as always.
- **Live read after ~2.5 days:** converged WR with CI (`calculate_converged_winrate.py`).
  With ~50–75 episodes/day, 2 days gives ±8–10% — enough to call a disaster, not a
  fine margin. The bar: clearly above the Phase-0 control's band.

### Phase 3 — conditional extensions (days 10–16)

Only if Phase 2 moves the live rating by ≥ +300 over the control:

- **Midgame reset (seed-conditioning, the mechanism the live v0.2.4 analysis pointed at):**
  the shop draw is fully observable as it happens (day 3, 6, 9, 12, … unlocks). By day ~12
  the demand profile for the *rest* of the game is largely known. With an additive engine
  whose products are uncontested, their future prices are predictable from the observed
  draw — so the module (and only the module) can re-weight tomato vs carrot vs egg as the
  draw reveals itself. This is portfolio routing (rejected in the README for same-meta
  routes, correctly — I re-measured route×seed flips at 1/70 today) made possible *only
  because Phase 2 creates genuinely different engines to route between*.
- If Phase 2 does not transfer: lock the best live configuration, spend remaining
  submissions on one clean A/B if any candidate is within one gate of passing, and stop.

## Calendar (submission slots: 5/day, ~80 left; live A/B slots: ~3)

| days | submissions | work |
| --- | --- | --- |
| 1–3 | control v0.2.7 (1) | Phase 1 mining; dump-403 fix; module skeleton |
| 3–4 | — | read control; decide spine; finish module |
| 5–7 | — | offline gates on module; iterate |
| 8–10 | hybrid v0.4.0 (1) | live read starts |
| 11–12 | — | converged-WR read; revert-or-iterate decision |
| 12–15 | iterate or hold (0–2) | Phase 3 midgame reset if earned |
| 16 | final lock | best live config plays out the deadline |

## What this proposal deliberately does not do

- **No more route selection or mutation.** Exhausted by measurement (4,315 candidates,
  one meta, six losing operators, 1/70 route×seed flips).
- **No retraining a learning agent.** v0.0.8 lost 26× for structural reasons the README
  correctly diagnoses; 16 days cannot rediscover a $90k meta from scratch.
- **No market-layer work.** #23/#25/#30 measured it shut from both directions.
- **No PvP heuristics.** All four measured losers; the latency argument survived its own
  strongest test.
- **No shipping on panel deltas < 5 points.** Five consecutive versions won their local
  gates and moved live rating by nothing or worse. Local gates are a veto, and from here
  the *live converged win rate against the Phase-0 control* is the only trusted gate.

## Honest expected value

The pot is worth roughly **+$15–30k of margin per game** against a ~$90k meta, realised
where the field is densest (games decided by <10%). If it transfers, that is the difference
between the $93k top quartile and the $129k top decile — call it +300–800 rating, not a
jump to 3200. The path from ~2000 to 3200, if it exists, runs through re-spying the
September leaders (Phase 1) — the leaders' episodes are downloadable today via the
competition API even though the dumps 403, and nobody has looked at what they do *this
month*.
