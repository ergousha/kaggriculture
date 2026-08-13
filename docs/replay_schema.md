# Kaggle Replay Schema (annotated)

Verified against `logs/episode-89987566-replay.json` (2026-08-13). All leaderboard
episode downloads (`logs/leaderboard_replays/episode-*-replay.json`, `logs/daily/*`)
use this same `kaggle_environments` envelope.

## Envelope (what `mine_replays.py` navigates)

```
{
  "id": ...,
  "configuration": { ..., "seed": null },   # seed is ALWAYS null — see caveat below
  "rewards": [125013.0, 123285.0],          # FINAL CASH per seat; index = seat/player id
  "statuses": [...],
  "steps": [ [seat0, seat1], ... ],         # 720 entries, one per turn, 2 seats each
  "specification": {...}, "info": {...}, ...
}
```

- **Final score**: top-level `rewards[seat]`. Cross-check: equals
  `steps[-1][0].observation.farms[seat].money`. Per-step `reward` is 0 while
  `status == "ACTIVE"`; only the last step carries the final value.
- **Step count**: 720 entries, but `steps[0]` is the initial state — its action is a
  `{"farmer": ["PASS"], "hands": [], "market": []}` placeholder injected by the
  framework, not a player decision. The actionable trace is `steps[1:]` → **719 actions**
  (matches the agent's 719-step route).

## Per-step, per-seat record

`steps[i][seat]` = `{action, info, observation, reward, status}`.

**Action** (the atom of the mined trace):

```
steps[i][seat]["action"] = {
  "farmer": ["PASS"],                       # or ["PLANT", ...] etc. — one action
  "hands":  [],                             # one action per hired hand
  "market": [["HIRE"], ["BUY_SEED", "MELON", 6], ["SELL", "WHEAT", 3], ...]
}
```

**Observation**: full obs dict per README "Observation Format" (`day`, `hour`, `farms`,
`market`, `town`, `private`, plus `remainingOverageTime`; seat 0 additionally has
`step`). `farms` and `town` are shared/public; `private` is per-seat.

## The seed IS recoverable — read `info.seed`, not `configuration.seed`

> **Corrected 2026-08-13.** This section previously claimed the seed was unrecoverable
> and that the original-seed fidelity check was impossible. That was wrong, and it
> would have removed the strongest validation available to any replay-mining work.

`configuration.seed` is `null` in every replay, which is what misleads. But the engine
does not discard the seed — `kaggle_environments.utils.resolve_episode_seed()` *clears*
it from `configuration` (so agents cannot read it out of the observation) and then
**stores it on `env.info["seed"]` "so it persists into the replay"**. It is right there
at the top level:

```
"info": { "Agents": [...], "EpisodeId": 90849277, "TeamNames": [...], "seed": 793678630 }
```

Present and numeric in all 3,413 replays under `replays/`. Note `info` sits at ~byte 500
and `rewards` at ~byte 880, both well before `steps` at ~6.6 KB, so a 64 KB head read
gets you score *and* seed without parsing 32 MB.

**A replay can therefore be reconstructed exactly.** Extract both seats' 719-step traces,
replay them verbatim on `configuration={"seed": info["seed"]}`, and the final rewards
reproduce to the cent — verified on episode 90849277, which reproduced
`[54528.0, 52963.0]` identically. Two requirements:

1. **Both** seats must replay their own recorded trace. The market is shared state, so a
   one-sided reconstruction diverges.
2. Replay **verbatim** — no WEED repair, no SELL-slot reordering, no hands alignment.
   Those layers perturb an action and destroy exactness, turning a precise check into a
   fuzzy one. `mining/verbatim_agent.py` exists for exactly this.

This is the fidelity gate in `mine_replays.py`, and it is worth the compute: it catches
extraction and format-conversion bugs that would otherwise silently poison every
downstream phase. All 3,089 mined candidates passed it exactly.

Beware a naive `"seed"` regex on the head: `specification` carries a `"seed": 10` field
for **every crop**, so the first match in the file is a decoy. Brace-match the `info`
object instead (`mining/common._find_object`).

All market randomness derives from `random.Random((seed * 1_000_003) ^ day)`, so shop
draws and weed spawns are fully seed-determined. The shop-draw *sequence* is also
readable directly from `steps[i][0].observation.town.unlocked_shops` (grows every 3 days),
which is useful for labeling which market a mined route was optimized for.
