"""Scoring objective for macro-strategy search.

The competition ranks agents by who finishes richer, not by how much richer, so
the quantity to maximise is P(win) — not E[cash]. Those two come apart exactly
where it matters: a strategy with a high mean and one collapsing seed in forty
is worse, under pairwise scoring, than a strategy with a lower mean and no tail.

`fitness` is a smooth P(win) surrogate plus a CVaR term on the **margin**:

  * the surrogate keeps a gradient after win rate saturates at 100% against a
    scripted opponent, which raw win rate does not;
  * CVaR@alpha averages only the worst `alpha` fraction of seeds, so improving
    the tail scores and improving an already-good seed barely does.

The tail is measured on the margin `(me - opp)`, not on own cash. Scoring own
cash was tried first and it is wrong: against a weak scripted opponent every
seed is won by a mile, so own-cash variance is noise, and optimising it away
buys tail safety with production. The candidate that came out of it raised the
worst own-cash seed by 6.5% and then lost 8W-22L head-to-head against the very
strategy it was derived from. What actually threatens a win is a thin margin,
and that is what this penalises.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

CVAR_ALPHA = 0.25
CVAR_WEIGHT = 0.5
MARGIN_FLOOR = 1_000.0
MARGIN_SHARPNESS = 4.0


def _sigmoid(x: float) -> float:
    if x < -30.0:
        return 0.0
    if x > 30.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def cvar(values: list[float], alpha: float = CVAR_ALPHA) -> float:
    """Mean of the worst `alpha` fraction of `values` (at least one sample)."""
    if not values:
        return 0.0
    k = max(1, int(math.ceil(alpha * len(values))))
    worst = sorted(values)[:k]
    return sum(worst) / len(worst)


def score_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-episode {me_cash, opp_cash, win, seed, swap} into a score."""
    if not results:
        return {"fitness": 0.0, "episodes": 0}

    me = [float(r["me_cash"]) for r in results]
    opp = [float(r["opp_cash"]) for r in results]

    margins = []
    for a, b in zip(me, opp, strict=False):
        denom = max(MARGIN_FLOOR, abs(a) + abs(b))
        margins.append((a - b) / denom)

    p_win = sum(_sigmoid(MARGIN_SHARPNESS * m) for m in margins) / len(margins)
    margin_tail = cvar(margins)

    return {
        "fitness": p_win + CVAR_WEIGHT * margin_tail,
        "p_win_surrogate": p_win,
        "cvar_margin": margin_tail,
        "win_rate": sum(1 for r in results if r.get("win")) / len(results),
        "mean_cash": sum(me) / len(me),
        "median_cash": statistics.median(me),
        "cvar_cash": cvar(me),
        "min_cash": min(me),
        "mean_opp_cash": sum(opp) / len(opp),
        "episodes": len(results),
    }


def worst_episodes(results: list[dict[str, Any]], k: int = 5) -> list[dict[str, Any]]:
    """The `k` episodes with the lowest own cash, for tail diagnosis."""
    return sorted(results, key=lambda r: float(r["me_cash"]))[:k]
