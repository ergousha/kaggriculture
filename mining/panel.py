"""Opponent-panel construction for Phase 2.

Why this exists. The first run evaluated every candidate against a single fixed
opponent (v0.2.4) and shipped the route that beat it **97.0%** of the time. Live,
that route wins ~47%. The gap is not seed luck: v0.2.4 is a strong opponent (the
median mined candidate beats it 0% of the time), so selecting from the 8.4% of the
pool that beats it >=90% selects for routes that exploit *that specific opponent's*
market timing on a shared order book — and at 97% the metric is saturated, so it
cannot discriminate among the finalists at all.

A panel fixes both. Candidates are scored against several structurally different
strong routes, so a route that only counters one opponent's timing cannot win, and
the win-rate spread reopens.

Panel selection is greedy max-min diversity over per-step action distance, seeded
with the strongest entry, and biased toward distinct teams. The incumbent anchor is
always included so results stay comparable with the previous run.

WHAT "STRONGEST" MEANS is the caller's choice, and v0.2.7 changed it. Under
`--panel-source screen-top` (the v0.2.6 behaviour) entries arrive ordered by win
rate against the incumbent anchor, so every member beats the incumbent ~100% by
construction and the panel measures "routes that counter us" rather than "routes
the ladder rewards". Under `--panel-source leaderboard-top` entries arrive ordered
by their team's ladder rank, so the panel is a sample of the band above us. The
selection algorithm below is identical either way; only the ordering and the
eligible pool differ.
"""

from __future__ import annotations

from mining.common import decode_route_b85


def route_distance(a: list[dict], b: list[dict]) -> float:
    """Fraction of the 719 steps whose action differs.

    Compares whole per-step actions (farmer + hands + market), which is the unit the
    engine consumes; two routes that differ only in market-order ordering are
    already normalized to the same canonical form upstream.
    """
    n = min(len(a), len(b))
    if n == 0:
        return 1.0
    return sum(1 for i in range(n) if a[i] != b[i]) / n


def select_panel(entries: list[dict], k: int, min_distance: float = 0.15) -> list[dict]:
    """Greedy max-min diversity pick of `k` opponents.

    `entries` must be pre-sorted strongest-first; each needs `hash`, `route`, `team`.
    The strongest entry seeds the panel, then each subsequent pick maximises the
    minimum distance to everything already chosen. A team already represented is
    penalised so the panel does not collapse onto one lineage — the first run's top
    20 were 18/20 from a single team.

    Entries closer than `min_distance` to an existing pick are skipped outright:
    near-duplicates add compute without adding a distinct strategy to beat.
    """
    if not entries or k <= 0:
        return []
    chosen = [entries[0]]
    teams = {entries[0].get("team")}
    while len(chosen) < k:
        best = None
        best_score = -1.0
        for e in entries:
            if any(e["hash"] == c["hash"] for c in chosen):
                continue
            d = min(route_distance(e["route"], c["route"]) for c in chosen)
            if d < min_distance:
                continue
            # Distinct teams first, then maximise distance to the current panel.
            score = d + (0.25 if e.get("team") not in teams else 0.0)
            if score > best_score:
                best_score, best = score, e
        if best is None:
            break
        chosen.append(best)
        teams.add(best.get("team"))
    return chosen


def min_distances(panel: list[dict]) -> list[float]:
    """Each member's distance to the nearest earlier member; the seed gets 1.0.

    This is the number the runbook gates on (>= 0.3): below it the panel is six
    variations on one route and the worst-opponent tiebreak stops meaning anything.
    """
    return [
        1.0 if i == 0 else min(route_distance(p["route"], q["route"]) for q in panel[:i])
        for i, p in enumerate(panel)
    ]


def describe_panel(panel: list[dict], start: int = 1) -> str:
    lines = []
    for i, (p, d) in enumerate(zip(panel, min_distances(panel), strict=True)):
        spread = "-" if i == 0 else f"{d:.2f}"
        rank = p.get("rank")
        rank_s = f"rank {rank:>4}" if rank is not None else "rank    ?"
        win = p.get("win")
        win_s = f"{win:>6.1%}" if isinstance(win, float) else "     ?"
        lines.append(
            f"    {i + start}. {p['hash'][:10]}  {str(p.get('team', '?'))[:18]:<18} "
            f"{rank_s}  screen-win-vs-anchor {win_s}  min-dist-to-earlier {spread}"
        )
    return "\n".join(lines)


def load_routes(candidates: list[dict]) -> dict[str, list[dict]]:
    return {c["hash"]: decode_route_b85(c["route_b85"]) for c in candidates}
