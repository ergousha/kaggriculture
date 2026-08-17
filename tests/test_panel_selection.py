"""Regression tests for opponent-panel construction.

The panel is the thing v0.2.7 changed, and it is only observable seven hours into a
pipeline run. These tests pin the two properties the runbook gates on -- that
`leaderboard-top` orders by ladder rank rather than by win rate against the
incumbent, and that `min_distances` reports what `describe_panel` prints -- so a
regression surfaces in CI instead of in a re-run.
"""

from __future__ import annotations

import argparse
from typing import Any

from mining import panel as panelmod
from simulate_candidates import build_panel_entries, check_panel_source

PASS_STEP: dict[str, Any] = {"farmer": ["PASS"], "hands": [], "market": []}
MOVE_STEP: dict[str, Any] = {"farmer": ["MOVE", 1, 1], "hands": [], "market": []}


def _route(pattern: str) -> list[dict[str, Any]]:
    """A route whose steps differ wherever `pattern` differs."""
    return [MOVE_STEP if ch == "m" else PASS_STEP for ch in pattern]


def _candidate(h: str, team: str, rank: int | None, pattern: str) -> dict[str, Any]:
    from mining.common import encode_route_b85

    return {
        "hash": h,
        "team": team,
        "team_rank": rank,
        "route_b85": encode_route_b85(_route(pattern)),
    }


def _ranked(*rows: tuple[str, str, int | None, str, float]) -> list[tuple[dict, dict]]:
    """`[(scores, candidate)]` as `simulate_candidates` builds it, strongest-first."""
    entries = [
        ({"mean_win": win}, _candidate(h, team, rank, pattern))
        for h, team, rank, pattern, win in rows
    ]
    entries.sort(key=lambda r: -r[0]["mean_win"])
    return entries


def _args(**kw: Any) -> argparse.Namespace:
    base: dict[str, Any] = {
        "panel_source": "leaderboard-top",
        "panel_from_top": 120,
        "panel_team_top": 30,
        "panel_rank_min": 1,
    }
    base.update(kw)
    return argparse.Namespace(**base)


def test_leaderboard_source_orders_by_ladder_rank_not_screen_win() -> None:
    """The whole point of v0.2.7: the seed is the best-ranked team, not the best
    counter to the incumbent."""
    ranked = _ranked(
        ("aaaa", "counter-team", 900, "pppp", 0.99),
        ("bbbb", "top-team", 2, "mmmm", 0.10),
        ("cccc", "mid-team", 20, "mmpp", 0.50),
    )
    entries = build_panel_entries(ranked, _args())
    assert [e["hash"] for e in entries] == ["bbbb", "cccc"]
    assert [e["rank"] for e in entries] == [2, 20]


def test_rank_min_turns_the_panel_into_a_band() -> None:
    """Issue #22 measured that the >3000 meta is not the ladder we are matched
    into, so the panel has to be able to exclude the top as well as the bottom."""
    ranked = _ranked(
        ("aaaa", "sealed-meta", 2, "pppp", 0.10),
        ("bbbb", "our-band", 200, "mmmm", 0.40),
        ("cccc", "below-us", 900, "mmpp", 0.90),
    )
    entries = build_panel_entries(ranked, _args(panel_rank_min=100, panel_team_top=500))
    assert [e["hash"] for e in entries] == ["bbbb"]


def test_leaderboard_source_breaks_rank_ties_by_screen_win() -> None:
    """Within one team, take the route that screened best; that choice cannot bias
    the panel across teams."""
    ranked = _ranked(
        ("aaaa", "top-team", 2, "pppp", 0.20),
        ("bbbb", "top-team", 2, "mmmm", 0.80),
    )
    entries = build_panel_entries(ranked, _args())
    assert [e["hash"] for e in entries] == ["bbbb", "aaaa"]


def test_screen_top_source_reproduces_the_v0_2_6_ordering() -> None:
    ranked = _ranked(
        ("aaaa", "counter-team", 900, "pppp", 0.99),
        ("bbbb", "top-team", 2, "mmmm", 0.10),
    )
    entries = build_panel_entries(ranked, _args(panel_source="screen-top"))
    assert [e["hash"] for e in entries] == ["aaaa", "bbbb"]


def test_unranked_candidates_are_ineligible_for_a_leaderboard_panel() -> None:
    """A team that left the board is unknown, not top-ranked -- `None` must not
    sort to the front."""
    ranked = _ranked(
        ("aaaa", "gone", None, "pppp", 0.99),
        ("bbbb", "top-team", 2, "mmmm", 0.10),
    )
    entries = build_panel_entries(ranked, _args())
    assert [e["hash"] for e in entries] == ["bbbb"]


def test_check_panel_source_rejects_a_pool_mined_before_ranks_existed() -> None:
    import pytest

    stale = [{"hash": "aaaa", "team": "x"}]
    with pytest.raises(SystemExit, match="team_rank"):
        check_panel_source(stale, "leaderboard-top", rank_min=1, rank_max=30, panel_size=6)


def test_check_panel_source_rejects_a_band_too_thin_for_the_panel() -> None:
    import pytest

    thin = [_candidate("aaaa", "top-team", 2, "pppp")]
    with pytest.raises(SystemExit, match="distinct teams"):
        check_panel_source(thin, "leaderboard-top", rank_min=1, rank_max=30, panel_size=6)


def test_min_distances_matches_what_describe_panel_prints() -> None:
    picked = [
        {"hash": "aaaa", "team": "a", "rank": 1, "win": 0.5, "route": _route("pppp")},
        {"hash": "bbbb", "team": "b", "rank": 2, "win": 0.5, "route": _route("mmmm")},
        {"hash": "cccc", "team": "c", "rank": 3, "win": 0.5, "route": _route("mmmp")},
    ]
    dists = panelmod.min_distances(picked)
    assert dists[0] == 1.0  # the seed has nothing earlier to be close to
    assert dists[1] == 1.0  # all four steps differ from the seed
    assert dists[2] == 0.25  # one step from `bbbb`
    text = panelmod.describe_panel(picked, start=2)
    assert "min-dist-to-earlier 0.25" in text
    assert "rank    3" in text


def test_select_panel_trades_distance_for_a_distinct_team() -> None:
    """The distinct-team bonus is 0.25 of action distance, so a new team wins any
    contest it loses on distance by less than that -- and loses one it gives up
    more. The v0.2.5 run's top 20 were 18/20 one team; this is what stops that."""
    seed = {"hash": "aaaa", "team": "a", "rank": 1, "win": 0.5, "route": _route("pppppppp")}
    same_team_far = {
        "hash": "bbbb",
        "team": "a",
        "rank": 1,
        "win": 0.5,
        "route": _route("mmmmmmmm"),
    }
    new_team_near = {
        "hash": "cccc",
        "team": "c",
        "rank": 2,
        "win": 0.5,
        "route": _route("mmmmmmmp"),
    }

    # 1.000 vs 0.875 + 0.25 -> the new team wins.
    picked = panelmod.select_panel([seed, same_team_far, new_team_near], k=2, min_distance=0.15)
    assert [p["hash"] for p in picked] == ["aaaa", "cccc"]

    # 1.000 vs 0.375 + 0.25 -> distance wins, and the panel doubles up on a team.
    new_team_very_near = {**new_team_near, "route": _route("mmmppppp")}
    picked = panelmod.select_panel(
        [seed, same_team_far, new_team_very_near], k=2, min_distance=0.15
    )
    assert [p["hash"] for p in picked] == ["aaaa", "bbbb"]
