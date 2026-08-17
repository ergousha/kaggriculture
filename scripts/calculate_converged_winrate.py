import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kaggle.api.kaggle_api_extended import KaggleApi


def agent_id(a, attr: str, key: str):
    """Read an id off an episode agent, or None if it is unset.

    kagglesdk defaults `_submission_id` and `_team_id` to **0**, not None, so the
    obvious `getattr(a, attr, None) or a.get(key)` chain silently converts an unset
    id into None and the agent gets misfiled as the opponent. Treat 0 as unset here
    and let the caller decide what to do about it.
    """
    if isinstance(a, dict):
        value = a.get(key)
    else:
        value = getattr(a, attr, None)
    return value if value else None


def wilson_interval(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval — honest error bars on a small episode count."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def main():
    from submit import load_credentials

    load_credentials()

    api = KaggleApi()
    api.authenticate()

    target_version = sys.argv[1] if len(sys.argv) > 1 else "v0.2.6"

    print("Fetching submissions for kaggriculture...")
    subs = api.competition_submissions("kaggriculture", page_size=100)

    # Anchor the match so "v0.2.1" cannot select a "v0.2.16" submission, and keep
    # every hit: logs/submission_history.md shows v0.2.6 was prepared twice, so
    # first-match-wins picks an arbitrary one of several same-version submissions.
    pattern = re.compile(rf"(?<![\w.]){re.escape(target_version)}(?![\w.])")
    matches = [s for s in subs if getattr(s, "description", None) and pattern.search(s.description)]

    if not matches:
        print(f"Could not find submission '{target_version}'")
        sys.exit(1)

    matches.sort(key=lambda s: int(s.ref), reverse=True)
    target_sub = matches[0]
    if len(matches) > 1:
        others = ", ".join(str(s.ref) for s in matches[1:])
        print(
            f"Warning: {len(matches)} submissions match '{target_version}'; "
            f"using the most recent (ID {target_sub.ref}); ignoring {others}."
        )

    sub_id = int(target_sub.ref)
    print(f"Found '{target_version}' submission (ID {sub_id}). Fetching episodes...")

    eps = api.competition_list_episodes(sub_id)
    if not eps:
        print("No episodes found.")
        sys.exit(0)

    print(f"Fetched {len(eps)} total episodes.")

    # Drop unfinished episodes *before* slicing. Filtering afterwards lets the
    # newest still-running episodes eat into the window, so the "last quartile"
    # silently covers fewer than 25% of completed play.
    completed = [ep for ep in eps if "COMPLETED" in str(getattr(ep, "state", ""))]
    unfinished = len(eps) - len(completed)
    if unfinished:
        print(f"Ignoring {unfinished} episodes that are not COMPLETED.")

    if not completed:
        print("No completed episodes to analyze.")
        sys.exit(1)

    # Sort by id; ids are monotonic, so this is a recency proxy.
    eps_sorted = sorted(completed, key=lambda x: getattr(x, "id", None) or 0)

    quartile_start = int(len(eps_sorted) * 0.75)
    last_quartile = eps_sorted[quartile_start:]

    wins = 0
    losses = 0
    ties = 0
    unresolved = 0

    print(f"Analyzing last quartile ({len(last_quartile)} of {len(eps_sorted)} completed)...")
    for ep in last_quartile:
        agents = getattr(ep, "agents", []) or (ep.get("agents", []) if isinstance(ep, dict) else [])
        if not agents and hasattr(ep, "_agents"):
            agents = ep._agents

        my_reward = None
        opp_reward = None

        for a in agents:
            a_sub_id = agent_id(a, "submission_id", "submissionId")

            raw_reward = a.get("reward") if isinstance(a, dict) else getattr(a, "reward", None)
            reward = 0.0 if raw_reward is None else float(raw_reward)

            if a_sub_id is not None and str(a_sub_id) == str(sub_id):
                my_reward = reward
            else:
                opp_reward = reward

        if my_reward is None or opp_reward is None:
            # Neither side resolved — count it rather than dropping it, so the
            # denominator below can never quietly disagree with the window size.
            unresolved += 1
            continue

        if my_reward > opp_reward:
            wins += 1
        elif my_reward < opp_reward:
            losses += 1
        else:
            ties += 1

    total_played = wins + losses + ties
    if unresolved:
        plural = "" if unresolved == 1 else "s"
        print(
            f"Warning: {unresolved} episode{plural} had unresolvable agent ids and went unscored."
        )

    if total_played > 0:
        successes = wins + 0.5 * ties
        win_rate = successes / total_played
        low, high = wilson_interval(successes, total_played)
        print(
            f"Converged Win Rate (last quartile): {win_rate:.2%} "
            f"({wins}W {losses}L {ties}T out of {total_played})"
        )
        print(f"95% CI: {low:.2%} - {high:.2%}")
        if high - low > 0.20:
            print(
                "That interval is too wide to call a plateau on its own; "
                "treat it as consistent-with, not evidence-for."
            )
    else:
        print("Could not parse agent rewards for the episodes.")
        sys.exit(1)


if __name__ == "__main__":
    main()
