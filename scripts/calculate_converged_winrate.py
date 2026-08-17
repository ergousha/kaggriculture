import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kaggle.api.kaggle_api_extended import KaggleApi


def main():
    from submit import load_credentials

    load_credentials()

    api = KaggleApi()
    api.authenticate()

    target_version = sys.argv[1] if len(sys.argv) > 1 else "v0.2.6"

    print("Fetching submissions for kaggriculture...")
    subs = api.competition_submissions("kaggriculture", page_size=100)

    target_sub = None
    for s in subs:
        desc = getattr(s, "description", None)
        if desc and target_version in desc:
            target_sub = s
            break

    if not target_sub:
        print(f"Could not find submission '{target_version}'")
        sys.exit(1)

    sub_id = int(target_sub.ref)
    print(f"Found '{target_version}' submission (ID {sub_id}). Fetching episodes...")

    eps = api.competition_list_episodes(sub_id)
    if not eps:
        print("No episodes found.")
        sys.exit(0)

    print(f"Fetched {len(eps)} total episodes.")

    # Sort episodes by id (which is always set)
    eps_sorted = sorted(eps, key=lambda x: getattr(x, "id", 0))

    quartile_start = int(len(eps_sorted) * 0.75)
    last_quartile = eps_sorted[quartile_start:]

    wins = 0
    losses = 0
    ties = 0
    skipped = 0

    print(f"Analyzing last quartile ({len(last_quartile)} episodes)...")
    for ep in last_quartile:
        state = str(getattr(ep, "state", ""))
        if "COMPLETED" not in state:
            skipped += 1
            continue

        agents = getattr(ep, "agents", []) or (ep.get("agents", []) if isinstance(ep, dict) else [])
        if not agents and hasattr(ep, "_agents"):
            agents = ep._agents

        my_reward = None
        opp_reward = None

        for a in agents:
            a_sub_id = getattr(a, "submission_id", None) or (
                a.get("submissionId") if isinstance(a, dict) else None
            )

            reward = getattr(a, "reward", None) or (
                a.get("reward") if isinstance(a, dict) else None
            )
            if reward is None:
                reward = 0.0
            else:
                reward = float(reward)

            if str(a_sub_id) == str(sub_id):
                my_reward = reward
            else:
                opp_reward = reward

        if my_reward is not None and opp_reward is not None:
            if my_reward > opp_reward:
                wins += 1
            elif my_reward < opp_reward:
                losses += 1
            else:
                ties += 1

    total_played = wins + losses + ties
    if skipped > 0:
        print(f"Skipped {skipped} episodes (not COMPLETED).")

    if total_played > 0:
        win_rate = (wins + 0.5 * ties) / total_played
        print(
            f"Converged Win Rate (last quartile): {win_rate:.2%} ({wins}W {losses}L {ties}T out of {total_played})"
        )
    else:
        print("Could not parse agent rewards for the episodes.")


if __name__ == "__main__":
    main()
