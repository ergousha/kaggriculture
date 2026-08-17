import glob
import json
import os
import sys
import time
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kaggle.api.kaggle_api_extended import KaggleApi


def with_retry(fn, *args, what: str, attempts: int = 5, base_delay: float = 2.0):
    """Call a Kaggle API method, backing off on failure.

    This sweep issues one `competition_list_episodes` call per *submission* per team,
    so request volume scales with the field's submission count and the API starts
    refusing partway through. Without a backoff the caller's `except` turns that into
    a silently truncated result set.
    """
    for attempt in range(attempts):
        try:
            return fn(*args)
        except Exception as exc:  # noqa: BLE001 - retried, then re-raised below
            if attempt == attempts - 1:
                raise
            delay = base_delay * (2**attempt)
            print(f"  ! {what} failed ({exc}); retrying in {delay:.0f}s")
            time.sleep(delay)
    return None


def main():
    from submit import load_credentials

    load_credentials()

    api = KaggleApi()
    api.authenticate()

    # 1. Load leaderboard
    lb_files = glob.glob("scratch/kaggriculture-publicleaderboard*.csv")
    if not lb_files:
        print("Leaderboard CSV not found in scratch/. Downloading...")
        os.makedirs("scratch", exist_ok=True)
        api.competition_leaderboard_download("kaggriculture", "scratch/")
        import zipfile

        with zipfile.ZipFile("scratch/kaggriculture.zip", "r") as zip_ref:
            zip_ref.extractall("scratch/")
        lb_files = glob.glob("scratch/kaggriculture-publicleaderboard*.csv")

    df = pd.read_csv(lb_files[0])

    top_teams = df.head(30)

    # Find our team
    my_team = df[df["TeamMemberUserNames"].str.contains("erginakin", case=False, na=False)]
    if my_team.empty:
        raise ValueError("Could not find our team 'erginakin' in the leaderboard CSV.")

    my_team_id = my_team.iloc[0]["TeamId"]
    if my_team.index[0] >= 30:
        top_teams = pd.concat([top_teams, my_team])

    results = []
    top_5_matchmaking = []
    failed_teams = []
    teams_without_submissions = []

    print(f"Fetching active submissions and episodes for {len(top_teams)} teams...")
    for _idx, row in top_teams.iterrows():
        team_id = row["TeamId"]
        team_name = row["TeamName"]
        score = row["Score"]

        try:
            subs = with_retry(
                api.competition_team_submissions, team_id, what=f"{team_name} submissions"
            )
            if not subs:
                print(f"[{team_name}] No active submissions")
                teams_without_submissions.append({"Rank": int(row["Rank"]), "TeamName": team_name})
                continue

            all_episodes = {}
            for sub in subs:
                sub_id = getattr(sub, "id", None)
                if sub_id is None:
                    sub_id = int(sub.ref)

                eps = with_retry(
                    api.competition_list_episodes,
                    sub_id,
                    what=f"{team_name} episodes (submission {sub_id})",
                )
                for ep in eps or []:
                    ep_id = getattr(ep, "id", None) or (
                        ep.get("id") if isinstance(ep, dict) else None
                    )
                    if not ep_id and hasattr(ep, "_id"):
                        ep_id = ep._id
                    all_episodes[ep_id] = ep

            unique_eps = list(all_episodes.values())
            episode_count = len(unique_eps)

            results.append(
                {
                    "Rank": row["Rank"],
                    "TeamId": int(team_id),
                    "TeamName": team_name,
                    "Score": score,
                    "EpisodeCount": episode_count,
                }
            )
            print(f"[{row['Rank']}] {team_name} (Score: {score}) -> {episode_count} episodes")

            if row["Rank"] <= 5:
                # Sort unique episodes by ID descending to get the most recent ones
                unique_eps.sort(
                    key=lambda x: getattr(x, "id", 0) or getattr(x, "_id", 0), reverse=True
                )

                opponents = []
                for ep in unique_eps[:20]:
                    agents = getattr(ep, "agents", []) or (
                        ep.get("agents", []) if isinstance(ep, dict) else []
                    )
                    if not agents and hasattr(ep, "_agents"):
                        agents = ep._agents

                    for a in agents:
                        a_team_id = getattr(a, "team_id", None) or (
                            a.get("teamId") if isinstance(a, dict) else None
                        )
                        if str(a_team_id) != str(team_id):
                            # We get the opponent's name if available, otherwise just use their ID
                            a_team_name = getattr(a, "team_name", None) or (
                                a.get("teamName") if isinstance(a, dict) else str(a_team_id)
                            )
                            opponents.append(a_team_name)

                top_5_matchmaking.append(
                    {
                        "TeamName": team_name,
                        "RecentOpponents": opponents,
                    }
                )

        except Exception as e:
            print(f"Error fetching data for {team_name}: {e}")
            failed_teams.append({"Rank": int(row["Rank"]), "TeamName": team_name, "Error": str(e)})

    coverage = {
        "TeamsRequested": int(len(top_teams)),
        "TeamsCollected": len(results),
        "TeamsFailed": failed_teams,
        "TeamsWithoutSubmissions": teams_without_submissions,
        "Complete": len(results) == len(top_teams),
    }

    # 2. Plot
    df_results = pd.DataFrame(results)
    if df_results.empty:
        print("No team data collected — see the errors above.")
        sys.exit(1)

    plt.figure(figsize=(10, 6))
    plt.scatter(df_results["EpisodeCount"], df_results["Score"], alpha=0.7)

    for _i, row in df_results.iterrows():
        if row["Rank"] <= 5 or row["TeamId"] == my_team_id:
            plt.annotate(
                row["TeamName"],
                (row["EpisodeCount"], row["Score"]),
                xytext=(5, 5),
                textcoords="offset points",
            )

    plt.xlabel("Episode Count (Active Submissions)")
    plt.ylabel("Leaderboard Score")
    plt.title(
        f"Score vs Episode Count ({coverage['TeamsCollected']} of "
        f"{coverage['TeamsRequested']} teams)"
    )
    plt.grid(True)

    os.makedirs("logs", exist_ok=True)
    plt.savefig("logs/score_vs_episodes.png")
    print("\nSaved plot to logs/score_vs_episodes.png")

    # 3. Output matchmaking analysis
    print("\n--- Matchmaking for Top 5 Teams (last 20 episodes) ---")
    for team_data in top_5_matchmaking:
        print(f"\nTeam: {team_data['TeamName']}")
        counts = Counter(team_data["RecentOpponents"])
        for opp, count in counts.most_common():
            print(f"  {opp}: {count}")

    # Save results. `coverage` goes in the artifact so a truncated run is
    # self-identifying rather than reading as a complete sweep.
    with open("logs/leaderboard_research.json", "w") as f:
        json.dump(
            {
                "coverage": coverage,
                "results": results,
                "top_5_matchmaking": top_5_matchmaking,
            },
            f,
            indent=2,
        )

    if not coverage["Complete"]:
        print(
            f"\nINCOMPLETE: collected {coverage['TeamsCollected']} of "
            f"{coverage['TeamsRequested']} teams "
            f"({len(failed_teams)} errored, {len(teams_without_submissions)} without submissions)."
        )
        print("Do not quote whole-field conclusions from this run; re-run before use.")
        sys.exit(1)


if __name__ == "__main__":
    main()
