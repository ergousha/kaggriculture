import pandas as pd
import numpy as np

# Load the fingerprints
df = pd.read_csv('logs/daily_fingerprints.csv')

# Drop rows without a team name
df = df.dropna(subset=['team'])
df = df[df['team'] != '']

# Group by team
team_stats = df.groupby('team').agg(
    matches=('episode_id', 'count'),
    mean_cash=('cash', 'mean'),
    median_cash=('cash', 'median'),
    max_cash=('cash', 'max')
).reset_index()

# Filter out teams with too few matches (e.g., less than 5) to have meaningful variance
team_stats = team_stats[team_stats['matches'] >= 5]

# Get top 5 scorers based on median cash
top_5 = team_stats.sort_values('median_cash', ascending=False).head(5)
print("Top 5 Teams by Median Cash:")
print(top_5)

# Columns to check for route similarity (op counts)
op_cols = [c for c in df.columns if c.startswith('op_') and c not in ('ops_productive', 'ops_logistics', 'ops_total')]

print("\n--- Route Analysis for Top 5 Teams ---")
for _, row in top_5.iterrows():
    team = row['team']
    team_data = df[df['team'] == team]
    
    print(f"\nTeam: {team} (Matches: {len(team_data)})")
    
    # Calculate step variance
    steps_std = team_data['steps'].std()
    steps_mean = team_data['steps'].mean()
    print(f"  Steps: {steps_mean:.1f} (std: {steps_std:.2f})")
    
    # Calculate standard deviation for each op column
    std_ops = team_data[op_cols].std().fillna(0)
    mean_ops = team_data[op_cols].mean().fillna(0)
    
    # Check if ops are identical (std == 0) across matches
    total_std = std_ops.sum()
    if total_std < 5:
        print("  Route Pattern: HIGHLY STATIC (Likely a recorded route)")
    elif total_std < 50:
        print("  Route Pattern: SEMI-STATIC (Likely a recorded route with slight runtime variations)")
    else:
        print("  Route Pattern: DYNAMIC (Plays differently each match)")
        
    print(f"  Total Op Variance (sum of std devs): {total_std:.2f}")
    
    # Print top 3 most used ops for context
    top_ops = mean_ops.sort_values(ascending=False).head(3)
    ops_str = ", ".join([f"{k.replace('op_', '')}: {v:.1f}" for k, v in top_ops.items()])
    print(f"  Primary Ops: {ops_str}")
    
    # Show variance of productive ops
    prod_std = team_data['ops_productive'].std()
    log_std = team_data['ops_logistics'].std()
    print(f"  Productive Ops std: {prod_std:.2f}, Logistics Ops std: {log_std:.2f}")
