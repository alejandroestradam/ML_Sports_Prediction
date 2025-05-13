import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Create output directory for processed data
os.makedirs("processed_data", exist_ok=True)

print("Starting feature engineering process...")

def load_team_data():
    """Load and combine team statistics from all teams"""
    all_team_stats = []
    match_data = []
    
    # Get all team folders
    team_folders = [f for f in os.listdir("collected_data") if os.path.isdir(os.path.join("collected_data", f)) 
                   and f not in ['team_stats_2020-2021', 'team_stats_2021-2022', 'team_stats_2022-2023', 'team_stats_2023-2024']]
    
    for team_folder in team_folders:
        # Process team match stats for each season
        season_folders = [f for f in sorted(os.listdir("collected_data")) if f.startswith("team_stats_")]
        
        for season in season_folders:
            season_name = season.replace("team_stats_", "")
            
            # Load matches data if available
            match_file = os.path.join("collected_data", team_folder, f"matches_{season_name}.csv")
            if os.path.exists(match_file):
                try:
                    df = pd.read_csv(match_file)
                    df['team_folder'] = team_folder
                    match_data.append(df)
                    print(f"Loaded matches for {team_folder} in season {season_name}")
                except Exception as e:
                    print(f"Error loading {match_file}: {e}")
            
            # Load team match stats
            team_stats_file = os.path.join("collected_data", team_folder, f"team_match_stats_{season_name}.csv")
            if os.path.exists(team_stats_file):
                try:
                    df = pd.read_csv(team_stats_file)
                    df['team_folder'] = team_folder
                    all_team_stats.append(df)
                    print(f"Loaded team stats for {team_folder} in season {season_name}")
                except Exception as e:
                    print(f"Error loading {team_stats_file}: {e}")
    
    # Combine all data
    team_stats_df = pd.concat(all_team_stats, ignore_index=True) if all_team_stats else pd.DataFrame()
    matches_df = pd.concat(match_data, ignore_index=True) if match_data else pd.DataFrame()
    
    return team_stats_df, matches_df

def preprocess_data(team_stats_df, matches_df):
    """Preprocess the raw data for feature engineering"""
    
    # Convert date columns to datetime
    if not team_stats_df.empty and 'match_date' in team_stats_df.columns:
        team_stats_df['match_date'] = pd.to_datetime(team_stats_df['match_date'], errors='coerce')
    
    if not matches_df.empty and 'match_date' in matches_df.columns:
        matches_df['match_date'] = pd.to_datetime(matches_df['match_date'], errors='coerce')
    
    return team_stats_df, matches_df

def calculate_rolling_averages(df, group_cols, value_cols, window_size=3):
    """Calculate rolling averages for specified columns"""
    if df.empty:
        return df
    
    # Sort by team and date
    df = df.sort_values(group_cols + ['match_date'])
    
    # Group by team
    grouped = df.groupby(group_cols)
    
    # Calculate rolling averages
    for col in value_cols:
        if col in df.columns:
            df[f'{col}_rolling_{window_size}'] = grouped[col].transform(
                lambda x: x.rolling(window=window_size, min_periods=1).mean()
            )
    
    return df

def create_form_indicators(team_stats_df):
    """Create form indicators based on recent performance"""
    if team_stats_df.empty:
        return team_stats_df
    
    # Extract relevant metrics for team form
    form_metrics = ['Goals_average', 'Goals Conceded_average', 'Team Wins_percentage']
    
    # Filter for form metrics
    form_df = team_stats_df[team_stats_df['statistic_type'].isin(form_metrics)].copy()
    
    # Create team form indicators
    form_df = calculate_rolling_averages(
        form_df,
        group_cols=['team_id', 'team_name'],
        value_cols=['value'],
        window_size=3
    )
    
    # Rename columns for clarity
    form_df = form_df.rename(columns={'value_rolling_3': 'form_value'})
    
    # Merge back to original dataframe
    result = pd.merge(
        team_stats_df,
        form_df[['team_id', 'team_name', 'match_date', 'statistic_type', 'form_value']],
        on=['team_id', 'team_name', 'match_date', 'statistic_type'],
        how='left'
    )
    
    return result

def generate_team_strength_metrics(team_stats_df):
    """Generate team strength metrics based on performance stats"""
    if team_stats_df.empty:
        return team_stats_df
    
    # Create a pivot table with key metrics for each team/match
    pivot_stats = team_stats_df[team_stats_df['statistic_type'].isin(['Goals_average', 'Goals Conceded_average', 'Team Wins_percentage'])]
    
    # Calculate team strength as a weighted sum of key metrics
    # Higher values for goals and wins, lower values for conceded goals
    strength_df = pivot_stats.copy()
    
    # Create weights dictionary
    weights = {
        'Goals_average': 1.0,
        'Goals Conceded_average': -0.8,  # Negative weight for goals conceded
        'Team Wins_percentage': 1.2
    }
    
    # Apply weights to each statistic
    strength_df['weighted_value'] = strength_df.apply(
        lambda row: row['value'] * weights.get(row['statistic_type'], 0),
        axis=1
    )
    
    # Group by team and match to calculate overall strength
    team_strength = strength_df.groupby(['team_id', 'team_name', 'match_date'])['weighted_value'].sum().reset_index()
    team_strength = team_strength.rename(columns={'weighted_value': 'team_strength'})
    
    # Normalize team strength to 0-1 scale
    min_strength = team_strength['team_strength'].min()
    max_strength = team_strength['team_strength'].max()
    
    if max_strength > min_strength:
        team_strength['team_strength_normalized'] = (team_strength['team_strength'] - min_strength) / (max_strength - min_strength)
    else:
        team_strength['team_strength_normalized'] = 0.5  # Default if all values are the same
    
    # Merge strength metrics back to the original dataframe
    result = pd.merge(
        team_stats_df,
        team_strength[['team_id', 'team_name', 'match_date', 'team_strength', 'team_strength_normalized']],
        on=['team_id', 'team_name', 'match_date'],
        how='left'
    )
    
    return result

def encode_match_context(team_stats_df, matches_df):
    """Encode match context including home/away advantage and opponent strength"""
    if team_stats_df.empty:
        return team_stats_df
    
    print(f"Encoding match context... Shape of team_stats_df: {team_stats_df.shape}")
    
    # Extract home/away information
    if 'is_home' in team_stats_df.columns:
        home_advantage = team_stats_df['is_home'].astype(int)
        team_stats_df['home_advantage'] = home_advantage
        print(f"Home advantage feature created based on is_home column")
    else:
        print("WARNING: is_home column not found, cannot create home advantage feature")
        team_stats_df['home_advantage'] = 0  # Default value
    
    # Check if we have team strength metrics
    if 'team_strength_normalized' not in team_stats_df.columns:
        print("WARNING: team_strength_normalized column not found, opponent strength cannot be calculated")
        return team_stats_df
    
    # Check if we have matches data with opponents
    if matches_df.empty:
        print("WARNING: No matches data available, opponent strength cannot be calculated")
        return team_stats_df
    
    # Check if required columns exist in matches_df
    required_cols = ['match_id', 'home_team_id', 'away_team_id']
    if not all(col in matches_df.columns for col in required_cols):
        print(f"WARNING: Required columns {required_cols} not found in matches_df")
        print(f"Available columns in matches_df: {matches_df.columns.tolist()}")
        return team_stats_df
    
    try:
        # Get the latest team strength metrics
        latest_strength = team_stats_df.groupby(['team_id', 'match_date'])['team_strength_normalized'].last().reset_index()
        print(f"Created team strength lookup with {len(latest_strength)} entries")
        
        # Create a lookup table
        strength_lookup = latest_strength.set_index(['team_id', 'match_date'])['team_strength_normalized'].to_dict()
        
        # Make sure match_id column exists in team_stats_df for joining
        if 'match_id' not in team_stats_df.columns:
            print("WARNING: match_id column not found in team_stats_df, cannot join with matches data")
            return team_stats_df
        
        # Join home and away team IDs from matches_df to team_stats_df
        team_stats_df = pd.merge(
            team_stats_df,
            matches_df[['match_id', 'home_team_id', 'away_team_id']],
            on='match_id',
            how='left'
        )
        
        print(f"Joined match data. Shape after join: {team_stats_df.shape}")
        
        # Function to look up opponent strength
        def get_opponent_strength(row):
            # First check if we have is_home to determine opponent
            if 'is_home' in row and pd.notnull(row['is_home']):
                if row['is_home']:
                    opponent_id = row['away_team_id'] if pd.notnull(row.get('away_team_id', np.nan)) else None
                else:
                    opponent_id = row['home_team_id'] if pd.notnull(row.get('home_team_id', np.nan)) else None
            else:
                # If no is_home, we can't determine opponent
                return np.nan
                
            # Check if we have opponent_id and match_date for lookup
            if opponent_id and pd.notnull(row.get('match_date', np.nan)):
                match_date = row['match_date']
                # Try to find exact match in lookup
                if (opponent_id, match_date) in strength_lookup:
                    return strength_lookup[(opponent_id, match_date)]
                
                # If no exact match, try to find the closest date
                opponent_dates = [d for (team, d) in strength_lookup.keys() if team == opponent_id]
                if opponent_dates:
                    closest_date = min(opponent_dates, key=lambda d: abs((d - match_date).total_seconds()))
                    return strength_lookup[(opponent_id, closest_date)]
            
            return np.nan
        
        # Apply the lookup to get opponent strength - wrapped in try/except for safety
        try:
            team_stats_df['opponent_strength'] = team_stats_df.apply(get_opponent_strength, axis=1)
            print(f"Created opponent_strength feature with {team_stats_df['opponent_strength'].count()} non-null values")
            
            # Calculate relative strength (team strength compared to opponent)
            team_stats_df['relative_strength'] = team_stats_df['team_strength_normalized'] - team_stats_df['opponent_strength']
            print(f"Created relative_strength feature")
        except Exception as e:
            print(f"Error calculating opponent strength: {e}")
    
    except Exception as e:
        print(f"Error in encode_match_context: {e}")
    
    return team_stats_df

def create_match_outcome_target(team_stats_df, matches_df):
    """Create target variables for match outcome prediction"""
    if team_stats_df.empty or matches_df.empty:
        return team_stats_df
    
    # Join match results to team stats
    match_results = matches_df[['match_id', 'home_team_id', 'away_team_id', 'home_score', 'away_score']].copy()
    
    # Create a match outcome column (win=2, draw=1, loss=0)
    match_results['home_outcome'] = match_results.apply(
        lambda row: 2 if row['home_score'] > row['away_score'] else (1 if row['home_score'] == row['away_score'] else 0),
        axis=1
    )
    
    match_results['away_outcome'] = match_results.apply(
        lambda row: 2 if row['away_score'] > row['home_score'] else (1 if row['away_score'] == row['home_score'] else 0),
        axis=1
    )
    
    # Merge to create match outcome targets
    team_stats_with_outcomes = pd.merge(
        team_stats_df,
        match_results,
        on='match_id',
        how='left'
    )
    
    # Assign the correct outcome based on whether team was home or away
    team_stats_with_outcomes['match_outcome'] = team_stats_with_outcomes.apply(
        lambda row: row['home_outcome'] if row['is_home'] else row['away_outcome'],
        axis=1
    )
    
    return team_stats_with_outcomes

def create_time_based_features(df):
    """Create time-based features like day of week, month, etc."""
    if df.empty or 'match_date' not in df.columns:
        return df
    
    # Extract time components
    df['match_day_of_week'] = df['match_date'].dt.dayofweek
    df['match_month'] = df['match_date'].dt.month
    df['match_is_weekend'] = df['match_day_of_week'].isin([5, 6]).astype(int)  # 5=Saturday, 6=Sunday
    
    # Create Liga MX tournament indicators (Apertura/Clausura)
    # In Liga MX: Apertura typically runs Jul-Dec, Clausura runs Jan-May
    df['tournament_type'] = df['match_date'].dt.month.apply(
        lambda m: 'Apertura' if m >= 7 else 'Clausura'
    )
    
    # Create match day indicators based on date ordering within tournament
    for season in df['season_name'].unique():
        for tournament in ['Apertura', 'Clausura']:
            # Filter for current season and tournament
            mask = (df['season_name'] == season) & (df['tournament_type'] == tournament)
            
            if sum(mask) > 0:
                # Sort dates within tournament
                tournament_dates = sorted(df.loc[mask, 'match_date'].unique())
                
                if tournament_dates:
                    # Create a dictionary mapping dates to matchdays
                    date_to_matchday = {date: idx + 1 for idx, date in enumerate(tournament_dates)}
                    
                    # Apply matchday mapping
                    df.loc[mask, 'matchday'] = df.loc[mask, 'match_date'].map(date_to_matchday)
                    
                    # Total matchdays in this tournament
                    total_matchdays = len(tournament_dates)
                    
                    # Create tournament phase (regular season vs liguilla/playoffs)
                    # Liga MX typically has 17 regular season matchdays, then liguilla
                    if total_matchdays > 0:
                        df.loc[mask, 'tournament_phase'] = df.loc[mask, 'matchday'].apply(
                            lambda day: 'liguilla' if day is not None and day > 17 else 'regular_season'
                        )
                        
                        # Tournament progress (0-1 scale)
                        df.loc[mask, 'tournament_progress'] = df.loc[mask, 'matchday'].apply(
                            lambda day: (day / 17) if day is not None and day <= 17 else 1.0
                        )
                        
                        # Create tournament phase with more granularity for regular season
                        df.loc[mask, 'detailed_phase'] = df.loc[mask, 'matchday'].apply(
                            lambda day: 'liguilla' if day is not None and day > 17 
                                else ('early' if day is not None and day <= 6 
                                      else ('mid' if day is not None and day <= 12 else 'late'))
                        )
    
    return df

def load_player_data():
    """Load player performance data from all teams"""
    all_player_data = []
    
    # Get all team folders
    team_folders = [f for f in os.listdir("collected_data") if os.path.isdir(os.path.join("collected_data", f)) 
                   and f not in ['team_stats_2020-2021', 'team_stats_2021-2022', 'team_stats_2022-2023', 'team_stats_2023-2024']]
    
    for team_folder in team_folders:
        # Find player_performance_data_all_seasons.csv file
        player_file = os.path.join("collected_data", team_folder, "player_performance_data_all_seasons.csv")
        if os.path.exists(player_file):
            try:
                df = pd.read_csv(player_file)
                # Add team folder for reference
                df['team_folder'] = team_folder  
                all_player_data.append(df)
                print(f"Loaded player data for {team_folder}")
            except Exception as e:
                print(f"Error loading {player_file}: {e}")
    
    # Combine all player data
    if all_player_data:
        return pd.concat(all_player_data, ignore_index=True)
    else:
        print("No player data found")
        return pd.DataFrame()

def process_player_data(player_df):
    """Process player performance data to use in feature engineering"""
    if player_df.empty:
        return pd.DataFrame()
    
    print("Processing player performance data...")
    
    # Handle potential missing columns
    required_cols = ['player_id', 'player_name', 'season_name', 'team_id', 'team_name', 
                     'position', 'appearances', 'minutes_played', 'goals', 'assists']
    
    missing_cols = [col for col in required_cols if col not in player_df.columns]
    if missing_cols:
        print(f"Warning: Missing columns in player data: {missing_cols}")
        for col in missing_cols:
            player_df[col] = 0 if col in ['appearances', 'minutes_played', 'goals', 'assists'] else ''
    
    # Calculate goals per minute and assists per minute (only for players with minutes)
    player_df['goals_per_minute'] = np.where(player_df['minutes_played'] > 0, 
                                           player_df['goals'] / player_df['minutes_played'], 
                                           0)
    
    player_df['assists_per_minute'] = np.where(player_df['minutes_played'] > 0, 
                                             player_df['assists'] / player_df['minutes_played'], 
                                             0)
    
    # Calculate goal contributions (goals + assists)
    player_df['goal_contributions'] = player_df['goals'] + player_df['assists']
    
    # Calculate goal contributions per 90 minutes
    player_df['goal_contributions_per_90'] = np.where(player_df['minutes_played'] > 0,
                                                    player_df['goal_contributions'] * 90 / player_df['minutes_played'],
                                                    0)
    
    # Flag for key players (played significant minutes)
    # Consider players who played in more than 10 matches or 900 minutes as key players
    player_df['is_key_player'] = ((player_df['appearances'] >= 10) | 
                                 (player_df['minutes_played'] >= 900)).astype(int)
    
    # Create a player efficiency rating
    # This is a simple formula that weights goals, assists and playing time
    player_df['player_efficiency'] = (
        (player_df['goals'] * 0.6) + 
        (player_df['assists'] * 0.4) + 
        (player_df['minutes_played'] / 1000 * 0.2)
    ) / (player_df['appearances'] + 1)  # +1 to avoid division by zero
    
    return player_df

def create_team_player_features(player_df, team_stats_df, matches_df):
    """Create team-level aggregated features from player data"""
    if player_df.empty:
        return team_stats_df
    
    print("Creating team-level features from player data...")
    
    # First, check column names in matches_df to determine the correct column names
    print(f"Available columns in matches_df: {matches_df.columns.tolist()}")
    
    # Map expected columns to actual columns
    match_id_col = 'match_id' if 'match_id' in matches_df.columns else None
    season_name_col = 'season_name' if 'season_name' in matches_df.columns else None
    home_team_col = next((col for col in matches_df.columns if 'home' in col.lower() and 'team' in col.lower() and 'name' in col.lower()), None)
    away_team_col = next((col for col in matches_df.columns if 'away' in col.lower() and 'team' in col.lower() and 'name' in col.lower()), None)
    
    # Alternative column names if standard ones not found
    if not match_id_col:
        match_id_col = next((col for col in matches_df.columns if 'match' in col.lower() and 'id' in col.lower()), None)
    if not season_name_col:
        season_name_col = next((col for col in matches_df.columns if 'season' in col.lower()), None)
    if not home_team_col:
        home_team_col = 'home_team' if 'home_team' in matches_df.columns else None
    if not away_team_col:
        away_team_col = 'away_team' if 'away_team' in matches_df.columns else None
    
    print(f"Using columns: match_id={match_id_col}, season_name={season_name_col}, home_team={home_team_col}, away_team={away_team_col}")
    
    # Check if we have the necessary columns
    if not all([match_id_col, season_name_col, home_team_col, away_team_col]):
        print("WARNING: Required columns not found in matches_df. Cannot integrate player data.")
        return team_stats_df
    
    # Group by team and season to calculate team-level metrics
    team_player_stats = player_df.groupby(['team_name', 'season_name']).agg({
        'player_id': 'count',  # Number of players with data
        'appearances': 'sum',  # Total appearances
        'minutes_played': 'sum',  # Total minutes
        'goals': 'sum',  # Total goals
        'assists': 'sum',  # Total assists
        'goal_contributions': 'sum',  # Total goal contributions
        'player_efficiency': 'mean',  # Average player efficiency
        'is_key_player': 'sum',  # Number of key players
        'goal_contributions_per_90': lambda x: np.average(x, weights=player_df.loc[x.index, 'minutes_played'])  # Weighted average
    }).reset_index()
    
    # Rename columns for clarity
    team_player_stats.columns = [
        'team_name', 'season_name', 'num_players', 'total_appearances', 
        'total_minutes', 'player_total_goals', 'player_total_assists', 
        'player_total_contributions', 'avg_player_efficiency',
        'num_key_players', 'weighted_contributions_per_90'
    ]
    
    # Create additional derived metrics
    # Team experience metric (total minutes accumulated by all players)
    team_player_stats['team_experience'] = team_player_stats['total_minutes'] / 1000  # Scale down
    
    # Key player ratio (percentage of key players in squad)
    team_player_stats['key_player_ratio'] = team_player_stats['num_key_players'] / team_player_stats['num_players']
    
    # Calculate top scorer goals for each team/season
    top_scorers = player_df.sort_values('goals', ascending=False).groupby(['team_name', 'season_name']).head(1)
    top_scorers = top_scorers[['team_name', 'season_name', 'player_name', 'goals']]
    top_scorers.columns = ['team_name', 'season_name', 'top_scorer', 'top_scorer_goals']
    
    # Add top scorer information
    team_player_stats = pd.merge(
        team_player_stats,
        top_scorers,
        on=['team_name', 'season_name'],
        how='left'
    )
    
    # Now merge these team-level player stats with the matches dataframe
    # Create a subset of matches_df with the required columns
    matches_subset = matches_df[[match_id_col, season_name_col, home_team_col, away_team_col]].copy()
    
    # First try to merge based on the home team
    try:
        # Join for home team
        team_player_stats_expanded = pd.merge(
            matches_subset,
            team_player_stats,
            left_on=[season_name_col, home_team_col],
            right_on=['season_name', 'team_name'],
            how='left'
        )
        
        # Rename for home team
        home_team_stats = {
            'num_players': 'home_num_players',
            'total_appearances': 'home_total_appearances',
            'total_minutes': 'home_total_minutes',
            'player_total_goals': 'home_player_total_goals',
            'player_total_assists': 'home_player_total_assists',
            'player_total_contributions': 'home_player_total_contributions',
            'avg_player_efficiency': 'home_avg_player_efficiency',
            'num_key_players': 'home_num_key_players',
            'weighted_contributions_per_90': 'home_weighted_contributions_per_90',
            'team_experience': 'home_team_experience',
            'key_player_ratio': 'home_key_player_ratio',
            'top_scorer': 'home_top_scorer',
            'top_scorer_goals': 'home_top_scorer_goals'
        }
        
        team_player_stats_expanded = team_player_stats_expanded.rename(columns=home_team_stats)
        
        # Join for away team
        team_player_stats_expanded = pd.merge(
            team_player_stats_expanded,
            team_player_stats,
            left_on=[season_name_col, away_team_col],
            right_on=['season_name', 'team_name'],
            how='left',
            suffixes=('', '_away')
        )
        
        # Rename for away team
        away_team_stats = {
            'num_players': 'away_num_players',
            'total_appearances': 'away_total_appearances',
            'total_minutes': 'away_total_minutes',
            'player_total_goals': 'away_player_total_goals',
            'player_total_assists': 'away_player_total_assists',
            'player_total_contributions': 'away_player_total_contributions',
            'avg_player_efficiency': 'away_avg_player_efficiency',
            'num_key_players': 'away_num_key_players',
            'weighted_contributions_per_90': 'away_weighted_contributions_per_90',
            'team_experience': 'away_team_experience',
            'key_player_ratio': 'away_key_player_ratio',
            'top_scorer': 'away_top_scorer',
            'top_scorer_goals': 'away_top_scorer_goals'
        }
        
        team_player_stats_expanded = team_player_stats_expanded.rename(columns=away_team_stats)
        
        # Calculate comparative features between home and away teams
        # Only calculate if both values are available
        team_player_stats_expanded['experience_diff'] = team_player_stats_expanded['home_team_experience'].sub(
            team_player_stats_expanded['away_team_experience'], fill_value=0)
        
        team_player_stats_expanded['key_player_diff'] = team_player_stats_expanded['home_num_key_players'].sub(
            team_player_stats_expanded['away_num_key_players'], fill_value=0)
        
        team_player_stats_expanded['efficiency_diff'] = team_player_stats_expanded['home_avg_player_efficiency'].sub(
            team_player_stats_expanded['away_avg_player_efficiency'], fill_value=0)
        
        team_player_stats_expanded['contribution_diff'] = team_player_stats_expanded['home_weighted_contributions_per_90'].sub(
            team_player_stats_expanded['away_weighted_contributions_per_90'], fill_value=0)
        
        # Select only the relevant columns for merging with team_stats_df
        player_features = [
            match_id_col, 
            'home_num_players', 'home_num_key_players', 
            'home_avg_player_efficiency', 'home_team_experience',
            'home_weighted_contributions_per_90', 'home_key_player_ratio',
            'away_num_players', 'away_num_key_players', 
            'away_avg_player_efficiency', 'away_team_experience',
            'away_weighted_contributions_per_90', 'away_key_player_ratio',
            'experience_diff', 'key_player_diff', 'efficiency_diff', 'contribution_diff'
        ]
        
        # Filter columns that actually exist
        player_features = [col for col in player_features if col in team_player_stats_expanded.columns]
        
        # Join with team_stats_df on match_id
        if 'match_id' in team_stats_df.columns:
            result = pd.merge(
                team_stats_df,
                team_player_stats_expanded[player_features],
                left_on='match_id',
                right_on=match_id_col,
                how='left'
            )
            print(f"Added player-based features to team stats. New shape: {result.shape}")
            return result
        else:
            print("WARNING: match_id column not found in team_stats_df. Cannot integrate player data.")
            return team_stats_df
    
    except Exception as e:
        print(f"Error integrating player data: {e}")
        return team_stats_df

def build_engineered_dataset(team_stats_df, matches_df, player_df=None):
    """Combine all feature engineering steps to build the final dataset"""
    print("Building engineered dataset...")
    
    # Preprocess the data
    team_stats_df, matches_df = preprocess_data(team_stats_df, matches_df)
    
    # 1. Create form indicators
    print("Creating form indicators...")
    enhanced_df = create_form_indicators(team_stats_df)
    
    # 2. Generate team strength metrics
    print("Generating team strength metrics...")
    enhanced_df = generate_team_strength_metrics(enhanced_df)
    
    # 3. Encode match context
    print("Encoding match context...")
    enhanced_df = encode_match_context(enhanced_df, matches_df)
    
    # 4. Create time-based features
    print("Creating time-based features...")
    enhanced_df = create_time_based_features(enhanced_df)
    
    # 5. Create match outcome targets
    print("Creating match outcome targets...")
    enhanced_df = create_match_outcome_target(enhanced_df, matches_df)
    
    # 6. Integrate player data if available
    if player_df is not None and not player_df.empty:
        print("Processing player data and integrating with team stats...")
        processed_player_df = process_player_data(player_df)
        enhanced_df = create_team_player_features(processed_player_df, enhanced_df, matches_df)
    
    # Save the engineered dataset
    enhanced_df.to_csv("processed_data/engineered_features.csv", index=False)
    print(f"Saved engineered dataset with {enhanced_df.shape[0]} rows and {enhanced_df.shape[1]} columns")
    
    return enhanced_df

def create_aggregate_features(df):
    """Create aggregated features for modeling"""
    if df.empty:
        return pd.DataFrame()
    
    print("Creating aggregated features for modeling...")
    
    # Get key statistics for each team and match
    key_stats = [
        'Goals_average', 'Goals Conceded_average', 'Team Wins_percentage',
        'Cleansheets_percentage', 'Failed To Score_percentage'
    ]
    
    # Filter for our key statistics
    model_df = df[df['statistic_type'].isin(key_stats)].copy()
    
    # Define required columns and check which ones are available
    required_index_cols = ['team_id', 'team_name', 'match_id', 'match_date', 'season_name']
    optional_index_cols = ['is_home', 'team_strength', 'opponent_strength', 'match_outcome', 'tournament_phase', 'detailed_phase']
    
    # Check which columns actually exist in the dataframe
    available_index_cols = required_index_cols + [col for col in optional_index_cols if col in model_df.columns]
    
    print(f"Available columns for pivot table: {available_index_cols}")
    
    try:
        # Pivot to create wide format with one row per team/match
        pivot_df = model_df.pivot_table(
            index=available_index_cols,
            columns='statistic_type',
            values=['value', 'form_value'] if 'form_value' in model_df.columns else ['value'],
            aggfunc='mean'  # Use mean in case there are duplicate entries
        ).reset_index()
        
        # Flatten the multi-level column names
        pivot_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in pivot_df.columns.values]
        
        # Save the aggregate features
        pivot_df.to_csv("processed_data/model_features.csv", index=False)
        print(f"Saved aggregate features with {pivot_df.shape[0]} rows and {pivot_df.shape[1]} columns")
        
        return pivot_df
    except Exception as e:
        print(f"Error creating aggregate features: {e}")
        print("Creating simplified aggregate features...")
        
        # Create a simplified version if the pivot fails
        simplified_pivot = model_df.pivot_table(
            index=['team_id', 'team_name', 'season_name'],
            columns='statistic_type',
            values='value',
            aggfunc='mean'
        ).reset_index()
        
        simplified_pivot.to_csv("processed_data/simplified_model_features.csv", index=False)
        print(f"Saved simplified aggregate features with {simplified_pivot.shape[0]} rows and {simplified_pivot.shape[1]} columns")
        
        return simplified_pivot

def visualize_engineered_features(df):
    """Create visualizations of the engineered features"""
    print("Creating visualizations of engineered features...")
    
    # Create output directory
    os.makedirs("plots/feature_engineering", exist_ok=True)
    
    # 1. Team strength distribution
    if 'team_strength_normalized' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.histplot(df['team_strength_normalized'].dropna(), bins=20, kde=True)
        plt.title('Distribution of Normalized Team Strength')
        plt.xlabel('Team Strength (0-1 scale)')
        plt.savefig('plots/feature_engineering/team_strength_distribution.png')
        # plt.show()
        
        # Team strength by team
        team_avg_strength = df.groupby('team_name')['team_strength_normalized'].mean().sort_values(ascending=False).reset_index()
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='team_strength_normalized', y='team_name', data=team_avg_strength)
        plt.title('Average Team Strength by Team')
        plt.xlabel('Average Normalized Strength')
        plt.tight_layout()
        plt.savefig('plots/feature_engineering/team_strength_by_team.png')
        plt.show()
    
    # 2. Home advantage effect
    if 'is_home' in df.columns and 'value' in df.columns:
        home_advantage = df[df['statistic_type'] == 'Team Wins_percentage'].groupby('is_home')['value'].mean().reset_index()
        
        plt.figure(figsize=(8, 6))
        sns.barplot(x='is_home', y='value', data=home_advantage)
        plt.title('Home Advantage Effect on Win Percentage')
        plt.xlabel('Home Team')
        plt.ylabel('Average Win Percentage')
        plt.xticks([0, 1], ['Away', 'Home'])
        plt.savefig('plots/feature_engineering/home_advantage_effect.png')
        plt.show()
    
    # 3. Tournament type effect
    if 'tournament_type' in df.columns and 'value' in df.columns:
        tournament_effect = df[df['statistic_type'] == 'Team Wins_percentage'].groupby('tournament_type')['value'].mean().reset_index()
        
        if not tournament_effect.empty and len(tournament_effect) > 1:
            plt.figure(figsize=(8, 6))
            sns.barplot(x='tournament_type', y='value', data=tournament_effect)
            plt.title('Tournament Type Effect on Win Percentage')
            plt.xlabel('Tournament')
            plt.ylabel('Average Win Percentage')
            plt.savefig('plots/feature_engineering/tournament_type_effect.png')
            plt.show()
    
    # 4. Tournament phase effect
    if 'tournament_phase' in df.columns and 'value' in df.columns:
        phase_effect = df[df['statistic_type'] == 'Team Wins_percentage'].groupby('tournament_phase')['value'].mean().reset_index()
        
        if not phase_effect.empty and len(phase_effect) > 1:
            plt.figure(figsize=(8, 6))
            sns.barplot(x='tournament_phase', y='value', data=phase_effect)
            plt.title('Tournament Phase Effect on Win Percentage')
            plt.xlabel('Tournament Phase')
            plt.ylabel('Average Win Percentage')
            plt.savefig('plots/feature_engineering/tournament_phase_effect.png')
            plt.show()
    
    # 5. Correlation matrix of engineered features
    try:
        # Try to read model features file, falling back to simplified version if needed
        model_features_path = "processed_data/model_features.csv"
        simplified_path = "processed_data/simplified_model_features.csv"
        
        if os.path.exists(model_features_path):
            model_features = pd.read_csv(model_features_path)
            print(f"Loaded model features from {model_features_path}")
        elif os.path.exists(simplified_path):
            model_features = pd.read_csv(simplified_path)
            print(f"Loaded simplified model features from {simplified_path}")
        else:
            model_features = pd.DataFrame()
            print("No model features file found for correlation analysis")
        
        if not model_features.empty:
            numeric_cols = model_features.select_dtypes(include=[np.number]).columns
            
            # Select a subset of important features to avoid overcrowding
            important_features = [col for col in numeric_cols if any(key in col for key in 
                                ['Goals', 'Wins', 'strength', 'form', 'Team', 'outcome', 'Cleansheets'])]
            
            # Limit to 10 features if we have too many
            if len(important_features) > 10:
                important_features = important_features[:10]
            
            if important_features:
                corr_matrix = model_features[important_features].corr()
                
                plt.figure(figsize=(12, 10))
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5)
                plt.title('Correlation Matrix of Key Engineered Features')
                plt.tight_layout()
                plt.savefig('plots/feature_engineering/feature_correlation.png')
                plt.show()
                
                print("Generated correlation matrix visualization")
    except Exception as e:
        print(f"Error generating correlation matrix: {e}")
    
    # 6. Tournament progress effect on performance (if available)
    if 'tournament_progress' in df.columns and 'value' in df.columns:
        try:
            # Create bins for tournament progress
            df['progress_bin'] = pd.cut(df['tournament_progress'], bins=5, labels=['Very Early', 'Early', 'Mid', 'Late', 'Very Late'])
            progress_effect = df[df['statistic_type'] == 'Team Wins_percentage'].groupby('progress_bin')['value'].mean().reset_index()
            
            if not progress_effect.empty:
                plt.figure(figsize=(10, 6))
                sns.barplot(x='progress_bin', y='value', data=progress_effect)
                plt.title('Tournament Progress Effect on Win Percentage')
                plt.xlabel('Tournament Progress')
                plt.ylabel('Average Win Percentage')
                plt.savefig('plots/feature_engineering/tournament_progress_effect.png')
                plt.show()
        except Exception as e:
            print(f"Error generating tournament progress visualization: {e}")

def generate_player_team_features(player_df):
    """Generate team-level aggregate features from player data for analysis"""
    if player_df.empty:
        print("No player data available to generate team features")
        return
    
    print("Generating player-based team features for analysis...")
    
    # Process player data first to get basic metrics
    processed_df = process_player_data(player_df)
    
    # Group by team and season
    team_features = []
    
    for (team_name, season_name), team_season_df in processed_df.groupby(['team_name', 'season_name']):
        # Basic counts
        total_players = len(team_season_df)
        players_min_5_appearances = sum(team_season_df['appearances'] >= 5)
        
        # Performance metrics
        avg_player_goals = team_season_df['goals'].mean()
        max_player_goals = team_season_df['goals'].max()
        
        # Get top 5 scorers' total goals
        top_scorers = team_season_df.nlargest(5, 'goals')
        total_goals_top5 = top_scorers['goals'].sum()
        
        # Assist metrics
        avg_player_assists = team_season_df['assists'].mean()
        max_player_assists = team_season_df['assists'].max()
        
        # Goal contribution of top 5 players
        top_contributors = team_season_df.nlargest(5, 'goal_contributions')
        goal_contribution_top5 = top_contributors['goal_contributions'].sum()
        
        # Playing time metrics
        avg_minutes_played = team_season_df['minutes_played'].mean()
        minutes_played_variation = team_season_df['minutes_played'].std()
        
        # Get team ID
        team_id = team_season_df['team_id'].iloc[0]
        
        # Calculate player continuity/retention across seasons
        # For simplicity, default to 1 until we have a previous season to compare
        player_continuity = 1.0
        player_retention_rate = 0.0
        retained_goal_contribution = 0.0
        
        # Store the team-season features
        team_features.append({
            'team_name': team_name,
            'season_name': season_name,
            'total_players': total_players,
            'players_min_5_appearances': players_min_5_appearances,
            'avg_player_goals': avg_player_goals,
            'max_player_goals': max_player_goals,
            'total_goals_top5': total_goals_top5,
            'avg_player_assists': avg_player_assists,
            'max_player_assists': max_player_assists,
            'goal_contribution_top5': goal_contribution_top5,
            'avg_minutes_played': avg_minutes_played,
            'minutes_played_variation': minutes_played_variation,
            'player_continuity': player_continuity,
            'team_id': team_id,
            'player_retention_rate': player_retention_rate,
            'retained_goal_contribution': retained_goal_contribution
        })
    
    # Convert to DataFrame
    team_features_df = pd.DataFrame(team_features)
    
    # Calculate player retention rate across seasons
    # This requires a second pass since we need data from all seasons
    seasons = sorted(team_features_df['season_name'].unique())
    teams = team_features_df['team_name'].unique()
    
    for team in teams:
        for i, season in enumerate(seasons[1:], 1):  # Skip first season as it has no previous
            prev_season = seasons[i-1]
            
            # Get player lists for current and previous seasons
            try:
                current_players = set(processed_df[(processed_df['team_name'] == team) & 
                                                 (processed_df['season_name'] == season)]['player_id'])
                
                prev_players = set(processed_df[(processed_df['team_name'] == team) & 
                                              (processed_df['season_name'] == prev_season)]['player_id'])
                
                # Calculate retention rate
                if prev_players:
                    retained_players = current_players.intersection(prev_players)
                    retention_rate = len(retained_players) / len(prev_players)
                    
                    # Update the retention rate in the dataframe
                    mask = (team_features_df['team_name'] == team) & (team_features_df['season_name'] == season)
                    team_features_df.loc[mask, 'player_retention_rate'] = retention_rate
                    
                    # Calculate retained goal contribution (what % of goals from previous season was retained)
                    prev_season_contributions = processed_df[(processed_df['team_name'] == team) & 
                                                         (processed_df['season_name'] == prev_season) &
                                                         (processed_df['player_id'].isin(retained_players))]['goal_contributions'].sum()
                    
                    total_prev_contributions = processed_df[(processed_df['team_name'] == team) & 
                                                       (processed_df['season_name'] == prev_season)]['goal_contributions'].sum()
                    
                    if total_prev_contributions > 0:
                        retained_contribution = prev_season_contributions / total_prev_contributions
                        team_features_df.loc[mask, 'retained_goal_contribution'] = retained_contribution
            except Exception as e:
                print(f"Error calculating retention for {team} in {season}: {e}")
    
    # Save to CSV
    output_file = "processed_data/player_team_features.csv"
    team_features_df.to_csv(output_file, index=False)
    print(f"Saved player-based team features to {output_file}")
    
    return team_features_df

def main():
    # Load the data
    print("Loading team data...")
    team_stats_df, matches_df = load_team_data()
    
    # Load player data
    print("Loading player data...")
    player_df = load_player_data()
    
    # Check if we have data to process
    if team_stats_df.empty:
        print("No team statistics data available for feature engineering.")
        return
    
    # Generate player team features for analysis if player data is available
    if not player_df.empty:
        print("Generating player team features...")
        generate_player_team_features(player_df)
    
    # Build the engineered dataset, including player data if available
    enhanced_df = build_engineered_dataset(team_stats_df, matches_df, player_df)
    
    # Create aggregate features for modeling
    model_features = create_aggregate_features(enhanced_df)
    
    # Visualize the engineered features
    visualize_engineered_features(enhanced_df)
    
    print("Feature engineering process completed successfully!")
    print(f"Generated {enhanced_df.shape[1]} engineered features")
    print("Engineered data saved to processed_data/engineered_features.csv")
    print("Modeling features saved to processed_data/model_features.csv")

if __name__ == "__main__":
    main()
