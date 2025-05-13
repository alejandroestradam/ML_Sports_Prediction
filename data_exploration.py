import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob
import re
from datetime import datetime

# Set plot style
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Create output directory for plots
os.makedirs("plots", exist_ok=True)

# Function to load team statistics data
def load_team_stats():
    """Load team statistics data from all teams"""
    all_team_stats = []
    
    # Get all team folders
    team_folders = [f for f in os.listdir("collected_data") if os.path.isdir(os.path.join("collected_data", f)) 
                   and f not in ['team_stats_2020-2021', 'team_stats_2021-2022', 'team_stats_2022-2023', 'team_stats_2023-2024']]
    
    for team_folder in team_folders:
        # Find all_team_stats.csv file
        team_stats_file = os.path.join("collected_data", team_folder, "all_team_stats.csv")
        if os.path.exists(team_stats_file):
            try:
                df = pd.read_csv(team_stats_file)
                df['team_folder'] = team_folder  # Add team folder for reference
                all_team_stats.append(df)
                print(f"Loaded stats for {team_folder}")
            except Exception as e:
                print(f"Error loading {team_stats_file}: {e}")
    
    # Combine all team stats
    if all_team_stats:
        return pd.concat(all_team_stats, ignore_index=True)
    else:
        return pd.DataFrame()

# Function to load player performance data
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
                all_player_data.append(df)
                print(f"Loaded player data for {team_folder}")
            except Exception as e:
                print(f"Error loading {player_file}: {e}")
    
    # Combine all player data
    if all_player_data:
        return pd.concat(all_player_data, ignore_index=True)
    else:
        return pd.DataFrame()

# Function to load seasonal team stats
def load_seasonal_team_stats():
    """Load season-specific team statistics"""
    season_stats = {}
    
    # Get all season folders
    season_folders = [f for f in os.listdir("collected_data") if f.startswith("team_stats_") and os.path.isdir(os.path.join("collected_data", f))]
    
    for season_folder in season_folders:
        season_name = season_folder.replace("team_stats_", "")
        season_data = {}
        
        # Categories of stats files
        stat_categories = ["general", "scoring", "defensive", "player"]
        
        for category in stat_categories:
            file_path = os.path.join("collected_data", season_folder, f"team_{category}_stats_{season_name}.csv")
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    season_data[category] = df
                    print(f"Loaded {category} stats for {season_name}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        season_stats[season_name] = season_data
    
    return season_stats

# Load the data
print("Loading team statistics data...")
team_stats_df = load_team_stats()

print("Loading player performance data...")
player_df = load_player_data()

print("Loading seasonal team statistics...")
seasonal_stats = load_seasonal_team_stats()

# ===== Data Exploration =====

def explore_team_stats(df):
    """Explore team statistics data"""
    print("\n===== Team Statistics Data Exploration =====")
    
    # Basic info and shape
    print(f"Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Column data types and non-null values
    print("\nDataset information:")
    df.info()
    
    # Descriptive statistics
    print("\nDescriptive statistics for numeric columns:")
    print(df.describe())
    
    # Missing values
    missing_values = df.isnull().sum()
    print("\nMissing values count:")
    print(missing_values[missing_values > 0])
    
    # Count unique teams and seasons
    print(f"\nNumber of unique teams: {df['team_name'].nunique()}")
    print(f"Teams: {sorted(df['team_name'].unique())}")
    
    print(f"\nNumber of unique seasons: {df['season_name'].nunique()}")
    print(f"Seasons: {sorted(df['season_name'].unique())}")
    
    # Count statistic types
    print(f"\nNumber of unique statistics: {df['statistic_type'].nunique()}")
    
    # Create a pivot table for key metrics by team and season
    # Extract only summary statistics - wins, goals, cleansheets
    key_stats = ['Team Wins_percentage', 'Goals_average', 'Goals Conceded_average', 'Cleansheets_percentage']
    
    try:
        # Filter for overall stats (all)
        df_filtered = df[df['statistic_type'].isin(key_stats)]
        pivot_df = df_filtered.pivot_table(
            index=['team_name'], 
            columns=['season_name', 'statistic_type'], 
            values='value',
            aggfunc='mean'
        )
        
        print("\nKey metrics by team and season:")
        print(pivot_df)
        
        # Save the pivot table
        pivot_df.to_csv("team_stats_summary.csv")
        print("Saved team stats summary to team_stats_summary.csv")
    except Exception as e:
        print(f"Error creating pivot table: {e}")
    
    return df

def explore_player_data(df):
    """Explore player performance data"""
    print("\n===== Player Performance Data Exploration =====")
    
    # Basic info and shape
    print(f"Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Column data types and non-null values
    print("\nDataset information:")
    df.info()
    
    # Descriptive statistics
    print("\nDescriptive statistics for numeric columns:")
    print(df.describe())
    
    # Missing values
    missing_values = df.isnull().sum()
    print("\nMissing values count:")
    print(missing_values[missing_values > 0])
    
    # Top goal scorers
    top_scorers = df.sort_values('goals', ascending=False).head(10)
    print("\nTop 10 goal scorers:")
    print(top_scorers[['player_name', 'team_name', 'season_name', 'goals', 'assists']])
    
    # Top assisters
    top_assisters = df.sort_values('assists', ascending=False).head(10)
    print("\nTop 10 assist providers:")
    print(top_assisters[['player_name', 'team_name', 'season_name', 'goals', 'assists']])
    
    # Players with most appearances
    top_appearances = df.sort_values('appearances', ascending=False).head(10)
    print("\nPlayers with most appearances:")
    print(top_appearances[['player_name', 'team_name', 'season_name', 'appearances', 'minutes_played']])
    
    # Positions distribution
    position_counts = df['position'].value_counts()
    print("\nPosition distribution:")
    print(position_counts)
    
    return df

def visualize_team_stats(df):
    """Create visualizations for team statistics"""
    print("\n===== Team Statistics Visualizations =====")
    
    # Only keep 'all' location for comparable metrics
    df_all = df[df['statistic_location'] == 'all'].copy() if 'statistic_location' in df.columns else df.copy()
    
    # 1. Goals scored per team per season
    try:
        goals_df = df_all[(df_all['statistic_type'] == 'Goals_average')]
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='team_name', y='value', hue='season_name', data=goals_df)
        plt.title('Average Goals Scored per Team by Season')
        plt.xlabel('Team')
        plt.ylabel('Average Goals per Match')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('plots/goals_per_team_season.png')
        plt.show()
        
        # 2. Goals conceded per team per season
        goals_conceded_df = df_all[(df_all['statistic_type'] == 'Goals Conceded_average')]
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='team_name', y='value', hue='season_name', data=goals_conceded_df)
        plt.title('Average Goals Conceded per Team by Season')
        plt.xlabel('Team')
        plt.ylabel('Average Goals Conceded per Match')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('plots/goals_conceded_per_team_season.png')
        plt.show()
        
        # 3. Win rate per team per season
        win_rate_df = df_all[(df_all['statistic_type'] == 'Team Wins_percentage')]
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='team_name', y='value', hue='season_name', data=win_rate_df)
        plt.title('Win Rate per Team by Season')
        plt.xlabel('Team')
        plt.ylabel('Win Rate (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('plots/win_rate_per_team_season.png')
        plt.show()
        
        # 4. Compare home vs away performance
        if 'is_home' in df.columns:
            home_away_df = df[df['statistic_type'].isin(['Goals_average', 'Goals Conceded_average'])]
            
            plt.figure(figsize=(14, 10))
            sns.barplot(x='team_name', y='value', hue='is_home', data=home_away_df[home_away_df['statistic_type'] == 'Goals_average'])
            plt.title('Home vs Away Goals Scored per Team')
            plt.xlabel('Team')
            plt.ylabel('Average Goals per Match')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('plots/home_away_goals.png')
            plt.show()
    except Exception as e:
        print(f"Error creating team stats visualizations: {e}")

def visualize_player_stats(df):
    """Create visualizations for player statistics"""
    print("\n===== Player Statistics Visualizations =====")
    
    try:
        # 1. Top goal scorers overall
        top_scorers = df.groupby('player_name')['goals'].sum().nlargest(15).reset_index()
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='goals', y='player_name', hue='player_name', data=top_scorers, legend=False)
        plt.title('Top 15 Goal Scorers Across All Seasons')
        plt.xlabel('Total Goals')
        plt.ylabel('Player')
        plt.tight_layout()
        plt.savefig('plots/top_goal_scorers.png')
        # plt.show()
        
        # 2. Goals vs Assists scatter plot for players
        player_totals = df.groupby('player_name').agg({
            'goals': 'sum',
            'assists': 'sum',
            'appearances': 'sum',
            'team_name': 'first'  # Just take the first team for simplicity
        }).reset_index()
        
        plt.figure(figsize=(12, 10))
        scatter = sns.scatterplot(
            data=player_totals[player_totals['appearances'] > 10],  # Filter for players with significant appearances
            x='goals',
            y='assists',
            size='appearances',
            hue='team_name',
            alpha=0.7,
            sizes=(50, 400)
        )
        
        # Annotate top performers
        top_performers = player_totals[
            (player_totals['goals'] > player_totals['goals'].quantile(0.95)) | 
            (player_totals['assists'] > player_totals['assists'].quantile(0.95))
        ]
        
        for _, row in top_performers.iterrows():
            scatter.annotate(
                row['player_name'],
                xy=(row['goals'], row['assists']),
                xytext=(5, 5),
                textcoords='offset points'
            )
            
        plt.title('Goals vs Assists by Player (Size represents appearances)')
        plt.xlabel('Total Goals')
        plt.ylabel('Total Assists')
        plt.tight_layout()
        plt.savefig('plots/goals_vs_assists.png')
        plt.show()
        
        # 3. Goal contribution by position
        df['goal_contributions'] = df['goals'] + df['assists']
        position_contribution = df.groupby('position')['goal_contributions'].sum().reset_index()
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='position', y='goal_contributions', hue='position', data=position_contribution, legend=False)
        plt.title('Goal Contributions by Position')
        plt.xlabel('Position ID')
        plt.ylabel('Total Goal Contributions (Goals + Assists)')
        plt.tight_layout()
        plt.savefig('plots/contribution_by_position.png')
        plt.show()
        
        # 4. Goals per minute efficiency
        df['mins_per_goal'] = df['minutes_played'] / df['goals'].replace(0, np.nan)
        efficient_scorers = df[df['goals'] >= 5].sort_values('mins_per_goal').head(15)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='mins_per_goal', y='player_name', hue='player_name', data=efficient_scorers, legend=False)
        plt.title('Most Efficient Goal Scorers (Minutes per Goal)')
        plt.xlabel('Minutes per Goal')
        plt.ylabel('Player')
        plt.tight_layout()
        plt.savefig('plots/goal_efficiency.png')
        plt.show()
        
    except Exception as e:
        print(f"Error creating player stats visualizations: {e}")

def correlation_analysis(df):
    """Perform correlation analysis on player statistics"""
    print("\n===== Correlation Analysis =====")
    
    try:
        # Select numeric columns for correlation
        numeric_cols = df.select_dtypes(include=['number']).columns
        corr_matrix = df[numeric_cols].corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix of Player Statistics')
        plt.tight_layout()
        plt.savefig('plots/correlation_matrix.png')
        plt.show()
        
        print("Correlation matrix saved to plots/correlation_matrix.png")
        
        # Print strongest correlations
        corr_pairs = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
        corr_pairs = corr_pairs[corr_pairs != 1.0]  # Remove self-correlations
        
        print("\nStrongest correlations:")
        print(corr_pairs.head(10))
        
    except Exception as e:
        print(f"Error in correlation analysis: {e}")
        
def analyze_team_performance_trends(df):
    """Analyze team performance trends over seasons"""
    print("\n===== Team Performance Trends Analysis =====")
    
    try:
        # Extract only the key performance metrics for trend analysis
        metrics = ['Team Wins_percentage', 'Goals_average', 'Goals Conceded_average', 'Cleansheets_percentage']
        trends_df = df[df['statistic_type'].isin(metrics) & (df['statistic_location'] == 'all')]
        
        # Convert season_name to numeric format for proper ordering
        trends_df['season_num'] = trends_df['season_name'].apply(lambda x: int(x.split('-')[0]))
        
        # Pivot the data for easier analysis
        pivot_df = trends_df.pivot_table(
            index=['team_name', 'season_num', 'season_name'],
            columns='statistic_type',
            values='value'
        ).reset_index()
        
        # Sort by team and season
        pivot_df = pivot_df.sort_values(['team_name', 'season_num'])
        
        # Plot trends for each metric
        for metric in metrics:
            if metric in pivot_df.columns:
                plt.figure(figsize=(14, 8))
                
                for team in pivot_df['team_name'].unique():
                    team_data = pivot_df[pivot_df['team_name'] == team]
                    plt.plot(team_data['season_name'], team_data[metric], marker='o', linewidth=2, label=team)
                
                plt.title(f'Trend of {metric} by Season')
                plt.xlabel('Season')
                plt.ylabel(metric)
                plt.xticks(rotation=45)
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f'plots/trend_{metric.replace(" ", "_").lower()}.png')
                plt.show()
                
        print("Team performance trend plots saved to plots/ directory")
                
    except Exception as e:
        print(f"Error analyzing team performance trends: {e}")

# Run the analysis
if not team_stats_df.empty:
    team_stats_df = explore_team_stats(team_stats_df)
    visualize_team_stats(team_stats_df)
    analyze_team_performance_trends(team_stats_df)
else:
    print("No team statistics data available for analysis.")

if not player_df.empty:
    player_df = explore_player_data(player_df)
    visualize_player_stats(player_df)
    correlation_analysis(player_df)
else:
    print("No player performance data available for analysis.")

# Print summary
print("\n===== Analysis Summary =====")
print(f"Analyzed team statistics for {team_stats_df['team_name'].nunique() if not team_stats_df.empty else 0} teams")
print(f"Analyzed player performance data for {player_df['player_name'].nunique() if not player_df.empty else 0} players")
print(f"Generated visualizations in the 'plots' directory")
print("Analysis complete!")
