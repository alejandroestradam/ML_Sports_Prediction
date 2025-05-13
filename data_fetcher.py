import requests
import json
import os
import pandas as pd
from dotenv import load_dotenv
import time
from datetime import datetime
import logging
import argparse
import unicodedata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_collection.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()
API_KEY = os.getenv('API_KEY')

if not API_KEY:
    raise ValueError("API key not found in environment variables")

BASE_URL = "https://api.sportmonks.com/v3/football"

# Create data directory if it doesn't exist
DATA_DIR = "collected_data"
os.makedirs(DATA_DIR, exist_ok=True)

def normalize_text(text):
    """Normalize text by removing accents and special characters"""
    if text is None:
        return None
    
    if not isinstance(text, str):
        return text
    
    text = unicodedata.normalize('NFD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    return text

def normalize_dataframe(df):
    """Normalize all string columns in a DataFrame to remove accents"""
    df_normalized = df.copy()
    
    for col in df_normalized.columns:
        if df_normalized[col].dtype == 'object':
            df_normalized[col] = df_normalized[col].apply(normalize_text)
    
    return df_normalized

def make_api_request(endpoint, params=None):
    """Make a request to the API with rate limiting and error handling"""
    if params is None:
        params = {}
        
    params["api_token"] = API_KEY
    
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/{endpoint}", params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:  # Rate limited
                retry_after = int(response.headers.get('Retry-After', retry_delay))
                logging.warning(f"Rate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
            elif response.status_code == 404:
                logging.error(f"Resource not found: {endpoint} with params {params}")
                raise
            else:
                logging.error(f"HTTP Error: {e}")
                time.sleep(retry_delay * (attempt + 1))
        except Exception as e:
            logging.error(f"Error accessing API: {e}")
            time.sleep(retry_delay * (attempt + 1))
            
    raise Exception(f"Failed to get data after {max_retries} attempts")

def get_team_data(team_id, includes=None):
    """Get detailed data for a specific team"""
    logging.info(f"Getting detailed data for team ID {team_id}...")
    
    endpoint = f"teams/{team_id}"
    params = {}
    
    if includes:
        params["include"] = ";".join(includes)
    
    team_data = make_api_request(endpoint, params)
    logging.info(f"Successfully retrieved data for {team_data['data']['name']}")
    
    return team_data

def get_seasons_for_team(team_id, limit=5):
    """Get the most recent seasons for Liga MX (specifically up to 5 recent seasons)"""
    logging.info(f"Getting recent Liga MX seasons for team ID {team_id}...")
    
    recent_liga_mx_seasons = [
        # 2024-2025 season
        {"id": 23586, "name": "2024/2025", "league_id": 743, "country_id": 1678, "start_date": "2024-07-05", "end_date": "2025-05-09"},
        # 2023-2024 season
        {"id": 21623, "name": "2023/2024", "league_id": 743, "country_id": 1678, "start_date": "2023-07-01", "end_date": "2024-05-27"},
        # 2022-2023 season
        {"id": 19684, "name": "2022/2023", "league_id": 743, "country_id": 1678, "start_date": "2022-07-01", "end_date": "2023-05-31"},
        # 2021-2022 season
        {"id": 18469, "name": "2021/2022", "league_id": 743, "country_id": 1678, "start_date": "2021-07-23", "end_date": "2022-05-30"},
        # 2020-2021 season
        {"id": 17191, "name": "2020/2021", "league_id": 743, "country_id": 1678, "start_date": "2020-07-25", "end_date": "2021-05-31"},
    ]
    
    seasons_to_return = recent_liga_mx_seasons[:limit]
    
    logging.info(f"Using hard-coded recent Liga MX seasons: {[s['name'] for s in seasons_to_return]}")
    return seasons_to_return

def get_players_for_team(team_id, season_id=None):
    """Get all players for a team through the squad endpoint"""
    logging.info(f"Getting players for team ID {team_id}...")
    
    endpoint = f"squads/teams/{team_id}"
    params = {}
    
    if season_id:
        params["filters"] = f"seasonIds:{season_id}"
    
    try:
        squad_data = make_api_request(endpoint, params)
        
        if 'data' in squad_data:
            players = squad_data['data']
            logging.info(f"Retrieved {len(players)} players from team squad")
            return players
        else:
            logging.warning(f"No squad data found for team ID {team_id}")
            return []
    except Exception as e:
        logging.error(f"Error getting squad: {e}")
        try:
            endpoint = f"players"
            params = {
                "filters": f"teamIds:{team_id}"
            }
            players_data = make_api_request(endpoint, params)
            
            if 'data' in players_data:
                players = players_data['data']
                logging.info(f"Retrieved {len(players)} players using fallback method")
                return players
            else:
                return []
        except Exception as fallback_e:
            logging.error(f"Fallback error getting players: {fallback_e}")
            return []

def get_player_statistics(player_id, season_id):
    """Get detailed statistics for a specific player in a specific season"""
    logging.info(f"Getting statistics for player ID {player_id} in season {season_id}...")
    
    endpoint = f"players/{player_id}"
    params = {
        "include": "statistics.details.type",
        "filters": f"playerStatisticSeasons:{season_id}"
    }
    
    try:
        player_data = make_api_request(endpoint, params)
        
        if 'data' not in player_data or 'statistics' not in player_data['data']:
            logging.warning(f"No statistics found for player {player_id} in season {season_id}")
            return None
            
        logging.info(f"Successfully retrieved statistics for player {player_id} in season {season_id}")
        return player_data['data']
    except Exception as e:
        logging.warning(f"Error retrieving statistics for player {player_id} in season {season_id}: {e}")
        return None

def get_player_info(player_id):
    """Get detailed information for a specific player"""
    logging.info(f"Getting detailed information for player ID {player_id}...")
    
    endpoint = f"players/{player_id}"
    params = {
        "include": "position;detailedPosition"
    }
    
    try:
        player_data = make_api_request(endpoint, params)
        
        if 'data' not in player_data:
            logging.warning(f"No information found for player {player_id}")
            return None
            
        logging.info(f"Successfully retrieved information for player {player_data['data'].get('display_name', '')}")
        return player_data['data']
    except Exception as e:
        logging.warning(f"Error retrieving information for player {player_id}: {e}")
        return None

def process_player_stats(seasons):
    """Process player statistics for all seasons and create a DataFrame"""
    all_player_stats = []
    
    player_file = os.path.join(DATA_DIR, "club_america_players.json")
    if os.path.exists(player_file):
        with open(player_file, 'r') as f:
            players = json.load(f)
    else:
        players = []
        for season in seasons:
            season_players = get_players_for_team(2687, season['id'])
            if season_players:
                players.extend(season_players)
    
    unique_players = {}
    for player in players:
        player_id = player.get('player_id') or player.get('id')
        if player_id and player_id not in unique_players:
            unique_players[player_id] = player
    
    for player_id, player_info in unique_players.items():
        player_details = get_player_info(player_id)
        
        if player_details:
            player_name = player_details.get('display_name') or player_details.get('name', '')
            position = player_details.get('position_id', '')
        else:
            player_name = player_info.get('display_name') or player_info.get('name', f"Player {player_id}")
            position = player_info.get('position_id', '')
        
        logging.info(f"Processing stats for player: {player_name}")
        
        for season in seasons:
            season_id = season['id']
            season_name = season['name']
            
            logging.info(f"Getting stats for {player_name} in season {season_name}")
            
            player_data = get_player_statistics(player_id, season_id)
            
            if player_data and 'statistics' in player_data:
                for stat_entry in player_data['statistics']:
                    if stat_entry.get('season_id') != season_id:
                        continue
                    
                    player_stats = {
                        'player_id': player_id,
                        'player_name': player_name,
                        'season_id': season_id,
                        'season_name': season_name,
                        'team_id': 2687,
                        'position': position,
                        'jersey_number': stat_entry.get('jersey_number', 0),
                        'appearances': 0,
                        'minutes_played': 0,
                        'goals': 0,
                        'assists': 0,
                        'shots_total': 0,
                        'shots_on_target': 0,
                        'passes_total': 0,
                        'passes_accuracy': 0,
                        'tackles': 0,
                        'interceptions': 0
                    }
                    
                    if 'details' in stat_entry:
                        for detail in stat_entry['details']:
                            if 'type' in detail and 'value' in detail:
                                stat_type = detail['type']
                                stat_name = stat_type.get('name', '').lower()
                                stat_code = stat_type.get('code', '').lower()
                                stat_value = detail['value']
                                
                                if 'goal' in stat_name or 'goal' in stat_code:
                                    if isinstance(stat_value, dict) and 'total' in stat_value:
                                        player_stats['goals'] = stat_value['total']
                                    else:
                                        player_stats['goals'] = stat_value
                                
                                elif 'assist' in stat_name or 'assist' in stat_code:
                                    if isinstance(stat_value, dict) and 'total' in stat_value:
                                        player_stats['assists'] = stat_value['total']
                                    else:
                                        player_stats['assists'] = stat_value
                                        
                                elif 'minute' in stat_name or 'minute' in stat_code:
                                    if isinstance(stat_value, dict) and 'total' in stat_value:
                                        player_stats['minutes_played'] = stat_value['total']
                                    else:
                                        player_stats['minutes_played'] = stat_value
                                
                                elif 'appearance' in stat_name or 'appearance' in stat_code:
                                    if isinstance(stat_value, dict) and 'total' in stat_value:
                                        player_stats['appearances'] = stat_value['total']
                                    else:
                                        player_stats['appearances'] = stat_value
                                
                                elif 'shot' in stat_name or 'shot' in stat_code:
                                    if isinstance(stat_value, dict):
                                        if 'total' in stat_value:
                                            player_stats['shots_total'] = stat_value['total']
                                        if 'on_target' in stat_value:
                                            player_stats['shots_on_target'] = stat_value['on_target']
                                
                                elif 'pass' in stat_name or 'pass' in stat_code:
                                    if isinstance(stat_value, dict):
                                        if 'total' in stat_value:
                                            player_stats['passes_total'] = stat_value['total']
                                        if 'accuracy' in stat_value:
                                            player_stats['passes_accuracy'] = stat_value['accuracy']
                                
                                elif 'tackle' in stat_name or 'tackle' in stat_code:
                                    if isinstance(stat_value, dict) and 'total' in stat_value:
                                        player_stats['tackles'] = stat_value['total']
                                    else:
                                        player_stats['tackles'] = stat_value
                                
                                elif 'interception' in stat_name or 'interception' in stat_code:
                                    if isinstance(stat_value, dict) and 'total' in stat_value:
                                        player_stats['interceptions'] = stat_value['total']
                                    else:
                                        player_stats['interceptions'] = stat_value
                    
                    if (player_stats['appearances'] > 0 or 
                        player_stats['minutes_played'] > 0 or 
                        player_stats['goals'] > 0 or 
                        player_stats['assists'] > 0):
                        all_player_stats.append(player_stats)
                        logging.info(f"Added stats for {player_name} in season {season_name}")
            else:
                logging.warning(f"No statistics data found for player {player_name} in season {season_name}")
    
    if all_player_stats:
        df = pd.DataFrame(all_player_stats)
        
        for season in seasons:
            season_id = season['id']
            season_name = season['name'].replace("/", "-")
            season_df = df[df['season_id'] == season_id]
            
            if not season_df.empty:
                season_csv_path = os.path.join(DATA_DIR, f"player_performance_data_{season_name}.csv")
                season_df.to_csv(season_csv_path, index=False)
                logging.info(f"Player performance data for season {season_name} saved to {season_csv_path}")
        
        csv_path = os.path.join(DATA_DIR, "player_performance_data_all_seasons.csv")
        df.to_csv(csv_path, index=False)
        logging.info(f"All player performance data saved to {csv_path}")
        return df
    else:
        logging.warning("No player statistics were collected")
        return pd.DataFrame()

def get_team_statistics(team_id, season_id):
    """Get team statistics for a specific season using the direct team statistics endpoint"""
    logging.info(f"Getting team statistics for team ID {team_id} in season {season_id}...")
    
    endpoint = f"teams/{team_id}"
    params = {
        "include": "statistics.details.type",
        "filters": f"teamStatisticSeasons:{season_id}"
    }
    
    try:
        team_data = make_api_request(endpoint, params)
        
        if not team_data or 'data' not in team_data:
            logging.warning(f"No team data found for team {team_id}")
            return None
            
        team_info = team_data['data']
        team_name = team_info.get('name', f"Team {team_id}")
        
        if 'statistics' not in team_info or not team_info['statistics']:
            logging.warning(f"No statistics found for team {team_name} in season {season_id}")
            return None
            
        logging.info(f"Successfully retrieved statistics for team {team_name} in season {season_id}")
        return team_info
    except Exception as e:
        logging.warning(f"Error retrieving team statistics for team {team_id} in season {season_id}: {e}")
        return None

def process_team_statistics(team_info, season_id, season_name):
    """Process team statistics data into a structured format"""
    if not team_info or 'statistics' not in team_info:
        return []
    
    team_id = team_info.get('id')
    team_name = team_info.get('name', f"Team {team_id}")
    
    stats_data = []
    
    for stat_entry in team_info['statistics']:
        if stat_entry.get('season_id') != season_id:
            continue
            
        if 'details' in stat_entry:
            for detail in stat_entry['details']:
                stat_type = None
                if 'type' in detail and detail['type']:
                    type_data = detail['type']
                    stat_type = type_data.get('name', '')
                    stat_code = type_data.get('code', '')
                    stat_group = type_data.get('stat_group', '')
                
                stat_value = detail.get('value')
                
                stat_record = {
                    'team_id': team_id,
                    'team_name': team_name,
                    'season_id': season_id,
                    'season_name': season_name,
                    'statistic_type': stat_type,
                    'statistic_code': stat_code,
                    'statistic_group': stat_group,
                }
                
                if isinstance(stat_value, dict):
                    if 'all' in stat_value:
                        all_stats = stat_value['all']
                        for key, value in all_stats.items():
                            all_stat_record = stat_record.copy()
                            all_stat_record['location'] = 'all'
                            all_stat_record['metric'] = key
                            all_stat_record['value'] = value
                            stats_data.append(all_stat_record)
                    
                    if 'home' in stat_value:
                        home_stats = stat_value['home']
                        for key, value in home_stats.items():
                            home_stat_record = stat_record.copy()
                            home_stat_record['location'] = 'home'
                            home_stat_record['metric'] = key
                            home_stat_record['value'] = value
                            stats_data.append(home_stat_record)
                    
                    if 'away' in stat_value:
                        away_stats = stat_value['away']
                        for key, value in away_stats.items():
                            away_stat_record = stat_record.copy()
                            away_stat_record['location'] = 'away'
                            away_stat_record['metric'] = key
                            away_stat_record['value'] = value
                            stats_data.append(away_stat_record)
                    
                    if 'count' in stat_value:
                        simple_stat_record = stat_record.copy()
                        simple_stat_record['location'] = 'overall'
                        simple_stat_record['metric'] = 'count'
                        simple_stat_record['value'] = stat_value['count']
                        stats_data.append(simple_stat_record)
                        
                    if 'average' in stat_value:
                        avg_stat_record = stat_record.copy()
                        avg_stat_record['location'] = 'overall'
                        avg_stat_record['metric'] = 'average'
                        avg_stat_record['value'] = stat_value['average']
                        stats_data.append(avg_stat_record)
                    
                else:
                    simple_stat_record = stat_record.copy()
                    simple_stat_record['location'] = 'overall'
                    simple_stat_record['metric'] = 'value'
                    simple_stat_record['value'] = stat_value
                    stats_data.append(simple_stat_record)
    
    return stats_data

def get_matches_for_team(team_id, season_id, status="result"):
    """Get matches for a team in a specific season with a specific status"""
    logging.info(f"Getting {status} matches for team ID {team_id} in season {season_id}...")
    
    endpoint = f"teams/{team_id}"
    
    params = {
        "include": "statistics.details.type",
        "filters": f"teamStatisticSeasons:{season_id}"
    }
    
    try:
        team_data = make_api_request(endpoint, params)
        
        if 'data' not in team_data:
            logging.warning(f"No team data found for team {team_id}")
            return []
            
        team_info = team_data['data']
        team_name = team_info.get('name', f"Team {team_id}")
        
        if 'statistics' not in team_info or not team_info['statistics']:
            logging.warning(f"No statistics found for team {team_name} in season {season_id}")
            return []
            
        matches_data = []
        
        for stat_entry in team_info['statistics']:
            if stat_entry.get('season_id') != season_id:
                continue
                
            match_data = {
                'id': f"team_stat_{team_id}_{season_id}",
                'team_id': team_id,
                'season_id': season_id,
                'team_name': team_name,
                'starting_at': datetime.now().isoformat(),
                'status': 'team_statistics',
                'statistics': {
                    'data': [stat_entry]
                },
                'participants': {
                    'data': [{
                        'id': team_id,
                        'name': team_name,
                        'meta': {'location': 'home'}
                    }]
                }
            }
            
            matches_data.append(match_data)
            
        logging.info(f"Retrieved team statistics for {team_name} in season {season_id}")
        return matches_data
            
    except Exception as e:
        logging.error(f"Error getting team statistics: {e}")
        
        try:
            logging.info(f"Trying alternative method using fixtures endpoint...")
            endpoint = f"fixtures"
            params = {
                "filters": f"teamIds:{team_id};seasonIds:{season_id};leagueIds:743",
                "include": "participants;statistics.type;lineups.details.type",
                "sort": "starting_at"
            }
            
            fixtures_data = make_api_request(endpoint, params)
            
            if 'data' in fixtures_data:
                matches = fixtures_data['data']
                logging.info(f"Retrieved {len(matches)} matches using fixtures endpoint")
                
                liga_mx_matches = [match for match in matches if match.get('league_id') == 743]
                logging.info(f"Filtered to {len(liga_mx_matches)} Liga MX matches")
                
                return liga_mx_matches
            else:
                return []
        except Exception as fallback_e:
            logging.error(f"Fallback error getting matches data: {fallback_e}")
            return []

def save_data(data, filename):
    """Save data to a JSON file"""
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    logging.info(f"Data saved to {filepath}")

def process_match_data(matches, season_name):
    """Process and extract useful statistics from match data"""
    logging.info(f"Processing {len(matches)} matches/team statistics for season {season_name}...")
    
    match_stats = []
    player_match_stats = []
    
    all_matches_data = []
    
    team_general_stats = []
    team_scoring_stats = []
    team_defensive_stats = []
    team_offensive_stats = []
    team_player_stats = []
    
    for match in matches:
        match_id = match.get('id')
        match_date = match.get('starting_at', '')
        match_status = match.get('status') or match.get('state') or match.get('status_name', '')
        
        if match_status == 'team_statistics' or 'team_id' in match:
            team_id = match.get('team_id')
            team_name = match.get('team_name', '')
            
            team_name = normalize_text(team_name)
            
            if 'statistics' in match and 'data' in match['statistics']:
                for team_stat in match['statistics']['data']:
                    if 'details' in team_stat:
                        details = team_stat.get('details')
                        if isinstance(details, dict) and 'data' in details:
                            details = details['data']
                        
                        for detail in details:
                            stat_type = None
                            stat_name = None
                            stat_code = None
                            stat_group = None
                            
                            if 'type' in detail:
                                type_data = detail['type']
                                if isinstance(type_data, dict):
                                    if 'data' in type_data:
                                        type_data = type_data['data']
                                    
                                    stat_name = normalize_text(type_data.get('name', ''))
                                    stat_code = type_data.get('code', '')
                                    stat_group = type_data.get('stat_group', '')
                            
                            stat_value = detail.get('value')
                            
                            base_record = {
                                'team_id': team_id,
                                'team_name': team_name,
                                'season_name': season_name,
                                'statistic_name': stat_name,
                                'statistic_code': stat_code,
                                'statistic_group': stat_group,
                            }
                            
                            if stat_group == 'offensive' or 'goal' in stat_code.lower() or 'scoring' in stat_code.lower() or 'goal' in stat_name.lower():
                                process_scoring_stats(stat_name, stat_code, stat_value, team_id, team_name, season_name, team_scoring_stats)
                            
                            elif stat_group == 'defensive' or 'defensive' in stat_code.lower() or 'conceded' in stat_code.lower() or 'tackle' in stat_code.lower() or 'interception' in stat_code.lower():
                                process_defensive_stats(stat_name, stat_code, stat_value, team_id, team_name, season_name, team_defensive_stats)
                            
                            elif stat_group == 'offensive' or 'attack' in stat_code.lower() or 'shot' in stat_code.lower() or 'pass' in stat_code.lower():
                                process_offensive_stats(stat_name, stat_code, stat_value, team_id, team_name, season_name, team_offensive_stats)
                            
                            elif 'player' in stat_code.lower() or isinstance(stat_value, dict) and ('player' in stat_value or 'most_appearing_players' in stat_value or 'most_substituted_players' in stat_value):
                                process_player_stats(stat_name, stat_code, stat_value, team_id, team_name, season_name, team_player_stats)
                            
                            else:
                                process_general_stats(stat_name, stat_code, stat_value, team_id, team_name, season_name, team_general_stats)
                            
                            if isinstance(stat_value, dict):
                                for location in ['all', 'home', 'away']:
                                    if location in stat_value:
                                        location_data = stat_value[location]
                                        for metric, value in location_data.items():
                                            team_match_stat = {
                                                'match_id': match_id,
                                                'season_name': season_name,
                                                'match_date': match_date,
                                                'team_id': team_id,
                                                'team_name': team_name,
                                                'is_home': location == 'home',
                                                'statistic_type': f"{stat_name}_{metric}",
                                                'statistic_location': location,
                                                'value': value
                                            }
                                            match_stats.append(team_match_stat)
                            else:
                                team_match_stat = {
                                    'match_id': match_id,
                                    'season_name': season_name,
                                    'match_date': match_date,
                                    'team_id': team_id,
                                    'team_name': team_name,
                                    'is_home': True,
                                    'statistic_type': stat_name,
                                    'statistic_location': 'overall',
                                    'value': stat_value
                                }
                                match_stats.append(team_match_stat)
            
            continue
        
        match_info = {
            'match_id': match_id,
            'season_name': season_name,
            'match_date': match_date,
            'status': match_status,
            'home_team': None,
            'away_team': None,
            'home_score': None,
            'away_score': None,
            'venue': None,
            'referee': None
        }
        
        if 'venue' in match and 'data' in match['venue']:
            match_info['venue'] = normalize_text(match['venue']['data'].get('name', ''))
        
        if 'officials' in match and 'data' in match['officials']:
            for official in match['officials']['data']:
                if official.get('type_id') == 1:
                    match_info['referee'] = normalize_text(official.get('name', ''))
        
        if 'participants' in match and 'data' in match['participants']:
            for team in match['participants']['data']:
                is_home = team.get('meta', {}).get('location') == 'home'
                team_name = normalize_text(team.get('name', ''))
                team_id = team.get('id')
                team_score = team.get('score', {}).get('goals', 0) if team.get('score') else 0
                
                if is_home:
                    match_info['home_team'] = team_name
                    match_info['home_team_id'] = team_id
                    match_info['home_score'] = team_score
                else:
                    match_info['away_team'] = team_name
                    match_info['away_team_id'] = team_id
                    match_info['away_score'] = team_score
        
        all_matches_data.append(match_info)
        
        if 'statistics' in match and 'data' in match['statistics']:
            for team_stat in match['statistics']['data']:
                team_id = team_stat.get('participant_id')
                team_name = None
                is_home = False
                
                if 'participants' in match and 'data' in match['participants']:
                    for team in match['participants']['data']:
                        if team.get('id') == team_id:
                            team_name = normalize_text(team.get('name', ''))
                            is_home = team.get('meta', {}).get('location') == 'home'
                            break
                
                if 'details' in team_stat and 'data' in team_stat['details']:
                    for detail in team_stat['details']['data']:
                        stat_type = None
                        if 'type' in detail and 'data' in detail['type']:
                            stat_type = normalize_text(detail['type']['data'].get('name', ''))
                        
                        team_match_stat = {
                            'match_id': match_id,
                            'season_name': season_name,
                            'match_date': match_date,
                            'team_id': team_id,
                            'team_name': team_name,
                            'is_home': is_home,
                            'statistic_type': stat_type,
                            'value': detail.get('value')
                        }
                        match_stats.append(team_match_stat)
        
        if 'lineups' in match and 'data' in match['lineups']:
            for player_lineup in match['lineups']['data']:
                player_id = player_lineup.get('player_id')
                player_name = normalize_text(player_lineup.get('player_name', ''))
                team_id = player_lineup.get('team_id')
                team_name = None
                position = player_lineup.get('position', '')
                
                if 'participants' in match and 'data' in match['participants']:
                    for team in match['participants']['data']:
                        if team.get('id') == team_id:
                            team_name = normalize_text(team.get('name', ''))
                            break
                
                player_match_info = {
                    'match_id': match_id,
                    'season_name': season_name,
                    'match_date': match_date,
                    'player_id': player_id,
                    'player_name': player_name,
                    'team_id': team_id,
                    'team_name': team_name,
                    'position': position,
                    'statistic_type': 'appearance',
                    'value': 1
                }
                player_match_stats.append(player_match_info)
                
                if 'details' in player_lineup and 'data' in player_lineup['details']:
                    for detail in player_lineup['details']['data']:
                        stat_type = None
                        if 'type' in detail and 'data' in detail['type']:
                            stat_type = normalize_text(detail['type']['data'].get('name', ''))
                        
                        player_stat = {
                            'match_id': match_id,
                            'season_name': season_name,
                            'match_date': match_date,
                            'player_id': player_id,
                            'player_name': player_name,
                            'team_id': team_id,
                            'team_name': team_name,
                            'position': position,
                            'statistic_type': stat_type,
                            'value': detail.get('value')
                        }
                        player_match_stats.append(player_stat)
    
    save_directory = os.path.join(DATA_DIR, f"team_stats_{season_name}")
    os.makedirs(save_directory, exist_ok=True)
    
    if team_general_stats:
        df = pd.DataFrame(team_general_stats)
        df = normalize_dataframe(df)
        df.to_csv(os.path.join(save_directory, f"team_general_stats_{season_name}.csv"), index=False, encoding='utf-8-sig')
        logging.info(f"Saved {len(team_general_stats)} general statistics")
    
    if team_scoring_stats:
        df = pd.DataFrame(team_scoring_stats)
        df = normalize_dataframe(df)
        df.to_csv(os.path.join(save_directory, f"team_scoring_stats_{season_name}.csv"), index=False, encoding='utf-8-sig')
        logging.info(f"Saved {len(team_scoring_stats)} scoring statistics")
    
    if team_defensive_stats:
        df = pd.DataFrame(team_defensive_stats)
        df = normalize_dataframe(df)
        df.to_csv(os.path.join(save_directory, f"team_defensive_stats_{season_name}.csv"), index=False, encoding='utf-8-sig')
        logging.info(f"Saved {len(team_defensive_stats)} defensive statistics")
    
    if team_offensive_stats:
        df = pd.DataFrame(team_offensive_stats)
        df = normalize_dataframe(df)
        df.to_csv(os.path.join(save_directory, f"team_offensive_stats_{season_name}.csv"), index=False, encoding='utf-8-sig')
        logging.info(f"Saved {len(team_offensive_stats)} offensive statistics")
    
    if team_player_stats:
        df = pd.DataFrame(team_player_stats)
        df = normalize_dataframe(df)
        df.to_csv(os.path.join(save_directory, f"team_player_stats_{season_name}.csv"), index=False, encoding='utf-8-sig')
        logging.info(f"Saved {len(team_player_stats)} player-related statistics")
    
    if match_stats:
        match_stats_df = pd.DataFrame(match_stats)
        match_stats_df = normalize_dataframe(match_stats_df)
        match_stats_df.to_csv(os.path.join(DATA_DIR, f"team_match_stats_{season_name}.csv"), index=False, encoding='utf-8-sig')
        logging.info(f"Saved {len(match_stats)} team match statistics to CSV")
    
    if player_match_stats:
        player_match_stats_df = pd.DataFrame(player_match_stats)
        player_match_stats_df = normalize_dataframe(player_match_stats_df)
        player_match_stats_df.to_csv(os.path.join(DATA_DIR, f"player_match_stats_{season_name}.csv"), index=False, encoding='utf-8-sig')
        logging.info(f"Saved {len(player_match_stats)} player match statistics to CSV")
    
    if all_matches_data:
        matches_df = pd.DataFrame(all_matches_data)
        matches_df = normalize_dataframe(matches_df)
        matches_df.to_csv(os.path.join(DATA_DIR, f"matches_{season_name}.csv"), index=False, encoding='utf-8-sig')
        logging.info(f"Saved {len(all_matches_data)} matches to CSV")
    
    return match_stats, player_match_stats

def process_general_stats(stat_name, stat_code, stat_value, team_id, team_name, season_name, stats_list):
    """Process general team statistics"""
    base_record = {
        'team_id': team_id,
        'team_name': team_name,
        'season_name': season_name,
        'statistic_name': stat_name,
        'statistic_code': stat_code,
    }
    
    if isinstance(stat_value, dict):
        flat_dict = flatten_dict(stat_value, parent_key=stat_code)
        for key, value in flat_dict.items():
            record = base_record.copy()
            record['metric'] = key
            record['value'] = value
            stats_list.append(record)
    else:
        record = base_record.copy()
        record['metric'] = 'value'
        record['value'] = stat_value
        stats_list.append(record)

def process_scoring_stats(stat_name, stat_code, stat_value, team_id, team_name, season_name, stats_list):
    """Process team scoring statistics"""
    base_record = {
        'team_id': team_id,
        'team_name': team_name,
        'season_name': season_name,
        'statistic_name': stat_name,
        'statistic_code': stat_code,
    }
    
    if isinstance(stat_value, dict):
        if 'all' in stat_value:
            for location in ['all', 'home', 'away']:
                if location in stat_value:
                    location_data = stat_value[location]
                    if isinstance(location_data, dict):
                        for metric, value in location_data.items():
                            record = base_record.copy()
                            record['location'] = location
                            record['metric'] = metric
                            record['value'] = value
                            stats_list.append(record)
                    else:
                        record = base_record.copy()
                        record['location'] = location
                        record['metric'] = 'value'
                        record['value'] = location_data
                        stats_list.append(record)
        elif '0-15' in stat_value:
            for period, period_data in stat_value.items():
                if isinstance(period_data, dict):
                    for metric, value in period_data.items():
                        record = base_record.copy()
                        record['time_period'] = period
                        record['metric'] = metric
                        record['value'] = value
                        stats_list.append(record)
                else:
                    record = base_record.copy()
                    record['time_period'] = period
                    record['metric'] = 'value'
                    record['value'] = period_data
                    stats_list.append(record)
        else:
            flat_dict = flatten_dict(stat_value, parent_key=stat_code)
            for key, value in flat_dict.items():
                record = base_record.copy()
                record['metric'] = key
                record['value'] = value
                stats_list.append(record)
    else:
        record = base_record.copy()
        record['metric'] = 'value'
        record['value'] = stat_value
        stats_list.append(record)

def process_defensive_stats(stat_name, stat_code, stat_value, team_id, team_name, season_name, stats_list):
    """Process team defensive statistics"""
    base_record = {
        'team_id': team_id,
        'team_name': team_name,
        'season_name': season_name,
        'statistic_name': stat_name,
        'statistic_code': stat_code,
    }
    
    if isinstance(stat_value, dict):
        if 'all' in stat_value:
            for location in ['all', 'home', 'away']:
                if location in stat_value:
                    location_data = stat_value[location]
                    if isinstance(location_data, dict):
                        for metric, value in location_data.items():
                            record = base_record.copy()
                            record['location'] = location
                            record['metric'] = metric
                            record['value'] = value
                            stats_list.append(record)
                    else:
                        record = base_record.copy()
                        record['location'] = location
                        record['metric'] = 'value'
                        record['value'] = location_data
                        stats_list.append(record)
        else:
            flat_dict = flatten_dict(stat_value, parent_key=stat_code)
            for key, value in flat_dict.items():
                record = base_record.copy()
                record['metric'] = key
                record['value'] = value
                stats_list.append(record)
    else:
        record = base_record.copy()
        record['metric'] = 'value'
        record['value'] = stat_value
        stats_list.append(record)

def process_offensive_stats(stat_name, stat_code, stat_value, team_id, team_name, season_name, stats_list):
    """Process team offensive statistics"""
    base_record = {
        'team_id': team_id,
        'team_name': team_name,
        'season_name': season_name,
        'statistic_name': stat_name,
        'statistic_code': stat_code,
    }
    
    if isinstance(stat_value, dict):
        if 'count' in stat_value and 'average' in stat_value:
            for metric, value in stat_value.items():
                record = base_record.copy()
                record['metric'] = metric
                record['value'] = value
                stats_list.append(record)
        elif 'total' in stat_value and ('on_target' in stat_value or 'inside_box' in stat_value):
            for metric, value in stat_value.items():
                record = base_record.copy()
                record['metric'] = metric
                record['value'] = value
                stats_list.append(record)
        else:
            flat_dict = flatten_dict(stat_value, parent_key=stat_code)
            for key, value in flat_dict.items():
                record = base_record.copy()
                record['metric'] = key
                record['value'] = value
                stats_list.append(record)
    else:
        record = base_record.copy()
        record['metric'] = 'value'
        record['value'] = stat_value
        stats_list.append(record)

def process_player_stats(stat_name, stat_code, stat_value, team_id, team_name, season_name, stats_list):
    """Process player-related team statistics"""
    base_record = {
        'team_id': team_id,
        'team_name': team_name,
        'season_name': season_name,
        'statistic_name': stat_name,
        'statistic_code': stat_code,
    }
    
    if isinstance(stat_value, dict):
        if 'most_appearing_players' in stat_value:
            for i, player_data in enumerate(stat_value['most_appearing_players']):
                record = base_record.copy()
                record['metric'] = 'most_appearing'
                record['rank'] = i + 1
                record['player_id'] = player_data.get('player_id')
                record['matches'] = player_data.get('matches')
                stats_list.append(record)
                
        elif 'longest_appearing_players' in stat_value:
            for i, player_data in enumerate(stat_value['longest_appearing_players']):
                record = base_record.copy()
                record['metric'] = 'longest_appearing'
                record['rank'] = i + 1
                record['player_id'] = player_data.get('player_id')
                record['minutes'] = player_data.get('minutes')
                stats_list.append(record)
                
        elif 'most_substituted_players' in stat_value:
            for i, player_data in enumerate(stat_value['most_substituted_players']):
                record = base_record.copy()
                record['metric'] = 'most_substituted'
                record['rank'] = i + 1
                record['player_id'] = player_data.get('player_id')
                record['in'] = player_data.get('in')
                record['out'] = player_data.get('out')
                record['total'] = player_data.get('total')
                stats_list.append(record)
                
        elif 'most_injured_players' in stat_value:
            for i, player_data in enumerate(stat_value['most_injured_players']):
                record = base_record.copy()
                record['metric'] = 'most_injured'
                record['rank'] = i + 1
                record['player_id'] = player_data.get('player_id')
                record['total'] = player_data.get('total')
                stats_list.append(record)
                
        elif 'national_team_players' in stat_value:
            for i, player_data in enumerate(stat_value['national_team_players']):
                record = base_record.copy()
                record['metric'] = 'national_team'
                record['player_id'] = player_data.get('player')
                record['national_team'] = player_data.get('team')
                stats_list.append(record)
                
        elif 'player_id' in stat_value and 'player_name' in stat_value:
            record = base_record.copy()
            record['player_id'] = stat_value.get('player_id')
            record['player_name'] = stat_value.get('player_name')
            
            for key, value in stat_value.items():
                if key not in ['player_id', 'player_name']:
                    record[key] = value
                    
            stats_list.append(record)
            
        else:
            flat_dict = flatten_dict(stat_value, parent_key=stat_code)
            for key, value in flat_dict.items():
                record = base_record.copy()
                record['metric'] = key
                record['value'] = value
                stats_list.append(record)
    else:
        record = base_record.copy()
        record['metric'] = 'value'
        record['value'] = stat_value
        stats_list.append(record)

def flatten_dict(d, parent_key='', sep='_'):
    """Flatten a nested dictionary structure"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            if all(not isinstance(x, (dict, list)) for x in v):
                items.append((new_key, ', '.join(str(x) for x in v)))
            else:
                items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)

def main():
    parser = argparse.ArgumentParser(description="Collect football statistics from SportMonks API")
    parser.add_argument('--team', type=str, help='Process only the specified team name')
    parser.add_argument('--skip-players', action='store_true', help='Skip player data collection')
    args = parser.parse_args()
    
    SKIP_PLAYER_DATA = args.skip_players or True
    
    all_teams = [
        {"id": 2687, "name": "America"},
        {"id": 2626, "name": "Cruz Azul"},
        {"id": 2662, "name": "Monterrey"},
        {"id": 10036, "name": "Pachuca"},
        {"id": 967, "name": "Toluca"},
        {"id": 609, "name": "Tigres"},
        {"id": 10836, "name": "Leon"},
        {"id": 3951, "name": "Necaxa"}
    ]
    
    if args.team:
        teams = [team for team in all_teams if team["name"].lower() == args.team.lower()]
        if not teams:
            print(f"Error: Team '{args.team}' not found in the list. Available teams:")
            for team in all_teams:
                print(f"  - {team['name']}")
            return
    else:
        teams = all_teams
    
    seasons = get_seasons_for_team(teams[0]["id"], limit=5)
    save_data(seasons, "liga_mx_seasons.json")
    
    if len(seasons) == 0:
        logging.error("No seasons found. Cannot proceed.")
        return
    
    print(f"Processing data for {len(teams)} teams across {len(seasons)} seasons...")
    print(f"Player data collection {'SKIPPED' if SKIP_PLAYER_DATA else 'ENABLED'}")
    
    all_progress_tracker = pd.DataFrame(columns=['team', 'season', 'player_stats', 'match_stats', 'team_stats'])
    
    for team in teams:
        team_id = team["id"]
        team_name = team["name"]
        team_dir = os.path.join(DATA_DIR, team_name)
        os.makedirs(team_dir, exist_ok=True)
        
        print(f"\n--- Processing Team: {team_name} (ID: {team_id}) ---")
        
        progress_tracker = pd.DataFrame(columns=['team', 'season', 'player_stats', 'match_stats', 'team_stats'])
        
        if not SKIP_PLAYER_DATA:
            team_includes = ["venue", "country"]
            try:
                team_data = get_team_data(team_id, team_includes)
                save_data(team_data, f"{team_name}/team_data.json")
            except Exception as e:
                logging.error(f"Error getting team data for {team_name}: {e}")
                team_data = {"data": {"name": team_name}}
            
            all_players = []
            print(f"Getting players for {team_name} in each season...")
            for season in seasons:
                season_id = season['id']
                season_name = season['name'].replace("/", "-")
                try:
                    players = get_players_for_team(team_id, season_id)
                    if players:
                        all_players.extend(players)
                        save_data(players, f"{team_name}/players_season_{season_name}.json")
                except Exception as e:
                    logging.error(f"Error getting players for {team_name} in season {season_id}: {e}")
            
            unique_players = {}
            print(f"Deduplicating players for {team_name}...")
            for player in all_players:
                player_id = player.get('player_id') or player.get('id')
                if player_id and player_id not in unique_players:
                    unique_players[player_id] = player
            
            unique_player_list = list(unique_players.values())
            save_data(unique_player_list, f"{team_name}/players.json")
            
            print(f"Processing player statistics for {team_name}...")
            try:
                process_player_stats_for_team(team, seasons, team_dir)
                
                for season in seasons:
                    progress_tracker = pd.concat([progress_tracker, pd.DataFrame([{
                        'team': team_name,
                        'season': season['name'],
                        'player_stats': 1,
                        'match_stats': 0,
                        'team_stats': 0
                    }])], ignore_index=True)
            except Exception as e:
                logging.error(f"Error processing player statistics for {team_name}: {e}")
                for season in seasons:
                    progress_tracker = pd.concat([progress_tracker, pd.DataFrame([{
                        'team': team_name,
                        'season': season['name'],
                        'player_stats': 0,
                        'match_stats': 0,
                        'team_stats': 0,
                        'error': str(e)
                    }])], ignore_index=True)
        else:
            print(f"Skipping player data collection for {team_name}")
            for season in seasons:
                progress_tracker = pd.concat([progress_tracker, pd.DataFrame([{
                    'team': team_name,
                    'season': season['name'],
                    'player_stats': 1,
                    'match_stats': 0,
                    'team_stats': 0
                }])], ignore_index=True)
        
        print(f"Getting team statistics for {team_name} across all seasons...")
        all_team_stats = []
        all_team_match_stats = []
        
        for season in seasons:
            season_id = season['id']
            season_name = season['name'].replace("/", "-")
            print(f"  - Processing season {season_name}...")
            
            try:
                matches = get_matches_for_team(team_id, season_id)
                if matches:
                    save_data(matches, f"{team_name}/matches_{season_name}.json")
                    
                    team_match_stats, _ = process_match_data(matches, season_name)
                    
                    if team_match_stats:
                        match_stats_df = pd.DataFrame(team_match_stats)
                        match_stats_df.to_csv(os.path.join(team_dir, f"team_match_stats_{season_name}.csv"), index=False)
                        logging.info(f"Saved {len(team_match_stats)} team statistics for {team_name} in season {season_name}")
                        
                        all_team_match_stats.extend(team_match_stats)
                        
                    for idx, row in progress_tracker.iterrows():
                        if row['team'] == team_name and row['season'] == season['name']:
                            progress_tracker.at[idx, 'match_stats'] = len(matches)
                            progress_tracker.at[idx, 'team_stats'] = len(team_match_stats) if team_match_stats else 0
            except Exception as e:
                logging.error(f"Error getting team statistics for {team_name} in season {season_id}: {e}")
        
        if all_team_match_stats:
            all_stats_df = pd.DataFrame(all_team_match_stats)
            all_stats_df.to_csv(os.path.join(team_dir, "all_team_stats.csv"), index=False)
            logging.info(f"Saved {len(all_team_match_stats)} total team statistics for {team_name}")
        
        all_progress_tracker = pd.concat([all_progress_tracker, progress_tracker], ignore_index=True)
        
        print(f"Completed processing for {team_name}")
    
    all_progress_tracker.to_csv(os.path.join(DATA_DIR, "data_collection_summary.csv"), index=False)
    
    logging.info("Data collection complete!")
    print("\n----- Data Collection Complete -----")
    print(f"Processed {len(teams)} teams across {len(seasons)} seasons")
    print(f"Data has been saved to the '{DATA_DIR}' directory with a subfolder for each team")
    print("\nCollection Summary by Team and Season:")
    print(all_progress_tracker.to_string(index=False))
    print("\nCheck data_collection_summary.csv for detailed statistics")

def process_player_stats_for_team(team, seasons, team_dir):
    """Process player statistics for a specific team"""
    team_id = team["id"]
    team_name = team["name"]
    
    player_file = os.path.join(team_dir, "players.json")
    if not os.path.exists(player_file):
        logging.warning(f"No player data found for {team_name}")
        return pd.DataFrame()
    
    with open(player_file, 'r') as f:
        players = json.load(f)
    
    all_player_stats = []
    
    unique_players = {}
    for player in players:
        player_id = player.get('player_id') or player.get('id')
        if player_id and player_id not in unique_players:
            unique_players[player_id] = player
    
    for player_id, player_info in unique_players.items():
        player_details = get_player_info(player_id)
        
        if player_details:
            player_name = player_details.get('display_name') or player_details.get('name', '')
            position = player_details.get('position_id', '')
        else:
            player_name = player_info.get('display_name') or player_info.get('name', f"Player {player_id}")
            position = player_info.get('position_id', '')
        
        for season in seasons:
            season_id = season['id']
            season_name = season['name']
            
            player_data = get_player_statistics(player_id, season_id)
            
            if player_data and 'statistics' in player_data:
                for stat_entry in player_data['statistics']:
                    if stat_entry.get('season_id') != season_id or stat_entry.get('team_id') != team_id:
                        continue
                    
                    player_stats = {
                        'player_id': player_id,
                        'player_name': player_name,
                        'season_id': season_id,
                        'season_name': season_name,
                        'team_id': team_id,
                        'team_name': team_name,
                        'position': position,
                        'appearances': 0,
                        'minutes_played': 0,
                        'goals': 0,
                        'assists': 0
                    }
                    
                    if 'details' in stat_entry:
                        for detail in stat_entry['details']:
                            if 'type' in detail and 'value' in detail:
                                stat_type = detail['type']
                                stat_name = stat_type.get('name', '').lower()
                                stat_value = detail['value']
                                
                                if 'goal' in stat_name:
                                    player_stats['goals'] = stat_value.get('total', 0) if isinstance(stat_value, dict) else stat_value
                                elif 'assist' in stat_name:
                                    player_stats['assists'] = stat_value.get('total', 0) if isinstance(stat_value, dict) else stat_value
                                elif 'appearance' in stat_name:
                                    player_stats['appearances'] = stat_value.get('total', 0) if isinstance(stat_value, dict) else stat_value
                                elif 'minute' in stat_name:
                                    player_stats['minutes_played'] = stat_value.get('total', 0) if isinstance(stat_value, dict) else stat_value
                    
                    if player_stats['appearances'] > 0 or player_stats['minutes_played'] > 0:
                        all_player_stats.append(player_stats)
    
    if all_player_stats:
        df = pd.DataFrame(all_player_stats)
        
        for season in seasons:
            season_id = season['id']
            season_name = season['name'].replace("/", "-")
            season_df = df[df['season_id'] == season_id]
            
            if not season_df.empty:
                csv_path = os.path.join(team_dir, f"player_performance_data_{season_name}.csv")
                season_df.to_csv(csv_path, index=False)
        
        csv_path = os.path.join(team_dir, "player_performance_data_all_seasons.csv")
        df.to_csv(csv_path, index=False)
        return df
    
    return pd.DataFrame()

if __name__ == "__main__":
    main()
