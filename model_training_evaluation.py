import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
import json
from sklearn.inspection import permutation_importance

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

# Create directories for models and results
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("plots/models", exist_ok=True)

def load_processed_data():
    """Load the processed data from feature engineering step"""
    # Try to load the regular model features first
    if os.path.exists("processed_data/model_features.csv"):
        print("Loading model features...")
        df = pd.read_csv("processed_data/model_features.csv")
    # Fall back to simplified model features if needed
    elif os.path.exists("processed_data/simplified_model_features.csv"):
        print("Loading simplified model features...")
        df = pd.read_csv("processed_data/simplified_model_features.csv")
    else:
        print("No processed data found. Please run feature_engineering.py first.")
        return None
    
    # Try to load player-team features and merge with model features if available
    if os.path.exists("processed_data/player_team_features.csv"):
        print("Loading player-team features...")
        player_features = pd.read_csv("processed_data/player_team_features.csv")
        
        # Check if we have team_name and season_name columns for merging
        if 'team_name' in player_features.columns and 'team_name_' in df.columns:
            # Rename columns for proper merging
            player_features_renamed = player_features.rename(columns={'team_name': 'team_name_'})
            
            # Merge on team_name and season_name if both available
            if 'season_name' in player_features_renamed.columns and 'season_name' in df.columns:
                print("Merging player features based on team_name and season_name...")
                df = pd.merge(
                    df,
                    player_features_renamed,
                    on=['team_name_', 'season_name'],
                    how='left'
                )
            # Otherwise just merge on team_name
            else:
                print("Merging player features based on team_name only...")
                df = pd.merge(
                    df,
                    player_features_renamed,
                    on='team_name_',
                    how='left'
                )
            
            print(f"Added {len(player_features.columns)} player-based features")
            
            # Fill any NaN values from the merge
            df = df.fillna(0)
    
    # Check if we have a match_date column and convert to datetime
    if 'match_date' in df.columns:
        df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
    
    # Print information about the loaded data
    print(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
    # print(f"Column names: {df.columns.tolist()}")
    
    return df

def chronological_train_test_split(df, test_size=0.2, date_column='match_date'):
    """Split the dataset chronologically (never randomly with time series)"""
    if date_column not in df.columns:
        print(f"Warning: {date_column} not found in data. Using sequential split instead.")
        train_size = int((1 - test_size) * len(df))
        train_data = df.iloc[:train_size]
        test_data = df.iloc[train_size:]
        return train_data, test_data
    
    # Sort by date
    df = df.sort_values(by=date_column)
    
    # Calculate the split point
    split_idx = int((1 - test_size) * len(df))
    
    # Get the split date for information
    split_date = df.iloc[split_idx][date_column]
    print(f"Splitting data chronologically at date: {split_date}")
    
    # Split the data
    train_data = df.iloc[:split_idx]
    test_data = df.iloc[split_idx:]
    
    print(f"Training data: {train_data.shape[0]} samples from {train_data[date_column].min()} to {train_data[date_column].max()}")
    print(f"Testing data: {test_data.shape[0]} samples from {test_data[date_column].min()} to {test_data[date_column].max()}")
    
    return train_data, test_data

def prepare_features_and_target(df, target_column='match_outcome', problem_type='classification'):
    """Prepare features and target from the dataframe"""
    if target_column not in df.columns:
        print(f"Target column {target_column} not found in data.")
        return None, None, []
    
    # Identify feature columns (exclude date, ID columns and target)
    exclude_cols = ['match_id', 'match_date', target_column, 'team_id']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Identify different feature types for better documentation
    team_perf_cols = [col for col in feature_cols if any(stat in col for stat in ['Goals', 'Wins', 'Cleansheets', 'Score'])]
    strength_cols = [col for col in feature_cols if any(term in col for term in ['strength', 'form'])]
    context_cols = [col for col in feature_cols if any(term in col for term in ['tournament', 'is_home', 'matchday', 'phase', 'home_advantage'])]
    player_cols = [col for col in feature_cols if any(term in col for term in ['player', 'efficiency', 'experience', 'key_player', 'contributions'])]
    
    # Print summary of feature types
    print(f"Selected {len(feature_cols)} feature columns")
    print(f"  - {len(team_perf_cols)} team performance features")
    print(f"  - {len(strength_cols)} team strength features")
    print(f"  - {len(context_cols)} match context features")
    print(f"  - {len(player_cols)} player-based features")
    
    # Identify player feature metrics to help with interpretation
    if player_cols:
        print("\nPlayer-based features include:")
        player_metrics = set()
        for col in player_cols:
            # Extract the base metric name from the column
            if '_diff' in col:
                metric = col.replace('_diff', '')
                player_metrics.add(f"{metric} (difference between teams)")
            elif 'home_' in col:
                metric = col.replace('home_', '')
                player_metrics.add(f"{metric} (home team)")
            elif 'away_' in col:
                metric = col.replace('away_', '')
                player_metrics.add(f"{metric} (away team)")
            else:
                player_metrics.add(col)
        
        for metric in sorted(player_metrics):
            print(f"  - {metric}")
    
    # Extract features
    X = df[feature_cols].copy()
    
    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=['number']).columns.tolist()
    
    # Fix mixed type columns by converting all categorical data to strings
    # This prevents the "Encoders require their input argument must be uniformly strings or numbers" error
    for col in categorical_cols:
        X[col] = X[col].astype(str)
    
    print(f"\nNumeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    
    # For classification problems, convert target to int labels
    # For regression, keep as is
    if problem_type == 'classification':
        # For match outcome: win=2, draw=1, loss=0, make sure it's int type
        y = df[target_column].astype(int)
    else:
        y = df[target_column]
    
    # Save feature types to dictionary for reference
    feature_types = {
        'team_performance': team_perf_cols,
        'team_strength': strength_cols,
        'match_context': context_cols,
        'player_based': player_cols
    }
    
    # Save feature type information to a JSON file for reference
    try:
        os.makedirs("models", exist_ok=True)
        with open('models/feature_types.json', 'w') as f:
            json.dump(feature_types, f, indent=4)
        print("Feature type information saved to models/feature_types.json")
    except Exception as e:
        print(f"Warning: Could not save feature type information: {str(e)}")
    
    return X, y, feature_cols

def create_preprocessor(numeric_features, categorical_features):
    """Create a scikit-learn preprocessor for the data"""
    # Numeric features: impute missing values, then scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical features: impute missing values, then one-hot encode
    if categorical_features:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'  # Drop any columns not specified
        )
    else:
        # If no categorical features, just process numeric ones
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ],
            remainder='drop'
        )
    
    return preprocessor

def train_classification_models(X_train, y_train, preprocessor):
    """Train and return multiple classification models"""
    models = {}
    
    # Logistic Regression
    print("Training Logistic Regression model...")
    lr_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    lr_pipe.fit(X_train, y_train)
    models['logistic_regression'] = lr_pipe
    
    # Support Vector Machine for Classification (SVC)
    print("Training Support Vector Classification (SVC) model...")
    svm_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', SVC(probability=True, random_state=42))
    ])
    svm_pipe.fit(X_train, y_train)
    models['svm'] = svm_pipe
    
    # Decision Tree
    print("Training Decision Tree model...")
    dt_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])
    dt_pipe.fit(X_train, y_train)
    models['decision_tree'] = dt_pipe
    
    # Random Forest
    print("Training Random Forest model...")
    rf_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    rf_pipe.fit(X_train, y_train)
    models['random_forest'] = rf_pipe
    
    # Gradient Boosting
    print("Training Gradient Boosting model...")
    gb_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ])
    gb_pipe.fit(X_train, y_train)
    models['gradient_boosting'] = gb_pipe
    
    # AdaBoost
    print("Training AdaBoost model...")
    ada_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', AdaBoostClassifier(n_estimators=100, random_state=42))
    ])
    ada_pipe.fit(X_train, y_train)
    models['adaboost'] = ada_pipe
    
    # Neural Network (MLP)
    print("Training Neural Network (MLP) model...")
    mlp_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42))
    ])
    mlp_pipe.fit(X_train, y_train)
    models['neural_network'] = mlp_pipe
    
    return models

def train_regression_models(X_train, y_train, preprocessor):
    """Train and return multiple regression models"""
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import AdaBoostRegressor
    import numpy as np
    
    models = {}
    
    # Support Vector Regression (SVR)
    print("Training Support Vector Regression (SVR) model...")
    svm_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', SVR(kernel='rbf'))  # Using RBF kernel for non-linear relationships
    ])
    svm_pipe.fit(X_train, y_train)
    models['svm'] = svm_pipe
    
    # Decision Tree Regressor
    print("Training Decision Tree Regressor model...")
    dt_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', DecisionTreeRegressor(random_state=42))
    ])
    dt_pipe.fit(X_train, y_train)
    models['decision_tree'] = dt_pipe
    
    # Random Forest Regressor
    print("Training Random Forest Regressor model...")
    rf_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    rf_pipe.fit(X_train, y_train)
    models['random_forest'] = rf_pipe
    
    # Gradient Boosting Regressor
    print("Training Gradient Boosting Regressor model...")
    gb_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ])
    gb_pipe.fit(X_train, y_train)
    models['gradient_boosting'] = gb_pipe
    
    # AdaBoost Regressor
    print("Training AdaBoost Regressor model...")
    ada_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', AdaBoostRegressor(n_estimators=100, random_state=42))
    ])
    ada_pipe.fit(X_train, y_train)
    models['adaboost'] = ada_pipe
    
    # Neural Network Regressor (MLP)
    print("Training Neural Network (MLP) Regressor model...")
    mlp_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42))
    ])
    mlp_pipe.fit(X_train, y_train)
    models['neural_network'] = mlp_pipe
    
    return models

def evaluate_classification_models(models, X_test, y_test):
    """Evaluate classification models and return metrics"""
    results = {}
    
    # Create a summary dataframe for comparison
    summary_metrics = []
    
    print("\n===== Classification Model Evaluation =====")
    
    for name, model in models.items():
        print(f"\nEvaluating {name} model...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Calculate precision, recall, F1-score for each class
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        # Calculate ROC curve and AUC for multi-class
        if y_pred_proba is not None:
            # For multi-class, we use one-vs-rest approach
            n_classes = len(np.unique(y_test))
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            # Use one-hot encoding for actual values
            y_test_dummies = pd.get_dummies(y_test).values
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_pred_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test_dummies.ravel(), y_pred_proba.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        else:
            roc_auc = {"micro": 0.0}
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': report,
            'roc_auc': roc_auc["micro"]
        }
        
        # Print results
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc['micro']:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(conf_matrix)
        
        # Visualize confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Loss', 'Draw', 'Win'] if len(np.unique(y_test)) == 3 else 'auto',
                    yticklabels=['Loss', 'Draw', 'Win'] if len(np.unique(y_test)) == 3 else 'auto')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - {name}')
        plt.tight_layout()
        plt.savefig(f'plots/models/confusion_matrix_{name}.png')
        plt.close()
        
        # Add to summary metrics
        summary_metrics.append({
            'model': name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc["micro"]
        })
    
    # Create a summary dataframe and plot
    metrics_df = pd.DataFrame(summary_metrics)
    
    # Plot comparison of metrics
    plt.figure(figsize=(12, 8))
    metrics_melted = pd.melt(metrics_df, id_vars=['model'], value_vars=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
    sns.barplot(x='model', y='value', hue='variable', data=metrics_melted)
    plt.title('Model Performance Comparison')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.savefig('plots/models/classification_model_comparison.png')
    plt.close()
    
    # Save detailed results to JSON
    with open('results/classification_results.json', 'w') as f:
        # Convert numpy arrays and other non-serializable objects to lists
        serializable_results = {}
        for model_name, model_results in results.items():
            serializable_results[model_name] = {}
            for metric, value in model_results.items():
                if isinstance(value, (np.ndarray, list)):
                    serializable_results[model_name][metric] = np.array(value).tolist()
                elif isinstance(value, dict):
                    serializable_results[model_name][metric] = {k: float(v) if isinstance(v, np.float64) else v for k, v in value.items()}
                elif isinstance(value, np.float64):
                    serializable_results[model_name][metric] = float(value)
                else:
                    serializable_results[model_name][metric] = value
                    
        json.dump(serializable_results, f, indent=4)
    
    # Return the results
    return results, metrics_df

def evaluate_regression_models(models, X_test, y_test):
    """Evaluate regression models and return metrics"""
    results = {}
    
    # Create a summary dataframe for comparison
    summary_metrics = []
    
    print("\n===== Regression Model Evaluation =====")
    
    for name, model in models.items():
        print(f"\nEvaluating {name} model...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        results[name] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        # Print results
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        # Plot actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Actual vs Predicted - {name} (R² = {r2:.4f})')
        plt.tight_layout()
        plt.savefig(f'plots/models/actual_vs_predicted_{name}.png')
        plt.close()
        
        # Add to summary metrics
        summary_metrics.append({
            'model': name,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        })
    
    # Create a summary dataframe and plot
    metrics_df = pd.DataFrame(summary_metrics)
    
    # Plot comparison of metrics
    plt.figure(figsize=(12, 8))
    metrics_melted = pd.melt(metrics_df, id_vars=['model'], value_vars=['mse', 'rmse', 'mae'])
    sns.barplot(x='model', y='value', hue='variable', data=metrics_melted)
    plt.title('Model Error Metrics Comparison')
    plt.xlabel('Model')
    plt.ylabel('Error Value')
    plt.xticks(rotation=45)
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.savefig('plots/models/regression_error_comparison.png')
    plt.close()
    
    # Plot R² comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x='model', y='r2', data=metrics_df)
    plt.title('Model R² Score Comparison')
    plt.xlabel('Model')
    plt.ylabel('R² Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/models/regression_r2_comparison.png')
    plt.close()
    
    # Save detailed results to JSON
    with open('results/regression_results.json', 'w') as f:
        # Convert numpy values to Python native types for JSON serialization
        serializable_results = {}
        for model_name, model_results in results.items():
            serializable_results[model_name] = {
                k: float(v) if isinstance(v, np.float64) else v 
                for k, v in model_results.items()
            }
        json.dump(serializable_results, f, indent=4)
    
    # Return the results
    return results, metrics_df

def plot_feature_importance(models, feature_names, problem_type='classification'):
    """Plot feature importance for applicable models"""
    for name, model in models.items():
        # Extract the actual model from the pipeline if possible
        try:
            if hasattr(model, 'named_steps'):
                if problem_type == 'classification':
                    estimator = model.named_steps.get('classifier')
                else:
                    estimator = model.named_steps.get('regressor')
            else:
                # For custom model classes that don't follow the Pipeline structure
                print(f"Skipping feature importance for {name} - not a standard scikit-learn pipeline")
                continue
                
            # Check if model has feature_importances_ attribute (tree-based models)
            if hasattr(estimator, 'feature_importances_'):
                # Get preprocessed feature names (this is complex due to preprocessing)
                # For simplicity, we'll use the original feature names, but this may not be accurate for one-hot encoded features
                
                # Get feature importances
                importances = estimator.feature_importances_
                
                # Create feature importance dataframe
                importance_df = pd.DataFrame({
                    'feature': feature_names[:len(importances)] if len(feature_names) >= len(importances) else feature_names + ['Unknown'] * (len(importances) - len(feature_names)),
                    'importance': importances
                })
                
                # Sort by importance
                importance_df = importance_df.sort_values('importance', ascending=False)
                
                # Plot top 20 features (or all if less than 20)
                top_n = min(20, len(importance_df))
                plt.figure(figsize=(12, 8))
                sns.barplot(x='importance', y='feature', data=importance_df.head(top_n))
                plt.title(f'Feature Importance - {name}')
                plt.tight_layout()
                plt.savefig(f'plots/models/feature_importance_{name}.png')
                plt.close()
                
                # Save to CSV
                importance_df.to_csv(f'results/feature_importance_{name}.csv', index=False)
                
                print(f"Feature importance for {name} saved to results/feature_importance_{name}.csv")
        except Exception as e:
            print(f"Error plotting feature importance for {name}: {str(e)}")
            continue

def save_models(models, problem_type='classification'):
    """Save trained models to disk"""
    # Create timestamp for model version
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for name, model in models.items():
        try:
            filename = f"models/{name}_{problem_type}_{timestamp}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {name} model to {filename}")
        except Exception as e:
            print(f"Error saving {name} model: {str(e)}")
            print(f"Skipping saving {name} model")

# === Liguilla Prediction Functions ===

# Liguilla bracket structure (quarterfinals)
LIGUILLA_BRACKET = [
    {"round": "Quarterfinals", "match_id": 1, "home_team": 1, "away_team": 8},  # 1st vs 8th
    {"round": "Quarterfinals", "match_id": 2, "home_team": 2, "away_team": 7},  # 2nd vs 7th
    {"round": "Quarterfinals", "match_id": 3, "home_team": 3, "away_team": 6},  # 3rd vs 6th
    {"round": "Quarterfinals", "match_id": 4, "home_team": 4, "away_team": 5},  # 4th vs 5th
]

# Team standings for the current tournament
TEAM_STANDINGS = {
    1: "Toluca",
    2: "America",
    3: "Cruz Azul",
    4: "Tigres UANL",
    5: "Necaxa",
    6: "Leon",
    7: "Pachuca",
    8: "Monterrey"
}

def create_match_features(home_team, away_team, df):
    """Create features for a match between home_team and away_team"""
    # Get the most recent data for both teams
    home_team_data = df[df['team_name_'] == home_team].iloc[-1].to_dict() if any(df['team_name_'] == home_team) else None
    away_team_data = df[df['team_name_'] == away_team].iloc[-1].to_dict() if any(df['team_name_'] == away_team) else None
    
    if home_team_data is None or away_team_data is None:
        print(f"Could not create match features for {home_team} vs {away_team}")
        return None
    
    # Create a new dict for the match features
    match_features = home_team_data.copy()
    
    # Update team information
    match_features['team_name_'] = str(home_team)  # Ensure string type
    
    # Set home field advantage indicators
    match_features['is_home_'] = 1
    if 'home_advantage' in match_features:
        match_features['home_advantage'] = 1
    
    # Update opponent strength if available
    if 'team_strength_normalized' in away_team_data and 'opponent_strength' in match_features:
        match_features['opponent_strength'] = away_team_data['team_strength_normalized']
    
    # Set match date to today
    if 'match_date' in match_features:
        match_features['match_date'] = datetime.now().strftime('%Y-%m-%d')
    
    # It's a Liguilla match
    if 'tournament_phase' in match_features:
        match_features['tournament_phase'] = 'liguilla'
    
    # Add player-based comparative features
    player_feature_prefixes = ['home_', 'away_']
    player_comparison_features = ['num_players', 'num_key_players', 'avg_player_efficiency', 
                                'weighted_contributions_per_90', 'team_experience', 'key_player_ratio']
    
    # For each comparison feature, check if it exists and update if needed
    for prefix in player_feature_prefixes:
        for feature in player_comparison_features:
            feature_name = f"{prefix}{feature}"
            
            # If home team feature exists in the data
            if feature_name in home_team_data:
                match_features[feature_name] = home_team_data[feature_name]
                
                # If it's a home_ feature and we have away data, set the away_ feature too
                if prefix == 'home_' and f"away_{feature}" in match_features:
                    # Get the corresponding feature from away team
                    away_feature_name = f"away_{feature}"
                    away_feature_value = away_team_data.get(feature_name.replace('home_', 'away_'), 0)
                    match_features[away_feature_name] = away_feature_value
    
    # Add difference features for player metrics if both home and away values exist
    for feature in player_comparison_features:
        home_feature = f"home_{feature}"
        away_feature = f"away_{feature}"
        diff_feature = f"{feature}_diff"
        
        if home_feature in match_features and away_feature in match_features:
            match_features[diff_feature] = match_features[home_feature] - match_features[away_feature]
    
    # Convert any categorical/string features to string type to ensure consistent types
    # This prevents encoding errors during prediction
    categorical_features = [col for col in match_features.keys() if col not in ['match_date'] and not isinstance(match_features[col], (int, float, bool, np.number))]
    for col in categorical_features:
        match_features[col] = str(match_features[col])
    
    # Return the feature set
    return match_features

def predict_match_outcome(home_team, away_team, model, df, feature_cols, problem_type='classification'):
    """Predict the outcome of a match between two teams using the trained model"""
    try:
        # Create match features for home team
        match_features = create_match_features(home_team, away_team, df)
        
        if match_features is None:
            raise ValueError(f"Failed to create match features for {home_team} vs {away_team}")
        
        # Extract only the features used by the model
        match_features_df = pd.DataFrame([match_features])
        X_match = match_features_df[feature_cols]
        
        # Make prediction
        if problem_type == 'classification':
            prediction = model.predict(X_match)[0]
            probabilities = model.predict_proba(X_match)[0]
            
            # Map the prediction to outcome
            outcomes = ['Loss', 'Draw', 'Win']
            result = outcomes[prediction]
            
            probability_map = {
                outcomes[i]: round(probabilities[i] * 100, 2) for i in range(len(outcomes))
            }
        else:
            # For regression models (goals prediction)
            predicted_goals = model.predict(X_match)[0]
            result = f"{predicted_goals:.2f} goals"
            probability_map = {"Predicted Goals": predicted_goals}
        
        return {
            'prediction': result,
            'home_team': home_team,
            'away_team': away_team,
            'probabilities': probability_map,
            'error': None
        }
    except Exception as e:
        print(f"Error predicting match outcome: {str(e)}")
        return {
            'prediction': 'Error',
            'home_team': home_team,
            'away_team': away_team,
            'probabilities': {'Error': 100},
            'error': str(e)
        }

def simulate_liguilla_round(bracket, model, df, feature_cols, problem_type='classification'):
    """Simulate a round of the Liguilla tournament"""
    results = []
    
    for match in bracket:
        home_team = match['home_team']
        away_team = match['away_team']
        
        # For team IDs, fetch the actual team names
        if isinstance(home_team, int) and home_team in TEAM_STANDINGS:
            home_team = TEAM_STANDINGS[home_team]
        if isinstance(away_team, int) and away_team in TEAM_STANDINGS:
            away_team = TEAM_STANDINGS[away_team]
        
        try:
            # Predict first leg (away team at home)
            away_home_result = predict_match_outcome(
                away_team, home_team, model, df, feature_cols, problem_type
            )
            
            # Predict second leg (home team at home)
            home_home_result = predict_match_outcome(
                home_team, away_team, model, df, feature_cols, problem_type
            )
            
            # Determine the winner
            winner_explanation = ""
            if problem_type == 'classification':
                # Convert outcomes to points
                away_result = away_home_result['prediction']
                home_result = home_home_result['prediction']
                
                away_points = 0
                home_points = 0
                
                # Calculate points based on predictions (standard soccer points: 3 for win, 1 for draw, 0 for loss)
                if away_result == 'Win':  # Away team wins as home
                    away_points += 3
                elif away_result == 'Draw':
                    away_points += 1
                    home_points += 1
                elif away_result == 'Loss':  # Home team wins as away
                    home_points += 3
                
                if home_result == 'Win':  # Home team wins as home
                    home_points += 3
                elif home_result == 'Draw':
                    home_points += 1
                    away_points += 1
                elif home_result == 'Loss':  # Away team wins as away
                    away_points += 3
                
                # The team with more points advances
                if home_points > away_points:
                    winner = home_team
                    winner_explanation = f"More points: {home_team} ({home_points} pts) vs {away_team} ({away_points} pts)"
                elif away_points > home_points:
                    winner = away_team
                    winner_explanation = f"More points: {away_team} ({away_points} pts) vs {home_team} ({home_points} pts)"
                else:
                    # In case of a tie, higher seed advances (home team is always higher seed)
                    winner = home_team
                    winner_explanation = f"Equal points ({home_points} pts each) - {home_team} advances as higher seed"
            else:
                # For regression problems, compare predicted goals
                away_goals = away_home_result['probabilities'].get('Predicted Goals', 0)
                home_goals = home_home_result['probabilities'].get('Predicted Goals', 0)
                
                # Total goals for each team over two legs
                total_away_goals = away_goals
                total_home_goals = home_goals
                
                if total_home_goals > total_away_goals:
                    winner = home_team
                    winner_explanation = f"More goals over two legs: {home_team} ({total_home_goals:.2f}) vs {away_team} ({total_away_goals:.2f})"
                elif total_away_goals > total_home_goals:
                    winner = away_team
                    winner_explanation = f"More goals over two legs: {away_team} ({total_away_goals:.2f}) vs {home_team} ({total_home_goals:.2f})"
                else:
                    # In case of a tie, higher seed advances (home team)
                    winner = home_team
                    winner_explanation = f"Equal goals ({total_home_goals:.2f} each) - {home_team} advances as higher seed"
            
            # Store the results
            results.append({
                'round': match['round'],
                'match_id': match['match_id'],
                'home_team': home_team,
                'away_team': away_team,
                'first_leg': away_home_result,
                'second_leg': home_home_result,
                'winner': winner,
                'winner_explanation': winner_explanation
            })
        except Exception as e:
            print(f"Error simulating match between {home_team} and {away_team}: {str(e)}")
            # Add a partial result with the error
            results.append({
                'round': match['round'],
                'match_id': match['match_id'],
                'home_team': home_team,
                'away_team': away_team,
                'error': str(e),
                'winner': home_team,  # Default to home team (higher seed) on error
                'winner_explanation': f"Error in simulation - defaulting to {home_team} as higher seed"
            })
    
    return results

def create_next_round(current_results, next_round_name):
    """Create the bracket for the next round based on current results"""
    # Get winners 
    winners = []
    for r in current_results:
        if r.get('winner') is not None:
            winners.append(r['winner'])
        else:
            print(f"Warning: No winner found for match between {r['home_team']} and {r['away_team']}")
    
    # If we don't have enough winners, we can't create the next round
    if len(winners) < 2:
        print(f"Error: Not enough winners to create {next_round_name}")
        return []
    
    next_round = []
    
    if len(winners) == 4:  # Creating semifinals
        # Match first winner against fourth, second against third
        next_round.append({
            'round': next_round_name,
            'match_id': 1,
            'home_team': winners[0],  # First winner
            'away_team': winners[3]   # Fourth winner
        })
        
        next_round.append({
            'round': next_round_name,
            'match_id': 2,
            'home_team': winners[1],  # Second winner
            'away_team': winners[2]   # Third winner
        })
        
    elif len(winners) == 2:  # Creating final
        # In final, just match the two semifinal winners
        next_round.append({
            'round': next_round_name,
            'match_id': 1,
            'home_team': winners[0],  # First semifinal winner
            'away_team': winners[1]   # Second semifinal winner
        })
    
    return next_round

def save_liguilla_results(all_results):
    """Save the Liguilla results to a JSON file"""
    # Create output directory if it doesn't exist
    os.makedirs("predictions", exist_ok=True)
    
    # Convert results to a serializable format
    serializable_results = {}
    
    for round_name, matches in all_results.items():
        serializable_results[round_name] = []
        
        for match in matches:
            serializable_match = match.copy()
            
            # Convert numpy values to Python native types
            for leg in ['first_leg', 'second_leg']:
                if leg in serializable_match:
                    for k, v in serializable_match[leg].items():
                        if isinstance(v, dict):
                            serializable_match[leg][k] = {
                                sk: float(sv) if isinstance(sv, (np.float32, np.float64)) else sv
                                for sk, sv in v.items()
                            }
                        elif isinstance(v, (np.float32, np.float64, np.int64, np.int32)):
                            serializable_match[leg][k] = float(v) if isinstance(v, (np.float32, np.float64)) else int(v)
            
            serializable_results[round_name].append(serializable_match)
    
    # Save to JSON
    with open('predictions/liguilla_results.json', 'w') as f:
        json.dump(serializable_results, f, indent=4)
    
    print("Liguilla results saved to predictions/liguilla_results.json")

def print_liguilla_results(all_results):
    """Print the Liguilla results in a readable format"""
    print("\n===== LIGA MX LIGUILLA PREDICTIONS =====\n")
    
    # Print explanation of advancement criteria
    if 'Quarterfinals' in all_results:
        first_match = all_results['Quarterfinals'][0]
        if 'first_leg' in first_match and 'probabilities' in first_match['first_leg']:
            probs = first_match['first_leg']['probabilities']
            if isinstance(probs, dict) and 'Loss' in probs and 'Draw' in probs and 'Win' in probs:
                print("ADVANCEMENT CRITERIA (Classification Model):")
                print("- Teams play two legs (home and away)")
                print("- Win = 3 points, Draw = 1 point, Loss = 0 points")
                print("- Team with more points advances")
                print("- If tied on points, higher seed advances")
            else:
                print("ADVANCEMENT CRITERIA (Regression Model):")
                print("- Teams play two legs (home and away)")
                print("- Team with more total predicted goals advances")
                print("- If tied on goals, higher seed advances")
        print("\n" + "-"*50 + "\n")
    
    rounds = list(all_results.keys())
    rounds.sort(key=lambda x: {'Quarterfinals': 0, 'Semifinals': 1, 'Final': 2}.get(x, 3))
    
    for round_name in rounds:
        print(f"\n{round_name.upper()}")
        print("=" * len(round_name))
        
        for match in all_results[round_name]:
            print(f"\nMatch {match['match_id']}: {match['home_team']} vs {match['away_team']}")
            
            # Print first leg
            if 'first_leg' in match:
                away_result = match['first_leg']['prediction'] if 'prediction' in match['first_leg'] else 'N/A'
                
                # Print probabilities if available
                if 'probabilities' in match['first_leg']:
                    probs = match['first_leg']['probabilities']
                    if isinstance(probs, dict) and 'Loss' in probs and 'Draw' in probs and 'Win' in probs:
                        prob_str = f" (Loss: {probs['Loss']:.2f}%, Draw: {probs['Draw']:.2f}%, Win: {probs['Win']:.2f}%)"
                    else:
                        prob_str = f" ({probs})"
                else:
                    prob_str = ""
                    
                print(f"  First Leg: {match['away_team']} at home - Result: {away_result}{prob_str}")
            
            # Print second leg
            if 'second_leg' in match:
                home_result = match['second_leg']['prediction'] if 'prediction' in match['second_leg'] else 'N/A'
                
                # Print probabilities if available
                if 'probabilities' in match['second_leg']:
                    probs = match['second_leg']['probabilities']
                    if isinstance(probs, dict) and 'Loss' in probs and 'Draw' in probs and 'Win' in probs:
                        prob_str = f" (Loss: {probs['Loss']:.2f}%, Draw: {probs['Draw']:.2f}%, Win: {probs['Win']:.2f}%)"
                    else:
                        prob_str = f" ({probs})"
                else:
                    prob_str = ""
                    
                print(f"  Second Leg: {match['home_team']} at home - Result: {home_result}{prob_str}")
            
            # Print winner with explanation
            if 'winner_explanation' in match:
                print(f"  WINNER: {match['winner']} advances - {match['winner_explanation']}")
            else:
                print(f"  WINNER: {match['winner']} advances")
    
    # Print champion
    if 'Final' in all_results and all_results['Final']:
        champion = all_results['Final'][0]['winner']
        print(f"\nPREDICTED CHAMPION: {champion}\n")

def predict_liguilla(best_model, df, feature_cols, problem_type='classification'):
    """Predict the entire Liguilla tournament using the best model"""
    # Create output directory
    os.makedirs("plots/liguilla", exist_ok=True)
    
    print("\n===== Predicting Liga MX Liguilla =====")
    
    # Check if we have the necessary data
    if df is None or best_model is None:
        print("Error: Missing data or model for Liguilla prediction")
        return
    
    # Simulate Liguilla
    all_results = {}
    
    # Quarter-finals
    print("\nSimulating Quarter-finals...")
    qf_results = simulate_liguilla_round(LIGUILLA_BRACKET, best_model, df, feature_cols, problem_type)
    all_results['Quarterfinals'] = qf_results
    
    # Semi-finals
    print("\nSimulating Semi-finals...")
    sf_bracket = create_next_round(qf_results, "Semifinals")
    sf_results = simulate_liguilla_round(sf_bracket, best_model, df, feature_cols, problem_type)
    all_results['Semifinals'] = sf_results
    
    # Final
    print("\nSimulating Final...")
    final_bracket = create_next_round(sf_results, "Final")
    final_results = simulate_liguilla_round(final_bracket, best_model, df, feature_cols, problem_type)
    all_results['Final'] = final_results
    
    # Save results
    save_liguilla_results(all_results)
    
    # Print results
    print_liguilla_results(all_results)
    
    print("\nLiguilla prediction completed!")
    print("Results saved to predictions/liguilla_results.json")

def select_best_model(metrics_df, problem_type='classification'):
    """Select the best model based on metrics"""
    if problem_type == 'classification':
        # For classification, prioritize F1 score
        best_model = metrics_df.loc[metrics_df['f1'].idxmax()]['model']
        print(f"\nBest model based on F1 score: {best_model}")
        return best_model
    else:
        # For regression, prioritize R² score
        best_model = metrics_df.loc[metrics_df['r2'].idxmax()]['model']
        print(f"\nBest model based on R² score: {best_model}")
        return best_model

def analyze_player_feature_importance(models, X_test, y_test, feature_cols, problem_type='classification'):
    """Analyze the importance of player-based features and visualize their impact"""
    # Identify player-based features
    player_cols = [col for col in feature_cols if any(term in col for term in ['player', 'efficiency', 'experience', 'key_player', 'contributions'])]
    
    if not player_cols:
        print("No player-based features found for analysis")
        return
    
    print(f"\n===== Analyzing Impact of {len(player_cols)} Player-Based Features =====")
    
    # Filter models to only include standard scikit-learn compatible ones
    compatible_models = {}
    for name, model in models.items():
        # Skip custom model classes that might not be compatible
        if not hasattr(model, 'predict') or any(model_type in name.lower() for model_type in ['kmeans', 'softmax']):
            print(f"Skipping {name} for feature importance analysis - not compatible")
            continue
        compatible_models[name] = model
    
    if not compatible_models:
        print("No compatible models available for feature importance analysis")
        return
    
    # Select the most performant model
    if problem_type == 'classification':
        # Evaluate all models and track their metrics
        model_scores = {}
        for name, model in compatible_models.items():
            try:
                # Make predictions with all features
                y_pred = model.predict(X_test)
                f1 = precision_recall_fscore_support(y_test, y_pred, average='weighted')[2]
                model_scores[name] = f1
            except Exception as e:
                print(f"Error evaluating {name}: {str(e)}")
        
        if not model_scores:
            print("Could not evaluate any models for feature importance")
            return
            
        # Get best model
        best_model_name = max(model_scores, key=model_scores.get)
        best_model = compatible_models[best_model_name]
        print(f"Using best model ({best_model_name}) for feature importance analysis")
        
        try:
            # Baseline performance with all features
            baseline_pred = best_model.predict(X_test)
            baseline_score = precision_recall_fscore_support(y_test, baseline_pred, average='weighted')[2]
            print(f"Baseline F1 Score (all features): {baseline_score:.4f}")
            
            # Create a version of the test data without player features
            X_test_no_player = X_test.copy()
            for col in player_cols:
                if col in X_test_no_player.columns:
                    X_test_no_player[col] = 0  # Zero out player features
            
            # Predict without player features
            no_player_pred = best_model.predict(X_test_no_player)
            no_player_score = precision_recall_fscore_support(y_test, no_player_pred, average='weighted')[2]
            print(f"F1 Score without player features: {no_player_score:.4f}")
            print(f"Player feature contribution: {baseline_score - no_player_score:.4f} F1 points")
            
            # Analyze feature groups to see which player metrics are most important
            player_metric_groups = {
                'efficiency': [col for col in player_cols if 'efficiency' in col],
                'key_players': [col for col in player_cols if 'key_player' in col],
                'experience': [col for col in player_cols if 'experience' in col],
                'contributions': [col for col in player_cols if 'contribution' in col],
                'player_count': [col for col in player_cols if 'num_player' in col]
            }
            
            # Visualize impact of different player metric groups
            metric_impact = {}
            for group_name, group_cols in player_metric_groups.items():
                if not group_cols:
                    continue
                    
                # Create a version of the test data without this group of features
                X_test_no_group = X_test.copy()
                for col in group_cols:
                    if col in X_test_no_group.columns:
                        X_test_no_group[col] = 0  # Zero out group features
                
                # Predict without group features
                no_group_pred = best_model.predict(X_test_no_group)
                no_group_score = precision_recall_fscore_support(y_test, no_group_pred, average='weighted')[2]
                
                # Calculate impact
                impact = baseline_score - no_group_score
                metric_impact[group_name] = impact
                print(f"Impact of {group_name} features: {impact:.4f} F1 points")
            
            # Visualize the impacts
            if metric_impact:
                plt.figure(figsize=(10, 6))
                impact_df = pd.DataFrame({'Metric Group': list(metric_impact.keys()), 
                                         'F1 Score Impact': list(metric_impact.values())})
                impact_df = impact_df.sort_values('F1 Score Impact', ascending=False)
                
                sns.barplot(x='F1 Score Impact', y='Metric Group', data=impact_df)
                plt.title('Impact of Player Metric Groups on Prediction Accuracy')
                plt.tight_layout()
                plt.savefig('plots/models/player_feature_impact.png')
                plt.close()
                
                print(f"Player feature impact visualization saved to plots/models/player_feature_impact.png")
        except Exception as e:
            print(f"Error in feature importance analysis: {str(e)}")
    
    else:  # regression
        # Evaluate all models and track their metrics
        model_scores = {}
        for name, model in compatible_models.items():
            try:
                # Make predictions with all features
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                model_scores[name] = r2
            except Exception as e:
                print(f"Error evaluating {name}: {str(e)}")
        
        if not model_scores:
            print("Could not evaluate any models for feature importance")
            return
            
        # Get best model
        best_model_name = max(model_scores, key=model_scores.get)
        best_model = compatible_models[best_model_name]
        print(f"Using best model ({best_model_name}) for feature importance analysis")
        
        try:
            # Baseline performance with all features
            baseline_pred = best_model.predict(X_test)
            baseline_r2 = r2_score(y_test, baseline_pred)
            print(f"Baseline R² Score (all features): {baseline_r2:.4f}")
            
            # Create a version of the test data without player features
            X_test_no_player = X_test.copy()
            for col in player_cols:
                if col in X_test_no_player.columns:
                    X_test_no_player[col] = 0  # Zero out player features
            
            # Predict without player features
            no_player_pred = best_model.predict(X_test_no_player)
            no_player_r2 = r2_score(y_test, no_player_pred)
            print(f"R² Score without player features: {no_player_r2:.4f}")
            print(f"Player feature contribution: {baseline_r2 - no_player_r2:.4f} R² points")
            
            # Analyze feature groups to see which player metrics are most important
            player_metric_groups = {
                'efficiency': [col for col in player_cols if 'efficiency' in col],
                'key_players': [col for col in player_cols if 'key_player' in col],
                'experience': [col for col in player_cols if 'experience' in col],
                'contributions': [col for col in player_cols if 'contribution' in col],
                'player_count': [col for col in player_cols if 'num_player' in col]
            }
            
            # Visualize impact of different player metric groups
            metric_impact = {}
            for group_name, group_cols in player_metric_groups.items():
                if not group_cols:
                    continue
                    
                # Create a version of the test data without this group of features
                X_test_no_group = X_test.copy()
                for col in group_cols:
                    if col in X_test_no_group.columns:
                        X_test_no_group[col] = 0  # Zero out group features
                
                # Predict without group features
                no_group_pred = best_model.predict(X_test_no_group)
                no_group_r2 = r2_score(y_test, no_group_pred)
                
                # Calculate impact
                impact = baseline_r2 - no_group_r2
                metric_impact[group_name] = impact
                print(f"Impact of {group_name} features: {impact:.4f} R² points")
            
            # Visualize the impacts
            if metric_impact:
                plt.figure(figsize=(10, 6))
                impact_df = pd.DataFrame({'Metric Group': list(metric_impact.keys()), 
                                         'R² Score Impact': list(metric_impact.values())})
                impact_df = impact_df.sort_values('R² Score Impact', ascending=False)
                
                sns.barplot(x='R² Score Impact', y='Metric Group', data=impact_df)
                plt.title('Impact of Player Metric Groups on Prediction Accuracy')
                plt.tight_layout()
                plt.savefig('plots/models/player_feature_impact.png')
                plt.close()
                
                print(f"Player feature impact visualization saved to plots/models/player_feature_impact.png")
        except Exception as e:
            print(f"Error in feature importance analysis: {str(e)}")            

def evaluate_model_behavior(best_model, X_test, y_test, feature_cols, problem_type='classification'):
    """Analyze model behavior and explain why certain models perform better than others"""
    # Get the actual model from the pipeline
    if problem_type == 'classification':
        estimator_name = 'classifier'
    else:
        estimator_name = 'regressor'
    
    # Try to extract the model
    try:
        model = best_model.named_steps.get(estimator_name)
        model_type = type(model).__name__
        print(f"Best model type: {model_type}")
    except:
        print("Could not extract model from pipeline")
        model_type = str(type(best_model))
    
    # Get feature importances if available
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Get feature names after preprocessing
        try:
            # This is complex because preprocessing may have transformed feature names
            # For simplicity, we'll use the original feature names
            top_features = pd.DataFrame({
                'feature': feature_cols[:len(importances)] if len(feature_cols) >= len(importances) else feature_cols + ['Unknown'] * (len(importances) - len(feature_cols)),
                'importance': importances
            }).sort_values('importance', ascending=False).head(10)
            
            print("\nTop 10 important features:")
            for i, (feature, importance) in enumerate(zip(top_features['feature'], top_features['importance'])):
                print(f"{i+1}. {feature}: {importance:.4f}")
        except:
            print("Could not determine top features")
    
    # For regression models, analyze residuals to understand model fit
    if problem_type == 'regression':
        try:
            # Make predictions
            y_pred = best_model.predict(X_test)
            
            # Calculate residuals
            residuals = y_test - y_pred
            
            # Create residual plot
            plt.figure(figsize=(10, 6))
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='-')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title('Residual Plot - Analyzing Model Fit')
            plt.tight_layout()
            plt.savefig('plots/models/residual_analysis.png')
            plt.close()
            
            # Check for patterns in residuals
            # print("\nResidual Analysis:")
            # print(f"Mean of residuals: {np.mean(residuals):.4f}")
            # print(f"Standard deviation of residuals: {np.std(residuals):.4f}")
            
            # Check if residuals are normally distributed
            from scipy import stats
            stat, p = stats.shapiro(residuals)
            print(f"Shapiro-Wilk test for normality: p-value = {p:.4f}")
            if p > 0.05:
                print("Residuals appear to be normally distributed (good)")
            else:
                print("Residuals do not appear to be normally distributed (indicates model misspecification)")
            
            # Check for heteroscedasticity (non-constant variance)
            if np.corrcoef(y_pred, np.abs(residuals))[0, 1] > 0.3:
                print("Possible heteroscedasticity detected (non-constant variance in residuals)")
            else:
                print("No strong heteroscedasticity detected")
            
            print("\nModel fit analysis saved to plots/models/residual_analysis.png")
            
            # Analyze linearity assumptions
            if 'linear' in model_type.lower():
                print("\nLinear Regression Assumption Analysis:")
                
                # Create actual vs predicted plot with perfect prediction line
                plt.figure(figsize=(10, 6))
                plt.scatter(y_test, y_pred, alpha=0.5)
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                plt.xlabel('Actual Values')
                plt.ylabel('Predicted Values')
                plt.title('Actual vs Predicted - Assessing Linearity')
                plt.tight_layout()
                plt.savefig('plots/models/linearity_analysis.png')
                plt.close()
                
                print("Linearity assumption analysis saved to plots/models/linearity_analysis.png")
                
                # Check for evidence of non-linearity
                nonlinearity = np.corrcoef(y_pred, np.square(residuals))[0, 1]
                if abs(nonlinearity) > 0.3:
                    print(f"Possible non-linear relationship detected (correlation: {nonlinearity:.4f})")
                    print("This suggests a more complex model like Gradient Boosting might be more appropriate.")
                else:
                    print("No strong evidence of non-linearity in the relationship")
        except Exception as e:
            print(f"Error analyzing model fit: {str(e)}")
    
    # Compare linear and tree-based models to understand data characteristics
    print("\nComparison of Model Types:")


def tune_gradient_boosting(X_train, y_train, X_test, y_test, preprocessor, problem_type='classification'):
    """Tune hyperparameters for Gradient Boosting to improve performance"""
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    import numpy as np
    
    print("\n===== Tuning Gradient Boosting Model =====")
    
    # Create parameter grid
    param_grid = {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__learning_rate': [0.01, 0.05, 0.1],
        'regressor__max_depth': [3, 5, 7],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__subsample': [0.8, 1.0]
    } if problem_type == 'regression' else {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__max_depth': [3, 5, 7],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__subsample': [0.8, 1.0]
    }
    
    # Create pipeline
    if problem_type == 'regression':
        gb_pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(random_state=42))
        ])
        scoring = 'r2'
    else:
        gb_pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', GradientBoostingClassifier(random_state=42))
        ])
        scoring = 'f1_weighted'
    
    # Create grid search
    grid_search = GridSearchCV(
        gb_pipe,
        param_grid,
        cv=3,  # Use 3-fold cross-validation
        scoring=scoring,
        n_jobs=-1  # Use all available cores
    )
    
    # Fit grid search
    print("Starting grid search for Gradient Boosting... (this may take a while)")
    grid_search.fit(X_train, y_train)
    
    # Print best parameters
    print("\nBest parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")
    
    # Get best estimator
    best_gb = grid_search.best_estimator_
    
    # Evaluate on test set
    if problem_type == 'regression':
        y_pred = best_gb.predict(X_test)
        from sklearn.metrics import r2_score, mean_squared_error
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"\nTuned Gradient Boosting performance:")
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
    else:
        y_pred = best_gb.predict(X_test)
        from sklearn.metrics import classification_report
        print(f"\nTuned Gradient Boosting performance:")
        print(classification_report(y_test, y_pred))
    
    return best_gb

def main():
    # Load the processed data
    df = load_processed_data()
    if df is None:
        return
    
    # Set the target column based on what's available in the data
    target_options = ['match_outcome', 'value_Goals_average']
    target_column = None
    problem_type = None
    
    for target in target_options:
        if target in df.columns:
            target_column = target
            problem_type = 'classification' if target == 'match_outcome' else 'regression'
            print(f"Using {target_column} as target variable for {problem_type}")
            break
    
    if target_column is None:
        print("No suitable target column found in the data.")
        return
    
    # Split the data chronologically
    train_data, test_data = chronological_train_test_split(df)
    
    # Prepare features and targets
    X_train, y_train, feature_cols = prepare_features_and_target(train_data, target_column, problem_type)
    X_test, y_test, _ = prepare_features_and_target(test_data, target_column, problem_type)
    
    if X_train is None or X_test is None:
        return
    
    # Identify numeric and categorical features
    numeric_features = X_train.select_dtypes(include=['number']).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=['number']).columns.tolist()
    
    # Create preprocessor
    preprocessor = create_preprocessor(numeric_features, categorical_features)
    
    # Train models based on problem type
    if problem_type == 'classification':
        models = train_classification_models(X_train, y_train, preprocessor)
        
        # Evaluate models
        results, metrics_df = evaluate_classification_models(models, X_test, y_test)
        
        # Plot feature importance for applicable models
        plot_feature_importance(models, feature_cols, problem_type='classification')
        
        # Select the best model
        best_model_name = select_best_model(metrics_df, problem_type='classification')
        best_model = models[best_model_name]
    else:  # regression
        # Train models without Linear Regression
        models = train_regression_models(X_train, y_train, preprocessor)
        
        # Evaluate models
        results, metrics_df = evaluate_regression_models(models, X_test, y_test)
        
        # Plot feature importance for applicable models
        plot_feature_importance(models, feature_cols, problem_type='regression')
        
        # Select the best model
        best_model_name = select_best_model(metrics_df, problem_type='regression')
        best_model = models[best_model_name]
        
        # If Gradient Boosting is selected as the best model, tune it
        if 'gradient_boosting' in best_model_name.lower():
            print(f"\n{best_model_name} is the best performing model. Tuning hyperparameters...")
            tuned_gb = tune_gradient_boosting(X_train, y_train, X_test, y_test, preprocessor, problem_type)
            
            # Save the tuned model
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            tuned_model_path = f"models/tuned_gradient_boosting_{problem_type}_{timestamp}.pkl"
            with open(tuned_model_path, 'wb') as f:
                pickle.dump(tuned_gb, f)
            print(f"Saved tuned Gradient Boosting model to {tuned_model_path}")
            
            # Use the tuned Gradient Boosting model
            best_model = tuned_gb
            best_model_name = "tuned_gradient_boosting"
            models[best_model_name] = best_model
        else:
            print(f"\n{best_model_name} is the best performing model. Using this for predictions.")
    
    # Save the trained models
    save_models(models, problem_type)
    
    print("Model training and evaluation completed successfully!")
    print(f"Trained and evaluated {len(models)} {problem_type} models")
    print(f"Best model: {best_model_name}")
    print(f"Evaluation results saved to results/{problem_type}_results.json")
    print(f"Visualizations saved to plots/models/ directory")
    
    # Save the feature lists for later reference
    feature_info = {
        'all_features': feature_cols,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features
    }
    
    with open('models/feature_info.pkl', 'wb') as f:
        pickle.dump(feature_info, f)
    
    print("Saved feature information to models/feature_info.pkl")
    
    # Analyze model behavior to understand model characteristics
    evaluate_model_behavior(best_model, X_test, y_test, feature_cols, problem_type)
    
    # Only use compatible models for Liguilla prediction
    # Check if model is compatible with predict_liguilla function
    is_compatible = True
    if not hasattr(best_model, 'predict_proba') and problem_type == 'classification':
        print(f"\nWarning: {best_model_name} does not support predict_proba, which is required for Liguilla prediction.")
        print("Using Gradient Boosting model for Liguilla prediction instead.")
        is_compatible = False
    
    # If best model is not compatible, use Gradient Boosting instead
    if not is_compatible and 'gradient_boosting' in models:
        prediction_model = models['gradient_boosting']
        print("Using Gradient Boosting model for Liguilla predictions")
    else:
        prediction_model = best_model
        print(f"Using {best_model_name} model for Liguilla predictions")
    
    # Predict Liguilla with the selected model
    predict_liguilla(prediction_model, df, feature_cols, problem_type)
    
    # Analyze the importance of player-based features
    player_cols = [col for col in feature_cols if any(term in col for term in 
                                                    ['player', 'efficiency', 'experience', 
                                                     'key_player', 'contributions'])]
    if player_cols:
        print("\nAnalyzing player feature importance...")
        analyze_player_feature_importance(models, X_test, y_test, feature_cols, problem_type)

if __name__ == "__main__":
    main()
