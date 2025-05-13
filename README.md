# ML Sports Prediction

## Overview

An advanced machine learning system for predicting match outcomes in Mexican soccer (Liga MX). This project combines data collection, feature engineering, and various machine learning algorithms to develop predictive models for football match results.

## Key Features

- **Comprehensive Data Collection:** Fetches team, player, and match data from the SportMonks API for Liga MX seasons
- **Advanced Feature Engineering:** Creates predictive features based on team performance, player statistics, and match context
- **Multiple ML Models:** Implements and compares various machine learning algorithms including Random Forests, Gradient Boosting, and Neural Networks
- **Chronological Evaluation:** Uses time-series-aware evaluation to ensure realistic model assessment
- **Liguilla (Playoff) Simulation:** Predicts outcomes for the Mexican league playoff system
- **Visualizations:** Generates insightful plots for feature importance and model performance

## Technologies Used

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Pandas](https://img.shields.io/badge/Pandas-Latest-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-blue)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Latest-blue)

- Python (pandas, numpy, scikit-learn)
- Statistical and Machine Learning algorithms
- Data visualization (matplotlib, seaborn)
- API integration (requests)

## Project Structure

```
ML_Sports_Prediction/
│
├── data_fetcher.py          # Collects team and player data from SportMonks API
├── feature_engineering.py   # Creates advanced features for prediction models
├── model_training_evaluation.py # Trains, evaluates and compares ML models
│
├── collected_data/          # Raw collected data from the API
├── processed_data/          # Engineered features ready for modeling
├── models/                  # Trained model files
├── results/                 # Prediction results and evaluations
└── plots/                   # Visualizations of features and model performance
```

## Getting Started

1. Clone this repository
2. Install required packages: `pip install -r requirements.txt`
3. Set up your SportMonks API key in a `.env` file
4. Run data collection: `python data_fetcher.py`
5. Generate features: `python feature_engineering.py`
6. Train and evaluate models: `python model_training_evaluation.py`

## Workflow

1. **Data Collection**: The `data_fetcher.py` script connects to the SportMonks API to collect comprehensive data on Liga MX teams, players, and matches.
2. **Feature Engineering**: The `feature_engineering.py` script processes raw data to create predictive features including:
   - Team performance metrics
   - Player statistics
   - Home/away advantage
   - Historical performance
   - Form indicators
3. **Model Training and Evaluation**: The `model_training_evaluation.py` script:
   - Trains multiple ML models
   - Evaluates performance using chronological validation
   - Generates visualizations of results
   - Simulates Liguilla outcomes

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Perfect for sports analytics enthusiasts, data scientists, and machine learning practitioners interested in applying ML to soccer predictions.
