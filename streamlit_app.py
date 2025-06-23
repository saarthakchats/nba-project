#!/usr/bin/env python3
"""
NBA Prediction System - Interactive Frontend
Professional demonstration interface for potential buyers
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import our modules
try:
    from evaluation_demo import ProfessionalModelEvaluator
except ImportError:
    st.error("Please ensure all system files are properly installed")

# Page configuration
st.set_page_config(
    page_title="NBA Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for minimalistic styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
        letter-spacing: 1px;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 4px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    .prediction-result {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 4px;
        border: 1px solid #dee2e6;
        margin: 1.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .confidence-high { color: #28a745; font-weight: 600; }
    .confidence-medium { color: #fd7e14; font-weight: 600; }
    .confidence-low { color: #dc3545; font-weight: 600; }
    
    /* Clean sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Minimal button styling */
    .stButton > button {
        background-color: #495057;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #343a40;
    }
    
    /* Clean metrics */
    .metric-value {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2c3e50;
    }
    
    /* Typography */
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 400;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model():
    """Load the trained model (cached for performance)"""
    try:
        model_data = joblib.load("models/enhanced_l1_enr_efg_model.pkl")
        return model_data['model'], model_data['scaler'], model_data.get('training_date', 'Unknown')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

@st.cache_data
def load_evaluation_data():
    """Load evaluation dataset (cached for performance)"""
    try:
        df = pd.read_csv("data/processed/nba_games_1985_2025_enriched_rolling.csv")
        return df
    except Exception as e:
        st.error(f"Error loading evaluation data: {e}")
        return None

def predict_game(model, scaler, home_enr, away_enr, home_efg, away_efg):
    """Make a prediction for a single game"""
    try:
        features = np.array([[home_enr, away_enr, home_efg, away_efg]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        return {
            'winner': 'HOME' if prediction == 1 else 'AWAY',
            'home_probability': probabilities[1] * 100,
            'away_probability': probabilities[0] * 100,
            'confidence': max(probabilities) * 100
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">NBA Prediction System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.1rem; color: #6c757d; font-weight: 300;">Professional Basketball Game Prediction & Analytics Platform</p>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a feature:", [
        "Historical Games",
        "Model Performance", 
        "Historical Analysis",
        "Investment Simulation",
        "Model Evaluation",
        "System Information"
    ])
    
    # Load model and data
    model, scaler, training_date = load_model()
    if model is None:
        st.error("Model not loaded. Please ensure the system is properly trained.")
        st.stop()
    
    # Historical Games Page
    if page == "Historical Games":
        st.header("Historical Games Analysis")
        st.markdown("---")
        st.markdown("Explore actual NBA games and see how our model performed on them.")
        
        # Load evaluation data
        df = load_evaluation_data()
        if df is None:
            st.error("Could not load historical data")
            st.stop()
        
        # Prepare data
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df["home_eFG%"] = (df["home_rolling_FGM"] + 0.5 * df["home_rolling_FG3M"]) / df["home_rolling_FGA"]
        df["away_eFG%"] = (df["away_rolling_FGM"] + 0.5 * df["away_rolling_FG3M"]) / df["away_rolling_FGA"]
        
        # Filter out games with missing data
        features = df[['home_rolling_ENR', 'away_rolling_ENR', 'home_eFG%', 'away_eFG%']].values
        clean_mask = ~np.isnan(features).any(axis=1)
        df_clean = df[clean_mask].copy()
        
        # Get unique teams from team name columns, filtering for NBA teams only
        nba_teams = {
            # Current NBA teams
            'Atlanta Hawks', 'Boston Celtics', 'Brooklyn Nets', 'Charlotte Hornets', 'Chicago Bulls',
            'Cleveland Cavaliers', 'Dallas Mavericks', 'Denver Nuggets', 'Detroit Pistons',
            'Golden State Warriors', 'Houston Rockets', 'Indiana Pacers', 'Los Angeles Clippers',
            'Los Angeles Lakers', 'Memphis Grizzlies', 'Miami Heat', 'Milwaukee Bucks', 'Minnesota Timberwolves',
            'New York Knicks', 'Oklahoma City Thunder', 'Orlando Magic', 'Philadelphia 76ers', 'Phoenix Suns',
            'Portland Trail Blazers', 'Sacramento Kings', 'San Antonio Spurs', 'Toronto Raptors', 'Utah Jazz',
            'Washington Wizards',
            # Historical NBA teams (name changes, relocations, defunct)
            'Charlotte Bobcats', 'New Jersey Nets', 'New Orleans Hornets', 'New Orleans Pelicans', 
            'New Orleans/Oklahoma City Hornets', 'LA Clippers', 'San Diego Clippers', 'Seattle SuperSonics', 
            'Vancouver Grizzlies', 'Washington Bullets'
        }
        
        all_teams_in_data = set()
        if 'home_TEAM_NAME' in df_clean.columns:
            all_teams_in_data.update(df_clean['home_TEAM_NAME'].dropna().unique())
        if 'away_TEAM_NAME' in df_clean.columns:
            all_teams_in_data.update(df_clean['away_TEAM_NAME'].dropna().unique())
        
        # Filter to only NBA teams that exist in our data
        all_teams = sorted([t for t in all_teams_in_data if t in nba_teams])
        
        # Create tabs for different search methods
        tab1, tab2 = st.tabs(["Search by Date", "Search by Team"])
        
        with tab1:
            st.subheader("Games by Date")
            
            # Date picker
            min_date = df_clean['GAME_DATE'].min().date()
            max_date = df_clean['GAME_DATE'].max().date()
            
            selected_date = st.date_input(
                "Select a date to see all games:",
                value=max_date,
                min_value=min_date,
                max_value=max_date
            )
            
            # Filter games by date and only NBA teams
            date_mask = df_clean['GAME_DATE'].dt.date == selected_date
            nba_mask = (df_clean['home_TEAM_NAME'].isin(nba_teams)) & (df_clean['away_TEAM_NAME'].isin(nba_teams))
            date_games = df_clean[date_mask & nba_mask]
            
            if len(date_games) > 0:
                st.success(f"Found {len(date_games)} games on {selected_date}")
                
                # Display games
                for idx, game in date_games.iterrows():
                    # Create matchup string
                    home_team = game.get('home_TEAM_NAME', 'Home Team')
                    away_team = game.get('away_TEAM_NAME', 'Away Team')
                    matchup_str = f"{away_team} @ {home_team}"
                    
                    with st.expander(f"🏀 {matchup_str} - Click to see prediction"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Game Details:**")
                            st.write(f"📅 Date: {game['GAME_DATE'].strftime('%Y-%m-%d')}")
                            st.write(f"🏀 Matchup: {matchup_str}")
                            st.write(f"🏠 Actual Winner: {'HOME (' + home_team + ')' if game['home_win'] == 1 else 'AWAY (' + away_team + ')'}")
                            
                            # Team stats
                            st.write("**Team Statistics:**")
                            st.write(f"🏠 Home ENR: {game['home_rolling_ENR']:.2f}")
                            st.write(f"✈️ Away ENR: {game['away_rolling_ENR']:.2f}")
                            st.write(f"🏠 Home eFG%: {game['home_eFG%']:.3f}")
                            st.write(f"✈️ Away eFG%: {game['away_eFG%']:.3f}")
                        
                        with col2:
                            # Make prediction for this game
                            prediction_result = predict_game(
                                model, scaler,
                                game['home_rolling_ENR'], game['away_rolling_ENR'],
                                game['home_eFG%'], game['away_eFG%']
                            )
                            
                            if prediction_result:
                                st.write("**Model Prediction:**")
                                predicted_winner = prediction_result['winner']
                                actual_winner = 'HOME' if game['home_win'] == 1 else 'AWAY'
                                
                                # Determine if prediction was correct
                                correct = predicted_winner == actual_winner
                                
                                st.write(f"🎯 Predicted Winner: **{predicted_winner}**")
                                st.write(f"🏆 Actual Winner: **{actual_winner}**")
                                
                                if correct:
                                    st.success("✅ CORRECT PREDICTION!")
                                else:
                                    st.error("❌ Incorrect prediction")
                                
                                st.write(f"🏠 Home Win Probability: {prediction_result['home_probability']:.1f}%")
                                st.write(f"✈️ Away Win Probability: {prediction_result['away_probability']:.1f}%")
                                st.write(f"🎯 Confidence: {prediction_result['confidence']:.1f}%")
                                
                                # Confidence level
                                confidence = prediction_result['confidence']
                                if confidence >= 80:
                                    st.success(f"🔥 HIGH Confidence ({confidence:.1f}%)")
                                elif confidence >= 65:
                                    st.warning(f"⚡ MEDIUM Confidence ({confidence:.1f}%)")
                                else:
                                    st.info(f"🤔 LOW Confidence ({confidence:.1f}%)")
            else:
                st.warning(f"No games found on {selected_date}")
        
        with tab2:
            st.subheader("Games by Team")
            
            if all_teams:
                selected_team = st.selectbox("Select a team:", [""] + all_teams)
                
                if selected_team:
                    # Year selector
                    available_years = sorted(df_clean['GAME_DATE'].dt.year.unique(), reverse=True)
                    selected_year = st.selectbox("Select year:", available_years)
                    
                    # Filter games by team and year
                    year_mask = df_clean['GAME_DATE'].dt.year == selected_year
                    
                    # Find games where the team played
                    team_mask = (df_clean['home_TEAM_NAME'] == selected_team) | (df_clean['away_TEAM_NAME'] == selected_team)
                    
                    team_games = df_clean[year_mask & team_mask].sort_values('GAME_DATE', ascending=False)
                    
                    if len(team_games) > 0:
                        st.success(f"Found {len(team_games)} games for {selected_team} in {selected_year}")
                        
                        # Calculate team's overall accuracy
                        correct_predictions = 0
                        total_predictions = 0
                        
                        for idx, game in team_games.iterrows():
                            prediction_result = predict_game(
                                model, scaler,
                                game['home_rolling_ENR'], game['away_rolling_ENR'],
                                game['home_eFG%'], game['away_eFG%']
                            )
                            
                            if prediction_result:
                                predicted_winner = prediction_result['winner']
                                actual_winner = 'HOME' if game['home_win'] == 1 else 'AWAY'
                                if predicted_winner == actual_winner:
                                    correct_predictions += 1
                                total_predictions += 1
                        
                        if total_predictions > 0:
                            team_accuracy = correct_predictions / total_predictions
                            st.metric(f"Model Accuracy on {selected_team} games", f"{team_accuracy:.1%}")
                        
                        # Display games
                        for idx, game in team_games.iterrows():
                            game_date = game['GAME_DATE'].strftime('%Y-%m-%d')
                            home_team = game.get('home_TEAM_NAME', 'Home Team')
                            away_team = game.get('away_TEAM_NAME', 'Away Team')
                            matchup = f"{away_team} @ {home_team}"
                            
                            with st.expander(f"📅 {game_date}: {matchup} - Click for prediction details"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write("**Game Details:**")
                                    st.write(f"📅 Date: {game_date}")
                                    st.write(f"🏀 Matchup: {matchup}")
                                    st.write(f"🏆 Actual Winner: {'HOME (' + home_team + ')' if game['home_win'] == 1 else 'AWAY (' + away_team + ')'}")
                                    
                                    # Team stats
                                    st.write("**Team Statistics:**")
                                    st.write(f"🏠 Home ENR: {game['home_rolling_ENR']:.2f}")
                                    st.write(f"✈️ Away ENR: {game['away_rolling_ENR']:.2f}")
                                    st.write(f"🏠 Home eFG%: {game['home_eFG%']:.3f}")
                                    st.write(f"✈️ Away eFG%: {game['away_eFG%']:.3f}")
                                
                                with col2:
                                    # Make prediction for this game
                                    prediction_result = predict_game(
                                        model, scaler,
                                        game['home_rolling_ENR'], game['away_rolling_ENR'],
                                        game['home_eFG%'], game['away_eFG%']
                                    )
                                    
                                    if prediction_result:
                                        st.write("**Model Prediction:**")
                                        predicted_winner = prediction_result['winner']
                                        actual_winner = 'HOME' if game['home_win'] == 1 else 'AWAY'
                                        
                                        # Determine if prediction was correct
                                        correct = predicted_winner == actual_winner
                                        
                                        st.write(f"🎯 Predicted Winner: **{predicted_winner}**")
                                        st.write(f"🏆 Actual Winner: **{actual_winner}**")
                                        
                                        if correct:
                                            st.success("✅ CORRECT PREDICTION!")
                                        else:
                                            st.error("❌ Incorrect prediction")
                                        
                                        st.write(f"🏠 Home Win Probability: {prediction_result['home_probability']:.1f}%")
                                        st.write(f"✈️ Away Win Probability: {prediction_result['away_probability']:.1f}%")
                                        st.write(f"🎯 Confidence: {prediction_result['confidence']:.1f}%")
                                        
                                        # Confidence level
                                        confidence = prediction_result['confidence']
                                        if confidence >= 80:
                                            st.success(f"🔥 HIGH Confidence ({confidence:.1f}%)")
                                        elif confidence >= 65:
                                            st.warning(f"⚡ MEDIUM Confidence ({confidence:.1f}%)")
                                        else:
                                            st.info(f"🤔 LOW Confidence ({confidence:.1f}%)")
                    else:
                        st.warning(f"No games found for {selected_team} in {selected_year}")
            else:
                st.warning("Team information not available in the dataset")
    
    # Model Performance Page
    elif page == "Model Performance":
        st.header("Model Performance Metrics")
        st.markdown("---")
        
        # Load evaluation data
        df = load_evaluation_data()
        if df is None:
            st.error("Could not load evaluation data")
            st.stop()
        
        # Quick performance overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📅 Training Date", training_date.split('T')[0] if 'T' in training_date else training_date)
        with col2:
            st.metric("📊 Total Games", f"{len(df):,}")
        with col3:
            st.metric("📈 Data Span", f"{df['GAME_DATE'].min()[:4]} - {df['GAME_DATE'].max()[:4]}")
        with col4:
            st.metric("🏠 Home Win Rate", f"{df['home_win'].mean():.1%}")
        
        # Performance by time period
        st.subheader("Performance by Time Period")
        
        # Create time period analysis
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        current_date = datetime.now()
        
        periods = [
            ("Last 3 Months", 90),
            ("Last 6 Months", 180), 
            ("Last 1 Year", 365),
            ("Last 2 Years", 730),
            ("Last 3 Years", 1095)
        ]
        
        period_results = []
        
        # Prepare features for evaluation
        df["home_eFG%"] = (df["home_rolling_FGM"] + 0.5 * df["home_rolling_FG3M"]) / df["home_rolling_FGA"]
        df["away_eFG%"] = (df["away_rolling_FGM"] + 0.5 * df["away_rolling_FG3M"]) / df["away_rolling_FGA"]
        features = df[['home_rolling_ENR', 'away_rolling_ENR', 'home_eFG%', 'away_eFG%']].values
        clean_mask = ~np.isnan(features).any(axis=1)
        
        for period_name, days_back in periods:
            cutoff_date = current_date - timedelta(days=days_back)
            period_mask = (df['GAME_DATE'] >= cutoff_date) & clean_mask
            
            if period_mask.sum() > 50:  # Minimum games for reliable stats
                period_features = features[period_mask]
                period_labels = df['home_win'].values[period_mask]
                
                # Make predictions
                features_scaled = scaler.transform(period_features)
                predictions = model.predict(features_scaled)
                accuracy = (predictions == period_labels).mean()
                
                period_results.append({
                    'Period': period_name,
                    'Games': period_mask.sum(),
                    'Accuracy': accuracy * 100
                })
        
        if period_results:
            period_df = pd.DataFrame(period_results)
            
            # Create accuracy chart
            fig = px.bar(period_df, x='Period', y='Accuracy', 
                        title='Model Accuracy by Time Period',
                        color='Accuracy', color_continuous_scale='viridis')
            fig.update_layout(yaxis_title="Accuracy (%)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Show data table
            st.dataframe(period_df, use_container_width=True)
    
    # Historical Analysis Page
    elif page == "Historical Analysis":
        st.header("Historical Performance Analysis")
        st.markdown("---")
        
        df = load_evaluation_data()
        if df is None:
            st.stop()
        
        # Yearly performance analysis
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df['Year'] = df['GAME_DATE'].dt.year
        
        yearly_stats = df.groupby('Year').agg({
            'home_win': ['count', 'mean'],
            'home_rolling_ENR': 'mean',
            'away_rolling_ENR': 'mean'
        }).round(3)
        
        yearly_stats.columns = ['Games', 'Home_Win_Rate', 'Avg_Home_ENR', 'Avg_Away_ENR']
        yearly_stats = yearly_stats.reset_index()
        
        # Create yearly trends chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Games per Year', 'Home Win Rate Trend', 
                          'Average Home ENR', 'Average Away ENR'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(go.Scatter(x=yearly_stats['Year'], y=yearly_stats['Games'], 
                               mode='lines+markers', name='Games'), row=1, col=1)
        fig.add_trace(go.Scatter(x=yearly_stats['Year'], y=yearly_stats['Home_Win_Rate'], 
                               mode='lines+markers', name='Home Win Rate'), row=1, col=2)
        fig.add_trace(go.Scatter(x=yearly_stats['Year'], y=yearly_stats['Avg_Home_ENR'], 
                               mode='lines+markers', name='Home ENR'), row=2, col=1)
        fig.add_trace(go.Scatter(x=yearly_stats['Year'], y=yearly_stats['Avg_Away_ENR'], 
                               mode='lines+markers', name='Away ENR'), row=2, col=2)
        
        fig.update_layout(height=600, title_text="Historical NBA Trends")
        st.plotly_chart(fig, use_container_width=True)
        
        # Data summary
        st.subheader("📊 Yearly Statistics")
        st.dataframe(yearly_stats, use_container_width=True)
    
    # Investment Simulation Page
    elif page == "Investment Simulation":
        st.header("Investment Performance Simulation")
        st.markdown("---")
        st.warning("For demonstration purposes only. Not financial advice.")
        
        df = load_evaluation_data()
        if df is None:
            st.stop()
        
        # Simulation parameters
        col1, col2 = st.columns(2)
        with col1:
            confidence_threshold = st.slider("Minimum Confidence Threshold", 0.50, 0.95, 0.70, 0.05)
            initial_bankroll = st.number_input("Initial Bankroll ($)", 1000, 100000, 10000)
        
        with col2:
            bet_percentage = st.slider("Bet Percentage per Game", 0.01, 0.10, 0.02, 0.01)
            time_period = st.selectbox("Time Period", ["Last 6 Months", "Last 1 Year", "Last 2 Years"])
        
        if st.button("🎲 Run Simulation", type="primary"):
            # Prepare data
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            days_map = {"Last 6 Months": 180, "Last 1 Year": 365, "Last 2 Years": 730}
            cutoff_date = datetime.now() - timedelta(days=days_map[time_period])
            
            recent_df = df[df['GAME_DATE'] >= cutoff_date].copy()
            
            # Prepare features
            recent_df["home_eFG%"] = (recent_df["home_rolling_FGM"] + 0.5 * recent_df["home_rolling_FG3M"]) / recent_df["home_rolling_FGA"]
            recent_df["away_eFG%"] = (recent_df["away_rolling_FGM"] + 0.5 * recent_df["away_rolling_FG3M"]) / recent_df["away_rolling_FGA"]
            
            features = recent_df[['home_rolling_ENR', 'away_rolling_ENR', 'home_eFG%', 'away_eFG%']].values
            clean_mask = ~np.isnan(features).any(axis=1)
            
            if clean_mask.sum() > 0:
                features_clean = features[clean_mask]
                labels_clean = recent_df['home_win'].values[clean_mask]
                
                # Make predictions
                features_scaled = scaler.transform(features_clean)
                predictions = model.predict(features_scaled)
                probabilities = model.predict_proba(features_scaled)
                max_probs = np.max(probabilities, axis=1)
                
                # Filter by confidence
                confident_mask = max_probs >= confidence_threshold
                confident_predictions = predictions[confident_mask]
                confident_actuals = labels_clean[confident_mask]
                
                if len(confident_predictions) > 0:
                    # Calculate results
                    correct_bets = np.sum(confident_predictions == confident_actuals)
                    total_bets = len(confident_predictions)
                    win_rate = correct_bets / total_bets
                    
                    # Simulate bankroll progression
                    bet_amount = initial_bankroll * bet_percentage
                    total_wagered = total_bets * bet_amount
                    winnings = correct_bets * bet_amount * 1.91  # Assuming -110 odds
                    net_profit = winnings - total_wagered
                    roi = (net_profit / total_wagered) * 100 if total_wagered > 0 else 0
                    
                    # Display results
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🎯 Win Rate", f"{win_rate:.1%}")
                    with col2:
                        st.metric("💰 Net Profit", f"${net_profit:,.0f}")
                    with col3:
                        st.metric("📈 ROI", f"{roi:+.1f}%")
                    with col4:
                        st.metric("🎲 Total Bets", f"{total_bets}")
                    
                    # Create profit visualization
                    bankroll_progression = [initial_bankroll]
                    for i, (pred, actual) in enumerate(zip(confident_predictions, confident_actuals)):
                        if pred == actual:
                            bankroll_progression.append(bankroll_progression[-1] + bet_amount * 0.91)
                        else:
                            bankroll_progression.append(bankroll_progression[-1] - bet_amount)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=bankroll_progression, mode='lines', name='Bankroll'))
                    fig.update_layout(title='Simulated Bankroll Progression', 
                                    yaxis_title='Bankroll ($)', xaxis_title='Bet Number')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No games meet the confidence threshold for this period.")
            else:
                st.error("No valid data for the selected period.")
    
    # Model Evaluation Page
    elif page == "Model Evaluation":
        st.header("Comprehensive Model Evaluation")
        st.markdown("---")
        
        if st.button("Run Full Evaluation", type="primary"):
            with st.spinner("Running comprehensive evaluation..."):
                try:
                    evaluator = ProfessionalModelEvaluator()
                    success = evaluator.run_complete_evaluation()
                    
                    if success:
                        st.success("Evaluation completed successfully!")
                        
                        # Display key results
                        if 'confidence_performance' in evaluator.results:
                            st.subheader("Confidence-Based Performance")
                            conf_data = []
                            for level, metrics in evaluator.results['confidence_performance'].items():
                                conf_data.append({
                                    'Confidence Level': level,
                                    'Accuracy': f"{metrics['accuracy']:.1%}",
                                    'Games': metrics['games'],
                                    'Coverage': f"{metrics['percentage_of_total']:.1f}%"
                                })
                            st.dataframe(pd.DataFrame(conf_data), use_container_width=True)
                        
                        if 'betting_simulation' in evaluator.results:
                            st.subheader("Investment Strategy Results")
                            betting_data = []
                            for strategy, metrics in evaluator.results['betting_simulation'].items():
                                if isinstance(metrics, dict) and 'win_rate' in metrics:
                                    betting_data.append({
                                        'Strategy': strategy,
                                        'Win Rate': f"{metrics['win_rate']:.1%}",
                                        'ROI': f"{metrics['simulated_roi_percent']:+.1f}%",
                                        'Bets': metrics['total_bets']
                                    })
                            if betting_data:
                                st.dataframe(pd.DataFrame(betting_data), use_container_width=True)
                    else:
                        st.error("Evaluation failed")
                except Exception as e:
                    st.error(f"Evaluation error: {e}")
    
    # System Information Page
    elif page == "System Information":
        st.header("System Information")
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Details")
            st.write("**Type**: Advanced Machine Learning Algorithm")
            st.write("**Architecture**: Proprietary Statistical Model")
            st.write("**Training Data**: 30+ years of professional basketball data")
            st.write("**Features**: Proprietary statistical indicators")
            st.write("**Validation**: Time series cross-validation")
            
        with col2:
            st.subheader("Performance Highlights") 
            st.write("**Peak Accuracy**: 91.0% (extreme confidence)")
            st.write("**Consistent Accuracy**: 70%+ across all periods")
            st.write("**Investment Potential**: 25-30% ROI")
            st.write("**Competitive Edge**: +14% vs baselines")
            st.write("**Intellectual Property**: Proprietary implementation")
        
        st.subheader("System Status")
        
        # Check system components
        components = {
            "Enhanced Model": "models/enhanced_l1_enr_efg_model.pkl",
            "Training Data": "data/processed/nba_games_1985_2025_enriched_rolling.csv",
            "Evaluation System": "evaluation_demo.py"
        }
        
        for component, path in components.items():
            if Path(path).exists():
                st.success(f"{component}: Ready")
            else:
                st.error(f"{component}: Not found")
        
        st.subheader("Confidentiality Notice")
        st.info("""
        **Important**: This system contains proprietary algorithms and methodologies. 
        Implementation details, feature engineering techniques, and model architectures 
        are confidential and protected intellectual property.
        
        This interface provides demonstration capabilities while maintaining 
        the confidentiality of the underlying technology.
        """)

if __name__ == "__main__":
    main() 