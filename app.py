"""
EPL Match Predictor - Streamlit Web App
Interactive machine learning predictions for Premier League matches
"""

import streamlit as st
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="EPL Match Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Cache model loading
@st.cache_resource
def load_model():
    """Load trained model and scaler"""
    try:
        with open('models/model_phase1_logistic_regression.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/scaler_phase1.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        st.stop()

@st.cache_data
def load_data():
    """Load historical match data"""
    try:
        df = pd.read_csv('data/processed/epl_all_data_with_features.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Ensure points columns exist
        if 'HomePoints' not in df.columns:
            df['HomePoints'] = df['FTR'].map({'H': 3, 'D': 1, 'A': 0})
            df['AwayPoints'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 3})
        
        return df
    except FileNotFoundError as e:
        st.error(f"❌ Data file not found: {e}")
        st.stop()

def get_team_form(df, team, is_home=True):
    """Get team's recent form"""
    if is_home:
        matches = df[df['HomeTeam'] == team].tail(5)
        if len(matches) == 0:
            return None
        return {
            'Form5': matches['HomePoints'].sum(),
            'GoalsFor5': matches['FTHG'].mean(),
            'GoalsAgainst5': matches['FTAG'].mean(),
            'Matches': len(matches)
        }
    else:
        matches = df[df['AwayTeam'] == team].tail(5)
        if len(matches) == 0:
            return None
        return {
            'Form5': matches['AwayPoints'].sum(),
            'GoalsFor5': matches['FTAG'].mean(),
            'GoalsAgainst5': matches['FTHG'].mean(),
            'Matches': len(matches)
        }

def predict_match(model, scaler, df, home_team, away_team):
    """Predict match outcome"""
    home_form = get_team_form(df, home_team, is_home=True)
    away_form = get_team_form(df, away_team, is_home=False)
    
    if home_form is None or away_form is None:
        return None, None
    
    # Create features
    features = {
        'HomeForm5': home_form['Form5'],
        'AwayForm5': away_form['Form5'],
        'HomeGoalsFor5': home_form['GoalsFor5'],
        'HomeGoalsAgainst5': home_form['GoalsAgainst5'],
        'AwayGoalsFor5': away_form['GoalsFor5'],
        'AwayGoalsAgainst5': away_form['GoalsAgainst5'],
    }
    features['FormDiff'] = features['HomeForm5'] - features['AwayForm5']
    features['GoalDiffFor'] = features['HomeGoalsFor5'] - features['AwayGoalsFor5']
    features['GoalDiffAgainst'] = features['AwayGoalsAgainst5'] - features['HomeGoalsAgainst5']
    
    # Predict
    feature_cols = [
        'HomeForm5', 'AwayForm5', 'HomeGoalsFor5', 'HomeGoalsAgainst5',
        'AwayGoalsFor5', 'AwayGoalsAgainst5', 'FormDiff', 'GoalDiffFor', 'GoalDiffAgainst'
    ]
    X = pd.DataFrame([features])[feature_cols]
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[0]
    
    # Map to outcomes
    classes = model.classes_
    result = {}
    for i, cls in enumerate(classes):
        if cls == 'H':
            result['Home Win'] = proba[i]
        elif cls == 'D':
            result['Draw'] = proba[i]
        elif cls == 'A':
            result['Away Win'] = proba[i]
    
    # Add form data
    features['HomeFormData'] = home_form
    features['AwayFormData'] = away_form
    
    return result, features

def create_probability_chart(probabilities, home_team, away_team):
    """Create a visual chart of probabilities"""
    fig = go.Figure(data=[
        go.Bar(
            x=['Home Win', 'Draw', 'Away Win'],
            y=[probabilities['Home Win'], probabilities['Draw'], probabilities['Away Win']],
            text=[f"{probabilities['Home Win']:.1%}", f"{probabilities['Draw']:.1%}", f"{probabilities['Away Win']:.1%}"],
            textposition='auto',
            marker_color=['#4CAF50', '#FFC107', '#F44336']
        )
    ])
    
    fig.update_layout(
        title=f"Win Probabilities: {home_team} vs {away_team}",
        yaxis_title="Probability",
        yaxis_tickformat='.0%',
        height=400,
        showlegend=False
    )
    
    return fig

# ===== MAIN APP =====

# Load model and data
model, scaler = load_model()
df = load_data()

# Get list of teams
teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())

# Header
st.markdown('<p class="main-header">⚽ Premier League Match Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Machine Learning model trained on 1,800+ EPL matches | Accuracy: 52.5% | Brier Score: 0.197</p>',
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About This Model")
    
    st.markdown("""
    ### How It Works
    This predictor uses **Logistic Regression** trained on:
    - **5 seasons** of EPL data (2019-2024)
    - **Team form** (last 5 home/away matches)
    - **Goals scored/conceded** patterns
    - **Home/away** performance splits
    
    ### Model Performance
    - **Accuracy:** 52.5%
    - **Brier Score:** 0.197
    - **Beats bookmaker baseline**
    
    ### Training Data
    - **Training:** 1,466 matches
    - **Testing:** 373 matches
    - **Date Range:** 2019-2024
    
    ### Features Used
    - Recent form (points from last 5)
    - Goals scored per match
    - Goals conceded per match
    - Home/away splits
    - Form differentials
    """)
    
    st.markdown("---")
    st.markdown("""
    **Created by:** Brendan  
    [GitHub](#) | [LinkedIn](#)
    
    *Last updated: Feb 2026*
    """)

# Main content
st.header("🔮 Predict a Match")

col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox(
        "🏠 Home Team",
        teams,
        index=teams.index('Arsenal') if 'Arsenal' in teams else 0,
        key='home'
    )

with col2:
    away_team = st.selectbox(
        "✈️ Away Team",
        teams,
        index=teams.index('Tottenham') if 'Tottenham' in teams else 1,
        key='away'
    )

# Predict button
if st.button("⚽ Make Prediction", type="primary", use_container_width=True):
    if home_team == away_team:
        st.error("❌ Please select different teams")
    else:
        with st.spinner("🔄 Analyzing team form..."):
            proba, features = predict_match(model, scaler, df, home_team, away_team)
            
            if proba is None:
                st.error("❌ Not enough historical data for one or both teams")
            else:
                # Success message
                st.success("✅ Prediction Complete!")
                
                st.markdown("---")
                
                # Main prediction
                prediction = max(proba, key=proba.get)
                confidence = proba[prediction]
                
                if confidence > 0.6:
                    confidence_emoji = "🔥"
                    confidence_text = "High Confidence"
                elif confidence > 0.45:
                    confidence_emoji = "⚖️"
                    confidence_text = "Moderate Confidence"
                else:
                    confidence_emoji = "⚠️"
                    confidence_text = "Low Confidence"
                
                # Prediction banner
                st.markdown(f"""
                <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            border-radius: 1rem; color: white; margin: 1rem 0;'>
                    <h2 style='margin: 0; font-size: 2.5rem;'>{confidence_emoji} {prediction}</h2>
                    <p style='margin: 0.5rem 0; font-size: 1.5rem;'>{confidence:.1%} Probability</p>
                    <p style='margin: 0; font-size: 1rem; opacity: 0.9;'>{confidence_text}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability metrics
                st.subheader("📊 Win Probabilities")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        f"🏠 {home_team} Win",
                        f"{proba['Home Win']:.1%}",
                        delta=None,
                        delta_color="off"
                    )
                
                with col2:
                    st.metric(
                        "🤝 Draw",
                        f"{proba['Draw']:.1%}",
                        delta=None,
                        delta_color="off"
                    )
                
                with col3:
                    st.metric(
                        f"✈️ {away_team} Win",
                        f"{proba['Away Win']:.1%}",
                        delta=None,
                        delta_color="off"
                    )
                
                # Visual chart
                st.plotly_chart(
                    create_probability_chart(proba, home_team, away_team),
                    use_container_width=True
                )
                
                # Recent form
                st.markdown("---")
                st.subheader("📈 Recent Form Analysis")
                st.markdown("*Based on last 5 home/away matches*")
                
                col1, col2 = st.columns(2)
                
                home_form_data = features['HomeFormData']
                away_form_data = features['AwayFormData']
                
                with col1:
                    st.markdown(f"### 🏠 {home_team} (Home)")
                    
                    # Points
                    points_pct = (home_form_data['Form5'] / 15) * 100
                    st.metric("Points", f"{home_form_data['Form5']}/15", f"{points_pct:.0f}%")
                    
                    # Goals
                    col1a, col1b = st.columns(2)
                    with col1a:
                        st.metric("Goals/Match", f"{home_form_data['GoalsFor5']:.2f}")
                    with col1b:
                        st.metric("Conceded/Match", f"{home_form_data['GoalsAgainst5']:.2f}")
                
                with col2:
                    st.markdown(f"### ✈️ {away_team} (Away)")
                    
                    # Points
                    points_pct = (away_form_data['Form5'] / 15) * 100
                    st.metric("Points", f"{away_form_data['Form5']}/15", f"{points_pct:.0f}%")
                    
                    # Goals
                    col2a, col2b = st.columns(2)
                    with col2a:
                        st.metric("Goals/Match", f"{away_form_data['GoalsFor5']:.2f}")
                    with col2b:
                        st.metric("Conceded/Match", f"{away_form_data['GoalsAgainst5']:.2f}")
                
                # Form differential
                form_diff = features['FormDiff']
                st.markdown("---")
                
                if form_diff > 3:
                    form_analysis = f"**{home_team}** in significantly better form (+{form_diff:.0f} points)"
                elif form_diff < -3:
                    form_analysis = f"**{away_team}** in significantly better form ({form_diff:.0f} points)"
                else:
                    form_analysis = f"Teams in **similar form** ({form_diff:+.0f} points difference)"
                
                st.info(f"📊 **Form Analysis:** {form_analysis}")
                
                # Fair odds
                st.markdown("---")
                st.subheader("💰 Fair Betting Odds")
                st.markdown("*Model-implied fair odds (no bookmaker margin)*")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if proba['Home Win'] > 0:
                        odds = 1 / proba['Home Win']
                        st.metric(f"{home_team} Win", f"{odds:.2f}")
                
                with col2:
                    if proba['Draw'] > 0:
                        odds = 1 / proba['Draw']
                        st.metric("Draw", f"{odds:.2f}")
                
                with col3:
                    if proba['Away Win'] > 0:
                        odds = 1 / proba['Away Win']
                        st.metric(f"{away_team} Win", f"{odds:.2f}")

# Footer
st.markdown("---")
st.markdown("### 📈 Model Performance Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Test Accuracy", "52.5%", help="Percentage of correct predictions on test set")

with col2:
    st.metric("Brier Score", "0.197", help="Lower is better. Professional bookmakers: ~0.21")

with col3:
    st.metric("Training Matches", "1,466", help="Number of matches used for training")

with col4:
    st.metric("Test Matches", "373", help="Number of matches used for testing")

# Disclaimer
st.markdown("---")
st.info("""
**⚠️ Disclaimer:** This model is for educational and entertainment purposes only. 
Football matches are inherently unpredictable. This model achieves 52.5% accuracy, 
which is near the theoretical ceiling due to the random nature of football. 
Do not use for actual betting decisions.
""")
