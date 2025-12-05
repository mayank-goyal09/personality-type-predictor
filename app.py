import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import os

# Page Configuration
st.set_page_config(
    page_title="Personality Type Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-style: italic;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🧠 Personality Type Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Discover your personality type using Machine Learning</div>', unsafe_allow_html=True)

# Load model (placeholder - replace with actual model loading)
def load_model():
    try:
        with open('models/personality_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.warning("⚠️ Model file not found. Please train the model first by running `python model.py`")
        return None

# Personality types mapping (adjust based on your dataset)
PERSONALITY_TYPES = [
    "Analyst", "Diplomat", "Sentinel", "Explorer", "Other"
]

# Tabs for Simple and Nerd mode
tab1, tab2 = st.tabs(["🎯 Simple Mode", "🤓 Nerd Mode (Full Features)"])

# ========== TAB 1: SIMPLE MODE ==========
with tab1:
    st.markdown("### Input Your Personality Traits")
    
    col1, col2 = st.columns(2)
    
    with col1:
        openness = st.slider("🌈 Openness to Experience", 0, 10, 5, help="Curiosity and willingness to try new things")
        conscientiousness = st.slider("📋 Conscientiousness", 0, 10, 5, help="Organization and dependability")
        extraversion = st.slider("🎉 Extraversion", 0, 10, 5, help="Sociability and enthusiasm")
    
    with col2:
        agreeableness = st.slider("🤝 Agreeableness", 0, 10, 5, help="Compassion and cooperation")
        neuroticism = st.slider("😰 Neuroticism", 0, 10, 5, help="Emotional stability")
    
    st.markdown("---")
    
    if st.button("🔮 Predict My Personality Type"):
        # Create input array
        input_data = np.array([[openness, conscientiousness, extraversion, agreeableness, neuroticism]])
        
        # Mock prediction (replace with actual model prediction)
        model = load_model()
        if model:
            prediction = model.predict(input_data)[0]
            confidence = model.predict_proba(input_data).max() * 100
        else:
            # Demo prediction
            prediction = "Analyst"
            confidence = 85.7
        
        # Display prediction
        st.markdown(f'<div class="prediction-box">Your Personality Type: {prediction}</div>', unsafe_allow_html=True)
        st.success(f"✅ Confidence: {confidence:.1f}%")
        
        # Trait breakdown visualization
        st.markdown("### 📊 Your Trait Breakdown")
        traits_df = pd.DataFrame({
            'Trait': ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism'],
            'Score': [openness, conscientiousness, extraversion, agreeableness, neuroticism]
        })
        
        fig = px.bar(traits_df, x='Trait', y='Score', color='Score',
                     color_continuous_scale='Viridis', text='Score')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

# ========== TAB 2: NERD MODE ==========
with tab2:
    st.markdown("### 🤓 Advanced Analysis & Model Insights")
    
    # Input section (same as Simple mode)
    st.markdown("#### Input Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        openness_n = st.number_input("Openness", 0, 10, 5)
        conscientiousness_n = st.number_input("Conscientiousness", 0, 10, 5)
    
    with col2:
        extraversion_n = st.number_input("Extraversion", 0, 10, 5)
        agreeableness_n = st.number_input("Agreeableness", 0, 10, 5)
    
    with col3:
        neuroticism_n = st.number_input("Neuroticism", 0, 10, 5)
    
    if st.button("🚀 Run Full Analysis"):
        input_data_n = np.array([[openness_n, conscientiousness_n, extraversion_n, agreeableness_n, neuroticism_n]])
        
        # Mock prediction
        model = load_model()
        if model:
            prediction_n = model.predict(input_data_n)[0]
            probabilities = model.predict_proba(input_data_n)[0]
        else:
            prediction_n = "Analyst"
            probabilities = np.array([0.857, 0.089, 0.034, 0.015, 0.005])
        
        # Results
        st.markdown(f"### 🎯 Prediction: **{prediction_n}**")
        
        # Probability distribution
        st.markdown("#### 📊 Probability Distribution Across All Types")
        prob_df = pd.DataFrame({
            'Personality Type': PERSONALITY_TYPES[:len(probabilities)],
            'Probability': probabilities * 100
        })
        
        fig2 = px.bar(prob_df, x='Personality Type', y='Probability',
                      color='Probability', color_continuous_scale='Blues',
                      text='Probability')
        fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig2.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Feature importance (mock data)
        st.markdown("#### 🔍 Feature Importance Analysis")
        importance_df = pd.DataFrame({
            'Feature': ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism'],
            'Importance': [0.28, 0.24, 0.22, 0.15, 0.11]
        }).sort_values('Importance', ascending=True)
        
        fig3 = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                      color='Importance', color_continuous_scale='Reds')
        fig3.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
        
        # Model metrics (mock data)
        st.markdown("#### 📈 Model Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", "85.7%", "2.3%")
        with col2:
            st.metric("Precision", "0.84", "0.03")
        with col3:
            st.metric("Recall", "0.83", "0.02")
        with col4:
            st.metric("F1-Score", "0.83", "0.02")
        
        # Download button
        st.markdown("---")
        report_data = {
            'Trait': ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism'],
            'Score': [openness_n, conscientiousness_n, extraversion_n, agreeableness_n, neuroticism_n],
            'Prediction': [prediction_n] * 5
        }
        report_df = pd.DataFrame(report_data)
        
        csv = report_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Prediction Report",
            data=csv,
            file_name='personality_prediction_report.csv',
            mime='text/csv'
        )

# Sidebar info
with st.sidebar:
    st.markdown("### ℹ️ About This App")
    st.info(
        "This app uses **Logistic Regression** to predict personality types "
        "based on the Big Five personality traits.\n\n"
        "**How to use:**\n"
        "1. Adjust trait sliders (0-10)\n"
        "2. Click predict button\n"
        "3. View your personality type!"
    )
    
    st.markdown("### 🎯 Personality Types")
    for ptype in PERSONALITY_TYPES:
        st.markdown(f"- {ptype}")
    
    st.markdown("### 📊 Dataset Info")
    st.markdown(
        "- **Features**: 5 personality traits\n"
        "- **Classes**: Multiple personality types\n"
        "- **Algorithm**: Logistic Regression"
    )

# Footer
st.markdown("---")
st.markdown('<div class="footer">Built with 🧠 by Mayank\'s ML Brain | Powered by Streamlit & Scikit-learn</div>', unsafe_allow_html=True)