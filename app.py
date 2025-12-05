import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

# ------------------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Deep Personality Analysis",
    page_icon="🧠",
    layout="centered",
)

# ------------------------------------------------------------------------------
# PREMIUM DARK BLUE + ORANGE THEME
# ------------------------------------------------------------------------------
st.markdown(
    """
<style>
/* Main app background - Dark blue to black gradient */
.stApp {
    background: linear-gradient(135deg, #0A1128 0%, #001F54 30%, #000000 100%);
    color: #E8EAF6;
}

/* Glass morphism cards */
.glass-card {
    background: rgba(10, 17, 40, 0.85);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 152, 0, 0.3);
    border-radius: 20px;
    padding: 1.8rem;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.7);
    margin-bottom: 1.5rem;
}

/* Orange accent text */
.orange-glow {
    color: #FF9800;
    font-weight: 700;
    text-shadow: 0 0 20px rgba(255, 152, 0, 0.5);
}

/* Hero title styling */
.hero-title {
    font-size: 3rem;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg, #FF9800, #FFB74D, #FF6F00);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    letter-spacing: 0.02em;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #000814 0%, #001D3D 100%);
    border-right: 1px solid rgba(255, 152, 0, 0.2);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #FF6F00, #FF9800);
    color: #000000 !important;
    border-radius: 50px;
    border: none;
    font-weight: 800;
    padding: 0.7rem 2rem;
    font-size: 1.1rem;
    box-shadow: 0 8px 25px rgba(255, 152, 0, 0.4);
    transition: all 0.25s ease;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.stButton > button:hover {
    transform: translateY(-3px) scale(1.03);
    box-shadow: 0 12px 35px rgba(255, 152, 0, 0.6);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 1rem;
    background: rgba(0, 31, 84, 0.5);
    border-radius: 15px;
    padding: 0.5rem;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(255, 152, 0, 0.1);
    border-radius: 12px;
    color: #FFB74D;
    font-weight: 600;
    padding: 0.8rem 1.5rem;
    border: 1px solid rgba(255, 152, 0, 0.3);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #FF6F00, #FF9800);
    color: #000000;
    border: none;
}

/* Progress bars - Custom orange gradient */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #FF6F00, #FFB74D, #FF9800);
}

/* Sliders */
.stSlider > div > div > div > div {
    background: rgba(255, 152, 0, 0.2);
}

/* Result cards */
.result-card {
    background: linear-gradient(135deg, rgba(255, 111, 0, 0.15), rgba(0, 31, 84, 0.3));
    border: 2px solid rgba(255, 152, 0, 0.4);
    border-radius: 18px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
}

/* Metric styling */
[data-testid="stMetric"] {
    background: rgba(10, 17, 40, 0.9);
    border-radius: 16px;
    padding: 1rem;
    border: 1px solid rgba(255, 152, 0, 0.35);
}

/* Info boxes */
.stAlert {
    background: rgba(0, 31, 84, 0.6);
    border-left: 4px solid #FF9800;
    border-radius: 12px;
}
</style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        with open("personality_model.pkl", "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        return None


artifacts = load_model()

# Session state
if "page" not in st.session_state:
    st.session_state.page = "quiz"


def reset_quiz():
    st.session_state.page = "quiz"


# ------------------------------------------------------------------------------
# APP LOGIC
# ------------------------------------------------------------------------------
if artifacts:
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    feature_names = artifacts["features"]

    # ------------------ HERO HEADER ------------------ #
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 class="hero-title">🧠 DEEP PERSONALITY ANALYSIS</h1>
            <p style="font-size: 1.15rem; color: #B0BEC5;">
                Advanced AI-powered psychological profiling in <span class="orange-glow">15 questions</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------ QUIZ PAGE ------------------ #
    if st.session_state.page == "quiz":

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(
            """
            ### 📋 Assessment Instructions
            Rate each statement from **1 (Strongly Disagree)** to **5 (Strongly Agree)**.  
            Be honest — this AI model analyzes subtle patterns in your responses.
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        with st.form("deep_dive_form"):

            # TABBED CATEGORIES
            tab1, tab2, tab3 = st.tabs(
                ["💃 Social Dynamics", "🎭 Emotional Style", "🤝 Interaction Habits"]
            )

            # CATEGORY 1: SOCIAL DYNAMICS
            with tab1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.info("**How do you handle groups and energy?**")

                q1 = st.slider(
                    "1️⃣ I am the life of the party.",
                    1, 5, 3,
                    help="Do you naturally become the center of attention?",
                )
                q2 = st.slider(
                    "2️⃣ I feel comfortable around people.",
                    1, 5, 3,
                    help="How at ease are you in social settings?",
                )
                q3 = st.slider(
                    "3️⃣ I keep in the background. *(Reverse)*",
                    1, 5, 3,
                    help="Do you prefer staying out of the spotlight?",
                )
                q4 = st.slider(
                    "4️⃣ I have little to say. *(Reverse)*",
                    1, 5, 3,
                    help="Are you typically quiet in conversations?",
                )
                q5 = st.slider(
                    "5️⃣ I talk to many different people at parties.",
                    1, 5, 3,
                    help="Do you mingle with diverse groups?",
                )
                st.markdown("</div>", unsafe_allow_html=True)

            # CATEGORY 2: EMOTIONAL STYLE
            with tab2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.warning("**How do you handle attention and feelings?**")

                q6 = st.slider(
                    "6️⃣ I don't mind being the center of attention.",
                    1, 5, 3,
                )
                q7 = st.slider(
                    "7️⃣ I don't like to draw attention to myself. *(Reverse)*",
                    1, 5, 3,
                )
                q8 = st.slider(
                    "8️⃣ I get stressed easily in social situations.",
                    1, 5, 3,
                )
                q9 = st.slider(
                    "9️⃣ I am relaxed most of the time.",
                    1, 5, 3,
                )
                q10 = st.slider(
                    "🔟 I worry about things.",
                    1, 5, 3,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            # CATEGORY 3: INTERACTION HABITS
            with tab3:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.success("**How do you interact with strangers?**")

                q11 = st.slider(
                    "1️⃣1️⃣ I start conversations.",
                    1, 5, 3,
                )
                q12 = st.slider(
                    "1️⃣2️⃣ I find it difficult to approach others.",
                    1, 5, 3,
                )
                q13 = st.slider(
                    "1️⃣3️⃣ I am quiet around strangers.",
                    1, 5, 3,
                )
                q14 = st.slider(
                    "1️⃣4️⃣ I make friends easily.",
                    1, 5, 3,
                )
                q15 = st.slider(
                    "1️⃣5️⃣ I take charge.",
                    1, 5, 3,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("---")
            submitted = st.form_submit_button(
                "🔍 ANALYZE MY PERSONALITY",
                use_container_width=True,
            )

            if submitted:
                st.session_state.answers = {
                    "Life_of_Party": q1,
                    "Comfortable_People": q2,
                    "Keep_Background": q3,
                    "Talk_Little": q4,
                    "Talk_Different_People": q5,
                    "Dont_Mind_Center": q6,
                    "Dont_Like_Attention": q7,
                    "Q6_Get_Stressed_Easily": q8,
                    "Q7_Relaxed_Most_Time": q9,
                    "Q8_Worry_About_Things": q10,
                    "Start_Convos": q11,
                    "Talk_Lot": q14,
                    "Quiet_Stranger": q13,
                }
                st.session_state.page = "result"
                st.rerun()

    # ------------------ RESULTS PAGE ------------------ #
    elif st.session_state.page == "result":

        with st.spinner("🧠 Processing neural pathways..."):
            time.sleep(1.5)

        answers = st.session_state.answers

        # Prepare data
        input_df = pd.DataFrame(index=[0], columns=feature_names)
        input_df[:] = 3

        for col, val in answers.items():
            if col in input_df.columns:
                input_df[col] = val

        # Predict
        input_scaled = scaler.transform(input_df)
        prob_ext = model.predict_proba(input_scaled)[0][1]
        prob_int = 1 - prob_ext

        # Trigger celebration[web:73][web:75]
        if prob_ext > 0.5:
            st.balloons()
        else:
            st.snow()

        # RESULTS HEADER
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 2rem;">
                <h2 class="orange-glow" style="font-size: 2.5rem;">📊 ANALYSIS COMPLETE</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # SCORE CARDS
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("### 🦉 Introvert Score")
            st.progress(prob_int)
            st.markdown(
                f"<h1 class='orange-glow' style='font-size: 2.5rem; margin-top: 0.5rem;'>{prob_int:.1%}</h1>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("### 🦁 Extrovert Score")
            st.progress(prob_ext)
            st.markdown(
                f"<h1 class='orange-glow' style='font-size: 2.5rem; margin-top: 0.5rem;'>{prob_ext:.1%}</h1>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")

        # FINAL VERDICT
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        if prob_ext > 0.5:
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <h2 style="color: #FF9800; font-size: 2rem;">🦁 VERDICT: EXTROVERT</h2>
                    <p style="font-size: 1.3rem; color: #FFB74D;">{prob_ext:.1%} Confidence</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.success(
                """
                **Your Profile:**  
                - Energized by social interaction  
                - Seeks external stimulation  
                - Thrives in group settings  
                - Thinks out loud and processes through discussion  
                """
            )
        else:
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <h2 style="color: #FF9800; font-size: 2rem;">🦉 VERDICT: INTROVERT</h2>
                    <p style="font-size: 1.3rem; color: #FFB74D;">{prob_int:.1%} Confidence</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.info(
                """
                **Your Profile:**  
                - Energized by solitude and reflection  
                - Prefers deep 1-on-1 conversations  
                - Needs alone time to recharge  
                - Thinks internally before speaking  
                """
            )

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.button("🔄 RETAKE ASSESSMENT", on_click=reset_quiz, use_container_width=True)

else:
    st.error("⚠️ Model file not found. Please train the model first!")
