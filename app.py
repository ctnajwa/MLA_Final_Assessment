import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Student Burnout Predictor",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# LOAD MODEL AND ARTIFACTS

@st.cache_resource
def load_model():
    """Load the trained model, scaler, and features"""
    try:
        # Load the trained model
        with open('svm_burnout_model_selected.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load the scaler
        with open('scaler_selected.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Load the selected features
        with open('selected_features_svm.pkl', 'rb') as f:
            selected_features = pickle.load(f)
        
        # Convert to list if needed
        if isinstance(selected_features, (pd.Index, np.ndarray)):
            selected_features = selected_features.tolist()
        
        model_info = {
            'algorithm': 'SVM (RBF Kernel)',
            'feature_count': len(selected_features),
            'feature_selection_method': 'SelectFromModel with LinearSVC'
        }
        
        return model, scaler, selected_features, model_info
    
    except FileNotFoundError as e:
        st.error(f"""
        ❌ Missing file: {e.filename}
        
        Please ensure all model files are in the same directory:
        - svm_burnout_model_selected.pkl
        - scaler_selected.pkl
        - selected_features_svm.pkl
        """)
        st.stop()

# Load the model
model, scaler, selected_features, model_info = load_model()


# CUSTOM CSS - MOBILE FRIENDLY


st.markdown("""
    <style>
    /* Main header */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    /* Sub header */
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Prediction boxes */
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
        font-size: 1.5rem;
        font-weight: 600;
    }
    .prediction-high {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 2px solid #F87171;
    }
    .prediction-low {
        background-color: #D1FAE5;
        color: #065F46;
        border: 2px solid #34D399;
    }
    
    /* Mobile responsive adjustments */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.8rem;
        }
        .sub-header {
            font-size: 1rem;
        }
        .prediction-box {
            font-size: 1.2rem;
            padding: 15px;
        }
        .stSlider {
            padding: 10px 0;
        }
        .stRadio > div {
            flex-direction: row !important;
        }
    }
    
    /* Make sliders touch-friendly on mobile */
    .stSlider > div > div {
        min-height: 30px;
    }
    
    /* Info box */
    .info-box {
        background-color: #EFF6FF;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
    }
    </style>
""", unsafe_allow_html=True)


# HEADER

st.markdown('<div class="main-header">📚 Student Burnout Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict whether a student is at risk of high burnout using SVM</div>', unsafe_allow_html=True)


# MAIN CONTENT - INPUT FORM

st.markdown("## 📝 Enter Student Information")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🛌 Sleep & Academic Load")
    
    sleep_hours = st.slider(
        "Sleep Hours (per night)",
        min_value=3.0,
        max_value=11.0,
        value=7.5,
        step=0.5,
        help="Recommended: 7-9 hours"
    )
    
    homework_hours = st.slider(
        "Homework Hours (per day)",
        min_value=0.0,
        max_value=6.0,
        value=2.5,
        step=0.5,
        help="Average hours spent on homework daily"
    )
    
    screen_time_hours = st.slider(
        "Screen Time Hours (per day)",
        min_value=0.5,
        max_value=10.0,
        value=4.0,
        step=0.5,
        help="Average hours of screen time daily"
    )

with col2:
    st.markdown("### 🧠 Stress & Demographics")
    
    self_rated_stress = st.select_slider(
        "Self-Rated Stress Level",
        options=[1, 2, 3, 4, 5],
        value=3,
        help="1 = Low Stress, 5 = High Stress"
    )
    
    gender = st.radio(
        "Gender",
        options=["Female", "Male"],
        index=0,
        horizontal=True
    )
    gender_male = 1 if gender == "Male" else 0
    
    # Display stress level indicator
    stress_labels = {
        1: "🟢 Low",
        2: "🟢 Moderate-Low",
        3: "🟡 Moderate",
        4: "🟠 Moderate-High",
        5: "🔴 High"
    }
    st.markdown(f"**Stress Level:** {stress_labels[self_rated_stress]}")


# PREDICTION BUTTON

st.markdown("---")

if st.button("🔮 Predict Burnout Risk", type="primary", use_container_width=True):
    
    # Prepare input data with consistent column names
    input_data = pd.DataFrame({
        'sleep_hours': [sleep_hours],
        'homework_hours': [homework_hours],
        'screen_time_hours': [screen_time_hours],
        'self_rated_stress': [self_rated_stress],
        'gender_Male': [gender_male]
    })
    
    # Ensure features are in correct order
    try:
        input_data = input_data[selected_features]
    except KeyError as e:
        st.error(f"""
        ❌ Feature mismatch error!
        
        The model expects these features: {selected_features}
        But the input has: {input_data.columns.tolist()}
        
        Please check that the model files are correct.
        """)
        st.stop()
    
    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    

    # DISPLAY RESULTS
    
    st.markdown("---")
    st.markdown("## 📊 Prediction Results")
    
    # Create columns for results
    result_col1, result_col2 = st.columns([2, 1])
    
    with result_col1:
        if prediction == 1:
            st.markdown("""
            <div class="prediction-box prediction-high">
                ⚠️ High Burnout Risk Detected
            </div>
            """, unsafe_allow_html=True)
            st.warning("""
            **Recommendations:**
            - Consider reducing workload
            - Prioritize sleep hygiene (aim for 7-9 hours)
            - Take regular breaks from screens
            - Reach out to support networks
            """)
        else:
            st.markdown("""
            <div class="prediction-box prediction-low">
                ✅ Low Burnout Risk
            </div>
            """, unsafe_allow_html=True)
            st.success("""
            **Keep up the good habits!**
            - Maintain balanced sleep schedule
            - Continue managing stress effectively
            - Stay connected with support systems
            """)
    
    with result_col2:
        st.markdown("### Confidence Scores")
        prob_df = pd.DataFrame({
            'Status': ['No Burnout', 'High Burnout'],
            'Probability': probability
        })
        fig = px.bar(prob_df, 
                     x='Status', 
                     y='Probability',
                     color='Status',
                     color_discrete_map={'No Burnout': '#34D399', 'High Burnout': '#F87171'},
                     title='Prediction Probability')
        st.plotly_chart(fig, use_container_width=True)

    
    # FEATURE CONTRIBUTION ANALYSIS

    st.markdown("---")
    st.markdown("### 📈 Feature Contribution Analysis")
    
    input_values = input_data.iloc[0]
    
    # Create a gauge for each feature
    cols = st.columns(len(selected_features))
    for col, feature in zip(cols, selected_features):
        value = input_values[feature]
        
        # Normalize value for gauge
        if feature == 'sleep_hours':
            max_val = 11.0
            color = "green" if value >= 7 else "orange" if value >= 6 else "red"
        elif feature == 'self_rated_stress':
            max_val = 5
            color = "red" if value >= 4 else "orange" if value >= 3 else "green"
        elif feature in ['homework_hours', 'screen_time_hours']:
            max_val = 8.0
            color = "red" if value >= 5 else "orange" if value >= 3.5 else "green"
        else:
            max_val = 1
            color = "blue"
        
        progress = min(value / max_val, 1.0) if max_val > 0 else 0
        
        with col:
            st.markdown(f"**{feature}**")
            st.progress(progress)
            st.caption(f"{value:.1f}" if isinstance(value, float) else f"{value}")
