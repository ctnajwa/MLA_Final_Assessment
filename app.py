import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

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


# CUSTOM CSS

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
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
    .feature-box {
        background-color: #F3F4F6;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
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

# SIDEBAR - MODEL INFORMATION


with st.sidebar:
    st.markdown("## 🤖 Model Information")
    st.markdown(f"""
    <div class="info-box">
        <b>Algorithm:</b> {model_info['algorithm']}<br>
        <b>Features Used:</b> {model_info['feature_count']}<br>
        <b>Feature Selection:</b> {model_info['feature_selection_method']}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 Selected Features")
    for feature in selected_features:
        st.markdown(f"- {feature}")
    
    st.markdown("---")
    st.markdown("### 📈 Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", "82%", delta="+2% vs DT")
    with col2:
        st.metric("F1 Score", "0.82", delta="0.03")
    
    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.8rem; color: #6B7280; text-align: center;">
        Built with ❤️ using Streamlit & SVM<br>
        Version 1.0
    </div>
    """, unsafe_allow_html=True)

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
    
    # Prepare input data
    input_data = pd.DataFrame({
        'sleep_hours': [sleep_hours],
        'homework_hours': [homework_hours],
        'screen_time_hours': [screen_time_hours],
        'self_rated_stress': [self_rated_stress],
        'gender_Male': [gender_male]
    })
    
    # Ensure features are in correct order
    input_data = input_data[selected_features]
    
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
    
    # Create a simple contribution visualization based on input values
    # (This is a simplified version - actual SHAP values would be better)
    
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


# FOOTER - ADDITIONAL INFORMATION

st.markdown("---")
with st.expander("ℹ️ About This Predictor"):
    st.markdown("""
    **Model Details:**
    - **Algorithm:** Support Vector Machine (SVM) with RBF kernel
    - **Feature Selection:** SelectFromModel with LinearSVC
    - **Training Data:** 1,956 student records
    - **Features Selected:** Top 5 most important features
    
    **Feature Descriptions:**
    - **Sleep Hours:** Average hours of sleep per night
    - **Homework Hours:** Average hours spent on homework daily
    - **Screen Time Hours:** Average hours of screen time daily
    - **Self-Rated Stress:** Self-reported stress level (1-5)
    - **Gender:** Student's gender
    
    **Important Notes:**
    - This is a predictive tool, not a medical diagnosis
    - Results should be interpreted with professional guidance
    - The model was trained on student population data
    """)
