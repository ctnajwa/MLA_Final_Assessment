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
        with open('svm_burnout_model_selected.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('scaler_selected.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
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
        
        Please ensure all model files are in the same directory.
        """)
        st.stop()

# Load the model
model, scaler, selected_features, model_info = load_model()

# Display which features are required (for debugging)
st.write("📊 **Required Features:**", selected_features)


# CUSTOM CSS - MOBILE FRIENDLY

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
    @media (max-width: 768px) {
        .main-header { font-size: 1.8rem; }
        .sub-header { font-size: 1rem; }
        .prediction-box { font-size: 1.2rem; padding: 15px; }
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

# Show selected features info
with st.expander("ℹ️ Model Information"):
    st.markdown(f"""
    - **Algorithm:** {model_info['algorithm']}
    - **Features Used:** {model_info['feature_count']}
    - **Feature Selection:** {model_info['feature_selection_method']}
    - **Features:** {', '.join(selected_features)}
    """)


# INPUT FORM - MATCHING SELECTED FEATURES

st.markdown("## 📝 Enter Student Information")

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

with col2:
    st.markdown("### 🤝 Support Systems")
    
    family_support = st.slider(
        "Family Support Level",
        min_value=1,
        max_value=5,
        value=3,
        help="1 = Low Support, 5 = High Support"
    )
    
    friend_support = st.slider(
        "Friend Support Level",
        min_value=1,
        max_value=5,
        value=3,
        help="1 = Low Support, 5 = High Support"
    )
    
    teacher_support = st.slider(
        "Teacher Support Level",
        min_value=1,
        max_value=5,
        value=3,
        help="1 = Low Support, 5 = High Support"
    )

# PREDICTION BUTTON

st.markdown("---")

if st.button("🔮 Predict Burnout Risk", type="primary", use_container_width=True):
    
    # Prepare input data with correct feature names
    input_data = pd.DataFrame({
        'sleep_hours': [sleep_hours],
        'homework_hours': [homework_hours],
        'family_support': [family_support],
        'friend_support': [friend_support],
        'teacher_support': [teacher_support]
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
            - Seek more support from family and friends
            - Talk to teachers about concerns
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
    
    cols = st.columns(len(selected_features))
    for col, feature in zip(cols, selected_features):
        value = input_values[feature]
        
        # Normalize value for gauge
        if feature == 'sleep_hours':
            max_val = 11.0
            color = "green" if value >= 7 else "orange" if value >= 6 else "red"
        elif feature == 'homework_hours':
            max_val = 6.0
            color = "red" if value >= 4 else "orange" if value >= 2.5 else "green"
        else:  # support features (family_support, friend_support, teacher_support)
            max_val = 5
            color = "green" if value >= 4 else "orange" if value >= 3 else "red"
        
        progress = min(value / max_val, 1.0) if max_val > 0 else 0
        
        with col:
            st.markdown(f"**{feature}**")
            st.progress(progress)
            st.caption(f"{value:.1f}" if isinstance(value, float) else f"{value}")
