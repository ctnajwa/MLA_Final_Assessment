import streamlit as st
import pickle
import numpy as np

# ==========================================
# 1. LOAD THE UPDATED ML ARTIFACTS
# ==========================================
with open('svm_burnout_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# ==========================================
# 2. CONFIGURE VISUAL APP LAYOUT
# ==========================================
# Eye-catching header block
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🧠 Student Burnout Analyzer</h1>", unsafe_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>An intelligent predictive system designed to identify early signs of academic stress and burnout among students.</p>", unsafe_html=True)
st.markdown("---")

# Using columns to create a clean side-by-side aesthetic
st.subheader("📊 Step 1: Student Demographics")
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("⚡ Gender Identification", options=["Female", "Male", "Nonbinary"])
with col2:
    self_rated_stress = st.slider("🔥 Personal Stress Level Rating", 1, 5, 3, help="1 = Calm/Relaxed, 5 = Overwhelmed")

st.markdown("---")
st.subheader("⏳ Step 2: Time Allocation & Routines")

# Grouping inputs into neat sliders with custom emojis
sleep_hours = st.slider("😴 Average Daily Sleep (Hours)", 3.0, 12.0, 7.0, 0.5)
sleep_quality = st.slider("⭐ Sleep Quality Rating", 1, 5, 3, help="1 = Restless, 5 = Deep peaceful sleep")
homework_hours = st.slider("📝 Daily Homework Time (Hours)", 0.0, 10.0, 2.0, 0.5)
tests_per_week = st.slider("📅 Number of Tests / Assessments per Week", 0, 5, 1)
screen_time_hours = st.slider("📱 Average Daily Screen Time (Hours)", 0.0, 15.0, 4.0, 0.5)
extracurricular_hours = st.slider("⚽ Weekly Extracurricular Activities (Hours)", 0.0, 30.0, 5.0, 1.0)
num_activities = st.slider("🎒 Number of Organized School Clubs/Activities", 0, 5, 1)
commute_minutes = st.slider("🚌 Daily Commute Duration (Minutes)", 0, 90, 20)

st.markdown("---")
st.subheader("🤝 Step 3: Support Ecosystem")

# Social factors grouped together
family_support = st.slider("🏡 Family Support System", 1, 5, 3, help="1 = Isolated, 5 = Highly supportive home environment")
friend_support = st.slider("🎒 Peer & Friend Support Group", 1, 5, 3, help="1 = Isolated, 5 = Reliable friend circle")
teacher_support = st.slider("🏫 Institutional / Teacher Support", 1, 5, 3, help="1 = Minimal guidance, 5 = Highly encouraging teachers")

# ==========================================
# 3. PREPROCESS INPUTS (WITHOUT GRADE)
# ==========================================
gender_Male = 1 if gender == "Male" else 0
gender_Nonbinary = 1 if gender == "Nonbinary" else 0

# The array layout perfectly matches the updated model structure (14 variables total)
raw_inputs = np.array([[
    sleep_hours, sleep_quality, homework_hours, tests_per_week,
    extracurricular_hours, num_activities, screen_time_hours, commute_minutes,
    family_support, friend_support, teacher_support, self_rated_stress,
    gender_Male, gender_Nonbinary
]])

scaled_inputs = scaler.transform(raw_inputs)

# ==========================================
# 4. PREDICTION STYLING
# ==========================================
st.markdown("<br>", unsafe_html=True)
if st.button("🚀 RUN BURNOUT ANALYSIS", type="primary", use_container_width=True):
    prediction = model.predict(scaled_inputs)[0]
    
    st.markdown("### 🔍 Analysis Verdict:")
    if prediction == 1:
        st.error("""
        ### ⚠️ Warning: High Burnout Risk Detected!
        The predictive algorithm indicates significant indicators matching high chronic academic exhaustion.
        
        **Recommended Interventions:**
        * Schedule an appointment with a school counselor.
        * Establish digital boundaries to lower screen-time hours.
        * Discuss workload adjustments regarding homework or extracurricular commitments.
        """)
    else:
        st.success("""
        ### ✅ Status: Balanced Workload
        The student exhibits a stable routine with normal stress indices and adequate resting parameters. 
        
        Keep promoting a healthy balance between work, play, and restorative sleep!
        """)
