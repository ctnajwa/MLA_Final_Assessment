import streamlit as st
import pickle
import numpy as np

# ==========================================
# 1. LOAD THE EXPORTED ML ARTIFACTS
# ==========================================
with open('svm_burnout_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# ==========================================
# 2. CONFIGURE WEB INTERFACE
# ==========================================
st.set_page_config(page_title="Smart Campus Burnout Predictor", layout="centered")

st.title("🎓 Smart Campus Student Burnout Predictor")
st.write("This application leverages Machine Learning to identify early signs of academic stress and burnout among students.")

st.markdown("---")
st.subheader("💡 Input Student Daily/Weekly Metrics")

# Create user friendly entry tools (Sliders and dropdowns matching your columns)
gender = st.selectbox("Gender", options=["Male", "Female", "Nonbinary"])
sleep_hours = st.slider("Average Daily Sleep Hours", 3.0, 12.0, 7.0, 0.1)
sleep_quality = st.slider("Sleep Quality Rating (1-5)", 1, 5, 3)
homework_hours = st.slider("Daily Homework Hours", 0.0, 10.0, 2.0, 0.1)
tests_per_week = st.slider("Number of Tests per Week", 0, 5, 1)
extracurricular_hours = st.slider("Weekly Extracurricular Hours", 0.0, 30.0, 5.0, 0.5)
num_activities = st.slider("Number of Organized Activities", 0, 5, 1)
screen_time_hours = st.slider("Average Daily Screen Time (Hours)", 0.0, 15.0, 4.0, 0.1)
commute_minutes = st.slider("Daily Commute Time (Minutes)", 0, 90, 20)
family_support = st.slider("Family Support Level (1-5)", 1, 5, 3)
friend_support = st.slider("Friend Support Level (1-5)", 1, 5, 3)
teacher_support = st.slider("Teacher Support Level (1-5)", 1, 5, 3)
self_rated_stress = st.slider("Self-Rated Stress Level (1-5)", 1, 5, 3)

# ==========================================
# 3. PREPROCESS USER ENTRYS
# ==========================================
# Re-create the One-Hot Encoded variables for gender exactly like the training columns
gender_Male = 1 if gender == "Male" else 0
gender_Nonbinary = 1 if gender == "Nonbinary" else 0

# Construct raw input array (Must follow the EXACT column order of your training X dataframe!)
raw_inputs = np.array([[
    sleep_hours, sleep_quality, homework_hours, tests_per_week,
    extracurricular_hours, num_activities, screen_time_hours, commute_minutes,
    family_support, friend_support, teacher_support, self_rated_stress,
    gender_Male, gender_Nonbinary
]])

# Pass raw parameters through your saved scaler
scaled_inputs = scaler.transform(raw_inputs)

# ==========================================
# 4. RUN PREDICTION & DISPLAY RESULTS
# ==========================================
st.markdown("---")
if st.button("📊 Analyze Burnout Risk", type="primary"):
    prediction = model.predict(scaled_inputs)[0]
    
    if prediction == 1:
        st.error("⚠️ **High Burnout Risk Detected!**\n\nThis student exhibits significant behavioral markers associated with high academic stress. Intervention or counseling support is recommended.")
    else:
        st.success("✅ **Normal Stress Levels.**\n\nThe student shows a balanced routine with healthy boundaries. Continue monitoring.")
