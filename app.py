import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Screening Tool",
    layout="centered"
)

st.title("🩺 Diabetes Risk Screening Tool")
st.caption("Machine learning–based screening tool (not a diagnostic system)")

# --------------------------------------------------
# Load models and artifacts
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("diabetes_model.joblib")
    model_bal = joblib.load("diabetes_model_balanced.joblib")
    scaler = joblib.load("scaler.joblib")
    features = joblib.load("features.joblib")
    return model, model_bal, scaler, features

try:
    model, model_bal, scaler, features = load_artifacts()
except Exception as e:
    st.error("❌ Error loading model files")
    st.exception(e)
    st.stop()

# --------------------------------------------------
# 1. Screening strategy
# --------------------------------------------------
st.subheader("1️⃣ Screening strategy")
model_type = st.radio(
    "Select screening mode:",
    [
        "Moderate sensitivity (clinical pre-screening)",
        "High sensitivity (population screening)"
    ]
)

if model_type == "Moderate sensitivity (clinical pre-screening)":
    clf = model
    default_threshold = 0.3
else:
    clf = model_bal
    default_threshold = 0.3

# --------------------------------------------------
# 2. Probability threshold
# --------------------------------------------------
st.subheader("2️⃣ Probability threshold")
threshold = st.slider(
    "Decision threshold",
    min_value=0.1,
    max_value=0.9,
    value=default_threshold,
    step=0.05
)

# --------------------------------------------------
# 3. Patient information
# --------------------------------------------------
st.subheader("3️⃣ Patient information")

sex = st.selectbox("Sex", ["Male", "Female"])
education = st.selectbox("Education level", ["Low", "Medium", "High"])
marital = st.selectbox("Marital status", ["Single", "Married"])
labor = st.selectbox("Labor status", ["Employed", "Unemployed"])
smoking = st.selectbox("Smoking", ["No", "Yes"])
alcohol = st.selectbox("Alcohol drinking", ["No", "Yes"])
physical = st.selectbox("Physical inactivity", ["No", "Yes"])
salt = st.selectbox("High salt intake", ["No", "Yes"])

bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
waist = st.number_input(
    "Waist circumference (cm)",
    min_value=50,
    max_value=150,
    value=90
)

# --------------------------------------------------
# Automatically determine obesity (IMPORTANT FIX)
# --------------------------------------------------
obesity = "Yes" if bmi >= 30 else "No"

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔍 Estimate diabetes risk"):
    input_df = pd.DataFrame([{
        "Sex": sex,
        "Education_level": education,
        "Marital_status": marital,
        "Labor_status": labor,
        "Smoking": smoking,
        "Alcohol_drinking": alcohol,
        "Physical_inactivity": physical,
        "High_salt_intake": salt,
        "BMI": bmi,
        "Waist_circumference": waist,
        "Obesity": obesity
    }])

    # One-hot encoding + align with training features
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=features, fill_value=0)

    # Scaling
    input_scaled = scaler.transform(input_df)

    # Prediction
    risk = clf.predict_proba(input_scaled)[0][1]

    st.metric("Estimated diabetes risk", f"{risk:.1%}")

    if risk >= threshold:
        st.warning(
            "⚠️ High diabetes risk detected. "
            "Further clinical evaluation is recommended."
        )
    else:
        st.success("✅ Low diabetes risk.")

# --------------------------------------------------
# Disclaimer
# --------------------------------------------------
st.info(
    "This tool is intended for research and screening purposes only. "
    "It does not replace professional medical diagnosis."
)


