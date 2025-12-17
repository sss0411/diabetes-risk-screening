import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------------------------------
# Page config
# -----------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Screening Tool",
    layout="centered"
)

# -----------------------------------------------------
# Load model & scaler
# -----------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("diabetes_model.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

model, scaler = load_artifacts()

# -----------------------------------------------------
# Title
# -----------------------------------------------------
st.title("🩺 Diabetes Risk Screening Tool")

st.write(
    """
    This application estimates the risk of diabetes based on
    demographic, lifestyle, and anthropometric factors.

    ⚠️ This tool is for **screening purposes only** and does not provide a diagnosis.
    """
)

# -----------------------------------------------------
# 1. Screening mode
# -----------------------------------------------------
st.header("1. Select screening mode")

mode = st.radio(
    "Screening mode:",
    [
        "Moderate sensitivity (clinical pre-screening)",
        "High sensitivity (population screening)"
    ]
)

# Default thresholds
if mode == "Moderate sensitivity (clinical pre-screening)":
    default_threshold = 0.50
else:
    default_threshold = 0.30

# -----------------------------------------------------
# 2. Probability threshold
# -----------------------------------------------------
st.header("2. Probability threshold")

threshold = st.slider(
    "Decision threshold",
    min_value=0.10,
    max_value=0.90,
    value=default_threshold,
    step=0.01
)

# -----------------------------------------------------
# 3. Patient information
# -----------------------------------------------------
st.header("3. Patient information")

age = st.number_input(
    "Age (years)",
    min_value=18,
    max_value=90,
    value=45
)

sex = st.selectbox("Sex", ["Female", "Male"])
education = st.selectbox("Education level", ["Low", "Medium", "High"])
marital = st.selectbox("Marital status", ["Single", "Married", "Other"])
labor = st.selectbox("Labor status", ["Unemployed", "Employed", "Other"])

smoking = st.selectbox("Smoking", ["No", "Yes"])
alcohol = st.selectbox("Alcohol drinking", ["No", "Yes"])
physical = st.selectbox("Physical inactivity", ["No", "Yes"])
salt = st.selectbox("High salt intake", ["No", "Yes"])

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0
)

waist = st.number_input(
    "Waist circumference (cm)",
    min_value=50,
    max_value=160,
    value=90
)

obesity = st.selectbox("Obesity", ["No", "Yes"])

# -----------------------------------------------------
# Encode categorical variables (MUST MATCH TRAINING)
# -----------------------------------------------------
sex_val = 1 if sex == "Male" else 0
smoking_val = 1 if smoking == "Yes" else 0
alcohol_val = 1 if alcohol == "Yes" else 0
physical_val = 1 if physical == "Yes" else 0
salt_val = 1 if salt == "Yes" else 0
obesity_val = 1 if obesity == "Yes" else 0

education_map = {"Low": 4, "Medium": 5, "High": 6}
marital_map = {"Single": 2, "Married": 1, "Other": 0}
labor_map = {"Unemployed": 1, "Employed": 2, "Other": 0}

# -----------------------------------------------------
# Create input DataFrame (ORDER IS CRITICAL)
# -----------------------------------------------------
input_data = pd.DataFrame({
    "Age": [age],
    "Sex": [sex_val],
    "Education_level": [education_map[education]],
    "Marital_status": [marital_map[marital]],
    "Labor_status": [labor_map[labor]],
    "Smoking": [smoking_val],
    "Alcohol_drinking": [alcohol_val],
    "Physical_inactivity": [physical_val],
    "High_salt_intake": [salt_val],
    "BMI": [bmi],
    "Waist_circumference": [waist],
    "Obesity": [obesity_val]
})

# -----------------------------------------------------
# Prediction
# -----------------------------------------------------
if st.button("Predict risk"):
    X_scaled = scaler.transform(input_data)
    probability = model.predict_proba(X_scaled)[0, 1]
    prediction = int(probability >= threshold)

    st.subheader("Result")

    st.write(f"**Predicted probability of diabetes:** {probability:.2f}")
    st.write(f"**Decision threshold:** {threshold:.2f}")

    if prediction == 1:
        st.error("⚠️ High risk of diabetes (screen-positive)")
    else:
        st.success("✅ Low risk of diabetes (screen-negative)")

    st.info(
        "This result represents a screening assessment only. "
        "Clinical evaluation and laboratory tests are required for diagnosis."
    )
