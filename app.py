import streamlit as st
import pandas as pd
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
# Title & description (USER-FRIENDLY)
# -----------------------------------------------------
st.title("🩺 Diabetes Risk Screening Tool")

st.write(
    """
    This tool estimates the risk of diabetes based on common
    health and lifestyle factors.

    ⚠️ The result is for **screening purposes only** and does not
    provide a medical diagnosis.
    """
)

# -----------------------------------------------------
# Fixed screening settings (NOT visible to user)
# -----------------------------------------------------
THRESHOLD = 0.30  # High sensitivity screening

# -----------------------------------------------------
# Patient information
# -----------------------------------------------------
st.header("Patient information")

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
# Encoding (MUST match training)
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
    prediction = int(probability >= THRESHOLD)

    st.subheader("Result")

    st.write(f"**Estimated probability of diabetes:** {probability:.2f}")

    if prediction == 1:
        st.error("⚠️ High risk of diabetes (screen-positive)")
    else:
        st.success("✅ Low risk of diabetes (screen-negative)")

    st.info(
        "This screening tool prioritizes sensitivity to avoid missing individuals "
        "at high risk. Clinical evaluation and laboratory tests are required "
        "for diagnosis."
    )
