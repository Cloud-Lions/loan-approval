
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

# Load models and scaler
ensemble = joblib.load("ensemble_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.title("💳 Loan Approval Prediction App")

# --- Input form ---
st.subheader("Applicant Information")

age = st.slider("Age", 18, 80, 30)
income = st.number_input("Annual Income ($)", min_value=1000, max_value=200000, value=50000, step=1000)
emp_exp = st.slider("Employment Experience (years)", 0, 40, 5)
loan_amount = st.number_input("Loan Amount ($)", min_value=500, max_value=50000, value=10000, step=500)
interest_rate = st.slider("Loan Interest Rate (%)", 5.0, 40.0, 12.5, 0.1)
loan_percent_income = loan_amount / (income + 1e-5)

credit_score = st.slider("Credit Score", 300, 850, 650)
cred_hist_len = st.slider("Credit History Length (years)", 0, 30, 5)

default_history = st.selectbox("Previous Loan Default on File", ["No", "Yes"])
home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
gender = st.selectbox("Gender", ["male", "female"])
education = st.selectbox("Education", ["High School", "Bachelor", "Master", "Doctorate", "Other"])

# --- Prepare input ---
input_dict = {
    "person_age": age,
    "person_income": income,
    "person_emp_exp": emp_exp,
    "loan_amnt": loan_amount,
    "loan_int_rate": interest_rate,
    "loan_percent_income": loan_percent_income,
    "cb_person_cred_hist_length": cred_hist_len,
    "credit_score": credit_score,
    "previous_loan_defaults_on_file": 1 if default_history == "Yes" else 0,
    "person_home_ownership": ["RENT", "OWN", "MORTGAGE", "OTHER"].index(home_ownership),
    "loan_intent": ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"].index(loan_intent),
    "person_gender": 1 if gender == "male" else 0,
    "person_education": ["High School", "Bachelor", "Master", "Doctorate", "Other"].index(education)
}
new_app = pd.DataFrame([input_dict])

# Scale numerics
num_cols = [
    'person_age','person_income','person_emp_exp','loan_amnt',
    'loan_int_rate','loan_percent_income',
    'cb_person_cred_hist_length','credit_score'
]
new_app[num_cols] = scaler.transform(new_app[num_cols])

  --- Predict & Explain ---
if st.button("Predict Loan Approval"):
    pred = ensemble.predict(new_app)[0]
    prob = ensemble.predict_proba(new_app)[0,1]

    if pred == 1:
        st.success(f"✅ Loan Approved with probability {prob:.2f}")
    else:
        st.error(f"❌ Loan Denied with probability {prob:.2f}")

    # SHAP Explanation (for XGBoost sub-model only)
    st.subheader("Feature Contribution (SHAP)")
    xgb_model = ensemble.named_estimators_["xgb"]
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(new_app)

    shap.initjs()
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.waterfall_plot(shap.Explanation(values=shap_values[0],
                                         base_values=explainer.expected_value,
                                         data=new_app.iloc[0],
                                         feature_names=new_app.columns))
    st.pyplot(fig)
