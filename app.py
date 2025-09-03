import streamlit as st
import pandas as pd
import joblib
import numpy as np
import requests
import os

# Function to download files
def download_file(url, filename):
    if not os.path.exists(filename):
        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
        else:
            st.error(f"Failed to download {filename}. Status code: {response.status_code}")
            st.stop()

# Download .pkl files from Google Drive
download_file('YOUR_GOOGLE_DRIVE_LINK_1', 'xgb_model.pkl')
download_file('YOUR_GOOGLE_DRIVE_LINK_2', 'scaler.pkl')
download_file('YOUR_GOOGLE_DRIVE_LINK_3', 'label_encoders.pkl')

# Load the saved XGBoost model, scaler, and encoders
try:
    model = joblib.load('xgb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
except FileNotFoundError as e:
    st.error(f"Error loading file: {e}")
    st.stop()

# Define the columns in the exact order used during training
features = [
    'person_age',
    'person_income',
    'person_emp_exp',
    'loan_amnt',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_cred_hist_length',
    'credit_score',
    'previous_loan_defaults_on_file',
    'person_home_ownership',
    'loan_intent',
    'person_gender',
    'person_education'
]

num_cols = [
    'person_age',
    'person_income',
    'person_emp_exp',
    'loan_amnt',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_cred_hist_length',
    'credit_score'
]

categoricals = [
    'person_home_ownership',
    'loan_intent',
    'previous_loan_defaults_on_file',
    'person_gender',
    'person_education'
]

# Define possible values for categorical features
home_ownership_options = ['RENT', 'OWN', 'MORTGAGE', 'OTHER']
loan_intent_options = ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION']
default_options = ['No', 'Yes']
gender_options = ['male', 'female']
education_options = ['High School', 'Associate', 'Bachelor', 'Master', 'PhD']

# Streamlit app
st.title('Loan Approval Prediction App')

st.write('Enter the details below to predict loan approval.')

# Numerical inputs with sliders or number inputs
person_age = st.slider('Person Age', min_value=18, max_value=100, value=30)
person_income = st.number_input('Person Income', min_value=0, max_value=10000000, value=50000)
person_emp_exp = st.slider('Person Employment Experience (years)', min_value=0, max_value=60, value=5)
loan_amnt = st.number_input('Loan Amount', min_value=500, max_value=1000000, value=10000)
loan_int_rate = st.slider('Loan Interest Rate (%)', min_value=5.0, max_value=25.0, value=10.0, step=0.1)
cb_person_cred_hist_length = st.slider('Credit History Length (years)', min_value=0, max_value=30, value=5)
credit_score = st.slider('Credit Score', min_value=300, max_value=850, value=700)

# Categorical inputs with selectboxes
person_home_ownership = st.selectbox('Person Home Ownership', home_ownership_options)
loan_intent = st.selectbox('Loan Intent', loan_intent_options)
previous_loan_defaults_on_file = st.selectbox('Previous Loan Defaults on File', default_options)
person_gender = st.selectbox('Person Gender', gender_options)
person_education = st.selectbox('Person Education', education_options)

# Calculate loan_percent_income
if person_income > 0:
    loan_percent_income = min(1.0, loan_amnt / person_income)  # Cap at 1.0 (100% as fraction)
    st.write(f"Loan Percent of Income: {loan_percent_income:.2f}")  # Display as fraction (e.g., 0.02)
else:
    st.error("Person Income must be greater than 0 to calculate Loan Percent of Income.")
    st.stop()

# Button to predict
if st.button('Predict'):
    # Create a dictionary with inputs
    input_data = {
        'person_age': person_age,
        'person_income': person_income,
        'person_emp_exp': person_emp_exp,
        'loan_amnt': loan_amnt,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'credit_score': credit_score,
        'person_home_ownership': person_home_ownership,
        'loan_intent': loan_intent,
        'previous_loan_defaults_on_file': previous_loan_defaults_on_file,
        'person_gender': person_gender,
        'person_education': person_education
    }

    # Convert to DataFrame with explicit column order
    input_df = pd.DataFrame([input_data])[features]

    # Encode categorical features
    for col in categoricals:
        le = label_encoders[col]
        try:
            input_df[col] = le.transform(input_df[col])
        except ValueError:
            st.error(f"Unknown value for {col}. Valid options: {list(le.classes_)}")
            st.stop()

    # Scale numerical features
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # Make prediction
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]  # Probability of approval (class 1)
    except ValueError as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    # Display result
    if prediction == 1:
        st.success(f'Loan Approved! Probability: {probability:.2f}')
    else:
        st.error(f'Loan Denied. Probability of Approval: {probability:.2f}')
