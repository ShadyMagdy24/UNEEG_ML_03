import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("healthcare_model.pkl")

# Streamlit UI
st.title("🏥 Health Care Diagnosis Prediction")

st.write("Enter patient details to predict the disease diagnosis.")

# User input fields
age = st.number_input("Age", min_value=1, max_value=120, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
billing_amount = st.number_input("Billing Amount ($)", min_value=0, max_value=100000, value=5000)
hospital_stay = st.number_input("Hospital Stay (Days)", min_value=1, max_value=365, value=5)

# Categorical Inputs
blood_type = st.selectbox("Blood Type", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
admission_type = st.selectbox("Admission Type", ["Elective", "Emergency", "Urgent"])
medication = st.selectbox("Medication", ["Aspirin", "Lipitor", "Penicillin", "Paracetamol", "Ibuprofen"])
test_results = st.selectbox("Test Results", ["Normal", "Abnormal", "Inconclusive"])

# Convert categorical inputs to numerical
gender = 1 if gender == "Female" else 0
blood_types = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
blood_type_encoded = [1 if blood_type == bt else 0 for bt in blood_types]

admission_types = ["Elective", "Emergency", "Urgent"]
admission_encoded = [1 if admission_type == at else 0 for at in admission_types]

medications = ["Aspirin", "Lipitor", "Penicillin", "Paracetamol", "Ibuprofen"]
medication_encoded = [1 if medication == med else 0 for med in medications]

test_results_types = ["Normal", "Abnormal", "Inconclusive"]
test_encoded = [1 if test_results == tr else 0 for tr in test_results_types]

# Create feature array
features = [age, gender, billing_amount, hospital_stay] + blood_type_encoded + admission_encoded + medication_encoded + test_encoded
features = np.array(features).reshape(1, -1)

# Predict button
if st.button("Predict Disease"):
    prediction = model.predict(features)
    disease_classes = ["Diabetes", "Asthma", "Obesity", "Arthritis", "Hypertension", "Cancer"]
    predicted_disease = disease_classes[int(prediction[0])]
    
    st.success(f"🩺 Predicted Diagnosis: **{predicted_disease}**")
