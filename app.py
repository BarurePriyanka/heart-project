import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model.pkl")

st.title("Heart Disease Prediction App ❤️")
st.write("Enter the patient details below to predict heart disease risk.")

# Inputs
age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", ["Male", "Female"])
bp = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol", min_value=100, max_value=400, value=200)
max_hr = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)

sex_val = 1 if sex == "Male" else 0

# Input for model (make sure columns match your trained model)
input_data = pd.DataFrame([[age, sex_val, bp, chol, max_hr]],
                          columns=['age', 'sex', 'bp', 'chol', 'max_hr'])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)
    result = "High Risk ❤️" if prediction[0] == 1 else "Low Risk 💚"
    st.success(f"Prediction: {result}")
