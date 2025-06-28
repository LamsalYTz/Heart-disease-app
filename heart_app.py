import streamlit as st 
import numpy as np
import joblib

model = joblib.load("heart_model.pkl")


# Streamlit UI
st.title("❤️ Heart Disease Prediction App")
st.write("Enter the following medical info to predict risk")

# Input fields
age = st.number_input("Age", 1, 120)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200)
chol = st.number_input("Cholesterol Level", 100, 600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak", 0.0, 6.0)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thal (0 = normal; 1 = fixed defect; 2 = reversible defect)", [0, 1, 2])

# Convert sex to numeric
sex_val = 1 if sex == "Male" else 0

# Predict button
if st.button("Predict"):
    input_data = np.array([[age, sex_val, cp, trestbps, chol, fbs, restecg, thalach, exang,
                            oldpeak, slope, ca, thal]])
    st.write("Input Model: ", input_data)
    prediction = model.predict(input_data)
    result = "🟢 No Heart Disease Detected" if prediction[0] == 0 else "🔴 Risk of Heart Disease"
    st.subheader("Prediction Result:")
    st.success(result)

