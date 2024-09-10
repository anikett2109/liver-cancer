import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model and transformer
def load_model():
    model = joblib.load('liver_disease_model.joblib')
    return model

def load_trans():
    transformer = joblib.load('transformer.joblib')
    return transformer

model = load_model()
transformer = load_trans()

# Title of the Streamlit app
st.title("Liver Disease Prediction App")

# User input fields
st.header("Enter the following details:")

# Collecting user inputs
age = st.number_input("Age", min_value=0, max_value=100, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])
total_bilirubin = st.number_input("Total Bilirubin (TB)", min_value=0.0, step=0.1)
direct_bilirubin = st.number_input("Direct Bilirubin (DB)", min_value=0.0, step=0.1)
alkaline_phosphotase = st.number_input("Alkaline Phosphotase (Alkphos)", min_value=0, step=1)
sgpt = st.number_input("SGPT (Sgpt)", min_value=0, step=1)
sgot = st.number_input("SGOT (Sgot)", min_value=0, step=1)
tp = st.number_input("Total Proteins (TP)", min_value=0.0, step=0.1)
alb = st.number_input("Albumin (ALB)", min_value=0.0, step=0.1)
ag_ratio = st.number_input("A/G Ratio", min_value=0.0, step=0.1)

# Encoding the categorical Gender feature
gender_encoded = 1 if gender == "Male" else 0

# When the user clicks the 'Predict' button
if st.button("Predict"):
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender_encoded],
        'TB': [total_bilirubin],
        'DB': [direct_bilirubin],
        'Alkphos': [alkaline_phosphotase],
        'Sgpt': [sgpt],
        'Sgot': [sgot],
        'TP': [tp],
        'ALB': [alb],
        'A/G Ratio': [ag_ratio]
    })

    # Transform the input data using the loaded transformer
    input_data_transformed = transformer.transform(input_data)

    # Make the prediction using the model
    prediction = model.predict(input_data_transformed)
    print(prediction)

    # Output the result
    if prediction[0] == 1:
        st.success("The model predicts that the patient has liver disease.")
    else:
        st.success("The model predicts that the patient does not have liver disease.")
