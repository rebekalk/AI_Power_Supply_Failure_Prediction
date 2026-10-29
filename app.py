import streamlit as st
import pandas as pd
import pickle

with open("power_supply_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("AI-Based Power Supply Failure Prediction System")

st.write("Enter the power supply readings below.")

input_voltage = st.number_input(
    "Input Voltage (V)", min_value=0.0, value=230.0
)

output_voltage = st.number_input(
    "Output Voltage (V)", min_value=0.0, value=12.0
)

current = st.number_input(
    "Current (A)", min_value=0.0, value=1.5
)

temperature = st.number_input(
    "Temperature (°C)", value=35.0
)

if st.button("Predict Power Supply Condition"):
    readings = pd.DataFrame(
        [[input_voltage, output_voltage, current, temperature]],
        columns=[
            "input_voltage",
            "output_voltage",
            "current",
            "temperature"
        ]
    )

    prediction = model.predict(readings)[0]

    if prediction == "Normal":
        st.success("Predicted Condition: NORMAL")
    else:
        st.error("Warning: ABNORMAL CONDITION PREDICTED")

    st.caption(
        "Educational prototype using synthetic data. "
        "Predictions are not validated for real equipment."
    )