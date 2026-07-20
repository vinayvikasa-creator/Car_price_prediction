from huggingface_hub import hf_hub_download
import os

MODEL_PATH = hf_hub_download(
    repo_id="gojoXgeto/car-price-model",
    filename="best_model.pkl"
)

# Import Required Libraries

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load Trained Model

from huggingface_hub import hf_hub_download
import joblib
import streamlit as st

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="gojoXgeto/car-price-model",
        filename="best_model.pkl"
    )

    model = joblib.load(model_path)
    encoders = joblib.load("models/label_encoders.pkl")

    return model, encoders

model, encoders = load_model()
# Streamlit Page Configuration

st.set_page_config(
    page_title="Car Price Prediction System",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Car Price Prediction System")
st.write("Predict the price of a used car using Machine Learning.")

# Sidebar

page = st.sidebar.selectbox(
    "Select Page",
    [
        "Single Prediction",
        "Batch Prediction",
        "Dashboard"
    ]
)

# Single Prediction Page

if page == "Single Prediction":

    st.header("Single Car Price Prediction")

maker = st.number_input("Maker", min_value=0, step=1)

car_model = st.number_input("Model", min_value=0, step=1)

location = st.number_input("Location", min_value=0, step=1)

distance = st.number_input("Distance", min_value=0.0)

owner_type = st.number_input("Owner Type", min_value=0, step=1)

manufacture_year = st.number_input("Manufacture Year", min_value=1990, max_value=2030)

age_of_car = st.number_input("Age of Car", min_value=0)

engine_displacement = st.number_input("Engine Displacement", min_value=0)

engine_power = st.number_input("Engine Power", min_value=0.0)

body_type = st.number_input("Body Type", min_value=0, step=1)

vroom_audit_rating = st.number_input("Vroom Audit Rating", min_value=0)

transmission = st.number_input("Transmission", min_value=0, step=1)

door_count = st.number_input("Door Count", min_value=1)

seat_count = st.number_input("Seat Count", min_value=1)

fuel_type = st.number_input("Fuel Type", min_value=0, step=1)

predict = st.button("Predict Price")

if page == "Batch Prediction":

    st.header("Batch Prediction")

    uploaded_file = st.file_uploader(
        "Upload Test CSV File",
        type=["csv", "xlsx"]
    )
# Dashboard Page

if page == "Dashboard":

    st.header("Car Price Dashboard")

    train = pd.read_excel("Cap_Training_Data_2025.xlsx")

    train.columns = train.columns.str.strip()

    st.subheader("Dataset Preview")
    st.dataframe(train.head())

    st.subheader("Price Distribution")
    st.bar_chart(train["Price"])

    st.subheader("Age of Car vs Price")
    chart1 = train[["Age of car", "Price"]].set_index("Age of car")
    st.line_chart(chart1)

    st.subheader("Distance vs Price")
    chart2 = train[["Distance", "Price"]].set_index("Distance")
    st.line_chart(chart2)

    st.subheader("Engine Power Distribution")
    st.bar_chart(train["engine_power"])

    if uploaded_file is not None:

        if uploaded_file.name.endswith(".csv"):
            batch_data = pd.read_csv(uploaded_file)
        else:
            batch_data = pd.read_excel(uploaded_file)

        st.write("Uploaded Data")
        st.dataframe(batch_data.head())

        if st.button("Predict Batch"):

            predictions = model.predict(batch_data)

            batch_data["Predicted Price"] = predictions

            st.success("Prediction Completed")

            st.dataframe(batch_data.head())

            csv = batch_data.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Predictions",
                csv,
                "Predicted_Cars.csv",
                "text/csv"
            )

if predict:

    input_data = pd.DataFrame({
        "ID": [0],
        "Maker": [maker],
        "model": [car_model],
        "Location": [location],
        "Distance": [distance],
        "Owner Type": [owner_type],
        "manufacture_year": [manufacture_year],
        "Age of car": [age_of_car],
        "engine_displacement": [engine_displacement],
        "engine_power": [engine_power],
        "body_type": [body_type],
        "Vroom Audit Rating": [vroom_audit_rating],
        "transmission": [transmission],
        "door_count": [door_count],
        "seat_count": [seat_count],
        "fuel_type": [fuel_type]
    })

    prediction = model.predict(input_data)

    st.success(f"Predicted Price: ₹ {prediction[0]:,.2f}")

    lower = prediction[0] * 0.95
    upper = prediction[0] * 1.05

    st.info(f"Estimated Fair Price Range: ₹ {lower:,.2f} - ₹ {upper:,.2f}")