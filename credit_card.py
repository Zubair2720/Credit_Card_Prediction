
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('random.pkl')

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to detect fraudulent transactions.")

st.divider()

# Get exact feature order from trained model
features = list(model.feature_names_in_)

# Input form
with st.form("fraud_form"):

    st.subheader("Transaction Details")

    input_data = {}

    # Display inputs in 4 columns
    cols = st.columns(4)

    for i, feature in enumerate(features):

        with cols[i % 4]:

            if feature == "Time":
                value = st.number_input(
                    "Time",
                    min_value=0.0,
                    value=0.0
                )

            elif feature == "Amount":
                value = st.number_input(
                    "Amount",
                    min_value=0.0,
                    value=100.0
                )

            else:
                value = st.number_input(
                    feature,
                    value=0.0,
                    format="%.6f"
                )

            input_data[feature] = value

    submit = st.form_submit_button(
        "Predict Transaction",
        use_container_width=True
    )

# Prediction
if submit:

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("🚨 Fraudulent Transaction Detected!")

    else:

        st.success("✅ Normal Transaction")

    # Prediction probability
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(input_df)[0]

        fraud_index = list(model.classes_).index(1)

        fraud_probability = probabilities[fraud_index] * 100

        st.metric(
            "Fraud Probability",
            f"{fraud_probability:.2f}%"
        )

        st.progress(float(fraud_probability / 100))
