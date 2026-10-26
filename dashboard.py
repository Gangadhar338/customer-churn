
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered",
)

st.title("Customer Churn Prediction")
st.write("Predict whether a customer may leave the service.")

st.subheader("Customer Details")

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=1000,
    value=2,
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=100000.0,
    value=70.5,
    step=5.0,
)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"],
)

if st.button("Predict Churn", type="primary"):
    payload = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "Contract": contract,
    }

    try:
        response = requests.post(
            API_URL,
            json=payload,
            timeout=10,
        )

        response.raise_for_status()
        result = response.json()

        if "error" in result:
            st.error(result["error"])
        else:
            st.subheader("Prediction Result")

            if result["churn_prediction"] == "Yes":
                st.warning("The model predicts that this customer may churn.")
            else:
                st.success("The model predicts that this customer will stay.")

            st.metric(
                "Estimated Churn Probability",
                f"{result['churn_probability']:.1%}",
            )

            st.info(result["message"])

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to the API. Start the API server first "
            "using: python -m uvicorn api:app --reload"
        )
    except requests.exceptions.RequestException as exc:
        st.error(f"API request failed: {exc}")
