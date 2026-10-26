
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "churn_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model not found. Run python model.py first."
    )

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn from customer details.",
    version="1.0.0",
)


class Customer(BaseModel):
    tenure: int = Field(ge=0, le=1000)
    MonthlyCharges: float = Field(ge=0, le=100000)
    Contract: str


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}


@app.post("/predict")
def predict_churn(customer: Customer):
    if customer.Contract not in [
        "Month-to-month",
        "One year",
        "Two year",
    ]:
        return {
            "error": "Contract must be Month-to-month, One year, or Two year"
        }

    customer_data = pd.DataFrame(
        [{
            "tenure": customer.tenure,
            "MonthlyCharges": customer.MonthlyCharges,
            "Contract": customer.Contract,
        }]
    )

    prediction = int(model.predict(customer_data)[0])
    probability = model.predict_proba(customer_data)[0][1]

    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(float(probability), 4),
        "message": (
            "Customer may leave; consider retention support."
            if prediction == 1
            else "Model predicts this customer will stay."
        ),
    }
