
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "customer.csv"
MODEL_PATH = BASE_DIR / "churn_model.joblib"


def load_data():
    """Load the customer dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)

    required_columns = [
        "tenure",
        "MonthlyCharges",
        "Contract",
        "Churn",
    ]

    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    if data[required_columns].isnull().any().any():
        raise ValueError("Dataset contains missing values in required columns.")

    if not data["Churn"].isin(["Yes", "No"]).all():
        raise ValueError("Churn must contain only 'Yes' or 'No'.")

    return data


def build_model():
    """Create the preprocessing and classification pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "contract",
                OneHotEncoder(handle_unknown="ignore"),
                ["Contract"],
            )
        ],
        remainder="passthrough",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=3,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def train_model():
    """Train, evaluate, and save the churn model."""
    data = load_data()

    features = ["tenure", "MonthlyCharges", "Contract"]
    X = data[features]
    y = data["Churn"].map({"No": 0, "Yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nCustomer Churn Model Evaluation")
    print("-" * 35)
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No Churn", "Churn"],
            zero_division=0,
        )
    )

    # Save the complete pipeline, including preprocessing.
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved successfully: {MODEL_PATH}")

    return model


if __name__ == "__main__":
    train_model()
