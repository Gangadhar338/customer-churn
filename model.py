
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


DATA_PATH = Path(__file__).parent / "data" / "customer.csv"


def load_data():
    """Load the customer dataset."""
    return pd.read_csv(DATA_PATH)


def train_model():
    """Train and evaluate a customer churn classifier."""
    data = load_data()

    features = ["tenure", "MonthlyCharges", "Contract"]
    X = data[features]
    y = data["Churn"].map({"No": 0, "Yes": 1})

    preprocessor = ColumnTransformer(
        transformers=[
            ("contract", OneHotEncoder(handle_unknown="ignore"), ["Contract"])
        ],
        remainder="passthrough",
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(max_depth=3, random_state=42)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Customer Churn Model Evaluation")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["No Churn", "Churn"],
        zero_division=0,
    ))

    return model


if __name__ == "__main__":
    train_model()
