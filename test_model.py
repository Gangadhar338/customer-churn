import unittest

import pandas as pd

from model import DATA_PATH, load_data, build_model


class TestCustomerChurn(unittest.TestCase):

    def test_dataset_exists(self):
        self.assertTrue(DATA_PATH.exists())

    def test_required_columns(self):
        data = load_data()
        expected = {"tenure", "MonthlyCharges", "Contract", "Churn"}
        self.assertTrue(expected.issubset(data.columns))

    def test_both_churn_classes_exist(self):
        data = load_data()
        self.assertEqual(set(data["Churn"].unique()), {"Yes", "No"})

    def test_model_predicts(self):
        data = load_data()
        X = data[["tenure", "MonthlyCharges", "Contract"]]
        y = data["Churn"].map({"No": 0, "Yes": 1})

        model = build_model()
        model.fit(X, y)

        sample = pd.DataFrame([{
            "tenure": 12,
            "MonthlyCharges": 70.0,
            "Contract": "One year",
        }])

        prediction = model.predict(sample)
        self.assertIn(int(prediction[0]), [0, 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)