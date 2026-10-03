"""
Customer Churn Prediction - Predict on a new customer
--------------------------------------------------------
Loads the saved trained pipeline and predicts on a sample customer.
Edit the `new_customer` dictionary below to try your own values.

Run from the project root (after running train.py at least once):
    python src/predict.py
"""

import pandas as pd
import joblib

MODEL_PATH = "models/churn_model.pkl"


def main():
    model = joblib.load(MODEL_PATH)

    new_customer = pd.DataFrame([{
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.0,
        "TotalCharges": 475.0,
    }])

    prediction = model.predict(new_customer)[0]
    probability = model.predict_proba(new_customer)[0]

    result = "Likely to Churn" if prediction == 1 else "Likely to Stay"
    churn_probability = probability[1]

    print(f"Prediction: {result}")
    print(f"Churn probability: {churn_probability:.2%}")


if __name__ == "__main__":
    main()
