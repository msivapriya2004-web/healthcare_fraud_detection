import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os


def train_model():
    # Load Datasets [cite: 33-35]
    train_fraud = pd.read_csv("data/Train-1542865627584.csv")
    # train_beneficiary = pd.read_csv("data/Train_Beneficiarydata-1542865627584.csv")
    train_inpatient = pd.read_csv("data/Train_Inpatientdata-1542865627584.csv")
    train_outpatient = pd.read_csv("data/Train_Outpatientdata-1542865627584.csv")

    # Simple merging logic for the example [cite: 32]
    # In a full pipeline, you would aggregate claims by Provider
    train_inpatient["IsInpatient"] = 1
    train_outpatient["IsInpatient"] = 0
    all_claims = pd.concat([train_inpatient, train_outpatient])

    # Merge with Fraud labels [cite: 36, 37]
    data = pd.merge(all_claims, train_fraud, on="Provider", how="inner")

    # Feature Selection & Preprocessing [cite: 41-43]
    # Selecting numerical columns for this baseline
    features = ["InscClaimAmtReimbursed", "DeductibleAmtPaid", "IsInpatient"]
    data[features] = data[features].fillna(0)

    X = data[features]
    y = data["PotentialFraud"].map({"Yes": 1, "No": 0})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Random Forest Model [cite: 47-52]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Model Persistence [cite: 18, 102]
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/trained_model.joblib")
    print("Model trained and saved successfully.")


if __name__ == "__main__":
    train_model()
