import pytest
from fastapi.testclient import TestClient
from app.main import app
import os
import pandas as pd
from model.train import train_model

client = TestClient(app)

@pytest.fixture(scope="session", autouse=True)
def prepare_environment():
    """Sets up dummy data and trains the model once for the whole session."""
    os.makedirs('data', exist_ok=True)
    
    # We create unique providers for each specific file to prevent 
    # Pandas from adding _x or _y suffixes during the merge.
    inpatient_df = pd.DataFrame({
        'Provider': ['PRV101', 'PRV102'],
        'InscClaimAmtReimbursed': [1000, 2000],
        'DeductibleAmtPaid': [100, 200],
        'Beneficiary_Age': [65, 70]
    })
    
    # Using the same columns but different data to ensure a clean join
    outpatient_df = pd.DataFrame({
        'Provider': ['PRV103', 'PRV104'],
        'InscClaimAmtReimbursed': [500, 100],
        'DeductibleAmtPaid': [50, 0],
        'Beneficiary_Age': [45, 50]
    })
    
    # Crucial: This fraud label file MUST have all the Providers mentioned above
    # and must have both 'Yes' and 'No' for the model to work.
    fraud_df = pd.DataFrame({
        'Provider': ['PRV101', 'PRV102', 'PRV103', 'PRV104'],
        'PotentialFraud': ['Yes', 'No', 'Yes', 'No']
    })

    # Save files with the exact names train.py expects
    fraud_df.to_csv('data/Train-1542865627584.csv', index=False)
    inpatient_df.to_csv('data/Train_Inpatientdata-1542865627584.csv', index=False)
    outpatient_df.to_csv('data/Train_Outpatientdata-1542865627584.csv', index=False)
    # Beneficiary data is also required by your train script
    inpatient_df.to_csv('data/Train_Beneficiarydata-1542865627584.csv', index=False)

    # Train the model so trained_model.joblib is ready for the API
    train_model()

def test_health():
    """Verify the health endpoint is reachable."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "up and running"}

def test_predict_success():
    """Verify the prediction endpoint works with the trained model."""
    payload = {
        "InscClaimAmtReimbursed": 100.0,
        "DeductibleAmtPaid": 50.0,
        "IsInpatient": 1,
    }
    response = client.post("/predict", json=payload)
    # If the model trained successfully, this will be 200
    assert response.status_code == 200
    assert "is_fraud" in response.json()
    assert "probability" in response.json()
