import pytest
from fastapi.testclient import TestClient
from app.main import app
import os
from model.train import train_model

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def ensure_model_exists():
    """Ensures the model is trained before running API tests."""
    if not os.path.exists('model/trained_model.joblib'):
        # This creates the dummy data needed by train_model
        os.makedirs('data', exist_ok=True)
        import pandas as pd
        df = pd.DataFrame({
            'Provider': ['P1', 'P2'],
            'InscClaimAmtReimbursed': [100, 200],
            'DeductibleAmtPaid': [10, 20],
            'PotentialFraud': ['Yes', 'No'],
            'Beneficiary_Age': [30, 40]
        })
        df.to_csv('data/Train-1542865627584.csv', index=False)
        df.to_csv('data/Train_Inpatientdata-1542865627584.csv', index=False)
        df.to_csv('data/Train_Outpatientdata-1542865627584.csv', index=False)
        df.to_csv('data/Train_Beneficiarydata-1542865627584.csv', index=False)
        
        train_model() # [cite: 45, 100]

def test_health():
    response = client.get("/health") # [cite: 89]
    assert response.status_code == 200

def test_predict_success():
    payload = {
        "InscClaimAmtReimbursed": 100.0,
        "DeductibleAmtPaid": 50.0,
        "IsInpatient": 1,
    }
    response = client.post("/predict", json=payload) # [cite: 86, 90]
    # This will now be 200 because the model file exists
    assert response.status_code == 200 
    assert "is_fraud" in response.json()
