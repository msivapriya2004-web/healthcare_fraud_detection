import os
import pandas as pd
import pytest
from model.train import train_model
from model.predict import make_prediction
from app.schemas import ClaimInput

@pytest.fixture(scope="session", autouse=True)
def setup_dummy_data():
    """Creates temporary data with BOTH classes to prevent predict_proba index errors."""
    os.makedirs('data', exist_ok=True)
    
    # We provide distinct providers and DIFFERENT labels (Yes and No) [cite: 37]
    inpatient_df = pd.DataFrame({
        'Provider': ['PRV51001', 'PRV51002'],
        'InscClaimAmtReimbursed': [500, 100],
        'DeductibleAmtPaid': [50, 0],
        'Beneficiary_Age': [65, 30]
    })
    
    outpatient_df = pd.DataFrame({
        'Provider': ['PRV51003', 'PRV51004'],
        'InscClaimAmtReimbursed': [200, 50],
        'DeductibleAmtPaid': [10, 0],
        'Beneficiary_Age': [70, 25]
    })
    
    # Critical: Must have at least one 'Yes' and one 'No' 
    fraud_df = pd.DataFrame({
        'Provider': ['PRV51001', 'PRV51002', 'PRV51003', 'PRV51004'],
        'PotentialFraud': ['Yes', 'No', 'Yes', 'No']
    })

    # Save files exactly as train.py expects
    inpatient_df.to_csv('data/Train_Inpatientdata-1542865627584.csv', index=False)
    outpatient_df.to_csv('data/Train_Outpatientdata-1542865627584.csv', index=False)
    fraud_df.to_csv('data/Train-1542865627584.csv', index=False)
    # Re-use outpatient as beneficiary data
    outpatient_df.to_csv('data/Train_Beneficiarydata-1542865627584.csv', index=False)

def test_train_logic():
    """Ensures the model trains and saves the joblib file[cite: 16, 102]."""
    train_model()
    assert os.path.exists('model/trained_model.joblib')

def test_prediction_logic():
    """Tests prediction logic. Now works because model has two classes."""
    # Ensure model is fresh
    train_model()
    
    sample_input = ClaimInput(
        InscClaimAmtReimbursed=300.0,
        DeductibleAmtPaid=20.0,
        IsInpatient=1
    )
    is_fraud, prob = make_prediction(sample_input)
    
    assert is_fraud in [0, 1]
    assert 0.0 <= prob <= 1.0 
