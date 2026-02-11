import os
import pandas as pd
import pytest
#import joblib
from model.train import train_model
from model.predict import make_prediction
from app.schemas import ClaimInput

@pytest.fixture(scope="session", autouse=True)
def setup_dummy_data():
    """Creates temporary data with the EXACT columns train.py expects."""
    os.makedirs('data', exist_ok=True)
    
    # These columns MUST match what is in your train.py merging/cleaning logic
    dummy_df = pd.DataFrame({
        'Provider': ['PRV51001', 'PRV51001'],
        'InscClaimAmtReimbursed': [500, 200],
        'DeductibleAmtPaid': [50, 10],
        'PotentialFraud': ['Yes', 'No'],
        'Beneficiary_Age': [65, 70] # Add any other columns your train.py uses
    })
    
    # Create all four files to satisfy the pd.read_csv calls in train.py [cite: 33-35]
    dummy_df.to_csv('data/Train-1542865627584.csv', index=False)
    dummy_df.to_csv('data/Train_Inpatientdata-1542865627584.csv', index=False)
    dummy_df.to_csv('data/Train_Outpatientdata-1542865627584.csv', index=False)
    dummy_df.to_csv('data/Train_Beneficiarydata-1542865627584.csv', index=False)

def test_train_logic():
    """Tests the training script. This must pass to create the .joblib file."""
    train_model()
    # Ensure the directory and file exist after training [cite: 18, 102]
    assert os.path.exists('model/trained_model.joblib')

def test_prediction_logic():
    """Tests prediction logic once the model file exists."""
    # Ensure model is there
    if not os.path.exists('model/trained_model.joblib'):
        train_model()
        
    sample_input = ClaimInput(
        InscClaimAmtReimbursed=300.0,
        DeductibleAmtPaid=20.0,
        IsInpatient=1
    )
    is_fraud, prob = make_prediction(sample_input)
    
    assert is_fraud in [0, 1]
    assert 0.0 <= prob <= 1.0
