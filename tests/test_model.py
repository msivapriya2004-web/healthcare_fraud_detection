import os
import pandas as pd
import pytest
from model.train import train_model
from model.predict import make_prediction
from app.schemas import ClaimInput

@pytest.fixture(scope="session", autouse=True)
def setup_dummy_data():
    """Creates temporary data with columns that survive merging."""
    os.makedirs('data', exist_ok=True)
    
    # We create unique providers for each file to prevent column suffixing (_x, _y)
    # during the inner join merge in train.py
    inpatient_df = pd.DataFrame({
        'Provider': ['PRV51001'],
        'InscClaimAmtReimbursed': [500],
        'DeductibleAmtPaid': [50],
        'Beneficiary_Age': [65]
    })
    
    outpatient_df = pd.DataFrame({
        'Provider': ['PRV51002'],
        'InscClaimAmtReimbursed': [200],
        'DeductibleAmtPaid': [10],
        'Beneficiary_Age': [70]
    })
    
    fraud_df = pd.DataFrame({
        'Provider': ['PRV51001', 'PRV51002'],
        'PotentialFraud': ['Yes', 'No']
    })

    # Save exactly as named in your train.py read_csv calls
    inpatient_df.to_csv('data/Train_Inpatientdata-1542865627584.csv', index=False)
    outpatient_df.to_csv('data/Train_Outpatientdata-1542865627584.csv', index=False)
    fraud_df.to_csv('data/Train-1542865627584.csv', index=False)
    # Re-use outpatient as dummy beneficiary data
    outpatient_df.to_csv('data/Train_Beneficiarydata-1542865627584.csv', index=False)

def test_train_logic():
    """Tests the training script. Successful run creates the joblib file."""
    train_model()
    assert os.path.exists('model/trained_model.joblib')

def test_prediction_logic():
    """Tests prediction logic once the model file exists."""
    sample_input = ClaimInput(
        InscClaimAmtReimbursed=300.0,
        DeductibleAmtPaid=20.0,
        IsInpatient=1
    )
    is_fraud, prob = make_prediction(sample_input)
    assert is_fraud in [0, 1]
    assert 0.0 <= prob <= 1.0
