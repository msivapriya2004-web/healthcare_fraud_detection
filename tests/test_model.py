import os
import pandas as pd
import pytest
from model.train import train_model
from model.predict import make_prediction
from app.schemas import ClaimInput


@pytest.fixture(scope="session", autouse=True)
def setup_dummy_data():
    """Creates a temporary data directory and CSV files for testing."""
    os.makedirs("data", exist_ok=True)

    # Create a minimal version of the expected Train CSV
    dummy_df = pd.DataFrame(
        {
            "Provider": ["PRV51001", "PRV51003"],
            "InscClaimAmtReimbursed": [500, 200],
            "DeductibleAmtPaid": [50, 10],
            "PotentialFraud": ["Yes", "No"],
        }
    )

    # Create the files your train.py looks for
    dummy_df.to_csv("data/Train-1542865627584.csv", index=False)
    dummy_df.to_csv("data/Train_Inpatientdata-1542865627584.csv", index=False)
    dummy_df.to_csv("data/Train_Outpatientdata-1542865627584.csv", index=False)
    dummy_df.to_csv("data/Train_Beneficiarydata-1542865627584.csv", index=False)


def test_train_logic():
    """Tests the training script and ensures model file is created."""
    train_model()
    assert os.path.exists("model/trained_model.joblib")


def test_prediction_logic():
    """Tests that the prediction function returns expected types."""
    sample_input = ClaimInput(
        InscClaimAmtReimbursed=300.0, DeductibleAmtPaid=20.0, IsInpatient=1
    )
    is_fraud, prob = make_prediction(sample_input)

    assert isinstance(is_fraud, int)
    assert 0.0 <= prob <= 1.0
