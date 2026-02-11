import os

# import pytest
from model.train import train_model
from model.predict import make_prediction
from app.schemas import ClaimInput


def test_train_and_predict_flow():
    # Triggering the training script to ensure code coverage for train.py
    train_model()
    assert os.path.exists("model/trained_model.joblib")

    # Testing prediction logic
    sample = ClaimInput(
        InscClaimAmtReimbursed=10.0, DeductibleAmtPaid=0.0, IsInpatient=0
    )
    is_fraud, prob = make_prediction(sample)
    assert is_fraud in [0, 1]
    assert 0 <= prob <= 1.0
