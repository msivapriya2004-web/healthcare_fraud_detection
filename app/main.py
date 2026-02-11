from fastapi import FastAPI, HTTPException
from app.schemas import ClaimInput, FraudPrediction
import joblib
import os

app = FastAPI(title="Healthcare Fraud Detection Service")

MODEL_PATH = "model/trained_model.joblib"


@app.get("/health")
def health():
    return {"status": "up and running"}


@app.post("/predict", response_model=FraudPrediction)
def predict(data: ClaimInput):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=500, detail="Model file not found. Train the model first."
        )

    model = joblib.load(MODEL_PATH)
    features = [[data.InscClaimAmtReimbursed, data.DeductibleAmtPaid, data.IsInpatient]]

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return {"is_fraud": int(prediction), "probability": float(probability)}
