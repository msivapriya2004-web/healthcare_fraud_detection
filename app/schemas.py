from pydantic import BaseModel


class ClaimInput(BaseModel):
    InscClaimAmtReimbursed: float
    DeductibleAmtPaid: float
    IsInpatient: int  # 1 for Inpatient, 0 for Outpatient


class FraudPrediction(BaseModel):
    is_fraud: int
    probability: float
