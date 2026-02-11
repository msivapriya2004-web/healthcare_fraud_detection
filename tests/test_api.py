from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "up and running"}


def test_predict_success():
    payload = {
        "InscClaimAmtReimbursed": 100.0,
        "DeductibleAmtPaid": 50.0,
        "IsInpatient": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "is_fraud" in response.json()


def test_predict_invalid_data():
    # Testing how the API handles missing fields to increase coverage
    response = client.post("/predict", json={"InscClaimAmtReimbursed": 100.0})
    assert response.status_code == 422
