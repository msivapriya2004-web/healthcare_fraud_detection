import joblib
import os
import pandas as pd

# Path to the saved model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained_model.joblib")


def make_prediction(input_data):
    """
    Loads the model and returns the fraud prediction and probability.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Trained model file not found at model/trained_model.joblib"
        )

    model = joblib.load(MODEL_PATH)

    # Prepare features in the exact order used during training [cite: 42, 101]
    features = pd.DataFrame(
        [
            {
                "InscClaimAmtReimbursed": input_data.InscClaimAmtReimbursed,
                "DeductibleAmtPaid": input_data.DeductibleAmtPaid,
                "IsInpatient": input_data.IsInpatient,
            }
        ]
    )

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return int(prediction), float(probability)
