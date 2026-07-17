import joblib
import pandas as pd
from pathlib import Path

# ---------------------------------------------------
# Paths
# ---------------------------------------------------

# This file is inside:
# freight_unit_predict/inference/predict_freight.py
#
# So parent.parent points to:
# freight_unit_predict/

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "fregit_unit_predict"
    / "models"
    / "predict_freight_model.pkl"
)


# ---------------------------------------------------
# Load Model
# ---------------------------------------------------

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# ---------------------------------------------------
# Predict Freight
# ---------------------------------------------------

def predict_freight_cost(input_data):
    """
    Predict freight cost.

    Parameters
    ----------
    input_data : dict

    Example
    -------
    {
        "Dollars": [18500, 9000]
    }
    """

    model = load_model()

    input_df = pd.DataFrame(input_data)

    required_columns = ["Dollars"]

    missing = [
        col for col in required_columns
        if col not in input_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    predictions = model.predict(input_df)

    result = input_df.copy()
    result["Predicted_Freight"] = predictions.round(2)

    return result


# ---------------------------------------------------
# Main
# ---------------------------------------------------

if __name__ == "__main__":
    sample_data = {
        "Dollars": [18500, 9000]
    }

    prediction = predict_freight_cost(sample_data)

    print("\nPrediction")
    print(prediction)