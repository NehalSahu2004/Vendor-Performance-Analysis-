import joblib
import pandas as pd
from pathlib import Path

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(r"D:\invoice intelligence")

MODEL_PATH = (
    PROJECT_ROOT
    / "invoice_flagging"
    / "model"
    / "predict_flag_invoice.pkl"
)

SCALER_PATH = (
    PROJECT_ROOT
    / "invoice_flagging"
    / "model"
    / "scaler.pkl"
)

# ==========================================================
# Load Model
# ==========================================================

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# ==========================================================
# Load Scaler
# ==========================================================

def load_scaler():

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found:\n{SCALER_PATH}"
        )

    return joblib.load(SCALER_PATH)


# ==========================================================
# Predict Invoice Flag
# ==========================================================


def predict_invoice_flag(input_data):

    model = load_model()
    scaler = load_scaler()

    input_df = pd.DataFrame(input_data)

    required_columns = [
        "invoice_quantity",
        "invoice_dollars",
        "Freight",
        "total_items_quantity",
        "total_item_dollars"
    ]

    missing = [
        col for col in required_columns
        if col not in input_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Keep the same order used during training
    input_df = input_df[required_columns]

    input_scaled = scaler.transform(input_df)

    input_scaled = pd.DataFrame(
        input_scaled,
        columns=input_df.columns
    )

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    result = input_df.copy()

    result["Prediction"] = [
        "Flagged" if p == 1 else "Normal"
        for p in prediction
    ]

    result["Confidence"] = probability.max(axis=1).round(3)

    return result

# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    sample_invoice = {

        "invoice_quantity": [100],
        "invoice_dollars": [25000],
        "Freight": [180],
        "total_items_quantity": [100],
        "total_item_dollars": [24990]

    }

    prediction = predict_invoice_flag(sample_invoice)

    print(prediction)