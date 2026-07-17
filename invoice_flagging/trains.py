import joblib
from pathlib import Path

from data_preprocessing import (
    load_invoice_data,
    apply_label,
    split_data
)

from modeling_evaluation import (
    train_random_forest,
    evaluate_classifier
)


DB_PATH = r"D:\invoice intelligence\data\inventory.db"

MODEL_DIR = Path("model")

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_items_quantity",
    "total_item_dollars"
]

TARGET = "flag_invoice"


def main():

    print("=" * 60)
    print("Invoice Flagging Model Training")
    print("=" * 60)

    # ---------------------------------------------------
    # Create model directory
    # ---------------------------------------------------
    MODEL_DIR.mkdir(exist_ok=True)

    # ---------------------------------------------------
    # Load data
    # ---------------------------------------------------
    print("\nLoading data...")

    df = load_invoice_data(DB_PATH)

    print(f"Total records : {len(df)}")

    # ---------------------------------------------------
    # Create labels & features
    # ---------------------------------------------------
    print("Creating labels...")

    df = apply_label(df)

    print(df[TARGET].value_counts())

    # ---------------------------------------------------
    # Train/Test Split
    # ---------------------------------------------------
    print("\nSplitting dataset...")

    X_train, X_test, y_train, y_test = split_data(
        df,
        FEATURES,
        TARGET
    )

    print(f"Training Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")

    # ---------------------------------------------------
    # Train Model
    # ---------------------------------------------------
    print("\nTraining Random Forest...")

    model = train_random_forest(
        X_train,
        y_train
    )

    # ---------------------------------------------------
    # Evaluate Model
    # ---------------------------------------------------
    print("\nEvaluating model...\n")

    results = evaluate_classifier(
        model,
        X_test,
        y_test,
        FEATURES
    )

    # ---------------------------------------------------
    # Save Model
    # ---------------------------------------------------
    model_path = MODEL_DIR / "predict_flag_invoice.pkl"

    joblib.dump(
        model.best_estimator_,
        model_path
    )

    print("\nModel Saved Successfully")

    print(model_path)

    # ---------------------------------------------------
    # Save Feature Importance
    # ---------------------------------------------------
    feature_path = MODEL_DIR / "feature_importance.csv"

    results["feature_importance"].to_csv(
        feature_path,
        index=False
    )

    print("Feature Importance Saved")

    print(feature_path)

    print("\nTraining Completed Successfully")


if __name__ == "__main__":
    main()