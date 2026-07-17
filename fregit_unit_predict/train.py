import joblib
from pathlib import Path

from data_preprocessing import *
from model_evalution import *

def main():
    db_path = r"D:\invoice intelligence\data\inventory.db"
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    #load data
    df = load_vendor_invoice_data(db_path)

    #prepare data
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    #train models
    lr_model = train_linear_regression(X_train, y_train)
    dt_model = train_decision_tree(X_train, y_train, max_depth=10)
    rf_model = train_random_forest(X_train, y_train, max_depth=10)

    #evaluate models
    result = []
    result.append(evaluate_model(lr_model, X_test, y_test, "Linear Regression"))
    result.append(evaluate_model(dt_model, X_test, y_test, "Decision Tree"))
    result.append(evaluate_model(rf_model, X_test, y_test, "Random Forest"))

    #select best model (lowest Mae)
    best_model_info = min(result, key=lambda x: x['mae'])
    best_model_name = best_model_info['model_name']

    best_model = {
        "Linear Regression": lr_model,
        "Decision Tree": dt_model,
        "Random Forest": rf_model
    }[best_model_name]

    #save best model
    model_path = model_dir / "predict_freight_model.pkl"
    joblib.dump(best_model, model_path)

    print(f'\nBest model name: {best_model_name}')
    print(f'Best model path: {model_path}')

if __name__ == "__main__":
    main()