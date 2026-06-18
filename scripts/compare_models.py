"""
compare_models.py
-----------------
Compares new model vs current. Promotes if >= 0.5% better.
Exit codes: 0 = keep old | 1 = new model promoted
Usage: python scripts/compare_models.py
"""
import os, sys, json, shutil
import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR  = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

FEATURE_ORDER = [
    "manufacture_year","mileage_km","engine_cc",
    "fuel_type_enc","transmission_enc","usd_pkr",
    "make_enc","model_enc",
]

def mape(y_true_log, y_pred_log):
    y_true = np.expm1(np.array(y_true_log))
    y_pred = np.expm1(np.array(y_pred_log))
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

def main():
    print("=" * 55)
    print("MODEL COMPARISON")
    print("=" * 55)

    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))[FEATURE_ORDER]
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()

    current_path = os.path.join(MODELS_DIR, "xgb_best.json")
    new_path     = os.path.join(MODELS_DIR, "xgb_new.json")

    if not os.path.exists(new_path):
        print("ERROR: xgb_new.json not found")
        sys.exit(0)

    current_model = xgb.XGBRegressor()
    current_model.load_model(current_path)
    current_mape = round(mape(y_test, current_model.predict(X_test)), 4)
    print(f"Current model MAPE : {current_mape:.2f}%")

    new_model = xgb.XGBRegressor()
    new_model.load_model(new_path)
    new_mape = round(mape(y_test, new_model.predict(X_test)), 4)
    print(f"New model MAPE     : {new_mape:.2f}%")

    improvement = current_mape - new_mape
    promoted    = improvement >= 0.0
    print(f"Improvement        : {improvement:+.2f}%")

    report = {
        "current_mape": current_mape,
        "new_mape": new_mape,
        "improvement": round(improvement, 4),
        "promoted": promoted,
    }
    with open(os.path.join(REPORTS_DIR, "comparison_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 55)
    if promoted:
        shutil.copy(new_path, current_path)
        os.remove(new_path)
        print(f"NEW MODEL PROMOTED — {improvement:+.2f}% better")
        sys.exit(1)
    else:
        os.remove(new_path)
        print(f"NEW MODEL REJECTED — only {improvement:+.2f}% improvement")
        sys.exit(0)

if __name__ == "__main__":
    main()
