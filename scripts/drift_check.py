"""
drift_check.py
--------------
Checks if model performance has drifted beyond threshold.
Exit codes: 0 = no drift | 1 = drift detected
Usage: python scripts/drift_check.py --threshold 12.0
"""
import os, sys, argparse, json
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=12.0)
    args = parser.parse_args()

    print("=" * 55)
    print("DRIFT CHECK")
    print(f"Threshold : {args.threshold}% MAPE")
    print("=" * 55)

    model_path = os.path.join(MODELS_DIR, "xgb_best.json")
    if not os.path.exists(model_path):
        print("ERROR: model file not found — triggering retrain")
        sys.exit(1)

    model = xgb.XGBRegressor()
    model.load_model(model_path)

    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))[FEATURE_ORDER]
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()

    overall_mape = mape(y_test, model.predict(X_test))
    print(f"\nOverall MAPE : {overall_mape:.2f}%")

    df_full      = pd.read_csv(os.path.join(DATA_DIR, "pakwheels_final.csv"))
    df_test_full = df_full[df_full["manufacture_year"] >= 2023].reset_index(drop=True)
    X_test_r     = X_test.reset_index(drop=True)
    y_test_r     = y_test.reset_index(drop=True)

    yearly = {}
    print("\nPer-year MAPE:")
    for yr in sorted(df_test_full["manufacture_year"].unique()):
        mask = (df_test_full["manufacture_year"] == yr).values
        if mask.sum() == 0: continue
        m = mape(y_test_r[mask], model.predict(X_test_r[mask]))
        yearly[int(yr)] = round(m, 2)
        flag = "DRIFT" if m > args.threshold else "OK"
        print(f"  {yr}: {m:.2f}%  [{flag}]")

    drift_detected = overall_mape > args.threshold
    recent_years   = [yr for yr in yearly if yr >= 2024]
    if recent_years:
        recent_mape    = max(yearly[yr] for yr in recent_years)
        drift_detected = drift_detected or (recent_mape > args.threshold)

    report = {
        "overall_mape": round(overall_mape, 2),
        "threshold": args.threshold,
        "drift_detected": drift_detected,
        "yearly_mape": yearly,
    }
    with open(os.path.join(REPORTS_DIR, "drift_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 55)
    if drift_detected:
        print(f"DRIFT DETECTED — {overall_mape:.2f}% > {args.threshold}%")
        sys.exit(1)
    else:
        print(f"NO DRIFT — {overall_mape:.2f}% <= {args.threshold}%")
        sys.exit(0)

if __name__ == "__main__":
    main()
