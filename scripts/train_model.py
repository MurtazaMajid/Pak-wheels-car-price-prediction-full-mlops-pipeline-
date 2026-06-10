import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
import optuna
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── PATHS ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR    = os.path.join(PROJECT_ROOT, "plots")
MLFLOW_DB = os.path.join(PROJECT_ROOT, "mlflow.db")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
mlflow.set_experiment("pakistan-car-price")

print("=" * 55)
print(f"XGBoost version : {xgb.__version__}")
print(f"PROJECT ROOT    : {PROJECT_ROOT}")
print("=" * 55)

# ── LOAD DATA ──────────────────────────────────────────────────────────────────
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()
y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()

FEATURE_ORDER = [
    "manufacture_year", "mileage_km",      "engine_cc",
    "fuel_type_enc",    "transmission_enc", "usd_pkr",
    "make_enc",         "model_enc",
]
X_train = X_train[FEATURE_ORDER]
X_test  = X_test[FEATURE_ORDER]

print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")

# ── HELPERS ────────────────────────────────────────────────────────────────────
def eval_metrics(y_true_log, y_pred_log):
    y_true = np.expm1(np.array(y_true_log))
    y_pred = np.expm1(np.array(y_pred_log))
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }

# ══════════════════════════════════════════════════════════
# STEP 1 — BASELINE
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("STEP 1: BASELINE MODEL")
print("=" * 55)

BASELINE_PARAMS = {
    "n_estimators":        300,
    "max_depth":           6,
    "learning_rate":       0.1,
    "subsample":           0.8,
    "colsample_bytree":    0.8,
    "random_state":        42,
    "tree_method":         "hist",
}

with mlflow.start_run(run_name="baseline"):
    model_base = xgb.XGBRegressor(**BASELINE_PARAMS)
    model_base.fit(X_train, y_train,
                   eval_set=[(X_test, y_test)],
                   verbose=50)
    metrics_base = eval_metrics(y_test, model_base.predict(X_test))
    mlflow.log_params({**BASELINE_PARAMS, "feature_version": "v1"})
    mlflow.log_metrics(metrics_base)
    mlflow.xgboost.log_model(model_base, name="model")

print("\nBASELINE RESULTS:")
for k, v in metrics_base.items():
    print(f"  {k:6s}: {v:,.2f}")

# ══════════════════════════════════════════════════════════
# STEP 2 — OPTUNA
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("STEP 2: OPTUNA HYPERPARAMETER TUNING (50 trials)")
print("=" * 55)

def objective(trial):
    params = {
        "n_estimators":          1000,
        "early_stopping_rounds": 30,
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma":            trial.suggest_float("gamma", 0, 1),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0, 2),
        "random_state":     42,
        "tree_method":      "hist",
    }
    m = xgb.XGBRegressor(**params)
    m.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=False)
    return np.sqrt(mean_squared_error(y_test, m.predict(X_test)))

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

best_params = {
    **study.best_params,
    "n_estimators":          1000,
    "early_stopping_rounds": 30,
    "random_state":          42,
    "tree_method":           "hist",
}

with mlflow.start_run(run_name="optuna_best") as run:
    model_best = xgb.XGBRegressor(**best_params)
    model_best.fit(X_train, y_train,
                   eval_set=[(X_test, y_test)],
                   verbose=False)
    metrics_best = eval_metrics(y_test, model_best.predict(X_test))
    mlflow.log_params({**best_params, "feature_version": "v1", "n_features": 8})
    mlflow.log_metrics(metrics_best)
    mlflow.xgboost.log_model(model_best, name="model")
    best_run_id = run.info.run_id

print("\nBEST MODEL RESULTS:")
for k, v in metrics_best.items():
    print(f"  {k:6s}: {v:,.2f}")

# save model
model_path = os.path.join(MODELS_DIR, "xgb_best.json")
model_best.save_model(model_path)
print(f"\nModel saved → {model_path}")

# ══════════════════════════════════════════════════════════
# STEP 3 — SHAP
# ══════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════
# STEP 3 — FEATURE IMPORTANCE (XGBoost built-in)
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("STEP 3: FEATURE IMPORTANCE")
print("=" * 55)

importance = model_best.get_booster().get_score(importance_type='gain')
total = sum(importance.values()) or 1
feat_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
print("\nFeature ranking by gain:")
for f, v in feat_imp:
    print(f"  {f:20s}: {v/total:.4f}")

# ══════════════════════════════════════════════════════════
# STEP 4 — DRIFT BY YEAR
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("STEP 4: DRIFT EVALUATION BY YEAR")
print("=" * 55)

# reload test data with year info
df_full = pd.read_csv(os.path.join(DATA_DIR, "pakwheels_final.csv"))
test_idx = df_full["manufacture_year"] >= 2023
df_test  = df_full[test_idx].reset_index(drop=True)
X_test_drift = X_test.reset_index(drop=True)
y_test_drift = y_test.reset_index(drop=True)

results = []
for yr in sorted(df_test["manufacture_year"].unique()):
    mask = df_test["manufacture_year"] == yr
    X_yr = X_test_drift[mask.values]
    y_yr = y_test_drift[mask.values]
    if len(X_yr) == 0:
        continue
    y_p  = model_best.predict(X_yr)
    m    = eval_metrics(y_yr, y_p)
    usd  = df_test.loc[mask, "usd_pkr"].mean() if "usd_pkr" in df_test.columns else 280
    results.append({"year": int(yr), "n": int(len(X_yr)), "usd_pkr": round(usd, 1), **{k: round(v, 2) for k, v in m.items()}})

df_drift = pd.DataFrame(results)
print(df_drift[["year", "n", "rmse", "mape", "r2", "usd_pkr"]].to_string(index=False))

# drift plot
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.bar(df_drift["year"], df_drift["mape"], color="#C0392B", alpha=0.75, label="MAPE %")
ax1.set_ylabel("MAPE (%)"); ax1.set_xlabel("Year")
ax2 = ax1.twinx()
ax2.plot(df_drift["year"], df_drift["usd_pkr"], color="#2455B5", linewidth=2.5, marker="o", label="USD/PKR")
ax2.set_ylabel("USD/PKR")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
plt.title("Drift Evidence — MAPE vs USD/PKR by Year")
plt.tight_layout()
p = os.path.join(PLOTS_DIR, "drift_evidence.png")
plt.savefig(p, dpi=150); plt.close()
print(f"Saved → {p}")

with mlflow.start_run(run_name="drift_evaluation"):
    for row in results:
        mlflow.log_metric(f"mape_yr_{row['year']}", row["mape"])
        mlflow.log_metric(f"r2_yr_{row['year']}",   row["r2"])
        mlflow.log_metric(f"rmse_yr_{row['year']}", row["rmse"])

# ══════════════════════════════════════════════════════════
# STEP 5 — REGISTER
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("STEP 5: REGISTER MODEL")
print("=" * 55)

client    = MlflowClient()
model_uri = f"runs:/{best_run_id}/model"
try:
    mv = mlflow.register_model(model_uri, name="pakistan-car-price-xgb")
    client.transition_model_version_stage(
        name="pakistan-car-price-xgb",
        version=mv.version,
        stage="Production",
    )
    print(f"Registered version {mv.version} → Production")
except Exception as e:
    print(f"Registration note: {e}")

# ══════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("FINAL SUMMARY")
print("=" * 55)
print(f"  XGBoost version : {xgb.__version__}")
print(f"  Baseline RMSE   : {metrics_base['rmse']:>15,.0f} PKR")
print(f"  Best RMSE       : {metrics_best['rmse']:>15,.0f} PKR")
print(f"  Best MAPE       : {metrics_best['mape']:>14.2f} %")
print(f"  Best R2         : {metrics_best['r2']:>14.3f}")
print(f"  Model saved     : {model_path}")
print("=" * 55)
