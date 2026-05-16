import os
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

from feature_config import FUEL_TYPE_ENC, TRANSMISSION_ENC, FEATURE_ORDER

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(BASE_DIR, "xgb_best.json")
TRAIN_REF_PATH = os.path.join(BASE_DIR, "X_train.csv")

# ── APP ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Pakistan Car Price Prediction API",
    description=(
        "XGBoost model trained on 6,543 Pakwheels listings (2015-2022). "
        "Predicts used car price in PKR with MAPE ~10.5%. "
        "Includes SHAP explainability and drift monitoring."
    ),
    version="1.0.0",
)

# ── GLOBALS ───────────────────────────────────────────────────────────────────
BOOSTER        = None   # raw xgb.Booster — avoids base_score issues with XGBRegressor
EXPLAINER      = None
GLOBAL_MEAN    = 15.3144
PREDICTION_LOG = []
TRAIN_REF      = None

# ── STARTUP ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
def load_artifacts():
    global BOOSTER, EXPLAINER, GLOBAL_MEAN, TRAIN_REF

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}")

    # load as raw Booster — avoids XGBRegressor base_score rescaling bug in XGB 2.x
    BOOSTER = xgb.Booster()
    BOOSTER.load_model(MODEL_PATH)
    print(f"✅ Booster loaded from {MODEL_PATH}")

    # verify with a test prediction
    test_data = pd.DataFrame([{
        'manufacture_year': 2021, 'mileage_km': 45000, 'engine_cc': 1800,
        'fuel_type_enc': 0, 'transmission_enc': 1, 'usd_pkr': 280,
        'make_enc': 15.3144, 'model_enc': 15.3144
    }])[FEATURE_ORDER]
    test_dmatrix = xgb.DMatrix(test_data)
    test_pred    = BOOSTER.predict(test_dmatrix)
    test_price   = int(np.expm1(float(test_pred[0])))
    print(f"✅ Startup test prediction: PKR {test_price:,} (expected ~5,059,281)")

    if not os.path.exists(TRAIN_REF_PATH):
        raise RuntimeError(f"X_train.csv not found at {TRAIN_REF_PATH}")
    train_df    = pd.read_csv(TRAIN_REF_PATH)
    TRAIN_REF   = train_df[FEATURE_ORDER].copy()
    GLOBAL_MEAN = float(train_df["make_enc"].mean())
    print(f"✅ Training reference loaded — {len(TRAIN_REF)} rows, GLOBAL_MEAN={GLOBAL_MEAN:.4f}")

    background = TRAIN_REF.sample(n=min(100, len(TRAIN_REF)), random_state=42)
    EXPLAINER  = shap.TreeExplainer(BOOSTER, background)
    print("✅ SHAP explainer ready")
    print("✅ Startup complete — API is live")


# ── SCHEMAS ───────────────────────────────────────────────────────────────────
class CarFeatures(BaseModel):
    model_config = {"protected_namespaces": ()}

    manufacture_year: int             = Field(..., ge=1990, le=2026, example=2021)
    mileage_km:       int             = Field(..., ge=0,    le=500000, example=45000)
    engine_cc:        int             = Field(..., ge=600,  le=8000,   example=1800)
    fuel_type:        str             = Field(..., example="Petrol",
                                           description="Petrol | Diesel | Hybrid | Electric | CNG | LPG")
    transmission:     str             = Field(..., example="Automatic",
                                           description="Manual | Automatic | CVT | Semi-Auto")
    usd_pkr:          Optional[float] = Field(None, example=280.0,
                                           description="USD/PKR rate. Leave null to use 280.0")
    make_enc:         Optional[float] = Field(None, example=15.3144,
                                           description="Leave null to use global mean (~15.31)")
    model_enc:        Optional[float] = Field(None, example=15.3144,
                                           description="Leave null to use global mean (~15.31)")


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    predicted_price_pkr:       float
    predicted_price_formatted: str
    confidence_range:          dict
    shap_top_features:         list
    model_version:             str = "v1-xgb-8features"
    timestamp:                 str


# ── HELPERS ───────────────────────────────────────────────────────────────────
def build_feature_row(car: CarFeatures) -> pd.DataFrame:
    row = {
        "manufacture_year": car.manufacture_year,
        "mileage_km":       car.mileage_km,
        "engine_cc":        car.engine_cc,
        "fuel_type_enc":    FUEL_TYPE_ENC.get(car.fuel_type, 0),
        "transmission_enc": TRANSMISSION_ENC.get(car.transmission, 0),
        "usd_pkr":          car.usd_pkr   if car.usd_pkr   is not None else 280.0,
        "make_enc":         car.make_enc  if car.make_enc  is not None else GLOBAL_MEAN,
        "model_enc":        car.model_enc if car.model_enc is not None else GLOBAL_MEAN,
    }
    return pd.DataFrame([row])[FEATURE_ORDER]


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "service":  "Pakistan Car Price Prediction API",
        "status":   "running",
        "docs":     "/docs",
        "predict":  "/predict",
        "drift":    "/drift-report",
        "version":  "1.0.0",
    }


@app.get("/health", tags=["Health"])
def health():
    return {
        "status":             "healthy",
        "model_loaded":       BOOSTER is not None,
        "predictions_logged": len(PREDICTION_LOG),
        "timestamp":          datetime.utcnow().isoformat(),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(car: CarFeatures):
    if BOOSTER is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X        = build_feature_row(car)
    dmatrix  = xgb.DMatrix(X)
    raw_pred = BOOSTER.predict(dmatrix)
    log_pred = float(raw_pred[0])
    price    = float(np.expm1(log_pred))

    print(f"DEBUG log_pred={log_pred:.4f}  price={price:,.0f}")

    low  = float(price * 0.85)
    high = float(price * 1.15)

    # SHAP top 3
    shap_vals = EXPLAINER.shap_values(X)[0]
    shap_dict = dict(zip(FEATURE_ORDER, [float(v) for v in shap_vals]))
    top3      = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    shap_top  = [
        {
            "feature":   f,
            "impact":    round(v, 4),
            "direction": "increases price" if v > 0 else "decreases price",
        }
        for f, v in top3
    ]

    log_entry = X.iloc[0].to_dict()
    log_entry["predicted_log_price"] = log_pred
    log_entry["timestamp"]           = datetime.utcnow().isoformat()
    PREDICTION_LOG.append(log_entry)

    return PredictionResponse(
        predicted_price_pkr=price,
        predicted_price_formatted=f"PKR {price:,.0f}",
        confidence_range={
            "low":  f"PKR {low:,.0f}",
            "high": f"PKR {high:,.0f}",
        },
        shap_top_features=shap_top,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/drift-summary", tags=["Monitoring"])
def drift_summary():
    if len(PREDICTION_LOG) < 10:
        return {
            "status":             "not enough data",
            "predictions_logged": len(PREDICTION_LOG),
            "required":           10,
            "message":            "Send at least 10 /predict requests to enable drift monitoring",
        }

    pred_df  = pd.DataFrame(PREDICTION_LOG)
    features = [f for f in FEATURE_ORDER if f in pred_df.columns]

    summary = {}
    for feat in features:
        train_mean = float(TRAIN_REF[feat].mean())
        pred_mean  = float(pred_df[feat].mean())
        drift_pct  = abs(pred_mean - train_mean) / (abs(train_mean) + 1e-9) * 100
        summary[feat] = {
            "train_mean": round(train_mean, 4),
            "live_mean":  round(pred_mean,  4),
            "drift_%":    round(drift_pct,  2),
            "alert":      drift_pct > 15,
        }

    alerts = [f for f, v in summary.items() if v["alert"]]
    return {
        "predictions_logged": len(PREDICTION_LOG),
        "features_monitored": len(features),
        "alerts":             alerts,
        "drift_details":      summary,
        "timestamp":          datetime.utcnow().isoformat(),
    }


@app.get("/drift-report", response_class=HTMLResponse, tags=["Monitoring"])
def drift_report():
    if len(PREDICTION_LOG) < 5:
        return HTMLResponse(
            "<h2 style='font-family:sans-serif;padding:2rem'>"
            "Not enough predictions yet. Send at least 5 /predict requests.</h2>",
            status_code=200,
        )

    pred_df  = pd.DataFrame(PREDICTION_LOG)
    features = [f for f in FEATURE_ORDER if f in pred_df.columns]

    rows = ""
    for feat in features:
        train_mean = float(TRAIN_REF[feat].mean())
        pred_mean  = float(pred_df[feat].mean())
        drift_pct  = abs(pred_mean - train_mean) / (abs(train_mean) + 1e-9) * 100
        status     = "🔴 ALERT" if drift_pct > 15 else "🟢 OK"
        rows += f"""
        <tr>
          <td>{feat}</td>
          <td>{train_mean:.4f}</td>
          <td>{pred_mean:.4f}</td>
          <td>{drift_pct:.2f}%</td>
          <td>{status}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
  <title>Drift Monitor — Pakistan Car Price API</title>
  <style>
    body  {{ font-family: sans-serif; padding: 2rem; background: #0f0f0f; color: #eee; }}
    h1    {{ color: #4ade80; margin-bottom: 0.5rem; }}
    p     {{ color: #aaa; margin-bottom: 1.5rem; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th    {{ background: #1f1f1f; padding: 12px 16px; text-align: left; color: #4ade80; }}
    td    {{ padding: 10px 16px; border-bottom: 1px solid #2a2a2a; }}
    tr:hover {{ background: #1a1a1a; }}
    .badge {{ background: #1f2f1f; border: 1px solid #4ade80;
              padding: 3px 12px; border-radius: 20px; color: #4ade80;
              font-size: 0.85rem; margin-right: 8px; }}
    .note  {{ margin-top: 2rem; color: #555; font-size: 0.85rem; }}
  </style>
</head>
<body>
  <h1>📊 Drift Monitor</h1>
  <p>
    <span class="badge">Predictions logged: {len(PREDICTION_LOG)}</span>
    <span class="badge">Features: {len(features)}</span>
    <span class="badge">Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</span>
  </p>
  <table>
    <tr>
      <th>Feature</th><th>Train Mean</th><th>Live Mean</th><th>Drift %</th><th>Status</th>
    </tr>
    {rows}
  </table>
  <p class="note">
    🔴 ALERT = feature distribution drifted &gt;15% from training data<br>
    🟢 OK    = within acceptable range
  </p>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.get("/fuel-types", tags=["Reference"])
def fuel_types():
    return {"fuel_types": list(FUEL_TYPE_ENC.keys())}


@app.get("/transmission-types", tags=["Reference"])
def transmission_types():
    return {"transmission_types": list(TRANSMISSION_ENC.keys())}