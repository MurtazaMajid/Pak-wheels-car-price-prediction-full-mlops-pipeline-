"""
prepare_data.py
---------------
Reads pakwheels_final.csv, applies feature engineering,
does time-based split, saves X_train/X_test/y_train/y_test to data/.

Run this when you have new raw data before retraining.

Usage:
    cd car-price-mlops
    python scripts/prepare_data.py
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf

# ── PATHS ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")

print("Loading data...")
df = pd.read_csv(os.path.join(DATA_DIR, "pakwheels_final.csv"))
print(f"Loaded {len(df):,} rows")

# ── USD/PKR ────────────────────────────────────────────────────────────────────
print("Fetching USD/PKR rates from yfinance...")
try:
    ticker   = yf.Ticker("USDPKR=X")
    hist     = ticker.history(start="2015-01-01", interval="1mo")
    usd_map  = hist["Close"].resample("YE").mean().to_dict()
    usd_map  = {k.year: v for k, v in usd_map.items()}
    df["usd_pkr"] = df["manufacture_year"].map(usd_map).fillna(280.0)
    print(f"USD/PKR mapped: {usd_map}")
except Exception as e:
    print(f"yfinance failed ({e}), using fallback rates")
    fallback = {
        2015: 102, 2016: 104, 2017: 105, 2018: 121,
        2019: 150, 2020: 161, 2021: 176, 2022: 204,
        2023: 278, 2024: 278, 2025: 280,
    }
    df["usd_pkr"] = df["manufacture_year"].map(fallback).fillna(280.0)

# ── ENCODINGS ──────────────────────────────────────────────────────────────────
FUEL_ENC  = {"Petrol": 0, "Diesel": 1, "Hybrid": 2,
             "Electric": 3, "CNG": 4, "LPG": 5}
TRANS_ENC = {"Manual": 0, "Automatic": 1, "CVT": 2, "Semi-Auto": 3}

df["fuel_type_enc"]    = df["fuel_type"].map(FUEL_ENC).fillna(0).astype(int)
df["transmission_enc"] = df["transmission"].map(TRANS_ENC).fillna(0).astype(int)

# ── LOG PRICE ──────────────────────────────────────────────────────────────────
df["log_price"] = np.log1p(df["price_pkr"])

# ── TIME-BASED SPLIT ───────────────────────────────────────────────────────────
train = df[df["manufacture_year"] <= 2022].copy()
test  = df[df["manufacture_year"] >= 2023].copy()

print(f"\nTrain: {len(train):,} rows (2015-2022)")
print(f"Test : {len(test):,} rows  (2023-2025)")

# ── TARGET ENCODING — train-only ───────────────────────────────────────────────
global_mean   = train["log_price"].mean()
make_enc_map  = train.groupby("make")["log_price"].mean()
model_enc_map = train.groupby("model")["log_price"].mean()

train["make_enc"]  = train["make"].map(make_enc_map).fillna(global_mean)
train["model_enc"] = train["model"].map(model_enc_map).fillna(global_mean)
test["make_enc"]   = test["make"].map(make_enc_map).fillna(global_mean)
test["model_enc"]  = test["model"].map(model_enc_map).fillna(global_mean)

# ── FEATURE ORDER ──────────────────────────────────────────────────────────────
FEATURES = [
    "manufacture_year", "mileage_km",      "engine_cc",
    "fuel_type_enc",    "transmission_enc", "usd_pkr",
    "make_enc",         "model_enc",
]

X_train = train[FEATURES]
X_test  = test[FEATURES]
y_train = train["log_price"]
y_test  = test["log_price"]

# ── SAVE ───────────────────────────────────────────────────────────────────────
X_train.to_csv(os.path.join(DATA_DIR, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(DATA_DIR,  "X_test.csv"),  index=False)
y_train.to_csv(os.path.join(DATA_DIR, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(DATA_DIR,  "y_test.csv"),  index=False)

print(f"\nSaved to {DATA_DIR}/")
print(f"  X_train : {X_train.shape}  nulls: {X_train.isnull().sum().sum()}")
print(f"  X_test  : {X_test.shape}   nulls: {X_test.isnull().sum().sum()}")
print(f"  Features: {FEATURES}")
print("\nDone — ready to retrain.")
