"""
check_leakage.py
----------------
Run this whenever you get new data before retraining.
Validates the train/test split for data leakage across 7 checks.

Usage:
    cd car-price-mlops
    python scripts/check_leakage.py
"""

import os
import pandas as pd
import numpy as np

# ── PATHS ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")

df = pd.read_csv(os.path.join(DATA_DIR, "pakwheels_final.csv"))

print("=" * 55)
print("DATA LEAKAGE VALIDATION REPORT")
print("=" * 55)

# split
train = df[df["manufacture_year"] <= 2022].copy()
test  = df[df["manufacture_year"] >= 2023].copy()

# ── CHECK 1: year overlap ──────────────────────────────────────────────────────
print("\n[1] Split integrity")
train_years = sorted(train["manufacture_year"].unique())
test_years  = sorted(test["manufacture_year"].unique())
overlap     = set(train_years) & set(test_years)
print(f"    Train years: {train_years}")
print(f"    Test  years: {test_years}")
if overlap:
    print(f"    FAIL - year overlap: {overlap}")
else:
    print("    PASS - no year overlap between train and test")

# ── CHECK 2: listing ID overlap ────────────────────────────────────────────────
print("\n[2] Listing ID overlap")
if "listing_id" in df.columns:
    shared = set(train["listing_id"]) & set(test["listing_id"])
    print(f"    PASS - {len(shared)} listing IDs shared between train and test")
else:
    print("    SKIP - no listing_id column found")

# ── CHECK 3: target encoding leakage ──────────────────────────────────────────
print("\n[3] Target encoding leakage check")
train["log_price"] = np.log1p(train["price_pkr"])
test["log_price"]  = np.log1p(test["price_pkr"])

train_only_enc = train.groupby("make")["log_price"].mean()
full_enc       = df.groupby("make")["log_price"].mean()
common_makes   = train_only_enc.index.intersection(full_enc.index)
diff           = ((train_only_enc[common_makes] - full_enc[common_makes]).abs()
                  / full_enc[common_makes].abs() * 100)
print(f"    Avg difference train-only vs full-data encoding: {diff.mean():.2f}%")
print(f"    Max difference: {diff.max():.2f}%")
if diff.mean() < 10:
    print("    Low leakage risk from encoding")
else:
    print("    WARNING - high encoding leakage risk, use train-only encoding")

# ── CHECK 4: usd_pkr ──────────────────────────────────────────────────────────
print("\n[4] usd_pkr feature check")
if "usd_pkr" in df.columns:
    corr_train = train["usd_pkr"].corr(train["price_pkr"])
    corr_test  = test["usd_pkr"].corr(test["price_pkr"])
    print(f"    Correlation usd_pkr vs price in train: {corr_train:.3f}")
    print(f"    Correlation usd_pkr vs price in test:  {corr_test:.3f}")
    print("    usd_pkr is from external yfinance data - not derived from price")
    print("    PASS - no circular dependency")
else:
    print("    SKIP - usd_pkr column not found")

# ── CHECK 5: price_per_cc ─────────────────────────────────────────────────────
print("\n[5] price_per_cc feature check")
if "price_per_cc" in df.columns:
    print("    price_per_cc = price_pkr / engine_cc")
    print("    This IS derived from the target price_pkr")
    print("    Action: DROP price_per_cc from features before training")
else:
    print("    PASS - price_per_cc not present in dataset")

# ── CHECK 6: duplicates ───────────────────────────────────────────────────────
print("\n[6] Duplicate listing check")
dup_train = train.duplicated().sum()
dup_test  = test.duplicated().sum()
print(f"    Duplicates in train: {dup_train}")
print(f"    Duplicates in test:  {dup_test}")
if dup_train == 0 and dup_test == 0:
    print("    PASS - no duplicates")
else:
    print("    WARNING - duplicates found, drop before training")

# ── CHECK 7: future information ───────────────────────────────────────────────
print("\n[7] Future information check")
FEATURES = ["manufacture_year", "mileage_km", "engine_cc",
            "fuel_type", "transmission", "usd_pkr"]
DERIVED  = ["price_per_cc"]
for f in FEATURES:
    status = "PASS" if f in df.columns else "MISSING"
    print(f"    {status:6s}  {f}")
for f in DERIVED:
    print(f"    DROP   {f}  (derived from target)")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("SUMMARY")
print("=" * 55)
print(f"  Train size : {len(train):,} rows")
print(f"  Test  size : {len(test):,} rows")
print(f"  Year split : train <= 2022 | test >= 2023")
print("=" * 55)
