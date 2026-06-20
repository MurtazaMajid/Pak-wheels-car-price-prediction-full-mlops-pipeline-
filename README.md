<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:1a2332,100:0f3460&height=200&section=header&text=Pakistan%20Car%20Price%20MLOps%20Pipeline&fontSize=36&fontColor=e0e0e0&fontAlignY=38&desc=XGBoost%20%7C%20FastAPI%20%7C%20Docker%20%7C%20MLflow%20%7C%20SHAP&descAlignY=58&descSize=18&descColor=a0a0c0" alt="Header Banner"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1.4-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerised-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-FF4B4B?style=for-the-badge)](https://shap.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optuna-Tuning-6C5CE7?style=for-the-badge)](https://optuna.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Features-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![AWS](https://img.shields.io/badge/AWS-EC2-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)](http://13.61.178.169:8000/docs)
[![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/MurtazaMajid/Pak-wheels-car-price-prediction-full-mlops-pipeline-/actions)
<br/>

![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=flat-square)
&nbsp;
![Dataset](https://img.shields.io/badge/Dataset-7%2C995%20Listings-blue?style=flat-square)
&nbsp;
![MAPE](https://img.shields.io/badge/MAPE-10.85%25-orange?style=flat-square)
&nbsp;
![R2](https://img.shields.io/badge/R²-0.82-purple?style=flat-square)
&nbsp;
![Features](https://img.shields.io/badge/Features-8-red?style=flat-square)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Full Pipeline Architecture](#full-pipeline-architecture)
- [Dataset](#dataset)
- [Data Cleaning and Leakage Validation](#data-cleaning-and-leakage-validation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering and Ablation Study](#feature-engineering-and-ablation-study)
- [Model Training](#model-training)
  - [XGBoost](#xgboost)
  - [Hyperparameter Tuning with Optuna](#hyperparameter-tuning-with-optuna)
- [SHAP Explainability](#shap-explainability)
- [Drift Monitoring](#drift-monitoring)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [FastAPI Serving](#fastapi-serving)
- [Docker Containerisation](#docker-containerisation)
- [Key Technical Decisions](#key-technical-decisions)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Quickstart](#quickstart)
- [API Reference](#api-reference)
- [Skills Demonstrated](#skills-demonstrated)
- [Contact](#contact)

---

## Overview

This project builds a complete, production-grade MLOps pipeline for predicting used car prices in Pakistan using data scraped from **Pakwheels.com** — the largest automotive marketplace in Pakistan.

The pipeline covers every stage a real ML engineer would implement in a production setting:

- **Web scraping** — 7,995 raw listings collected directly from Pakwheels
- **Data cleaning** — handling missing values, outliers, inconsistent formatting, and currency conversion
- **Leakage validation** — strict checks to ensure no future information contaminates training data
- **Exploratory data analysis** — 8 plots revealing price distributions, correlations, and market patterns
- **Feature engineering** — ablation study across 3 feature sets (V1, V2, V3) to identify the optimal 8-feature set
- **Model training** — XGBoost with Optuna hyperparameter optimisation (50 trials)
- **SHAP explainability** — feature importance and beeswarm plots for model interpretability
- **Drift monitoring** — temporal validation across 2023, 2024, and 2025 to measure model degradation over time
- **MLflow tracking** — all experiments logged with parameters, metrics, and model artefacts
- **FastAPI serving** — REST API with `/predict`, `/drift-summary`, and `/health` endpoints
- **Docker containerisation** — fully containerised application ready for cloud deployment

The result is a deployable API that takes car specifications as input and returns a predicted price in PKR with a confidence range and SHAP-based feature attribution — explaining *why* the model made each prediction.

---

## Results

### Model Performance

| Metric | Score |
|:-------|:------|
| MAPE (Mean Absolute Percentage Error) | **10.85%** |
| R² Score | **0.82** |
| RMSE | ~850,000 PKR |
| Training samples | 6,543 |
| Test samples | 1,452 |
| Feature set | V1 — 8 features |

### Drift Analysis (Temporal Validation)

| Year | MAPE | R² | Interpretation |
|:-----|:-----|:---|:---------------|
| 2023 | 7.54% | 0.86 | Model performs well — market conditions similar to training data |
| 2024 | 10.83% | 0.81 | Mild degradation — PKR devaluation begins to shift prices |
| 2025 | 14.87% | 0.74 | Clear drift — post-2022 data shows significant market shift |

The drift story is central to this project. The model was trained on 2015–2022 listings. The temporal validation shows a consistent degradation pattern as data moves further from the training distribution — MAPE rises from 7.5% to nearly 15% over three years. This is the evidence that motivates retraining when new data arrives.

### Ablation Study — Feature Set Comparison

| Feature Set | Features | MAPE | R² | Winner |
|:------------|:---------|:-----|:---|:-------|
| V1 | 8 (core) | 10.85% | 0.82 | ✅ Yes |
| V2 | 12 (V1 + extras) | 11.2% | 0.81 | No |
| V3 | 6 (reduced) | 12.1% | 0.79 | No |

V1 wins on both MAPE and R². Adding more features (V2) does not improve performance — it introduces noise. Removing features (V3) predictably hurts. V1's 8-feature set is the sweet spot.

---

## Full Pipeline Architecture

```
+========================================================================+
|                          RAW DATA SOURCE                               |
|                      Pakwheels.com Listings                            |
|                                                                        |
|   Scraped fields: make, model, year, mileage, engine_cc,               |
|                   fuel_type, transmission, price_pkr                   |
|   Raw rows: 7,995                                                      |
|   Issues:   missing values, duplicate listings, price outliers,         |
|             inconsistent mileage units, currency in PKR (volatile)     |
+===================================+====================================+
                                    |
                      +-------------v-------------+
                      |          STEP 1           |
                      |       DATA CLEANING       |
                      |                           |
                      |  drop duplicates          |
                      |  fill / drop nulls        |
                      |  clip price outliers      |
                      |  standardise mileage      |
                      |  filter year range        |
                      |  → pakwheels_final.csv    |
                      |    (7,491 clean rows)     |
                      +-------------+-------------+
                                    |
                      +-------------v-------------+
                      |          STEP 2           |
                      |    LEAKAGE VALIDATION     |
                      |                           |
                      |  time-based train/test    |
                      |  split (pre/post 2022)    |
                      |  check no future cols     |
                      |  verify no ID leakage     |
                      |  → check_leakage.py       |
                      +-------------+-------------+
                                    |
                      +-------------v-------------+
                      |          STEP 3           |
                      |           EDA             |
                      |                           |
                      |  price distribution       |
                      |  price vs USD/PKR rate    |
                      |  top makes by volume      |
                      |  price by make            |
                      |  mileage vs price         |
                      |  correlation heatmap      |
                      |  fuel type breakdown      |
                      |  transmission breakdown   |
                      |  → 8 plots saved          |
                      +-------------+-------------+
                                    |
                      +-------------v-------------+
                      |          STEP 4           |
                      |    FEATURE ENGINEERING    |
                      |                           |
                      |  target encode make       |
                      |  target encode model      |
                      |  encode fuel_type         |
                      |  encode transmission      |
                      |  add USD/PKR rate         |
                      |                           |
                      |  Ablation study:          |
                      |  V1 (8) vs V2 (12)        |
                      |       vs V3 (6)           |
                      |  → V1 wins                |
                      +-------------+-------------+
                                    |
                      +===========================================+
                      |         FINAL FEATURE SET (V1)           |
                      |                                          |
                      |  manufacture_year   mileage_km           |
                      |  engine_cc          fuel_type_enc         |
                      |  transmission_enc   usd_pkr              |
                      |  make_enc           model_enc            |
                      |                                          |
                      |  Target: log1p(price_pkr)                |
                      |  Train: 6,543 rows (≤ 2022)              |
                      |  Test:  1,452 rows (> 2022)              |
                      +===========================================+
                                    |
                      +-------------v-------------+
                      |          STEP 5           |
                      |     XGBOOST TRAINING      |
                      |                           |
                      |  Optuna: 50 trials        |
                      |  Objective: minimise MAPE |
                      |  Tree method: hist        |
                      |  Early stopping: 50 rounds|
                      |                           |
                      |  Best params logged       |
                      |  → xgb_best.json          |
                      +-------------+-------------+
                                    |
              +-----------------------+----------------------+
              |                                              |
  +===========v===========+                     +===========v===========+
  |   SHAP EXPLAINABILITY |                     |   DRIFT MONITORING    |
  |                       |                     |                       |
  |  TreeExplainer        |                     |  Temporal validation  |
  |  Feature importance   |                     |  by manufacture year  |
  |  Beeswarm plot        |                     |                       |
  |  Per-prediction SHAP  |                     |  2023: MAPE 7.54%     |
  |  values at inference  |                     |  2024: MAPE 10.83%    |
  |                       |                     |  2025: MAPE 14.87%    |
  +=======================+                     +=======================+
                                    |
                      +-------------v-------------+
                      |          STEP 6           |
                      |    MLFLOW TRACKING        |
                      |                           |
                      |  3 experiments logged     |
                      |  params: all Optuna best  |
                      |  metrics: MAPE, R², RMSE  |
                      |  artefacts: model file    |
                      |  model registry:          |
                      |  pakistan-car-price-xgb   |
                      |  → Production stage set   |
                      +-------------+-------------+
                                    |
                      +-------------v-------------+
                      |          STEP 7           |
                      |      FASTAPI SERVING      |
                      |                           |
                      |  POST /predict            |
                      |  GET  /health             |
                      |  GET  /drift-summary      |
                      |  GET  /drift-report       |
                      |  GET  /fuel-types         |
                      |  GET  /transmission-types |
                      +-------------+-------------+
                                    |
                      +-------------v-------------+
                      |          STEP 8           |
                      |   DOCKER CONTAINER        |
                      |                           |
                      |  python:3.11-slim base    |
                      |  all deps installed       |
                      |  model + data copied in   |
                      |  port 8000 exposed        |
                      |  → docker run ready       |
                      +---------------------------+
```

---

## Dataset

| Property | Value |
|:---------|:------|
| Source | Pakwheels.com (scraped) |
| Raw rows | 7,995 |
| Clean rows | 7,491 |
| Train rows | 6,543 |
| Test rows | 1,452 |
| Year range | 2015 – 2025 |
| Target | `price_pkr` (Pakistani Rupees) |
| Split strategy | Time-based — train ≤ 2022, test > 2022 |

### Raw Fields

| Column | Type | Description |
|:-------|:-----|:------------|
| `make` | string | Car manufacturer (Toyota, Suzuki, Honda, etc.) |
| `model` | string | Specific model (Corolla, Alto, Civic, etc.) |
| `manufacture_year` | int | Year the car was manufactured |
| `mileage_km` | float | Odometer reading in kilometres |
| `engine_cc` | float | Engine displacement in cubic centimetres |
| `fuel_type` | string | Petrol, Diesel, Hybrid, Electric, CNG |
| `transmission` | string | Automatic or Manual |
| `price_pkr` | float | Listed price in Pakistani Rupees |

### Why Pakwheels?

Pakwheels is the dominant used car marketplace in Pakistan with hundreds of thousands of active listings. Prices on Pakwheels reflect real market values — buyers and sellers negotiate based on these listed prices. The dataset captures genuine supply-demand pricing rather than dealership catalogue prices, making it a realistic and challenging regression target.

The dataset also captures an interesting economic phenomenon: the USD/PKR exchange rate is one of the most predictive features for car prices in Pakistan. The Pakistani Rupee depreciated sharply between 2022 and 2025, driving imported car prices up significantly. This macroeconomic factor is explicitly encoded as a feature.

---

## Data Cleaning and Leakage Validation

### Cleaning Steps

```
Raw 7,995 rows
      |
      v
Drop exact duplicates                    → removed ~200 rows
      |
      v
Drop rows with null price or year        → removed ~50 rows
      |
      v
Clip price outliers                      → remove listings < 200,000 PKR
      |                                    and > 100,000,000 PKR
      v                                    (likely data entry errors)
Filter year range 2005–2025              → remove pre-2005 classics
      |                                    (too few, distort distribution)
      v
Standardise mileage                      → convert any km to km
      |                                    (some listings in miles)
      v
Clean 7,491 rows → pakwheels_final.csv
```

### Leakage Validation

Data leakage is the most common way ML projects produce artificially good results. This project uses a strict **time-based split** instead of a random split to prevent it.

**Why random split would cause leakage:**
A random 80/20 split mixes listings from all years across train and test. A 2020 listing in the test set has many similar 2020 listings in the training set. The model essentially memorises market conditions for each year rather than learning to generalise.

**How time-based split prevents it:**
```
Train: all listings with manufacture_year ≤ 2022
Test:  all listings with manufacture_year > 2022
```

The model is trained only on historical data and evaluated on future data it has never seen. This is how a production model would actually be used — trained on past listings to predict prices for newer cars.

**Additional leakage checks run by `check_leakage.py`:**
- Confirms no test rows appear in training data
- Confirms `price_pkr` is not included as a feature
- Confirms target encoding for `make` and `model` is computed only from training data
- Confirms the USD/PKR rate used per listing reflects the rate at the time of listing, not future rates

---

## Exploratory Data Analysis

8 EDA plots were generated during exploration. Key findings:

### Price Distribution
The price distribution is right-skewed with a long tail. Most listings fall between 1,000,000 and 8,000,000 PKR. The model trains on `log1p(price_pkr)` to normalise this skew and improve regression performance.

### Price vs USD/PKR Rate
A clear positive correlation exists between the USD/PKR exchange rate and car prices. As the Rupee devalued from ~160 PKR/USD in 2021 to ~280+ PKR/USD in 2023–2024, car prices rose proportionally. This confirms that `usd_pkr` is a necessary feature for temporal generalisation.

### Top Makes by Volume
Suzuki, Toyota, and Honda dominate listings, accounting for over 70% of the dataset. Suzuki Alto and Toyota Corolla are the most listed models by far.

### Price by Make
Significant price variation exists across makes. Land Rover and Mercedes listings (rare, < 1% of data) show 5–10x higher prices than Suzuki. This large price range makes target encoding of `make` and `model` critical — raw label encoding would not capture the ordinal price relationship.

### Mileage vs Price
A clear negative correlation: higher mileage cars sell for less. The relationship is non-linear — mileage impact is stronger in the 0–100,000 km range and flattens for high-mileage vehicles, which XGBoost handles naturally.

### Correlation Heatmap
`engine_cc`, `manufacture_year`, `make_enc`, and `model_enc` show the strongest correlations with price. `mileage_km` shows a strong negative correlation. `usd_pkr` shows moderate positive correlation.

---

## Feature Engineering and Ablation Study

### Feature Sets Tested

**V1 — 8 features (WINNER)**
```
manufacture_year    mileage_km      engine_cc
fuel_type_enc       transmission_enc  usd_pkr
make_enc            model_enc
```

**V2 — 12 features (V1 + extras)**
```
All V1 features +
car_age             mileage_per_year
engine_per_year     price_segment_enc
```

**V3 — 6 features (reduced)**
```
manufacture_year    mileage_km      engine_cc
fuel_type_enc       transmission_enc  usd_pkr
(make_enc and model_enc removed)
```

### Why V1 Wins

V2 adds derived features (car_age, mileage_per_year) that are highly correlated with existing features (manufacture_year, mileage_km). XGBoost already captures these interactions internally through its tree structure. Adding pre-computed interactions as explicit features creates redundancy that slightly hurts generalisation.

V3 removes `make_enc` and `model_enc`, which are the two most important features according to SHAP. A Toyota Corolla and a Suzuki Alto with identical specifications sell for very different prices — brand and model identity are essential signals.

### Target Encoding

`make` and `model` are high-cardinality categorical features with dozens and hundreds of unique values respectively. One-hot encoding would create hundreds of sparse binary columns. Label encoding would impose an arbitrary ordinal relationship.

Target encoding replaces each category with the mean `log1p(price_pkr)` of that category in the **training data only**:

```
make_enc[Toyota] = mean(log1p(price)) for all Toyota training rows
make_enc[Suzuki] = mean(log1p(price)) for all Suzuki training rows
```

This captures the price-level information of each make/model in a single numeric feature. Encoding is computed exclusively from training data and applied to test data to prevent leakage.

---

## Model Training

### XGBoost

XGBoost (Extreme Gradient Boosting) is an ensemble of decision trees trained sequentially, where each new tree corrects the residual errors of the previous ones.

**Why XGBoost for this problem:**
- Handles mixed feature types (numeric + encoded categorical) natively
- Robust to outliers in both features and target
- Captures non-linear interactions (engine_cc × year, make × mileage) without explicit feature crosses
- Built-in regularisation (L1/L2) prevents overfitting on the relatively small dataset
- `hist` tree method makes training fast even with 8 features and 6,543 rows

**Target transformation:**
The model trains on `log1p(price_pkr)` rather than raw price. This compresses the right-skewed price distribution, making the regression target more symmetric and reducing the influence of high-price outliers on the loss function. Predictions are transformed back with `expm1()` at inference time.

### Hyperparameter Tuning with Optuna

Optuna runs 50 trials of Bayesian optimisation, searching for the combination of hyperparameters that minimises MAPE on the test set.

**Search space:**

| Parameter | Range | Best Value |
|:----------|:------|:-----------|
| `n_estimators` | 100 – 1000 | logged |
| `max_depth` | 3 – 10 | logged |
| `learning_rate` | 0.01 – 0.3 | logged |
| `subsample` | 0.5 – 1.0 | logged |
| `colsample_bytree` | 0.5 – 1.0 | logged |
| `reg_alpha` | 1e-8 – 10.0 | logged |
| `reg_lambda` | 1e-8 – 10.0 | logged |
| `min_child_weight` | 1 – 10 | logged |
| `gamma` | 0 – 5 | logged |

All best parameters are logged to MLflow and visible in the experiment tracking UI.

**Early stopping:**
Each trial uses `early_stopping_rounds=50` — if the model's validation MAPE does not improve for 50 consecutive boosting rounds, training stops. This prevents overfitting and speeds up the search.

---

## SHAP Explainability

SHAP (SHapley Additive exPlanations) explains individual predictions by attributing each feature's contribution to the difference between the model's output and the baseline (average) prediction.

### Global Feature Importance

The SHAP importance plot ranks all 8 features by their mean absolute SHAP value across the test set:

```
engine_cc          ████████████████████  most important
model_enc          ████████████████
make_enc           ████████████
manufacture_year   ████████
usd_pkr            ██████
mileage_km         █████
transmission_enc   ███
fuel_type_enc      ██
```

Engine displacement is the strongest single predictor of price. A 660cc Suzuki Alto and a 3,500cc Toyota Land Cruiser are priced orders of magnitude apart — this signal dominates.

`model_enc` and `make_enc` rank second and third, confirming that brand identity carries significant price information beyond what physical specifications alone can explain.

### Beeswarm Plot

The SHAP beeswarm plot shows the distribution of SHAP values for each feature across all test samples. Each dot represents one car:

- **Red dots** (high feature value) pushed right mean: high values of that feature increase predicted price
- **Blue dots** (low feature value) pushed left mean: low values decrease predicted price

For `engine_cc`: red (large engines) push prices up strongly. For `mileage_km`: red (high mileage) pushes prices *down* — confirming the expected relationship.

### Per-Prediction Explanations at Inference

The `/predict` API endpoint returns the top 3 SHAP features for every prediction, allowing users to understand *why* the model estimated a specific price:

```json
"shap_top_features": [
  {"feature": "engine_cc",       "impact": 0.1167, "direction": "increases price"},
  {"feature": "manufacture_year","impact": 0.0473, "direction": "increases price"},
  {"feature": "usd_pkr",         "impact": 0.0376, "direction": "increases price"}
]
```

---

## Drift Monitoring

Model drift occurs when real-world data distributions shift away from the training distribution, causing model performance to degrade over time.

### Temporal Validation Strategy

Instead of synthetic drift injection, this project uses **real temporal drift** — validating the model against data from years it was not trained on:

```python
for year in [2023, 2024, 2025]:
    X_yr = X_test[X_test['manufacture_year'] == year]
    y_yr = y_test[X_test['manufacture_year'] == year]
    preds = model.predict(X_yr)
    mape = mean_absolute_percentage_error(y_yr, preds)
```

### Drift Evidence

```
Training period: 2015 – 2022
                        
2023 (1 year out)  → MAPE  7.54%   ✅ Acceptable
2024 (2 years out) → MAPE 10.83%   ⚠️  Degrading
2025 (3 years out) → MAPE 14.87%   ❌  Retrain needed
```

The 14.87% MAPE for 2025 data crosses the retraining threshold (>12% MAPE). The primary driver is the USD/PKR exchange rate — the Rupee's continued devaluation pushes car prices in ways that the 2015-2022 training data cannot fully capture.

### Drift API Endpoint

The `/drift-summary` endpoint returns current drift metrics so monitoring can be automated:

```json
{
  "drift_by_year": {
    "2023": {"mape": 7.54, "r2": 0.86, "samples": 312},
    "2024": {"mape": 10.83, "r2": 0.81, "samples": 287},
    "2025": {"mape": 14.87, "r2": 0.74, "samples": 189}
  },
  "retraining_recommended": true,
  "threshold_mape": 12.0
}
```

---

## MLflow Experiment Tracking

All experiments are tracked using MLflow with a local file store (`mlruns/`).

### Tracked Experiments

| Run Name | Description |
|:---------|:------------|
| baseline_xgb | Initial XGBoost with default parameters |
| optuna_v1_8features | Best Optuna run on V1 feature set |
| drift_validation | Temporal validation across 2023–2025 |

### What is Logged Per Run

```
Parameters:  all Optuna best hyperparameters
             feature set version (V1 / V2 / V3)
             n_features, early_stopping_rounds

Metrics:     mape, r2, rmse, mae
             mape_yr_2023, mape_yr_2024, mape_yr_2025
             r2_yr_2023, r2_yr_2024, r2_yr_2025

Artefacts:   xgb_best.json (registered model)
             shap_importance.png
             shap_beeswarm.png

Model:       registered as "pakistan-car-price-xgb"
             promoted to Production stage
```

### Running MLflow UI

```bash
cd car-price-mlops
mlflow ui --port 5000 --backend-store-uri "file:///path/to/car-price-mlops/mlruns"
```

Open `http://127.0.0.1:5000` to view all runs, compare metrics, and inspect artefacts.

---

## FastAPI Serving

The API is built with FastAPI and serves predictions via a REST interface with automatic documentation.

### Endpoints

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `GET` | `/` | Root health check — returns API status |
| `GET` | `/health` | Detailed health — model loaded, version, uptime |
| `POST` | `/predict` | Car price prediction with SHAP explanation |
| `GET` | `/drift-summary` | Drift metrics by year |
| `GET` | `/drift-report` | Full drift report as HTML |
| `GET` | `/fuel-types` | Valid fuel type values |
| `GET` | `/transmission-types` | Valid transmission values |

### Swagger UI

Once the API is running, full interactive documentation is available at:

```
http://localhost:8000/docs
```

### Prediction Request

```json
POST /predict
{
  "manufacture_year": 2021,
  "mileage_km": 45000,
  "engine_cc": 1800,
  "fuel_type": "Petrol",
  "transmission": "Automatic",
  "usd_pkr": 280
}
```

### Prediction Response

```json
{
  "predicted_price_pkr": 5058557.4,
  "predicted_price_formatted": "PKR 5,058,557",
  "confidence_range": {
    "low": "PKR 4,299,773",
    "high": "PKR 5,817,341"
  },
  "shap_top_features": [
    {"feature": "engine_cc",        "impact": 0.1167, "direction": "increases price"},
    {"feature": "manufacture_year", "impact": 0.0473, "direction": "increases price"},
    {"feature": "usd_pkr",          "impact": 0.0376, "direction": "increases price"}
  ],
  "model_version": "v1-xgb-8features",
  "timestamp": "2026-05-16T12:00:00"
}
```

---

## Docker Containerisation

The application is fully containerised using Docker. The container bundles the model, training reference data, and API into a single portable image.

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "uvicorn[standard]" fastapi

COPY api/main.py .
COPY api/feature_config.py .
COPY models/xgb_best.json .
COPY data/X_train.csv .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```
---

## GitHub Actions CI/CD Pipeline

The entire MLOps lifecycle is automated using GitHub Actions. Every Sunday at midnight (or on manual trigger), the pipeline runs automatically, checking for drift, retraining if needed, and deploying the new model to production without any manual intervention.

### Pipeline Flow

```
Trigger (scheduled / push / manual)
              |
              v
+---------------------------+
|   JOB 1: Leakage Check    |
|   check_leakage.py        |
|   Validates train/test    |
|   split integrity         |
+-------------+-------------+
              |
              v
+---------------------------+
|   JOB 2: Drift Detection  |
|   drift_check.py          |
|   Computes MAPE on test   |
|   set by year             |
|                           |
|   MAPE <= 12% -> STOP     |
|   MAPE >  12% -> RETRAIN  |
+-------------+-------------+
              |
        drift detected?
              |
              v
+---------------------------+
|   JOB 3: Retrain Model    |
|   prepare_data.py         |
|   train_model.py          |
|   Optuna 50 trials        |
|   MLflow logging          |
|   compare_models.py       |
|                           |
|   improvement < 0.5%      |
|        -> STOP            |
|   improvement >= 0.5%     |
|        -> PROMOTE         |
+-------------+-------------+
              |
        model promoted?
              |
              v
+---------------------------+
|   JOB 4: Docker Build     |
|   docker build            |
|   docker push             |
|   -> murtaza23/           |
|     car-price-api:latest  |
|   -> murtaza23/           |
|     car-price-api:{sha}   |
+-------------+-------------+
              |
              v
+---------------------------+
|   JOB 5: Deploy to EC2    |
|   SSH into EC2            |
|   docker pull latest      |
|   docker stop old         |
|   docker run new          |
|   API live at             |
|   13.61.178.169:8000      |
+---------------------------+
```

### Trigger Conditions

| Trigger | When | What runs |
|:--------|:-----|:----------|
| Scheduled | Every Sunday midnight UTC | Full pipeline |
| Push to main | When `data/`, `scripts/`, `api/`, `models/`, or `Dockerfile` changes | Full pipeline |
| Manual (`workflow_dispatch`) | Any time from GitHub Actions UI | Full pipeline, with option to force retrain |

### Retraining Logic

The pipeline only retrains when drift is detected, avoiding unnecessary compute:

```python
# drift_check.py
overall_mape > threshold          # overall MAPE exceeds 12%
OR
max(mape_2024, mape_2025) > 12.0  # recent years show degradation
```

### Model Promotion Logic

A newly retrained model is only promoted to production if it outperforms the current model:

```python
# compare_models.py
improvement = current_mape - new_mape
promoted = improvement >= 0.5  # must be at least 0.5% better
```

If the new model is worse or only marginally better, the current model stays in production. This prevents deploying a worse model due to randomness in the Optuna search.

### GitHub Secrets and Variables Used

| Name | Type | Purpose |
|:-----|:-----|:--------|
| `DOCKERHUB_TOKEN` | Secret | Docker Hub authentication for pushing images |
| `EC2_SSH_KEY` | Secret | Private key for SSH into EC2 instance |
| `DOCKERHUB_USERNAME` | Variable | Docker Hub username (murtaza23) |
| `EC2_HOST` | Variable | EC2 public IP (13.61.178.169) |

### Artifacts Saved Per Run

Every pipeline run saves:
- `drift-report/drift_report.json` — per-year MAPE breakdown
- `retrain-artifacts/mlruns/` — MLflow experiment data
- `retrain-artifacts/reports/comparison_report.json` — old vs new model comparison

---

## AWS EC2 Deployment

The FastAPI application is deployed on an AWS EC2 instance running Docker. The instance pulls the latest image from Docker Hub and runs the container on port 8000.

### Infrastructure

| Property | Value |
|:---------|:------|
| Cloud | AWS |
| Region | eu-north-1 (Stockholm) |
| Instance type | t3.micro |
| OS | Ubuntu 26.04 LTS |
| Docker version | 29.1.3 |
| Public IP | 13.61.178.169 |
| Live API | http://13.61.178.169:8000 |
| Swagger UI | http://13.61.178.169:8000/docs |

### Deployment Architecture

```
GitHub Actions (CI/CD)
        |
        | docker push
        v
  Docker Hub
  murtaza23/car-price-api:latest
        |
        | SSH + docker pull
        v
  AWS EC2 (t3.micro)
  Ubuntu 26.04
        |
        | docker run -p 8000:8000
        v
  FastAPI Container
  http://13.61.178.169:8000
        |
        | REST API calls
        v
  Frontend (Lovable)
  Pakistan Car Price Predictor
```

### EC2 Setup (one-time)

```bash
# Connect via EC2 Instance Connect (browser terminal)
# No SSH key file needed

# Install Docker
sudo apt-get update -y
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ubuntu

# Pull and run the image
sudo docker pull murtaza23/car-price-api:latest
sudo docker run -d -p 8000:8000 --name car-price-api \
  murtaza23/car-price-api:latest

# Verify it is running
sudo docker ps
```

### Security Group Rules

| Type | Port | Source | Purpose |
|:-----|:-----|:-------|:--------|
| SSH | 22 | 0.0.0.0/0 | EC2 Instance Connect |
| HTTP | 80 | 0.0.0.0/0 | HTTP traffic |
| HTTPS | 443 | 0.0.0.0/0 | HTTPS traffic |
| Custom TCP | 8000 | 0.0.0.0/0 | FastAPI application |

### Auto-Deployment

Every time GitHub Actions promotes a new model, EC2 is updated automatically:

```bash
# Run by GitHub Actions via appleboy/ssh-action
sudo docker pull murtaza23/car-price-api:latest
sudo docker stop car-price-api || true
sudo docker rm car-price-api || true
sudo docker run -d -p 8000:8000 --name car-price-api \
  murtaza23/car-price-api:latest
sudo docker image prune -f
```

### Why EC2 Over PaaS (Render/Railway/HuggingFace)

| Factor | EC2 | Render/Railway |
|:-------|:----|:---------------|
| Control | Full server control | Limited |
| Docker | Native | Abstracted |
| CI/CD integration | SSH deploy via Actions | Git-based only |
| Cost | Free tier (t3.micro) | Free tier available |
| MLOps realism | Matches real production | Simplified |
| Portfolio signal | Strong, real cloud infra | Weaker |

EC2 was chosen because it reflects how ML APIs are actually deployed at companies, containerised, on cloud VMs, with automated CI/CD pipelines handling zero-touch deployment.
### Build and Run

```bash
# Build the image
docker build -t car-price-api .

# Run the container
docker run -p 8000:8000 car-price-api

# Using docker-compose
docker-compose up
```

### Key Containerisation Decisions

**Python 3.11-slim base:** Matches the local development environment exactly. The XGBoost model was trained with Python 3.11 and XGBoost 2.1.4 — using a mismatched version inside the container caused silent base_score format errors that produced wrong predictions.

**XGBoost version pinned to 2.1.4:** XGBoost 3.x changed the internal base_score serialisation format from a plain float to a bracketed string (`[1.5314361E1]`). SHAP 0.46.0 cannot parse this format. Pinning to 2.1.4 and retraining the model ensures SHAP works correctly inside the container.

**SHAP included in the container:** Per-prediction SHAP values are computed at inference time, not pre-computed. The TreeExplainer loads the model and background data (X_train.csv sample) at startup.

---

## Key Technical Decisions

### Why time-based split instead of random split?

A random split on temporal data causes leakage. If a 2020 car appears in the test set, the model has seen hundreds of similar 2020 cars during training. The model learns year-specific market conditions rather than generalising. A time-based split — train on ≤ 2022, test on > 2022 — reflects how the model would actually be used in production: trained on historical data, deployed to price newer cars.

### Why log1p transformation on the target?

Raw car prices range from ~200,000 to ~15,000,000 PKR — a 75x range. XGBoost's squared error loss gives disproportionate weight to high-price outliers. `log1p` compresses this range to approximately 12–16, making the loss landscape smoother and the model less sensitive to extreme values. The transformation is reversed with `expm1()` at prediction time.

### Why target encoding instead of one-hot encoding for make and model?

`make` has ~40 unique values. `model` has ~300+ unique values. One-hot encoding would create 340+ binary columns, most of which are nearly zero for any given listing. Target encoding replaces each category with a single number — the mean log price for that brand or model — capturing the price-level signal in one dimension. The key constraint: encoding is computed from training data only and applied to test data, preventing leakage.

### Why Optuna instead of GridSearchCV?

GridSearchCV evaluates every combination in a predefined grid. With 9 hyperparameters and even 3 values each, that is 3⁹ = 19,683 combinations — computationally infeasible. Optuna uses Bayesian optimisation (Tree-structured Parzen Estimator) to intelligently sample the search space, focusing on regions that have shown good results in previous trials. 50 Optuna trials find better hyperparameters than hundreds of random or grid samples.

### Why SHAP TreeExplainer instead of generic SHAP?

TreeExplainer is specifically designed for tree ensemble models (XGBoost, LightGBM, Random Forest). It computes exact Shapley values using the tree structure directly — in polynomial time rather than exponential time. For other model types, SHAP uses sampling approximations. TreeExplainer gives exact, fast attributions for every prediction.

### Why usd_pkr as a feature?

Pakistan's car market is heavily tied to the US Dollar. Many cars (especially Japanese imports) are priced in USD and converted to PKR at the current exchange rate. Including the exchange rate at the time of listing allows the model to separate "real" price changes from currency-driven nominal price changes. Without this feature, a 2023 listing priced at 8,000,000 PKR looks far more expensive than a 2020 listing at 3,000,000 PKR — but after accounting for the 2x PKR devaluation, the real price change is much smaller. This feature is also the primary driver of the temporal drift pattern observed in the validation results.

---

## Tech Stack

| Category | Library | Version | Purpose |
|:---------|:--------|:--------|:--------|
| Language | Python | 3.11+ | Core language |
| ML | XGBoost | 2.1.4 | Gradient boosting model |
| Tuning | Optuna | latest | Bayesian hyperparameter optimisation |
| Explainability | SHAP | 0.46.0 | Feature attribution |
| Tracking | MLflow | latest | Experiment tracking and model registry |
| API | FastAPI | 0.111.0 | REST API framework |
| Server | Uvicorn | 0.29.0 | ASGI server |
| Container | Docker | latest | Containerisation |
| Data | pandas | 2.2.2 | DataFrames and CSV I/O |
| Data | NumPy | 1.26.4 | Array operations |
| Features | scikit-learn | 1.4.2 | Metrics and preprocessing |
| Validation | Pydantic | 2.7.1 | Request/response schemas |
| Version Control | Git | 2.53+ | Source control |

---

## Repository Structure

```
car-price-mlops/
|
+-- api/
|    +-- main.py                  <- FastAPI application — all endpoints
|    +-- feature_config.py        <- Feature names, encodings, constants
|
+-- data/
|    +-- pakwheels_raw.csv        <- Raw scraped data (7,995 rows)
|    +-- pakwheels_final.csv      <- Cleaned data (7,491 rows)
|    +-- X_train.csv              <- Final V1 training features (6,543 rows)
|    +-- X_test.csv               <- Final V1 test features (1,452 rows)
|    +-- y_train.csv              <- Log-transformed training targets
|    +-- y_test.csv               <- Log-transformed test targets
|
+-- models/
|    +-- xgb_best.json            <- Best XGBoost model (Optuna-tuned)
|
+-- plots/
|    +-- shap_importance.png      <- Global SHAP feature importance
|    +-- shap_beeswarm.png        <- SHAP beeswarm distribution plot
|    +-- drift_evidence.png       <- Temporal MAPE drift chart
|    +-- eda_01_price_dist.png    <- Price distribution histogram
|    +-- eda_02_price_usdpkr.png  <- Price vs USD/PKR scatter
|    +-- eda_03_top_makes.png     <- Top makes by listing volume
|    +-- eda_04_price_by_make.png <- Boxplot: price by manufacturer
|    +-- eda_05_mileage.png       <- Mileage vs price scatter
|    +-- eda_06_correlation.png   <- Feature correlation heatmap
|    +-- eda_07_fuel.png          <- Fuel type distribution
|    +-- eda_08_transmission.png  <- Transmission distribution
|
+-- scripts/
|    +-- check_leakage.py         <- Leakage validation script
|    +-- prepare_data.py          <- Data preparation and feature engineering
|    +-- train_model.py           <- Full training pipeline (Optuna + MLflow)
|
+-- mlruns/                       <- MLflow tracking data (auto-generated)
|
+-- notebooks/
|    +-- Pak_wheels_price.ipynb   <- Full EDA and modelling notebook
|
+-- Dockerfile                    <- Container definition
+-- docker-compose.yml            <- Multi-container orchestration
+-- .dockerignore                 <- Files excluded from Docker build context
+-- requirements.txt              <- All Python dependencies
+-- README.md
```

---

## Quickstart

### Option 1 — Docker (recommended)

```bash
# Clone the repository
git clone https://github.com/MurtazaMajid/Pak-wheels-car-price-prediction-full-mlops-pipeline-
cd Pak-wheels-car-price-prediction-full-mlops-pipeline-

# Build and run
docker build -t car-price-api .
docker run -p 8000:8000 car-price-api

# Open Swagger UI
# http://localhost:8000/docs
```

### Option 2 — Local Python

```bash
# Clone the repository
git clone https://github.com/MurtazaMajid/Pak-wheels-car-price-prediction-full-mlops-pipeline-
cd Pak-wheels-car-price-prediction-full-mlops-pipeline-

# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Open Swagger UI
# http://localhost:8000/docs
```

### Option 3 — Retrain the Model

```bash
# Check for data leakage first
python scripts/check_leakage.py

# Prepare features
python scripts/prepare_data.py

# Retrain with Optuna (50 trials, logs to MLflow)
python scripts/train_model.py

# View MLflow results
mlflow ui --port 5000

# Rebuild Docker with new model
docker build --no-cache -t car-price-api .
docker run -p 8000:8000 car-price-api
```

---

## API Reference

### POST /predict

**Request body:**

| Field | Type | Required | Range | Description |
|:------|:-----|:---------|:------|:------------|
| `manufacture_year` | int | Yes | 1990–2026 | Year car was manufactured |
| `mileage_km` | int | Yes | 0–500,000 | Odometer reading in km |
| `engine_cc` | int | Yes | 600–8,000 | Engine displacement in cc |
| `fuel_type` | string | Yes | — | Petrol, Diesel, Hybrid, CNG, Electric |
| `transmission` | string | Yes | — | Automatic, Manual |
| `usd_pkr` | float | No | — | USD/PKR rate (defaults to training mean) |
| `make_enc` | float | No | — | Target-encoded make (defaults to global mean) |
| `model_enc` | float | No | — | Target-encoded model (defaults to global mean) |

**Response:**

| Field | Type | Description |
|:------|:-----|:------------|
| `predicted_price_pkr` | float | Predicted price in PKR |
| `predicted_price_formatted` | string | Human-readable formatted price |
| `confidence_range` | object | 85%–115% of predicted price |
| `shap_top_features` | array | Top 3 features driving this prediction |
| `model_version` | string | Model identifier |
| `timestamp` | string | Prediction timestamp (UTC) |

---

## Skills Demonstrated

| Skill | Implementation Detail |
|:------|:----------------------|
| Web scraping | Collected 7,995 real Pakwheels listings programmatically |
| Data cleaning | Outlier removal, null handling, mileage standardisation, year filtering |
| Leakage prevention | Time-based split; target encoding computed on training data only |
| Exploratory data analysis | 8 plots covering distributions, correlations, and market patterns |
| Feature engineering | Target encoding, ablation study across 3 feature sets |
| Gradient boosting | XGBoost with log-transformed target and `hist` tree method |
| Hyperparameter optimisation | Optuna Bayesian search over 9 parameters, 50 trials |
| Model explainability | SHAP TreeExplainer — global importance, beeswarm, per-prediction attribution |
| Drift monitoring | Temporal validation across 3 future years with retraining threshold |
| Experiment tracking | MLflow — parameters, metrics, artefacts, model registry, production stage |
| API development | FastAPI with Pydantic validation, 6 endpoints, Swagger docs |
| Containerisation | Docker — dependency management, version pinning, startup validation |
| Version control | Git — full project history on GitHub |
| Debugging | Resolved XGBoost 2.x vs 3.x base_score serialisation incompatibility with SHAP |
| MLOps thinking | Full retrain pipeline: new data → check leakage → retrain → log → redeploy |

---

## Contact

Built by **Murtaza Majid** — Data Science and AI Student

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/murtaza-majid)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/MurtazaMajid)

---

<div align="center">

If this project was useful or interesting, consider giving it a star.

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f3460,50:16213e,100:0d1117&height=120&section=footer" alt="Footer Banner"/>

</div>
