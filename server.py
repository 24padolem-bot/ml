"""
FastAPI Backend Server for Machine Health Intelligence.
NASA C-MAPSS FD001 Turbofan Engine Degradation & RUL Prognostics.

Deployable on Render, Railway, Fly.io, or standard Docker/Linux environments.
Serves high-performance REST APIs and the Portfolio Web Dashboard.
"""

from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import config
from src.data_loader import load_raw_data, compute_train_rul, compute_test_rul
from src.feature_engineering import generate_engine_features

app = FastAPI(
    title="Machine Health Intelligence API",
    description="Explainable Turbofan Engine Degradation, RUL Prognostics & Risk Triage",
    version="1.0.0"
)

# Enable CORS for cross-origin or local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory cache for fast response times
CACHE = {
    "train_df": None,
    "test_df": None,
    "rul_truth": None,
    "regression_pipeline": None,
    "classification_pipeline": None,
    "kmeans_model": None,
    "kmeans_scaler": None,
    "kmeans_pca": None,
    "reg_metadata": {},
    "clf_metadata": {},
    "km_metadata": {},
    "benchmark_regression": [],
    "benchmark_classification": [],
}


def load_artifacts():
    """Preloads datasets, trained ML models, and benchmark tables."""
    # 1. Load Data
    try:
        train_raw, test_raw, rul_raw = load_raw_data()
        CACHE["train_df"] = compute_train_rul(train_raw, rul_cap=config.RUL_CAP)
        CACHE["test_df"] = compute_test_rul(test_raw, rul_raw, rul_cap=config.RUL_CAP)
        CACHE["rul_truth"] = rul_raw
        print("[INFO] Datasets successfully loaded and cached.")
    except Exception as e:
        print(f"[WARN] Error loading raw data: {e}")

    # 2. Load Models
    try:
        reg_path = config.REGRESSION_MODELS_DIR / "best_regression_pipeline.joblib"
        if reg_path.exists():
            CACHE["regression_pipeline"] = joblib.load(reg_path)

        clf_path = config.CLASSIFICATION_MODELS_DIR / "best_classification_pipeline.joblib"
        if clf_path.exists():
            CACHE["classification_pipeline"] = joblib.load(clf_path)

        km_path = config.MODELS_DIR / "kmeans_model.joblib"
        scaler_path = config.MODELS_DIR / "kmeans_scaler.joblib"
        pca_path = config.MODELS_DIR / "kmeans_pca.joblib"
        if km_path.exists():
            CACHE["kmeans_model"] = joblib.load(km_path)
            CACHE["kmeans_scaler"] = joblib.load(scaler_path)
            CACHE["kmeans_pca"] = joblib.load(pca_path)

        reg_meta = config.REGRESSION_MODELS_DIR / "regression_metadata.json"
        if reg_meta.exists():
            with open(reg_meta) as f:
                CACHE["reg_metadata"] = json.load(f)

        clf_meta = config.CLASSIFICATION_MODELS_DIR / "classification_metadata.json"
        if clf_meta.exists():
            with open(clf_meta) as f:
                CACHE["clf_metadata"] = json.load(f)

        km_meta = config.RESULTS_DIR / "metrics" / "clustering_metadata.json"
        if km_meta.exists():
            with open(km_meta) as f:
                CACHE["km_metadata"] = json.load(f)

        # Benchmarks CSVs
        reg_csv = config.RESULTS_DIR / "metrics" / "regression_metrics.csv"
        if reg_csv.exists():
            CACHE["benchmark_regression"] = pd.read_csv(reg_csv).to_dict(orient="records")

        clf_csv = config.RESULTS_DIR / "metrics" / "classification_metrics.csv"
        if clf_csv.exists():
            CACHE["benchmark_classification"] = pd.read_csv(clf_csv).to_dict(orient="records")

        print("[INFO] ML Models and metrics metadata successfully loaded.")
    except Exception as e:
        print(f"[WARN] Error loading models: {e}")


# Initialize artifacts at startup
load_artifacts()


# -----------------------------------------------------------------------------
# REST API Endpoints
# -----------------------------------------------------------------------------

@app.get("/api/health")
def health_check():
    """Health status check endpoint for Render monitoring."""
    return {
        "status": "healthy",
        "service": "machine-health-intelligence",
        "framework": "FastAPI + Uvicorn",
        "dataset": "NASA C-MAPSS FD001",
        "models_ready": CACHE["regression_pipeline"] is not None and CACHE["classification_pipeline"] is not None
    }


@app.get("/api/fleet/summary")
def get_fleet_summary():
    """Returns top-level fleet telemetry and model performance summary."""
    train_df = CACHE["train_df"]
    test_df = CACHE["test_df"]

    num_train_units = int(train_df["unit_nr"].nunique()) if train_df is not None else 100
    num_test_units = int(test_df["unit_nr"].nunique()) if test_df is not None else 100
    total_train_cycles = int(len(train_df)) if train_df is not None else 20631

    return {
        "fleet_size": num_test_units,
        "train_units": num_train_units,
        "total_historical_cycles": total_train_cycles,
        "rul_cap": config.RUL_CAP,
        "risk_threshold": config.MAINTENANCE_RISK_THRESHOLD,
        "models": {
            "best_regression": {
                "name": "Random Forest Regressor",
                "test_mae": 12.02,
                "test_rmse": 16.53,
                "test_r2": 0.8299
            },
            "best_classification": {
                "name": "Random Forest Classifier",
                "accuracy": 0.950,
                "f1_score": 0.8936,
                "recall": 0.840,
                "roc_auc": 0.9840
            },
            "screening_classification": {
                "name": "Logistic Regression",
                "accuracy": 0.930,
                "recall": 0.920,
                "roc_auc": 0.9861
            }
        },
        "feature_count": 78
    }


@app.get("/api/telemetry/engines")
def list_engine_units():
    """Lists available engine IDs for test and train datasets."""
    test_df = CACHE["test_df"]
    train_df = CACHE["train_df"]

    test_units = sorted(test_df["unit_nr"].unique().tolist()) if test_df is not None else list(range(1, 101))
    train_units = sorted(train_df["unit_nr"].unique().tolist()) if train_df is not None else list(range(1, 101))

    return {
        "test_units": test_units,
        "train_units": train_units
    }


@app.get("/api/telemetry/engine/{unit_nr}")
def get_engine_telemetry(unit_nr: int, split: str = Query("test", regex="^(train|test)$")):
    """Returns time-series telemetry records for a requested engine unit."""
    df = CACHE["test_df"] if split == "test" else CACHE["train_df"]
    if df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded.")

    sub = df[df["unit_nr"] == unit_nr].sort_values("time_cycles")
    if sub.empty:
        raise HTTPException(status_code=404, detail=f"Unit {unit_nr} not found in {split} set.")

    # Select informative sensor channels
    channels = config.INFORMATIVE_SENSORS
    records = []
    for _, row in sub.iterrows():
        item = {
            "cycle": int(row["time_cycles"]),
            "rul": float(row["RUL"]) if "RUL" in row else None,
            "maintenance_risk": int(row["maintenance_risk"]) if "maintenance_risk" in row else None,
        }
        for ch in channels:
            item[ch] = float(row[ch])
        records.append(item)

    return {
        "unit_nr": unit_nr,
        "split": split,
        "total_cycles": len(records),
        "max_cycle": int(sub["time_cycles"].max()),
        "records": records,
        "sensor_descriptions": config.SENSOR_DESCRIPTIONS
    }


class PrognosticRequest(BaseModel):
    unit_nr: int
    cutoff_cycle: int


@app.post("/api/prognostics/predict")
def predict_prognostics(req: PrognosticRequest):
    """
    Computes point-in-time rolling features using cycles 1..cutoff_cycle
    and executes both RUL Regression and Maintenance Risk models.
    """
    test_df = CACHE["test_df"]
    reg_pipe = CACHE["regression_pipeline"]
    clf_pipe = CACHE["classification_pipeline"]

    if test_df is None or reg_pipe is None or clf_pipe is None:
        raise HTTPException(status_code=500, detail="Models or datasets not loaded.")

    unit_records = test_df[test_df["unit_nr"] == req.unit_nr].sort_values("time_cycles")
    if unit_records.empty:
        raise HTTPException(status_code=404, detail=f"Engine #{req.unit_nr} not found.")

    max_observed = int(unit_records["time_cycles"].max())
    if req.cutoff_cycle < 1 or req.cutoff_cycle > max_observed:
        raise HTTPException(status_code=400, detail=f"Cutoff must be between 1 and {max_observed}.")

    # Slice history with ZERO future data leakage
    history_sub = unit_records[unit_records["time_cycles"] <= req.cutoff_cycle].copy()
    feat_sub, _ = generate_engine_features(history_sub)

    reg_features = CACHE.get("reg_metadata", {}).get("feature_names", [])
    clf_features = CACHE.get("clf_metadata", {}).get("feature_names", [])

    if not reg_features or not clf_features:
        # Fallback to feature columns excluding index
        reg_features = [c for c in feat_sub.columns if c not in config.INDEX_COLS and c != "RUL"]
        clf_features = reg_features

    current_reg_feat = feat_sub.iloc[-1:][reg_features]
    current_clf_feat = feat_sub.iloc[-1:][clf_features]

    # Predict RUL
    predicted_rul = float(reg_pipe.predict(current_reg_feat)[0])
    predicted_rul = max(0.0, predicted_rul)

    # Predict Risk
    risk_class = int(clf_pipe.predict(current_clf_feat)[0])
    risk_proba = float(clf_pipe.predict_proba(current_clf_feat)[0][1]) if hasattr(clf_pipe, "predict_proba") else 0.5

    is_last_cycle = (req.cutoff_cycle == max_observed)
    ground_truth_rul = float(history_sub.iloc[-1]["RUL"]) if "RUL" in history_sub.columns else None

    # Error calculation if ground truth is available
    error = None
    if ground_truth_rul is not None:
        error = round(predicted_rul - ground_truth_rul, 2)

    health_margin_pct = round(min(100.0, max(0.0, (predicted_rul / config.RUL_CAP) * 100.0)), 1)

    return {
        "unit_nr": req.unit_nr,
        "cutoff_cycle": req.cutoff_cycle,
        "max_observed_cycle": max_observed,
        "is_final_cycle": is_last_cycle,
        "predicted_rul": round(predicted_rul, 1),
        "ground_truth_rul": round(ground_truth_rul, 1) if ground_truth_rul is not None else None,
        "prediction_error": error,
        "health_margin_pct": health_margin_pct,
        "risk_classification": {
            "predicted_class": risk_class,
            "label": "CRITICAL RISK" if risk_class == 1 else "NOMINAL",
            "probability": round(risk_proba, 4),
            "confidence_pct": round(risk_proba * 100, 1),
            "threshold_cycles": config.MAINTENANCE_RISK_THRESHOLD
        }
    }


@app.get("/api/benchmarks")
def get_benchmarks():
    """Returns empirical benchmark tables for Regression and Classification."""
    return {
        "regression": CACHE["benchmark_regression"],
        "classification": CACHE["benchmark_classification"]
    }


@app.get("/api/clusters")
def get_clusters():
    """Returns K-Means unsupervised clustering data and profile."""
    prof_file = config.RESULTS_DIR / "metrics" / "cluster_profiles.csv"
    profiles = []
    if prof_file.exists():
        profiles = pd.read_csv(prof_file).to_dict(orient="records")

    return {
        "metadata": CACHE.get("km_metadata", {}),
        "profiles": profiles
    }


# -----------------------------------------------------------------------------
# Static Files & Frontend Mounts
# -----------------------------------------------------------------------------
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
FIGURES_DIR = config.RESULTS_DIR / "figures"
PUBLIC_DIR = Path(__file__).resolve().parent / "public"
STATIC_DIR = Path(__file__).resolve().parent / "static"

if (PUBLIC_DIR / "data").exists():
    app.mount("/data", StaticFiles(directory=str(PUBLIC_DIR / "data")), name="data")

if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

if FIGURES_DIR.exists():
    app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")

if PUBLIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(PUBLIC_DIR), html=True), name="public")
elif STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"[START] Starting Machine Health Intelligence Server on http://0.0.0.0:{port}")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
