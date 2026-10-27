"""
Vercel Serverless Function entrypoint for Machine Health Intelligence.
Serves FastAPI REST APIs on Vercel.
"""

from pathlib import Path
import json
import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Machine Health Intelligence - Vercel API",
    description="Prognostics and Predictive Maintenance Serverless API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "public" / "data"

def load_json(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "service": "machine-health-intelligence",
        "runtime": "Vercel Serverless",
        "dataset": "NASA C-MAPSS FD001"
    }

@app.get("/api/fleet/summary")
def get_fleet_summary():
    data = load_json(DATA_DIR / "fleet_summary.json")
    if data:
        return data
    return {
        "fleet_size": 100,
        "train_units": 100,
        "total_historical_cycles": 20631,
        "rul_cap": 125,
        "risk_threshold": 30,
        "models": {
            "best_regression": {"name": "Random Forest Regressor", "test_mae": 12.02, "test_rmse": 16.53, "test_r2": 0.8299},
            "best_classification": {"name": "Random Forest Classifier", "accuracy": 0.950, "f1_score": 0.8936, "recall": 0.840, "roc_auc": 0.9840}
        },
        "feature_count": 78
    }

@app.get("/api/telemetry/engines")
def get_engines():
    data = load_json(DATA_DIR / "engines_list.json")
    if data:
        return data
    return {"test_units": list(range(1, 101)), "train_units": list(range(1, 101))}

@app.get("/api/telemetry/engine/{unit_nr}")
def get_engine_telemetry(unit_nr: int, split: str = Query("test")):
    unit_file = DATA_DIR / "engines" / f"unit_{unit_nr}.json"
    data = load_json(unit_file)
    if data:
        return data
    raise HTTPException(status_code=404, detail=f"Engine #{unit_nr} not found")

class PrognosticRequest(BaseModel):
    unit_nr: int
    cutoff_cycle: int

@app.post("/api/prognostics/predict")
def predict_prognostics(req: PrognosticRequest):
    unit_file = DATA_DIR / "engines" / f"unit_{req.unit_nr}.json"
    data = load_json(unit_file)
    if not data:
        raise HTTPException(status_code=404, detail=f"Engine #{req.unit_nr} not found")
    
    preds = data.get("predictions", {})
    cycle_key = str(req.cutoff_cycle)
    if cycle_key in preds:
        return preds[cycle_key]
    
    # If exact cycle not in keys, fallback to closest available cycle
    available_cycles = sorted([int(k) for k in preds.keys()])
    if available_cycles:
        closest = min(available_cycles, key=lambda x: abs(x - req.cutoff_cycle))
        return preds[str(closest)]
    
    raise HTTPException(status_code=400, detail="Prediction not found for cutoff")

@app.get("/api/benchmarks")
def get_benchmarks():
    data = load_json(DATA_DIR / "benchmarks.json")
    if data:
        return data
    return {"regression": [], "classification": []}
