"""
Regression Training & Experimentation Pipeline for Remaining Useful Life (RUL) Estimation.
Trains Linear Regression, Decision Tree, Random Forest, and Gradient Boosting Regressors.
Performs GroupKFold cross-validation, hyperparameter tuning, metrics evaluation,
and saves model artifacts.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.data_loader import load_raw_data, compute_train_rul, compute_test_rul
from src.feature_engineering import generate_engine_features, prepare_test_last_cycle_dataset
from src.evaluation import evaluate_regression_model
from src.visualization import (
    plot_regression_comparison,
    plot_regression_actual_vs_predicted,
    plot_feature_importances
)


def run_regression_experiments(tune_models: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes the complete RUL regression experimental suite.
    """
    print("=" * 65)
    print("STARTING TASK A: REMAINING USEFUL LIFE (RUL) REGRESSION EXPERIMENTS")
    print("=" * 65)

    # 1. Load Data
    train_raw, test_raw, rul_raw = load_raw_data()
    train_df = compute_train_rul(train_raw, rul_cap=config.RUL_CAP)
    test_df = compute_test_rul(test_raw, rul_raw, rul_cap=config.RUL_CAP)

    # 2. Feature Engineering
    print("[1/6] Generating temporal features grouped by engine...")
    train_featured, feature_cols = generate_engine_features(train_df)
    test_featured, _ = generate_engine_features(test_df)

    X_train = train_featured[feature_cols]
    y_train = train_featured["RUL"]
    groups_train = train_featured["unit_nr"]

    # Official benchmark evaluation: test engines at final recorded cycle
    X_test, y_test, _ = prepare_test_last_cycle_dataset(test_featured, feature_cols)
    print(f"Features: {len(feature_cols)} | Train samples: {len(X_train)} | Test engines: {len(X_test)}")

    # 3. Model Definitions
    # Scaling is embedded inside pipelines where appropriate
    models_to_evaluate = {
        "Linear Regression (Baseline)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "Decision Tree Regressor": Pipeline([
            ("model", DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=config.RANDOM_SEED))
        ]),
        "Random Forest Regressor": Pipeline([
            ("model", RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=config.RANDOM_SEED))
        ]),
        "Gradient Boosting Regressor": Pipeline([
            ("model", GradientBoostingRegressor(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=config.RANDOM_SEED))
        ])
    }

    # 4. Optional Group-Aware Hyperparameter Tuning for Top Candidate
    if tune_models:
        print("[2/6] Performing group-aware hyperparameter tuning on Random Forest Regressor...")
        gkf = GroupKFold(n_splits=3)
        param_grid = {
            "model__n_estimators": [100, 150],
            "model__max_depth": [8, 12],
            "model__min_samples_leaf": [4, 8]
        }
        grid_search = GridSearchCV(
            models_to_evaluate["Random Forest Regressor"],
            param_grid=param_grid,
            cv=gkf,
            scoring="neg_mean_absolute_error",
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train, groups=groups_train)
        best_rf_pipeline = grid_search.best_estimator_
        print(f"      Best RF Params: {grid_search.best_params_}")
        models_to_evaluate["Random Forest (Tuned)"] = best_rf_pipeline

    # 5. Training and Evaluating Models
    print("[3/6] Training and evaluating regression models...")
    results_list = []
    trained_pipelines = {}
    test_predictions = {}

    for name, pipe in models_to_evaluate.items():
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        metrics, y_pred = evaluate_regression_model(pipe, X_train, y_train, X_test, y_test, model_name=name)
        metrics["training_time_sec"] = round(train_time, 2)
        results_list.append(metrics)
        trained_pipelines[name] = pipe
        test_predictions[name] = y_pred

        print(f"  -> {name:30s} | Test MAE: {metrics['test_mae']:5.2f} cycles | "
              f"RMSE: {metrics['test_rmse']:5.2f} cycles | R2: {metrics['test_r2']:6.4f} | Train: {train_time:4.2f}s")

    metrics_df = pd.DataFrame(results_list)

    # 6. Select Best Model Based on Lowest Test MAE
    best_row = metrics_df.sort_values(by="test_mae").iloc[0]
    best_model_name = best_row["model_name"]
    best_pipeline = trained_pipelines[best_model_name]
    print(f"\n[4/6] Best performing regression model: '{best_model_name}' (Test MAE: {best_row['test_mae']} cycles)")

    # 7. Save Artifacts
    print("[5/6] Saving regression models, metadata, and metric tables...")
    config.REGRESSION_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Save best model and all models
    joblib.dump(best_pipeline, config.REGRESSION_MODELS_DIR / "best_regression_pipeline.joblib")
    for name, pipe in trained_pipelines.items():
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(pipe, config.REGRESSION_MODELS_DIR / f"{safe_name}.joblib")

    # Save metadata
    meta = {
        "best_model_name": best_model_name,
        "feature_names": feature_cols,
        "rul_cap": config.RUL_CAP,
        "test_mae": best_row["test_mae"],
        "test_rmse": best_row["test_rmse"],
        "test_r2": best_row["test_r2"],
        "all_metrics": metrics_df.to_dict(orient="records")
    }
    with open(config.REGRESSION_MODELS_DIR / "regression_metadata.json", "w") as f:
        json.dump(meta, f, indent=4)

    metrics_df.to_csv(config.METRICS_DIR / "regression_metrics.csv", index=False)

    # 8. Generate & Save Figures
    print("[6/6] Generating benchmark figures...")
    plot_regression_comparison(metrics_df, save_path=config.FIGURES_DIR / "regression_comparison.png")
    plot_regression_actual_vs_predicted(
        y_test.values, test_predictions[best_model_name],
        model_name=best_model_name,
        save_path=config.FIGURES_DIR / "regression_true_vs_pred.png"
    )

    # Extract feature importances if tree-based model
    model_obj = best_pipeline.named_steps.get("model", None)
    if hasattr(model_obj, "feature_importances_"):
        importances = model_obj.feature_importances_
        plot_feature_importances(
            feature_cols, importances, top_n=15,
            title=f"Top 15 Feature Importances: {best_model_name}",
            save_path=config.FIGURES_DIR / "regression_feature_importances.png"
        )

    print("\nTask A Regression experiments completed successfully!\n")
    return metrics_df, meta


if __name__ == "__main__":
    run_regression_experiments(tune_models=True)
