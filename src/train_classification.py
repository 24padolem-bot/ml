"""
Classification Training & Experimentation Pipeline for Maintenance-Risk Prediction.
Trains Logistic Regression, Decision Tree, Random Forest, and Support Vector Classifier (SVC).
Addresses class imbalance with balanced weighting, evaluates Accuracy, Precision,
Recall, F1-Score, Macro-F1, ROC-AUC, Confusion Matrices, and saves model artifacts.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.data_loader import load_raw_data, compute_train_rul, compute_test_rul
from src.feature_engineering import generate_engine_features, prepare_test_last_cycle_dataset
from src.evaluation import evaluate_classification_model
from src.visualization import (
    plot_classification_confusion_matrices,
    plot_feature_importances
)


def run_classification_experiments(tune_models: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes the complete Maintenance-Risk classification experimental suite.
    """
    print("=" * 65)
    print("STARTING TASK B: MAINTENANCE-RISK CLASSIFICATION EXPERIMENTS")
    print("=" * 65)

    # 1. Load Data & Compute Binary Risk Target (RUL <= 30 cycles)
    train_raw, test_raw, rul_raw = load_raw_data()
    train_df = compute_train_rul(train_raw, rul_cap=config.RUL_CAP, risk_threshold=config.MAINTENANCE_RISK_THRESHOLD)
    test_df = compute_test_rul(test_raw, rul_raw, rul_cap=config.RUL_CAP, risk_threshold=config.MAINTENANCE_RISK_THRESHOLD)

    # 2. Feature Engineering
    print("[1/6] Generating temporal features grouped by engine...")
    train_featured, feature_cols = generate_engine_features(train_df)
    test_featured, _ = generate_engine_features(test_df)

    X_train = train_featured[feature_cols]
    y_train = train_featured["maintenance_risk"]
    groups_train = train_featured["unit_nr"]

    # Official benchmark evaluation: test engines at final recorded cycle
    X_test, _, y_test = prepare_test_last_cycle_dataset(test_featured, feature_cols)
    
    print(f"Features: {len(feature_cols)} | Train samples: {len(X_train)} | Test engines: {len(X_test)}")
    print(f"Train risk class balance: Negative (Nominal)={np.sum(y_train == 0)}, Positive (Risk)={np.sum(y_train == 1)} "
          f"({np.mean(y_train):.1%})")
    print(f"Test risk class balance: Negative (Nominal)={np.sum(y_test == 0)}, Positive (Risk)={np.sum(y_test == 1)} "
          f"({np.mean(y_test):.1%})")

    # 3. Model Definitions
    # Distance/gradient-based models use StandardScaler; tree-based models operate directly
    models_to_evaluate = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=config.RANDOM_SEED))
        ]),
        "Decision Tree Classifier": Pipeline([
            ("model", DecisionTreeClassifier(class_weight="balanced", max_depth=6, min_samples_leaf=15, random_state=config.RANDOM_SEED))
        ]),
        "Random Forest Classifier": Pipeline([
            ("model", RandomForestClassifier(n_estimators=120, max_depth=10, min_samples_leaf=4, class_weight="balanced", n_jobs=-1, random_state=config.RANDOM_SEED))
        ]),
        "Support Vector Classifier (SVM)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", C=1.0, class_weight="balanced", probability=True, random_state=config.RANDOM_SEED))
        ])
    }

    # 4. Group-Aware Hyperparameter Tuning for Random Forest
    if tune_models:
        print("[2/6] Performing group-aware hyperparameter tuning on Random Forest Classifier...")
        gkf = GroupKFold(n_splits=3)
        param_grid = {
            "model__n_estimators": [100, 150],
            "model__max_depth": [8, 12],
            "model__min_samples_leaf": [3, 6]
        }
        grid_search = GridSearchCV(
            models_to_evaluate["Random Forest Classifier"],
            param_grid=param_grid,
            cv=gkf,
            scoring="f1",
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train, groups=groups_train)
        best_rf = grid_search.best_estimator_
        print(f"      Best RF Classifier Params: {grid_search.best_params_}")
        models_to_evaluate["Random Forest (Tuned)"] = best_rf

    # 5. Training and Evaluating Models
    print("[3/6] Training and evaluating classification models...")
    results_list = []
    trained_pipelines = {}
    test_predictions = {}
    confusion_matrices = {}

    for name, pipe in models_to_evaluate.items():
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        metrics, y_pred, y_prob, cm = evaluate_classification_model(
            pipe, X_train, y_train, X_test, y_test, model_name=name
        )
        metrics["training_time_sec"] = round(train_time, 2)
        results_list.append(metrics)
        trained_pipelines[name] = pipe
        test_predictions[name] = y_pred
        confusion_matrices[name] = cm

        print(f"  -> {name:32s} | Acc: {metrics['accuracy']:6.2%} | Recall: {metrics['recall']:6.2%} | "
              f"Precision: {metrics['precision']:6.2%} | F1: {metrics['f1_score']:6.4f} | ROC-AUC: {metrics['roc_auc']} | Train: {train_time:4.2f}s")

    metrics_df = pd.DataFrame(results_list)

    # 6. Select Best Model Based on Highest F1-Score & Recall
    # In predictive maintenance, missing a failing engine (False Negative) is much more catastrophic
    # than inspecting an engine early (False Positive), so Recall and F1 are prioritized over raw Accuracy.
    best_row = metrics_df.sort_values(by=["f1_score", "recall"], ascending=False).iloc[0]
    best_model_name = best_row["model_name"]
    best_pipeline = trained_pipelines[best_model_name]
    print(f"\n[4/6] Best performing classification model: '{best_model_name}' (F1: {best_row['f1_score']}, Recall: {best_row['recall']:.2%})")

    # 7. Save Artifacts
    print("[5/6] Saving classification models, metadata, and metric tables...")
    config.CLASSIFICATION_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_pipeline, config.CLASSIFICATION_MODELS_DIR / "best_classification_pipeline.joblib")
    for name, pipe in trained_pipelines.items():
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(pipe, config.CLASSIFICATION_MODELS_DIR / f"{safe_name}.joblib")

    meta = {
        "best_model_name": best_model_name,
        "feature_names": feature_cols,
        "threshold_cycles": config.MAINTENANCE_RISK_THRESHOLD,
        "accuracy": best_row["accuracy"],
        "precision": best_row["precision"],
        "recall": best_row["recall"],
        "f1_score": best_row["f1_score"],
        "macro_f1": best_row["macro_f1"],
        "roc_auc": best_row["roc_auc"],
        "all_metrics": metrics_df.to_dict(orient="records")
    }
    with open(config.CLASSIFICATION_MODELS_DIR / "classification_metadata.json", "w") as f:
        json.dump(meta, f, indent=4)

    metrics_df.to_csv(config.METRICS_DIR / "classification_metrics.csv", index=False)

    # 8. Generate & Save Figures
    print("[6/6] Generating classification figures...")
    plot_classification_confusion_matrices(
        confusion_matrices,
        save_path=config.FIGURES_DIR / "classification_confusion_matrices.png"
    )

    # Feature Importance / Coefficients
    model_obj = best_pipeline.named_steps.get("model", None)
    if hasattr(model_obj, "feature_importances_"):
        plot_feature_importances(
            feature_cols, model_obj.feature_importances_, top_n=15,
            title=f"Top 15 Risk Classification Features: {best_model_name}",
            save_path=config.FIGURES_DIR / "classification_feature_importances.png"
        )
    elif hasattr(model_obj, "coef_"):
        coef_abs = np.abs(model_obj.coef_[0])
        plot_feature_importances(
            feature_cols, coef_abs, top_n=15,
            title=f"Top 15 Feature Weights: {best_model_name}",
            save_path=config.FIGURES_DIR / "classification_feature_importances.png"
        )

    print("\nTask B Classification experiments completed successfully!\n")
    return metrics_df, meta


if __name__ == "__main__":
    run_classification_experiments(tune_models=True)
