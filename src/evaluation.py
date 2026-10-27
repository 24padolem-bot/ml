"""
Evaluation Module for Machine Health Intelligence.
Computes comprehensive metrics for regression (MAE, RMSE, R2, latency)
and classification (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix).
"""

import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def evaluate_regression_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Regressor"
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Evaluates a regression model on both training and test data.
    Computes MAE, RMSE, R2, and inference latency.
    """
    # Inference timing on test set
    start_time = time.perf_counter()
    y_pred_test = model.predict(X_test)
    inference_time_ms = (time.perf_counter() - start_time) * 1000.0

    y_pred_train = model.predict(X_train)

    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_r2 = r2_score(y_train, y_pred_train)

    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)

    metrics = {
        "model_name": model_name,
        "train_mae": round(float(train_mae), 3),
        "train_rmse": round(float(train_rmse), 3),
        "train_r2": round(float(train_r2), 4),
        "test_mae": round(float(test_mae), 3),
        "test_rmse": round(float(test_rmse), 3),
        "test_r2": round(float(test_r2), 4),
        "inference_time_ms": round(float(inference_time_ms), 2),
        "inference_per_sample_ms": round(float(inference_time_ms / len(X_test)), 4)
    }

    return metrics, y_pred_test


def evaluate_classification_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Classifier"
) -> Tuple[Dict[str, Any], np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Evaluates a classification model on test set.
    Computes Accuracy, Precision, Recall, F1, Macro-F1, ROC-AUC, and Confusion Matrix.
    """
    start_time = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_time_ms = (time.perf_counter() - start_time) * 1000.0

    # Probabilities for ROC-AUC
    y_prob = None
    roc_auc = np.nan
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_prob)
        except Exception:
            pass
    elif hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(X_test)
            roc_auc = roc_auc_score(y_test, scores)
        except Exception:
            pass

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "model_name": model_name,
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "macro_f1": round(float(macro_f1), 4),
        "roc_auc": round(float(roc_auc), 4) if not np.isnan(roc_auc) else "N/A",
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "inference_time_ms": round(float(inference_time_ms), 2)
    }

    return metrics, y_pred, y_prob, cm
