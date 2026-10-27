"""
Integration tests for saved model predictions and inference pipelines.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.data_loader import load_raw_data, compute_test_rul
from src.feature_engineering import generate_engine_features, prepare_test_last_cycle_dataset


def test_saved_regression_model():
    """Verify saved best regression pipeline loads and predicts valid RUL values."""
    model_path = config.REGRESSION_MODELS_DIR / "best_regression_pipeline.joblib"
    assert model_path.exists(), "Regression model artifact missing!"
    
    pipeline = joblib.load(model_path)
    
    # Load test data
    _, test_raw, rul_raw = load_raw_data()
    test_df = compute_test_rul(test_raw, rul_raw)
    test_feat, feature_cols = generate_engine_features(test_df)
    X_test_last, _, _ = prepare_test_last_cycle_dataset(test_feat, feature_cols)
    
    predictions = pipeline.predict(X_test_last)
    assert len(predictions) == 100
    assert not np.isnan(predictions).any()
    # Predictions should be positive RUL cycles
    assert (predictions >= 0).all()


def test_saved_classification_model():
    """Verify saved best classification pipeline loads, outputs binary labels and probabilities."""
    model_path = config.CLASSIFICATION_MODELS_DIR / "best_classification_pipeline.joblib"
    assert model_path.exists(), "Classification model artifact missing!"
    
    pipeline = joblib.load(model_path)
    
    _, test_raw, rul_raw = load_raw_data()
    test_df = compute_test_rul(test_raw, rul_raw)
    test_feat, feature_cols = generate_engine_features(test_df)
    X_test_last, _, _ = prepare_test_last_cycle_dataset(test_feat, feature_cols)
    
    preds = pipeline.predict(X_test_last)
    assert len(preds) == 100
    assert set(np.unique(preds)).issubset({0, 1})
    
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(X_test_last)
        assert probs.shape == (100, 2)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()
