"""
Unit tests for leakage-free feature engineering.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.data_loader import load_raw_data, compute_train_rul
from src.feature_engineering import generate_engine_features, prepare_test_last_cycle_dataset


def test_no_cross_engine_leakage():
    """Verify rolling statistics do not leak across engine boundaries."""
    # Create synthetic test dataset with two engines
    data = {
        "unit_nr": [1, 1, 1, 2, 2, 2],
        "time_cycles": [1, 2, 3, 1, 2, 3],
        "setting_1": [0.0] * 6,
        "setting_2": [0.0] * 6,
        "s_2": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0]
    }
    df = pd.DataFrame(data)
    
    featured, cols = generate_engine_features(
        df, sensor_cols=["s_2"], windows=[2], include_lag=True
    )
    
    # Engine 2 at cycle 1 must have lag_diff_1 == 0 and rolling_mean_2 == 100.0 (not affected by Engine 1's 30.0)
    eng2_cycle1 = featured[(featured["unit_nr"] == 2) & (featured["time_cycles"] == 1)].iloc[0]
    assert eng2_cycle1["s_2_roll_mean_2"] == 100.0
    assert eng2_cycle1["s_2_lag_diff_1"] == 0.0


def test_real_feature_matrix_integrity():
    """Verify feature generation on real training data produces valid shape and zero NaNs."""
    train_raw, _, _ = load_raw_data()
    train_df = compute_train_rul(train_raw)
    
    # Test on subset of engines for fast test execution
    subset = train_df[train_df["unit_nr"].isin([1, 2, 3])].copy()
    featured, cols = generate_engine_features(subset)
    
    assert len(featured) == len(subset)
    assert len(cols) > 20
    assert featured[cols].isnull().sum().sum() == 0
