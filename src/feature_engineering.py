"""
Feature Engineering Module for Machine Health Intelligence.
Constructs leakage-free temporal features (rolling mean, rolling std, lag differences)
strictly grouped by engine unit (unit_nr) without crossing engine boundaries.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def generate_engine_features(
    df: pd.DataFrame,
    sensor_cols: Optional[List[str]] = None,
    windows: List[int] = config.ROLLING_WINDOWS,
    include_lag: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Generates temporal features grouped strictly by engine (unit_nr).
    Ensures observations are sorted by unit_nr and time_cycles.
    Rolling windows and differences never cross engine boundaries.
    
    Returns:
        featured_df: DataFrame containing original + newly engineered features.
        feature_columns: List of all numeric feature column names ready for modeling.
    """
    data = df.sort_values(by=["unit_nr", "time_cycles"]).copy()
    
    sensors = sensor_cols if sensor_cols is not None else config.INFORMATIVE_SENSORS
    settings = [c for c in config.SETTING_COLS if c not in config.CONSTANT_COLS and c in data.columns]
    
    # Base feature list
    feature_cols = ["time_cycles"] + settings + sensors
    
    # 1. Rolling Mean across windows (e.g., 5, 15)
    for w in windows:
        for s in sensors:
            col_name = f"{s}_roll_mean_{w}"
            # min_periods=1 ensures no NaNs at the beginning of an engine's trajectory
            data[col_name] = (
                data.groupby("unit_nr")[s]
                .rolling(window=w, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            feature_cols.append(col_name)

    # 2. Rolling Standard Deviation (window=15)
    w_std = 15
    for s in sensors:
        col_name = f"{s}_roll_std_{w_std}"
        data[col_name] = (
            data.groupby("unit_nr")[s]
            .rolling(window=w_std, min_periods=1)
            .std()
            .fillna(0.0)
            .reset_index(level=0, drop=True)
        )
        feature_cols.append(col_name)

    # 3. Lag-1 Differences (rate of change across consecutive cycles)
    if include_lag:
        for s in sensors:
            col_name = f"{s}_lag_diff_1"
            data[col_name] = (
                data.groupby("unit_nr")[s]
                .diff(periods=1)
                .fillna(0.0)
            )
            feature_cols.append(col_name)

    # Verify no NaN values introduced in feature columns
    nan_count = data[feature_cols].isnull().sum().sum()
    if nan_count > 0:
        raise ValueError(f"Feature engineering generated {nan_count} unexpected NaNs.")

    return data, feature_cols


def prepare_test_last_cycle_dataset(
    test_featured_df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Extracts the final observation for each engine in the test set.
    This corresponds directly to the official NASA C-MAPSS RUL evaluation benchmark,
    where RUL_FD001.txt specifies the remaining useful life at the final recorded cycle.
    
    Returns:
        X_test_last: Feature matrix of shape (100, n_features)
        y_test_last_rul: True RUL at final cycle (from ground truth)
        y_test_last_risk: True binary maintenance risk class
    """
    last_indices = test_featured_df.groupby("unit_nr")["time_cycles"].idxmax()
    last_obs_df = test_featured_df.loc[last_indices].sort_values("unit_nr").copy()
    
    X_test_last = last_obs_df[feature_cols]
    y_test_last_rul = last_obs_df["RUL"]
    y_test_last_risk = last_obs_df["maintenance_risk"]
    
    return X_test_last, y_test_last_rul, y_test_last_risk


if __name__ == "__main__":
    from src.data_loader import load_raw_data, compute_train_rul, compute_test_rul
    train_raw, test_raw, rul_raw = load_raw_data()
    train_labeled = compute_train_rul(train_raw)
    test_labeled = compute_test_rul(test_raw, rul_raw)
    
    train_feat, feat_names = generate_engine_features(train_labeled)
    test_feat, _ = generate_engine_features(test_labeled)
    
    print(f"Feature engineering verified:")
    print(f"Engineered feature count: {len(feat_names)}")
    print(f"Features preview: {feat_names[:10]} ...")
    
    X_test_last, y_rul, y_risk = prepare_test_last_cycle_dataset(test_feat, feat_names)
    print(f"Test final cycle dataset shape: {X_test_last.shape}")
    print(f"Test final cycle RUL mean: {y_rul.mean():.2f}, Risk positive rate: {y_risk.mean():.2%}")
