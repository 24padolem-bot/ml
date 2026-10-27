"""
Preprocessing Module for Machine Health Intelligence.
Provides leakage-free engine-level train/validation splitting, constant column removal,
and Scikit-learn scaler pipelines fitted strictly on training engines.
"""

from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def drop_constant_columns(df: pd.DataFrame, constant_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Drops zero-variance and constant columns from DataFrame.
    """
    cols_to_drop = constant_cols if constant_cols is not None else config.CONSTANT_COLS
    existing_drops = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=existing_drops)


def split_train_val_by_engine(
    train_df: pd.DataFrame,
    val_ratio: float = config.VAL_ENGINE_RATIO,
    random_seed: int = config.RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Performs engine-level group splitting.
    Ensures that observations from the same engine unit NEVER appear in both train and validation sets.
    
    Returns:
        train_split_df: Observations belonging to training engines.
        val_split_df: Observations belonging to validation engines.
        train_engine_ids: Array of engine unit IDs in training split.
        val_engine_ids: Array of engine unit IDs in validation split.
    """
    unique_engines = train_df["unit_nr"].unique()
    rng = np.random.default_rng(random_seed)
    shuffled_engines = rng.permutation(unique_engines)
    
    val_count = int(np.round(len(unique_engines) * val_ratio))
    val_engines = shuffled_engines[:val_count]
    train_engines = shuffled_engines[val_count:]
    
    train_split_df = train_df[train_df["unit_nr"].isin(train_engines)].copy()
    val_split_df = train_df[train_df["unit_nr"].isin(val_engines)].copy()
    
    return train_split_df, val_split_df, train_engines, val_engines


class SensorPreprocessor:
    """
    Preprocessing pipeline wrapper that:
    1. Keeps track of active feature columns.
    2. Drops zero-variance sensor measurements.
    3. Fits a standard or min-max scaler on training data only.
    4. Transforms training, validation, or inference data consistently.
    """
    def __init__(self, scaler_type: str = "standard", drop_constants: bool = True):
        self.scaler_type = scaler_type
        self.drop_constants = drop_constants
        self.scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        self.feature_names: List[str] = []
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, feature_cols: List[str]):
        """
        Fits the scaler on specified feature columns of training observations.
        """
        cols = [c for c in feature_cols if c not in config.CONSTANT_COLS] if self.drop_constants else feature_cols
        self.feature_names = cols
        self.scaler.fit(df[self.feature_names])
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies learned scaling transformation to feature columns.
        Preserves metadata columns (unit_nr, time_cycles, RUL, maintenance_risk).
        """
        if not self.is_fitted:
            raise ValueError("SensorPreprocessor must be fitted before transform.")
        
        # Verify required features exist
        missing = [c for c in self.feature_names if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required feature columns for preprocessing: {missing}")
        
        scaled_array = self.scaler.transform(df[self.feature_names])
        scaled_df = pd.DataFrame(scaled_array, columns=self.feature_names, index=df.index)
        
        # Keep non-feature metadata
        meta_cols = [c for c in df.columns if c not in self.feature_names]
        result = pd.concat([df[meta_cols], scaled_df], axis=1)
        return result

    def fit_transform(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        self.fit(df, feature_cols)
        return self.transform(df)

    def save(self, filepath: Path):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: Path) -> "SensorPreprocessor":
        return joblib.load(filepath)


if __name__ == "__main__":
    from src.data_loader import load_raw_data, compute_train_rul
    train_raw, _, _ = load_raw_data()
    train_labeled = compute_train_rul(train_raw)
    
    train_split, val_split, t_ids, v_ids = split_train_val_by_engine(train_labeled)
    print(f"Engine-level split: {len(t_ids)} train engines ({len(train_split)} rows), "
          f"{len(v_ids)} val engines ({len(val_split)} rows)")
    
    preprocessor = SensorPreprocessor(scaler_type="standard")
    scaled_train = preprocessor.fit_transform(train_split, config.INFORMATIVE_RAW_FEATURES)
    scaled_val = preprocessor.transform(val_split)
    print(f"Preprocessor fit verified. Feature columns count: {len(preprocessor.feature_names)}")
