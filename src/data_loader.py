"""
Data Loader Module for NASA C-MAPSS Turbofan Engine Dataset.
Handles downloading, parsing, validating, and calculating target RUL and risk labels.
"""

import os
import urllib.request
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def download_dataset_if_missing(raw_dir: Optional[Path] = None, subset: str = config.DATASET_SUBSET) -> bool:
    """
    Downloads the NASA C-MAPSS dataset subset if not already present.
    Returns True if files exist or were downloaded successfully, False otherwise.
    """
    target_dir = Path(raw_dir) if raw_dir else config.RAW_DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    files = [f"train_{subset}.txt", f"test_{subset}.txt", f"RUL_{subset}.txt"]
    all_present = all((target_dir / f).exists() and (target_dir / f).stat().st_size > 0 for f in files)
    if all_present:
        return True

    print(f"[DataLoader] Dataset files not found in {target_dir}. Attempting automated download...")
    base_url = config.DATASET_DOWNLOAD_URL

    for f in files:
        dest = target_dir / f
        if not dest.exists() or dest.stat().st_size == 0:
            url = base_url + f
            try:
                print(f"[DataLoader] Downloading {url} -> {dest}...")
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=30) as resp, open(dest, 'wb') as out_f:
                    out_f.write(resp.read())
            except Exception as e:
                print(f"[DataLoader] Failed to download {f} automatically: {e}")
                print(f"[DataLoader] Please manually download {f} and place it into {target_dir}.")
                return False
    return True


def load_raw_data(
    train_path: Optional[Path] = None,
    test_path: Optional[Path] = None,
    rul_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads raw C-MAPSS train, test, and test RUL datasets.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, test_df, rul_df)
    """
    train_file = Path(train_path) if train_path else config.TRAIN_FILE
    test_file = Path(test_path) if test_path else config.TEST_FILE
    rul_file = Path(rul_path) if rul_path else config.RUL_FILE

    if not (train_file.exists() and test_file.exists() and rul_file.exists()):
        success = download_dataset_if_missing()
        if not success:
            raise FileNotFoundError(
                f"Missing dataset files in {config.RAW_DATA_DIR}. "
                f"Please ensure train_FD001.txt, test_FD001.txt, and RUL_FD001.txt are present."
            )

    train_df = pd.read_csv(train_file, sep=r"\s+", header=None, names=config.ALL_RAW_COLS)
    test_df = pd.read_csv(test_file, sep=r"\s+", header=None, names=config.ALL_RAW_COLS)
    rul_df = pd.read_csv(rul_file, sep=r"\s+", header=None, names=["RUL_ground_truth"])

    # Basic structural validations
    assert not train_df.empty, "train_df is empty"
    assert not test_df.empty, "test_df is empty"
    assert not rul_df.empty, "rul_df is empty"
    assert len(train_df.columns) == len(config.ALL_RAW_COLS), "Unexpected column count in train_df"

    return train_df, test_df, rul_df


def compute_train_rul(
    train_df: pd.DataFrame,
    rul_cap: Optional[int] = config.RUL_CAP,
    risk_threshold: int = config.MAINTENANCE_RISK_THRESHOLD
) -> pd.DataFrame:
    """
    Computes Remaining Useful Life (RUL) for training engine trajectories.
    For run-to-failure engines:
        RUL_raw(t) = max_cycle_for_engine - t
    Optionally applies a piecewise linear cap:
        RUL_capped(t) = min(RUL_raw(t), rul_cap)
    Also defines binary maintenance risk target:
        maintenance_risk = 1 if RUL <= risk_threshold else 0
    """
    df = train_df.copy()
    
    # Calculate maximum cycle per engine unit
    max_cycles = df.groupby("unit_nr")["time_cycles"].max().reset_index()
    max_cycles.rename(columns={"time_cycles": "max_cycle"}, inplace=True)
    
    # Merge max cycle and calculate RUL
    df = df.merge(max_cycles, on="unit_nr", how="left")
    df["RUL_raw"] = df["max_cycle"] - df["time_cycles"]
    
    if rul_cap is not None:
        df["RUL"] = np.minimum(df["RUL_raw"], rul_cap)
    else:
        df["RUL"] = df["RUL_raw"]

    df["maintenance_risk"] = (df["RUL"] <= risk_threshold).astype(int)
    
    return df


def compute_test_rul(
    test_df: pd.DataFrame,
    rul_df: pd.DataFrame,
    rul_cap: Optional[int] = config.RUL_CAP,
    risk_threshold: int = config.MAINTENANCE_RISK_THRESHOLD
) -> pd.DataFrame:
    """
    Computes true RUL across all test engine observations.
    For test engines censored before failure:
        RUL_raw(t) = (max_recorded_cycle - t) + true_final_RUL
    At the last cycle (t = max_recorded_cycle), RUL_raw equals true_final_RUL.
    """
    df = test_df.copy()
    
    # Map engine ID (1-based index) to ground truth final RUL
    rul_truth = rul_df.copy()
    rul_truth["unit_nr"] = np.arange(1, len(rul_truth) + 1)
    
    # Max recorded cycle in test set
    max_cycles = df.groupby("unit_nr")["time_cycles"].max().reset_index()
    max_cycles.rename(columns={"time_cycles": "max_recorded_cycle"}, inplace=True)
    
    # Merge
    df = df.merge(max_cycles, on="unit_nr", how="left")
    df = df.merge(rul_truth, on="unit_nr", how="left")
    
    # Calculate RUL at each time cycle
    df["RUL_raw"] = (df["max_recorded_cycle"] - df["time_cycles"]) + df["RUL_ground_truth"]
    
    if rul_cap is not None:
        df["RUL"] = np.minimum(df["RUL_raw"], rul_cap)
    else:
        df["RUL"] = df["RUL_raw"]

    df["maintenance_risk"] = (df["RUL"] <= risk_threshold).astype(int)
    
    return df


if __name__ == "__main__":
    train_raw, test_raw, rul_raw = load_raw_data()
    train_labeled = compute_train_rul(train_raw)
    test_labeled = compute_test_rul(test_raw, rul_raw)
    print(f"Data loading verified:")
    print(f"Train samples: {len(train_labeled)}, Columns: {len(train_labeled.columns)}")
    print(f"Test samples: {len(test_labeled)}, Columns: {len(test_labeled.columns)}")
    print(f"Train risk balance:\n{train_labeled['maintenance_risk'].value_counts(normalize=True)}")
