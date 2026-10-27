"""
Unit tests for data loading and RUL target calculation logic.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.data_loader import load_raw_data, compute_train_rul, compute_test_rul


def test_load_raw_data():
    """Verify raw C-MAPSS dataset loads with expected shapes and columns."""
    train_df, test_df, rul_df = load_raw_data()
    
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert isinstance(rul_df, pd.DataFrame)
    
    assert len(train_df.columns) == 26
    assert len(test_df.columns) == 26
    assert len(train_df) == 20631
    assert len(test_df) == 13096
    assert len(rul_df) == 100
    
    # Null check
    assert train_df.isnull().sum().sum() == 0
    assert test_df.isnull().sum().sum() == 0


def test_compute_train_rul():
    """Verify train RUL calculations, monotonic decline, and piece-wise clipping."""
    train_df, _, _ = load_raw_data()
    labeled = compute_train_rul(train_df, rul_cap=125, risk_threshold=30)
    
    assert "RUL" in labeled.columns
    assert "RUL_raw" in labeled.columns
    assert "maintenance_risk" in labeled.columns
    
    # Check engine 1 trajectory
    eng1 = labeled[labeled["unit_nr"] == 1]
    assert eng1.iloc[-1]["RUL_raw"] == 0
    assert eng1.iloc[0]["RUL_raw"] == len(eng1) - 1
    
    # Check capping
    assert labeled["RUL"].max() <= 125
    
    # Check risk label definition
    risk_mask = labeled["RUL"] <= 30
    assert (labeled.loc[risk_mask, "maintenance_risk"] == 1).all()
    assert (labeled.loc[~risk_mask, "maintenance_risk"] == 0).all()


def test_compute_test_rul():
    """Verify test RUL aligns with ground truth at last cycle."""
    _, test_df, rul_df = load_raw_data()
    labeled = compute_test_rul(test_df, rul_df, rul_cap=None, risk_threshold=30)
    
    # Engine 1 final cycle RUL must match row 0 of RUL_FD001.txt
    eng1 = labeled[labeled["unit_nr"] == 1]
    final_cycle_rul = eng1.iloc[-1]["RUL_raw"]
    expected_rul = rul_df.iloc[0, 0]
    
    assert final_cycle_rul == expected_rul
