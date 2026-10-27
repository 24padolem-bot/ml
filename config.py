"""
Configuration Module for Machine Health Intelligence.
Centralized settings for dataset paths, feature definitions, model hyperparameters,
and experimental thresholds.
"""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"
REGRESSION_MODELS_DIR = MODELS_DIR / "regression"
CLASSIFICATION_MODELS_DIR = MODELS_DIR / "classification"

RESULTS_DIR = BASE_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = BASE_DIR / "reports"

# Dataset file names (NASA C-MAPSS FD001)
DATASET_SUBSET = "FD001"
TRAIN_FILE = RAW_DATA_DIR / f"train_{DATASET_SUBSET}.txt"
TEST_FILE = RAW_DATA_DIR / f"test_{DATASET_SUBSET}.txt"
RUL_FILE = RAW_DATA_DIR / f"RUL_{DATASET_SUBSET}.txt"

# Dataset download mirror
DATASET_DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData/"
)

# Column names according to NASA C-MAPSS specifications
INDEX_COLS = ["unit_nr", "time_cycles"]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
ALL_RAW_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

# Constant / zero-variance features in FD001 (empirically confirmed std = 0)
# These are uninformative and cause multicollinearity if retained.
CONSTANT_COLS = ["setting_3", "s_1", "s_5", "s_10", "s_16", "s_18", "s_19"]

# Informative features retained for baseline modeling
INFORMATIVE_SETTINGS = [col for col in SETTING_COLS if col not in CONSTANT_COLS]
INFORMATIVE_SENSORS = [col for col in SENSOR_COLS if col not in CONSTANT_COLS]
INFORMATIVE_RAW_FEATURES = INFORMATIVE_SETTINGS + INFORMATIVE_SENSORS

# Meaningful sensor physical descriptions for interpretability & UI
SENSOR_DESCRIPTIONS = {
    "s_1": "Fan inlet temperature (°R)",
    "s_2": "LPC outlet temperature (°R)",
    "s_3": "HPC outlet temperature (°R)",
    "s_4": "LPT outlet temperature (°R)",
    "s_5": "Fan inlet pressure (psia)",
    "s_6": "Bypass duct pressure (psia)",
    "s_7": "HPC outlet pressure (psia)",
    "s_8": "Physical fan speed (rpm)",
    "s_9": "Physical core speed (rpm)",
    "s_10": "Engine pressure ratio (P50/P2)",
    "s_11": "HPC outlet static pressure (psia)",
    "s_12": "Ratio of fuel flow to Ps30 (pps/psia)",
    "s_13": "Corrected fan speed (rpm)",
    "s_14": "Corrected core speed (rpm)",
    "s_15": "Bypass Ratio",
    "s_16": "Burner bleed pressure (psia)",
    "s_17": "Bleed Enthalpy",
    "s_18": "Demanded fan speed (rpm)",
    "s_19": "Demanded corrected fan speed (rpm)",
    "s_20": "HPT coolant bleed (lbm/s)",
    "s_21": "LPT coolant bleed (lbm/s)"
}

# Target configuration
# RUL clipping (piece-wise linear model)
# Rationale: early in engine life, degradation is imperceptible and negligible.
# Capping prevents models from fitting noise in the initial flat degradation phase.
RUL_CAP = 125

# Maintenance-risk classification threshold (in cycles)
# An engine with RUL <= 30 cycles is defined as approaching critical maintenance.
MAINTENANCE_RISK_THRESHOLD = 30

# Feature engineering parameters
ROLLING_WINDOWS = [5, 15]

# Reproducibility
RANDOM_SEED = 42

# Train-Validation split ratio (engine-level)
VAL_ENGINE_RATIO = 0.2
N_SPLITS_CV = 5
