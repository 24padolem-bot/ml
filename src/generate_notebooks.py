"""
Script to create the 5 clean, runnable Jupyter notebooks in notebooks/
using nbformat.
"""

from pathlib import Path
import json
import nbformat as nbf

NOTEBOOKS_DIR = Path("c:/Users/HP/OneDrive/Desktop/ml/notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

# 1. 01_data_inspection.ipynb
nb1 = nbf.v4.new_notebook()
nb1.cells = [
    nbf.v4.new_markdown_cell("""# 01. Dataset Inspection & Structural Verification
**Project:** Machine Health Intelligence  
**Dataset:** NASA C-MAPSS Turbofan Engine Degradation Simulation (FD001)

This notebook inspects the raw data files (`train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt`), confirms column schemas, verifies the absence of missing values, and identifies zero-variance constant features."""),
    nbf.v4.new_code_cell("""import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

import pandas as pd
import numpy as np
import config
from src.data_loader import load_raw_data, compute_train_rul, compute_test_rul

print("Loading raw C-MAPSS FD001 data...")
train_raw, test_raw, rul_raw = load_raw_data()
print(f"Train Raw Shape: {train_raw.shape}")
print(f"Test Raw Shape:  {test_raw.shape}")
print(f"RUL Truth Shape: {rul_raw.shape}")"""),
    nbf.v4.new_code_cell("""# Check missing values
print("Train Nulls:", train_raw.isnull().sum().sum())
print("Test Nulls: ", test_raw.isnull().sum().sum())
print("RUL Nulls:  ", rul_raw.isnull().sum().sum())"""),
    nbf.v4.new_code_cell("""# Identify zero-variance features in FD001
variances = train_raw[config.SETTING_COLS + config.SENSOR_COLS].std()
constant_cols = variances[variances == 0.0].index.tolist()
print("Zero-variance features identified for systematic removal:", constant_cols)""")
]

with open(NOTEBOOKS_DIR / "01_data_inspection.ipynb", "w") as f:
    nbf.write(nb1, f)
print("Created 01_data_inspection.ipynb")

# 2. 02_eda.ipynb
nb2 = nbf.v4.new_notebook()
nb2.cells = [
    nbf.v4.new_markdown_cell("""# 02. Exploratory Data Analysis (EDA)
**Project:** Machine Health Intelligence

Analyzes engine lifespan distributions, operational sensor trajectories across cycles, correlation heatmaps, and early-stage vs late-stage degradation patterns."""),
    nbf.v4.new_code_cell("""import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config
from src.data_loader import load_raw_data, compute_train_rul
from src.visualization import (
    plot_engine_cycle_distribution,
    plot_sensor_distributions,
    plot_correlation_heatmap,
    plot_sensor_trajectories,
    plot_early_vs_late_comparison
)

train_raw, _, _ = load_raw_data()
train_df = compute_train_rul(train_raw, rul_cap=config.RUL_CAP)
print(f"Loaded {len(train_df)} observations across {train_df['unit_nr'].nunique()} engines.")"""),
    nbf.v4.new_code_cell("""# Engine lifespan distribution
plot_engine_cycle_distribution(train_df)
plt.show()"""),
    nbf.v4.new_code_cell("""# Sensor trajectories for representative engines
plot_sensor_trajectories(train_df, engine_ids=[1, 2, 3], sensors=["s_2", "s_3", "s_4", "s_11"])
plt.show()"""),
    nbf.v4.new_code_cell("""# Early vs Late degradation boxplots
plot_early_vs_late_comparison(train_df, sensors=["s_2", "s_3", "s_4", "s_11", "s_12", "s_15"])
plt.show()""")
]

with open(NOTEBOOKS_DIR / "02_eda.ipynb", "w") as f:
    nbf.write(nb2, f)
print("Created 02_eda.ipynb")

# 3. 03_regression.ipynb
nb3 = nbf.v4.new_notebook()
nb3.cells = [
    nbf.v4.new_markdown_cell("""# 03. Machine Learning Task A: Remaining Useful Life (RUL) Regression
**Project:** Machine Health Intelligence

Evaluates Linear Regression, Decision Tree, Random Forest, and Gradient Boosting Regressors.
Includes GroupKFold cross-validation, hyperparameter tuning, and official test set evaluation."""),
    nbf.v4.new_code_cell("""import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

import pandas as pd
import config
from src.train_regression import run_regression_experiments

metrics_df, metadata = run_regression_experiments(tune_models=False)
display(metrics_df)""")
]

with open(NOTEBOOKS_DIR / "03_regression.ipynb", "w") as f:
    nbf.write(nb3, f)
print("Created 03_regression.ipynb")

# 4. 04_classification.ipynb
nb4 = nbf.v4.new_notebook()
nb4.cells = [
    nbf.v4.new_markdown_cell("""# 04. Machine Learning Task B: Maintenance-Risk Classification
**Project:** Machine Health Intelligence

Evaluates Logistic Regression, Decision Tree, Random Forest, and Support Vector Classifier (SVC).
Addresses class imbalance via balanced weighting and evaluates Precision, Recall, F1, ROC-AUC, and Confusion Matrices."""),
    nbf.v4.new_code_cell("""import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

import pandas as pd
import config
from src.train_classification import run_classification_experiments

metrics_df, metadata = run_classification_experiments(tune_models=False)
display(metrics_df)""")
]

with open(NOTEBOOKS_DIR / "04_classification.ipynb", "w") as f:
    nbf.write(nb4, f)
print("Created 04_classification.ipynb")

# 5. 05_clustering.ipynb
nb5 = nbf.v4.new_notebook()
nb5.cells = [
    nbf.v4.new_markdown_cell("""# 05. Machine Learning Task C: Unsupervised Operating Pattern Clustering
**Project:** Machine Health Intelligence

Investigates latent machine operating states using K-Means clustering.
Applies the Elbow method (Inertia), Silhouette Score analysis, and 2D PCA projection to correlate clusters with physical degradation stages."""),
    nbf.v4.new_code_cell("""import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

import pandas as pd
import config
from src.clustering import run_clustering_experiments

metadata, profiles = run_clustering_experiments(selected_k=3)
display(profiles)""")
]

with open(NOTEBOOKS_DIR / "05_clustering.ipynb", "w") as f:
    nbf.write(nb5, f)
print("Created 05_clustering.ipynb")
