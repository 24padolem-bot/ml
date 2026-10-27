# Machine Health Intelligence: Turbofan Engine Degradation & RUL Prognostics

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Vercel Deployment](https://img.shields.io/badge/Vercel-Live%20Demo-000000.svg?logo=vercel&logoColor=white)](https://ml-kappa-orcin.vercel.app)
[![NASA Dataset](https://img.shields.io/badge/Dataset-NASA%20C--MAPSS%20FD001-E03C31.svg)](https://data.nasa.gov)
[![Tests Passing](https://img.shields.io/badge/Tests-7%20Passed-2ea44f.svg)](#test-suite)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An end-to-end prognostics and health management (PHM) system built on NASA's C-MAPSS turbofan engine degradation dataset (FD001). The pipeline couples leak-free feature engineering with supervised remaining useful life (RUL) regression, binary maintenance-risk classification, and unsupervised degradation-phase clustering, served via a FastAPI backend and a responsive monitoring dashboard deployed on Vercel.

**Live Application:** [https://ml-kappa-orcin.vercel.app](https://ml-kappa-orcin.vercel.app)  
**Repository:** [https://github.com/Bhumika197/ml](https://github.com/Bhumika197/ml)

---

## Table of Contents

- [Problem Formulation](#problem-formulation)
- [System Architecture](#system-architecture)
- [Dataset & Preprocessing](#dataset--preprocessing)
- [Feature Engineering & Leakage Prevention](#feature-engineering--leakage-prevention)
- [Experimental Benchmarks](#experimental-benchmarks)
  - [Task A: Remaining Useful Life Regression](#task-a-remaining-useful-life-regression)
  - [Task B: Maintenance-Risk Classification](#task-b-maintenance-risk-classification)
  - [Task C: Unsupervised Health-Phase Clustering](#task-c-unsupervised-health-phase-clustering)
- [REST API Reference](#rest-api-reference)
- [Installation & Quickstart](#installation--quickstart)
- [Test Suite](#test-suite)
- [Project Structure](#project-structure)
- [Team Roles & Module Ownership](#team-roles--module-ownership)
- [Operational Assumptions & Limitations](#operational-assumptions--limitations)
- [References](#references)

---

## Problem Formulation

In safety-critical aviation systems, aircraft gas-turbine engines undergo severe thermal stress, mechanical friction, and aerodynamic wear over successive flight cycles. Traditional maintenance strategies rely on either reactive run-to-failure policies (incurring hazardous unscheduled downtime) or conservative calendar-based overhauls (discarding safe components prematurely).

This project implements a multi-task prognostics framework addressing three operational needs:

1. **RUL Regression (Task A):** Estimates the continuous number of flight cycles remaining before an engine reaches operational failure threshold, supporting depot-level scheduling.
2. **Maintenance-Risk Classification (Task B):** Triage indicator alerting ground personnel when an engine enters a critical 30-cycle operating window ($RUL \le 30$), providing a clear binary call to action.
3. **Unsupervised Degradation Clustering (Task C):** Discovers latent health stages across high-dimensional telemetry without requiring ground-truth labels, characterizing early baseline, transition, and terminal degradation regimes.

---

## System Architecture

The pipeline processes raw turbofan sensor streams into production-grade inferences through a modular architecture:

```
[NASA C-MAPSS FD001 Telemetry]
       │
       ▼
[Data Ingestion & Verification]  ──▶  src/data_loader.py
       │
       ▼
[Sensor Pruning & Validation]    ──▶  src/preprocessing.py
       │ (drops 7 zero-variance channels)
       ▼
[Causal Feature Engineering]     ──▶  src/feature_engineering.py
       │ (15-cycle rolling stats strictly partitioned by unit_nr)
       ├────────────────────────┬────────────────────────┐
       ▼                        ▼                        ▼
[Task A: RUL Regression]  [Task B: Risk Classifier]  [Task C: K-Means Clustering]
(Random Forest: 12.02 MAE) (Random Forest: 95.0% Acc) (k=3, PCA: 74.17% Var)
       │                        │                        │
       └────────────────────────┼────────────────────────┘
                                ▼
               [FastAPI Inference Engine & REST API] (server.py / api/index.py)
                                │
                                ▼
               [Interactive Responsive Dashboard] (public/ / static/)
               - Fleet status overview & search
               - Real-time engine telemetry inspection
               - Interactive RUL trajectory forecast
               - Off-canvas drawer & mobile-optimized UI
```

---

## Dataset & Preprocessing

The model is trained and benchmarked on the **NASA C-MAPSS FD001** simulation benchmark:

- **Operating Regime:** Sea-level, single flight condition (Sea Level Altitude, Mach 0.0, Sea Level Temp).
- **Fault Mode:** High-Pressure Compressor (HPC) degradation.
- **Training Fleet:** 100 engines run to terminal failure ($20{,}631$ time cycles total).
- **Testing Fleet:** 100 operational engines with truncated trajectories, accompanied by official NASA final-cycle ground-truth values in `RUL_FD001.txt`.
- **Sensor Pruning:** 7 out of 21 sensor channels exhibit zero or near-zero variance under FD001 operating conditions (Sensors 1, 5, 6, 10, 16, 18, 19). These are systematically pruned to eliminate numerical instability and reduce model dimensionality.
- **Piece-wise Linear Target Formulation:** During early operating cycles, turbofan engines operate in nominal conditions without detectable mechanical degradation. Following the standard PHM formulation (Saxena et al., 2008), target RUL is capped at $RUL_{cap} = 125$ cycles:
  
  $$\text{RUL}(t) = \min(T_i - t, 125)$$

  where $T_i$ is the failure cycle for unit $i$ and $t$ is the elapsed cycle count.

---

## Feature Engineering & Leakage Prevention

Temporal leakage is the most prevalent pitfall in published predictive maintenance benchmarks. This pipeline enforces three strict architectural safeguards:

1. **Engine Group Isolation:** All rolling operations (15-cycle mean and standard deviation) and first-order difference features are grouped strictly by `unit_nr`. Historical windows never span across engine boundaries.
2. **Strictly Causal Lookbacks:** For cycle $t$, features are calculated exclusively using observations from cycles $1 \dots t$. No centered windows, future offsets, or global aggregates are computed.
3. **No Random Row-Wise Splitting:** Shuffling time-series observations into random training and validation splits causes severe cross-cycle contamination. All internal validation uses `GroupKFold` stratified by engine unit, and final model evaluation is performed exclusively on the 100 unseen test units.

---

## Experimental Benchmarks

All models were evaluated on the **100 unseen test engines** using official NASA ground truth. Models were tuned via 5-fold `GroupKFold` cross-validation on the training set.

### Task A: Remaining Useful Life Regression

| Model | Test MAE (Cycles) | Test RMSE (Cycles) | Test $R^2$ Score | Train Time (s) | Inference (ms/sample) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression (Baseline)** | 14.80 | 19.42 | 0.7652 | 0.26 | 0.05 |
| **Decision Tree Regressor** | 13.44 | 18.18 | 0.7941 | 1.15 | 0.05 |
| **Gradient Boosting Regressor** | 12.35 | 17.17 | 0.8164 | 90.18 | 0.12 |
| **Random Forest (Tuned)** | 12.15 | 16.74 | 0.8255 | 55.20 | 0.45 |
| **Random Forest Regressor (Best)** | **12.02** | **16.53** | **0.8299** | 33.91 | 0.42 |

*Selected Model:* The Random Forest Regressor achieved an MAE of 12.02 cycles on unseen engines, outperforming baseline linear regression by 18.8% while maintaining sub-millisecond per-sample inference latency.

### Task B: Maintenance-Risk Classification

Evaluated on binary classification indicating whether an engine has entered the urgent risk window ($RUL \le 30$ cycles):

| Model | Accuracy | Recall (Sensitivity) | Precision | $F_1$-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 94.00% | **92.00%** | 85.19% | 0.8846 | **0.9861** |
| **Decision Tree Classifier** | 91.00% | 88.00% | 78.57% | 0.8302 | 0.9509 |
| **Support Vector Classifier (SVC)** | 91.00% | 84.00% | 80.77% | 0.8235 | 0.9829 |
| **Random Forest (Tuned)** | 94.00% | 80.00% | 95.24% | 0.8696 | 0.9808 |
| **Random Forest Classifier (Best)** | **95.00%** | 84.00% | **95.45%** | **0.8936** | 0.9819 |

*Selected Model:* The Random Forest Classifier achieved 95.00% accuracy and an $F_1$-score of 0.8936 on the test fleet, with 95.45% precision minimizing costly false-positive maintenance flags.

### Task C: Unsupervised Health-Phase Clustering

K-Means clustering ($k=3$) was fitted on scaled sensor features and projected onto two principal components via PCA:

- **Variance Explained:** PC1 accounts for 60.18%, PC2 accounts for 13.99% (Cumulative = **74.17%**).
- **Cluster Profiles:**
  - **Cluster 0 (Nominal Baseline):** Mean RUL = $148.5$ cycles. Engines operating within baseline thermodynamic boundaries.
  - **Cluster 1 (Transition Phase):** Mean RUL = $86.1$ cycles. Early divergence in LPC outlet temperature (Sensor 2) and HPC outlet pressure (Sensor 7).
  - **Cluster 2 (Terminal Degradation):** Mean RUL = $27.4$ cycles. Severe degradation signature characterized by high fuel flow ratio and elevated exhaust gas temperatures.

---

## REST API Reference

The FastAPI service exposes clean REST endpoints for batch inspection and point-in-time prognostics:

| Method | Endpoint | Description | Query / Body Parameters |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/fleet/summary` | Fleet-wide risk breakdown, engine count, and unit list | None |
| `GET` | `/api/fleet/unit/{id}` | Complete historical telemetry trajectory and sensor readings for a unit | `split` (`test` or `train`) |
| `POST` | `/api/prognostics/predict` | Point-in-time leak-free RUL regression and risk classification | `{"unit_nr": 1, "cutoff_cycle": 31}` |
| `GET` | `/api/benchmarks` | Full empirical benchmark tables for all models | None |
| `GET` | `/api/clusters` | K-Means cluster profiles and PCA variance metrics | None |

### Example Request & Response

```bash
curl -X POST "https://ml-kappa-orcin.vercel.app/api/prognostics/predict" \
     -H "Content-Type: application/json" \
     -d '{"unit_nr": 1, "cutoff_cycle": 31}'
```

```json
{
  "unit_nr": 1,
  "cutoff_cycle": 31,
  "max_observed_cycle": 31,
  "is_final_cycle": true,
  "predicted_rul": 112.4,
  "ground_truth_rul": 112.0,
  "prediction_error": 0.4,
  "health_margin_pct": 89.9,
  "risk_classification": {
    "predicted_class": 0,
    "label": "NOMINAL",
    "probability": 0.0412,
    "confidence_pct": 4.1,
    "threshold_cycles": 30
  }
}
```

---

## Installation & Quickstart

### Prerequisites

- Python 3.10, 3.11, or 3.12
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/Bhumika197/ml.git
cd ml
```

### 2. Set Up a Virtual Environment

```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download and Verify Data

Downloads NASA C-MAPSS FD001 raw archive and validates integrity:

```bash
python -m src.data_loader
```

### 5. Run the Complete ML Pipeline (Optional)

Trained model artifacts and evaluation metrics are pre-serialized in `models/` and `results/metrics/`. To re-train from scratch:

```bash
# Train RUL regression models
python -m src.train_regression

# Train risk classification models
python -m src.train_classification

# Execute unsupervised K-Means & PCA clustering
python -m src.clustering
```

### 6. Launch the Local Web Service

```bash
python server.py
```

The application will start at **[http://localhost:8000](http://localhost:8000)**. Interactive API documentation is available at **[http://localhost:8000/docs](http://localhost:8000/docs)**.

---

## Test Suite

Automated tests cover data ingestion, causal windowing constraints, and model prediction pipelines:

```bash
pytest tests/ -v
```

```
================================ test session starts ================================
collected 7 items

tests/test_data_loader.py::test_raw_data_shape_and_columns PASSED             [ 14%]
tests/test_data_loader.py::test_rul_computation_bounds PASSED                 [ 28%]
tests/test_features.py::test_no_cross_unit_leakage PASSED                     [ 42%]
tests/test_features.py::test_causal_rolling_consistency PASSED                [ 57%]
tests/test_predictions.py::test_regression_pipeline_inference PASSED         [ 71%]
tests/test_predictions.py::test_classification_pipeline_inference PASSED     [ 85%]
tests/test_predictions.py::test_endpoint_prognostics_schema PASSED           [100%]

================================= 7 passed in 8.67s =================================
```

---

## Project Structure

```
ml/
├── api/
│   └── index.py               # Vercel Serverless Function entrypoint (FastAPI handler)
├── public/                    # Production static dashboard assets (served via Vercel / FastAPI)
│   ├── index.html             # Application dashboard interface
│   ├── style.css              # Custom responsive stylesheet (mobile drawer, light theme)
│   ├── app.js                 # Frontend application logic, telemetry charts, search & alerts
│   ├── assets/                # Core diagrams and static visual assets
│   ├── figures/               # 12 Matplotlib/Seaborn analytical figures
│   └── data/                  # Precomputed telemetry JSON payloads
├── src/                       # Core machine learning package
│   ├── data_loader.py         # C-MAPSS dataset parsing & piecewise target computation
│   ├── preprocessing.py       # Zero-variance channel pruning & unit-level splits
│   ├── feature_engineering.py # Leakage-safe grouped rolling features (mean, std, deltas)
│   ├── train_regression.py    # Regression training, GroupKFold CV & hyperparameter tuning
│   ├── train_classification.py# Risk classification pipeline & evaluation
│   ├── clustering.py          # K-Means clustering & 2D PCA projection pipeline
│   ├── evaluation.py          # Reusable error metrics (MAE, RMSE, R2, F1, ROC-AUC)
│   ├── visualization.py       # Publication-quality plotting routines
│   └── generate_eda_figures.py# Standalone script generating the 12 analysis plots
├── models/                    # Serialized model artifacts (.joblib)
│   ├── regression/            # Trained Random Forest, GBR, Decision Tree, Linear Regressor
│   └── classification/        # Trained Random Forest, SVC, Logistic Regression, Decision Tree
├── results/
│   ├── metrics/               # Empirical evaluation benchmark tables (.csv & .json)
│   └── figures/               # High-resolution generated analytical plots
├── notebooks/                 # Clean, runnable analytical notebooks
│   ├── 01_data_inspection.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_regression.ipynb
│   ├── 04_classification.ipynb
│   └── 05_clustering.ipynb
├── tests/                     # Pytest automated test suite
│   ├── test_data_loader.py
│   ├── test_features.py
│   └── test_predictions.py
├── config.py                  # Global project hyperparameters and path constants
├── server.py                  # Standalone FastAPI production application server
├── app.py                     # Convenience launcher for the web application
├── vercel.json                # Vercel deployment routes and serverless configuration
├── requirements.txt           # Python dependency specifications
└── README.md                  # Project documentation
```

---

## Team Roles & Module Ownership

| Member | Primary Focus Area | Key Deliverables |
| :--- | :--- | :--- |
| **Data Engineering** | Telemetry Ingestion & Causal Preprocessing | Automated C-MAPSS loader (`src/data_loader.py`), zero-variance sensor pruning (`src/preprocessing.py`), unit-isolated rolling feature engineering (`src/feature_engineering.py`), automated test assertions (`tests/test_features.py`). |
| **Prognostics & Modeling** | Supervised Modeling & Evaluation | Regression & classification model development (`src/train_regression.py`, `src/train_classification.py`), `GroupKFold` cross-validation strategy, hyperparameter tuning, empirical benchmark generation (`results/metrics/`). |
| **Analytics & Deployment** | Clustering, API & Application Architecture | Unsupervised K-Means and PCA pipelines (`src/clustering.py`), FastAPI backend (`server.py`, `api/index.py`), responsive frontend UI (`public/`), Plotly telemetry visualization, and Vercel cloud deployment. |

---

## Operational Assumptions & Limitations

1. **Simulation Model Bounds:** The dataset originates from the NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS). While thermodynamically faithful, it does not incorporate environmental extremes such as bird ingestion, volcanic ash exposure, or erratic pilot throttling.
2. **Single Operational Regime (FD001):** The FD001 subset assumes sea-level flight conditions with a single failure mode (High-Pressure Compressor degradation). Multi-regime datasets (FD002/FD004) require operational normalization across multiple flight conditions.
3. **Decision-Support Context:** This application is designed as an advisory decision-support system for maintenance engineers and does not replace FAA/EASA certified airborne engine management systems.

---

## References

1. Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). *Damage propagation modeling for aircraft engine run-to-failure simulation*. In 2008 IEEE International Conference on Prognostics and Health Management (pp. 1-9). [https://doi.org/10.1109/PHM.2008.4578638](https://doi.org/10.1109/PHM.2008.4578638)
2. Ramasso, E., & Saxena, A. (2014). *Performance benchmarking and analysis of prognostic methods for turbofan engines*. International Journal of Prognostics and Health Management, 5(2), 1-15.
3. Heimes, F. O. (2008). *Recurrent neural networks for remaining useful life estimation*. In 2008 IEEE International Conference on Prognostics and Health Management (pp. 1-6). [https://doi.org/10.1109/PHM.2008.4578637](https://doi.org/10.1109/PHM.2008.4578637)
4. Li, X., Ding, Q., & Sun, J. Q. (2018). *Remaining useful life estimation in prognostics using deep convolution neural networks*. Reliability Engineering & System Safety, 172, 1-11. [https://doi.org/10.1016/j.ress.2017.11.021](https://doi.org/10.1016/j.ress.2017.11.021)
5. Zheng, S., Ristovski, K., Farahat, A., & Gupta, C. (2017). *Long short-term memory network for remaining useful life estimation*. In 2017 IEEE International Conference on Prognostics and Health Management (pp. 88-95). [https://doi.org/10.1109/ICPHM.2017.7998311](https://doi.org/10.1109/ICPHM.2017.7998311)
