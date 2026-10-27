# Machine Health Intelligence: An Explainable Machine Learning System for Predictive Maintenance and Remaining Useful Life Estimation

**Authors:** Student 1 (Data & Preprocessing Lead), Student 2 (Modeling & Evaluation Lead), Student 3 (Clustering & Deployment Lead)  
**Academic Degree:** Bachelor of Technology in Computer Science and Engineering  
**Academic Course:** Machine Learning Mini-Project Evaluation (15 Marks)  
**Date:** October 2026  

---

## Abstract
Modern industrial machinery and aerospace turbofan engines degrade continuously under operational, mechanical, and aerodynamic stresses. Unanticipated mechanical failures result in catastrophic downtime, unscheduled grounding, expensive emergency overhauls, and potential safety risks. In response, predictive maintenance leverages condition-monitoring sensor streams to anticipate equipment failure before occurrence. This study presents **Machine Health Intelligence**, a comprehensive, modular, and leak-free machine learning system built on NASA's benchmark Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) turbofan engine degradation dataset (FD001). 

The system implements three complementary paradigms: (i) **Task A: Remaining Useful Life (RUL) Regression**, predicting remaining operational cycles via Linear Regression, Decision Tree Regressors, Random Forest Regressors, and Gradient Boosting Regressors; (ii) **Task B: Maintenance-Risk Classification**, categorizing equipment approaching a defined operational risk threshold ($RUL \le 30$ cycles) using Logistic Regression, Decision Tree Classifiers, Random Forest Classifiers, and Support Vector Classifiers (SVC); and (iii) **Task C: Unsupervised Operating State Clustering**, discovering latent degradation phases via K-Means clustering and Principal Component Analysis (PCA) on physical sensor signatures. 

To eliminate data leakage, we enforce engine-level group cross-validation (`GroupKFold`) and verify performance strictly on 100 unseen test engine trajectories evaluated against official ground-truth labels. Random Forest achieved the highest prognostic accuracy for RUL estimation ($\text{MAE} = 12.02\text{ cycles}$, $\text{RMSE} = 16.53\text{ cycles}$, $R^2 = 0.8299$), outperforming baseline Linear Regression ($\text{MAE} = 14.80\text{ cycles}$, $R^2 = 0.7652$). In risk classification, Random Forest achieved an $F_1\text{-score}$ of $0.8936$ ($95.00\%$ accuracy, $84.00\%$ recall), while Logistic Regression delivered maximum fault sensitivity ($92.00\%$ recall, $F_1 = 0.8846$, $\text{ROC-AUC} = 0.9861$). K-Means ($k=3$) successfully reconstructed physical degradation regimes from healthy baseline to severe late-stage wear ($74.17\%$ variance explained by 2D PCA). All models, metrics, and diagnostics are integrated into an interactive Streamlit dashboard.

---

## 1. Introduction & Problem Statement
Industrial asset management historically oscillated between two inefficient extremes:
1. **Corrective (Run-to-Failure) Maintenance:** Equipment is operated until physical failure. While maximizing part utilization, it incurs unpredictable downtime, collateral subsystem damage, and flight safety risks.
2. **Preventive (Time-Based) Maintenance:** Parts are overhauled or retired according to conservative fixed operational schedules. This discards components with substantial remaining life, driving up lifecycle operating costs.

**Predictive Maintenance (PdM)** bridges this gap by continuously monitoring sensor parameters to estimate remaining machine health, forecasting the **Remaining Useful Life (RUL)** in operational flight cycles.

### 1.1 Engineering Objectives
This project addresses four key objectives:
1. **Accurate Continuous RUL Prognostics:** Estimate how many cycles an engine can safely continue operating.
2. **Actionable Risk Triage:** Provide binary alert classification for maintenance crews when an engine enters a critical inspection window ($RUL \le 30$ cycles).
3. **Unsupervised Operating State Discovery:** Characterize multivariate sensor clusters to understand operating dynamics without supervision.
4. **Transparent Feature Attribution:** Identify which engine sensors (such as high-pressure compressor discharge temperatures and static pressures) drive degradation warnings.
5. **Interactive Decision-Support Deployment:** Deliver an intuitive, non-proprietary web dashboard that exposes point-in-time sensor history and real-time inference.

---

## 2. Literature Review
The NASA C-MAPSS dataset represents the definitive benchmark in predictive maintenance literature. To ground our work, we review five foundational and contemporary peer-reviewed studies:

### 2.1 Reviewed Research Papers
1. **Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008).**  
   *Title:* Damage propagation modeling for aircraft engine run-to-failure simulation.  
   *Venue:* 2008 IEEE International Conference on Prognostics and Health Management.  
   *DOI:* [10.1109/PHM.2008.4578638](https://doi.org/10.1109/PHM.2008.4578638)  
   *Focus:* Introduced the C-MAPSS simulation model, detailing thermodynamic physics and high-pressure compressor degradation parameters.  
   *Findings:* Demonstrated that individual engines exhibit heterogeneous baseline lifespans (128 to 362 cycles in FD001) due to manufacturing tolerances and initial wear margins.  
   *Relationship to Project:* Establishes the foundational physical basis for why engine ID boundaries must be strictly observed during feature engineering.

2. **Heimes, F. O. (2008).**  
   *Title:* Recurrent neural networks for remaining useful life estimation.  
   *Venue:* 2008 International Conference on Prognostics and Health Management.  
   *DOI:* [10.1109/PHM.2008.4578637](https://doi.org/10.1109/PHM.2008.4578637)  
   *Focus:* Applied recurrent architectures to benchmark C-MAPSS data, proposing the piece-wise linear RUL target function.  
   *Findings:* Proved that early in engine life, wear is imperceptible; capping the RUL target (e.g., at 125–130 cycles) prevents estimators from fitting noise during healthy operation.  
   *Relationship to Project:* Directly inspired our adoption of $RUL_{cap} = 125$ cycles.

3. **Ramasso, E., & Saxena, A. (2014).**  
   *Title:* Performance benchmarking and analysis of prognostic methods for turbofan engines.  
   *Venue:* International Journal of Prognostics and Health Management, 5(2), 1-15.  
   *Focus:* Evaluated statistical learning algorithms and classical regression models across C-MAPSS datasets.  
   *Findings:* Showed that feature engineering (rolling averages and temporal trend differences) substantially improves traditional tree and linear model performance to near-deep-learning levels on FD001.  
   *Relationship to Project:* Guides our temporal rolling window and lag feature pipeline.

4. **Zheng, S., Ristovski, K., Farahat, A., & Gupta, C. (2017).**  
   *Title:* Long short-term memory network for remaining useful life estimation.  
   *Venue:* 2017 IEEE International Conference on Prognostics and Health Management (ICPHM).  
   *DOI:* [10.1109/ICPHM.2017.7998311](https://doi.org/10.1109/ICPHM.2017.7998311)  
   *Focus:* Investigated multi-layer LSTM networks for sequence learning without manual feature extraction.  
   *Findings:* Achieved RMSE of ~16.1 cycles on FD001 test data, but noted high computational demands and opacity in feature attribution.  
   *Relationship to Project:* Our ensemble Random Forest model achieves comparable test accuracy ($\text{RMSE} = 16.53$ cycles) with full interpretability and sub-millisecond inference latency on standard student hardware.

5. **Li, X., Ding, Q., & Sun, J. Q. (2018).**  
   *Title:* Remaining useful life estimation in prognostics using deep convolution neural networks.  
   *Venue:* Reliability Engineering & System Safety, 172, 1-11.  
   *DOI:* [10.1016/j.ress.2017.11.021](https://doi.org/10.1016/j.ress.2017.11.021)  
   *Focus:* Explored 2D CNN filters over sliding time windows of sensor measurements.  
   *Findings:* Confirmed that static sensors with zero variance contribute zero diagnostic value and should be purged.  
   *Relationship to Project:* Confirms the validity of pruning the 7 invariant sensors in FD001 prior to training.

### 2.2 Literature Comparison Table

| Study | Model / Method | Target Metric | Strengths | Limitations | Relevance to Our Project |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Saxena et al. (2008)** | C-MAPSS Physics Simulation | Physics verification | Realistic thermodynamics | No ML prognostic algorithm | Establishes ground truth physics of FD001 |
| **Heimes (2008)** | Recurrent Neural Net (RNN) | C-MAPSS Scoring Metric | Introduced Piecewise RUL | Slow convergence, overfitting | Formulation of capped $RUL=125$ target |
| **Ramasso & Saxena (2014)** | Statistical & Tree Ensembles | RMSE, Score | Low latency, transparent rules | Required extensive feature engineering | Proves power of rolling temporal features |
| **Zheng et al. (2017)** | LSTM Neural Network | RMSE ~16.14 cycles | Captures long temporal dependencies | High compute overhead, black-box | Benchmark against which our RF is evaluated |
| **Li et al. (2018)** | Deep Convolutional Network | RMSE ~15.38 cycles | Automated spatial feature maps | GPU-dependent, opaque features | Justifies purging zero-variance sensors |
| **Our System (2026)** | **RF, GBR, Logistic Reg, SVC, K-Means** | **MAE 12.02, RMSE 16.53, F1 0.8936** | **Leakage-safe, interpretable, laptop-ready, full UI** | Focused on single-condition FD001 | **Complete pedagogical & decision-support system** |

---

## 3. Dataset Description
The system is built on NASA C-MAPSS dataset subset **FD001**.
- **Operating Conditions:** 1 (Sea-Level flight condition)
- **Fault Mode:** Single fault mode (High-Pressure Compressor degradation)
- **Training Trajectories (`train_FD001.txt`):** 100 engines monitored from nominal state until catastrophic failure ($20,631$ records).
- **Test Trajectories (`test_FD001.txt`):** 100 engines whose records terminate at an arbitrary cycle prior to failure ($13,096$ records).
- **Ground Truth Target (`RUL_FD001.txt`):** Actual remaining useful cycles for each test engine at its last recorded cycle ($100$ records).

### 3.1 Sensor Measurements & Physical Annotations
The raw dataset contains 26 numerical columns: Unit ID, Time Cycle, 3 Operational Settings, and 21 Sensors:

| Column | Physical Description | Units | Variance in FD001 | Action Taken |
| :--- | :--- | :--- | :--- | :--- |
| `setting_1` | Flight condition: Altitude | Nominal | Active ($\sigma > 0$) | Retained |
| `setting_2` | Flight condition: Mach number | Nominal | Active ($\sigma > 0$) | Retained |
| `setting_3` | Throttle resolver angle | Deg | Invariant ($\sigma = 0.0$) | **Dropped** |
| `s_1` | Fan inlet temperature | °R | Invariant ($\sigma = 0.0$) | **Dropped** |
| `s_2` | LPC outlet temperature | °R | Active ($\sigma = 0.50$) | Retained |
| `s_3` | HPC outlet temperature | °R | Active ($\sigma = 6.13$) | Retained |
| `s_4` | LPT outlet temperature | °R | Active ($\sigma = 9.00$) | Retained |
| `s_5` | Fan inlet pressure | psia | Invariant ($\sigma = 0.0$) | **Dropped** |
| `s_6` | Bypass duct pressure | psia | Active ($\sigma = 0.07$) | Retained |
| `s_7` | HPC outlet pressure | psia | Active ($\sigma = 0.89$) | Retained |
| `s_8` | Physical fan speed | rpm | Active ($\sigma = 0.07$) | Retained |
| `s_9` | Physical core speed | rpm | Active ($\sigma = 22.08$) | Retained |
| `s_10` | Engine pressure ratio | -- | Invariant ($\sigma = 0.0$) | **Dropped** |
| `s_11` | HPC outlet static pressure | psia | Active ($\sigma = 0.27$) | Retained |
| `s_12` | Ratio of fuel flow to Ps30 | pps/psia | Active ($\sigma = 0.74$) | Retained |
| `s_13` | Corrected fan speed | rpm | Active ($\sigma = 0.11$) | Retained |
| `s_14` | Corrected core speed | rpm | Active ($\sigma = 19.08$) | Retained |
| `s_15` | Bypass Ratio | -- | Active ($\sigma = 0.04$) | Retained |
| `s_16` | Burner bleed pressure | psia | Invariant ($\sigma = 0.0$) | **Dropped** |
| `s_17` | Bleed Enthalpy | -- | Active ($\sigma = 1.55$) | Retained |
| `s_18` | Demanded fan speed | rpm | Invariant ($\sigma = 0.0$) | **Dropped** |
| `s_19` | Demanded corrected fan speed | rpm | Invariant ($\sigma = 0.0$) | **Dropped** |
| `s_20` | HPT coolant bleed | lbm/s | Active ($\sigma = 0.18$) | Retained |
| `s_21` | LPT coolant bleed | lbm/s | Active ($\sigma = 0.11$) | Retained |

---

## 4. Exploratory Data Analysis & Preprocessing

### 4.1 Data Integrity & Zero Missing Values
Inspection of all $20,631$ training records and $13,096$ test records revealed **zero missing or null entries**, reflecting clean thermodynamic simulation output.

### 4.2 Engine Lifespan Heterogeneity
Analysis of maximum operating cycles reveals substantial variance:
- Minimum lifespan: $128$ cycles (Engine #39)
- Maximum lifespan: $362$ cycles (Engine #69)
- Median lifespan: $199$ cycles
- Mean lifespan: $206.3$ cycles

This variation proves that predicting RUL purely as a linear function of elapsed cycle is insufficient; degradation depends fundamentally on multivariate sensor drift.

### 4.3 Sensor Trend Dynamics
During degradation, the following key thermodynamic shifts occur:
1. **Temperature Surge:** As compressor blades wear and aerodynamic efficiency degrades, higher fuel flows are required to maintain thrust, causing temperatures `s_2` (LPC outlet), `s_3` (HPC outlet), and `s_4` (LPT outlet) to climb steadily.
2. **Pressure Drop & Rise:** High Pressure Compressor outlet static pressure (`s_11`) increases, while bypass ratio (`s_15`) drifts upwards.
3. **Core Speed Deviation:** Core shaft speeds (`s_9`, `s_14`) exhibit systematic drift as mechanical clearances widen.

---

## 5. Methodology & Leakage-Free Feature Engineering

### 5.1 Piece-Wise Linear RUL Formulation
For engine $i$ with failure cycle $T_i$, raw RUL at cycle $t$ is $RUL_{raw} = T_i - t$. Following Heimes (2008), we implement a piece-wise linear target function capped at $RUL_{cap} = 125$:
$$RUL(t) = \min(T_i - t, 125)$$
*Rationale:* In pristine operational states ($t \ll T_i$), physical damage has not yet initiated. Forcing a regression model to predict $250$ versus $220$ cycles when sensor signatures are indistinguishable forces it to fit random noise. Capping stabilizes early-life gradients.

### 5.2 Maintenance-Risk Classification Target
The continuous RUL is mapped into a binary operational classification target:
$$\text{Risk Class} = \begin{cases} 1 \text{ (Critical Maintenance Urgency)}, & \text{if } RUL \le 30 \text{ cycles} \\ 0 \text{ (Nominal Health)}, & \text{if } RUL > 30 \text{ cycles} \end{cases}$$
In the training data, $3,100$ observations ($15.0\%$) are Class 1, and $17,531$ observations ($85.0\%$) are Class 0. We compensate for this class imbalance using inverse frequency weighting (`class_weight='balanced'`).

### 5.3 Leakage-Safe Temporal Features
To capture degradation velocity and volatility without future leakage:
1. **Engine Grouping:** Data is sorted by `['unit_nr', 'time_cycles']`. All rolling calculations are strictly partitioned by `unit_nr`.
2. **Rolling Means:** Windows of $w \in \{5, 15\}$ cycles capture smoothed operating trajectories.
3. **Rolling Volatility:** Standard deviation over a $15$-cycle window captures turbulent sensor fluctuations indicative of mechanical instability.
4. **Lag Differences:** Lag-1 differences ($\Delta s_t = s_t - s_{t-1}$) represent rate of sensor change.
This expands the 17 informative raw features into **78 engineered features**.

### 5.4 Engine-Level Validation Strategy
Shuffling individual rows randomly into train/test sets causes **severe temporal and operational leakage**, as consecutive cycles from the same engine would be present in both sets. We enforce:
- **Group-Aware Splitting:** 80 training engines ($16,779$ rows) and 20 validation engines ($3,852$ rows).
- **Official Benchmark Test Set:** Evaluated exclusively on the 100 unseen test engines from `test_FD001.txt` paired with `RUL_FD001.txt`.

---

## 6. Experimental Results & Discussion

### 6.1 Task A: RUL Regression Benchmarks
All four regression architectures were evaluated on the 100 test engines at their final recorded observation:

| Regression Model | Test MAE (Cycles) | Test RMSE (Cycles) | Test $R^2$ | Train Time (s) | Inference Time (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression (Baseline)** | 14.80 | 19.42 | 0.7652 | 0.26 | 0.05 |
| **Decision Tree Regressor** | 13.44 | 18.18 | 0.7941 | 1.15 | 0.05 |
| **Gradient Boosting Regressor** | 12.35 | 17.17 | 0.8164 | 90.18 | 0.12 |
| **Random Forest (Tuned)** | 12.15 | 16.74 | 0.8255 | 55.20 | 0.45 |
| **Random Forest Regressor (Default)** | **12.02** | **16.53** | **0.8299** | **33.91** | **0.42** |

#### Discussion of Regression Performance
1. **Ensemble Superiority:** Random Forest achieved the lowest error ($\text{MAE} = 12.02$ cycles, $\text{RMSE} = 16.53$ cycles, $R^2 = 0.8299$). By averaging 100 de-correlated trees, variance is substantially attenuated relative to single Decision Trees ($\text{MAE} = 13.44$ cycles).
2. **Error Significance:** In practical terms, an MAE of 12 cycles means that for an engine with 75 remaining cycles, the system predicts within $\pm 12$ cycles—providing a reliable operational horizon for scheduling hangar maintenance.
3. **Linear Baseline Deficit:** Linear Regression produced an MAE of $14.80$ cycles. While fast ($0.26$s), it assumes strictly additive linear wear, underestimating non-linear exponential acceleration near failure.

---

### 6.2 Task B: Maintenance-Risk Classification Benchmarks
Evaluated on the 100 test engines with ground-truth binary labels ($25$ engines at high risk, $75$ nominal):

| Classification Model | Accuracy | Recall | Precision | $F_1$-Score | Macro-$F_1$ | ROC-AUC | Train Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 94.00% | **92.00%** | 85.19% | 0.8846 | 0.9228 | **0.9861** | 0.52 |
| **Decision Tree Classifier** | 91.00% | 88.00% | 78.57% | 0.8302 | 0.8850 | 0.9509 | 1.17 |
| **Support Vector Machine (SVC)** | 91.00% | 84.00% | 80.77% | 0.8235 | 0.8824 | 0.9829 | 45.16 |
| **Random Forest (Tuned)** | 94.00% | 80.00% | 95.24% | 0.8696 | 0.9157 | 0.9808 | 4.70 |
| **Random Forest Classifier (Default)** | **95.00%** | 84.00% | **95.45%** | **0.8936** | **0.9321** | 0.9819 | 3.87 |

#### Confusion Matrix Breakdown (100 Test Engines)

```
Random Forest Classifier:
                 Predicted Nominal (0)   Predicted Risk (1)
Actual Nominal:           74                      1          (FP = 1)
Actual Risk:               4                     21          (FN = 4)

Logistic Regression:
                 Predicted Nominal (0)   Predicted Risk (1)
Actual Nominal:           71                      4          (FP = 4)
Actual Risk:               2                     23          (FN = 2)
```

#### Operational Trade-Off Analysis: False Negatives vs. False Positives
In industrial aviation, the economic and safety costs of diagnostic errors are highly asymmetric:
- **False Negative (FN):** Model predicts engine is safe, but it actually has $\le 30$ cycles left. The engine continues flying towards in-flight shutdown or severe secondary damage. **Cost: Extremely High.**
- **False Positive (FP):** Model flags an engine for early inspection when it actually has $>30$ cycles left. The engine is inspected slightly early in the hangar. **Cost: Minor inspection overhead.**
- *Conclusion:* While Random Forest maximizes overall Accuracy ($95.00\%$) and Precision ($95.45\%$), Logistic Regression achieved superior safety recall ($92.00\%$, only 2 false negatives) with an outstanding ROC-AUC of $0.9861$. The system enables maintenance supervisors to calibrate this decision threshold based on organizational risk tolerance.

---

### 6.3 Task C: Unsupervised Operating Pattern Clustering
K-Means clustering was executed on the 15 informative standardized physical sensors.
- **Elbow Analysis:** Inertia declined sharply from $k=2$ ($186,975$) to $k=3$ ($148,340$), forming a distinct inflection elbow.
- **Silhouette Validation:** Silhouette score peaked at $k=2$ ($0.3533$) and stabilized at $k=3$ ($0.2564$).
- **PCA Dimensionality Reduction:** Principal Component 1 (PC1) accounts for **$60.18\%$** of sensor variance, and PC2 accounts for **$13.99\%$**, yielding a cumulative explained variance of **$74.17\%$**.

#### Cluster Profile Characterization

| Cluster ID | Engine Records | Mean Time Cycle | Mean True RUL (Cycles) | Physical HPC Temp `s_3` (°R) | Physical Core Speed `s_9` (rpm) | Inferred Operating Regime |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | 7,812 | 64.2 | **148.5** | 1584.2 | 9051.4 | **Healthy Baseline Phase** |
| **1** | 8,945 | 128.6 | **86.1** | 1591.1 | 9062.8 | **Transition / Moderate Wear Phase** |
| **2** | 3,874 | 194.8 | **27.4** | 1602.8 | 9084.1 | **Severe Late Degradation Phase** |

*Key Insight:* Although K-Means was given zero RUL labels during training, the unsupervised clusters spontaneously sorted the engine data along the degradation continuum. Cluster 0 corresponds to early-life engines (mean RUL 148.5 cycles), whereas Cluster 2 captures engines in immediate proximity to failure (mean RUL 27.4 cycles).

---

## 7. Model Interpretation & Sensor Attribution
Feature importance analysis across Random Forest and Gradient Boosting reveals the dominant physical drivers of degradation:
1. **`s_11_roll_mean_15` (HPC Outlet Static Pressure):** Relative importance of $18.4\%$. Pressure build-up reflects compressor blade fouling and stall margin erosion.
2. **`s_4_roll_mean_15` (LPT Outlet Temperature):** Relative importance of $14.2\%$. Direct indicator of thermodynamic combustion efficiency decline.
3. **`s_3_roll_mean_15` (HPC Outlet Temperature):** Relative importance of $11.8\%$.
4. **`time_cycles`:** Cumulative mechanical runtime ($9.5\%$).
5. **`s_15_roll_mean_15` (Bypass Ratio):** Relative importance of $7.1\%$.

*Predictive Importance vs. Causal Influence:* While these features possess the highest statistical predictive power, high feature importance reflects correlation with damage accumulation rather than root-cause mechanical causation. Compressor blade wear causally induces aerodynamic drag, which in turn causes the observed rises in temperatures and pressures.

---

## 8. Streamlit Dashboard Architecture
The complete system is deployed as an interactive Streamlit application (`app.py`), structured across seven modular sections:
1. **Project Overview:** High-level problem formulation, dataset overview, and architecture diagram.
2. **Data Explorer:** Dynamic trajectory plotting, summary statistics, and missing value verification.
3. **RUL Prediction (Regression):** Interactive engine selection (1–100) and cycle slider. Computes point-in-time features on historical cycles and outputs estimated remaining cycles, health index, and trend projections.
4. **Maintenance-Risk Classification:** Dynamic risk gauge, threshold alert status (Critical vs Nominal), and probability scores.
5. **Model Benchmarks & Comparison:** Live display of saved experimental metric tables, interactive Plotly bar charts, and confusion matrix heatmaps.
6. **Operating Clusters (K-Means):** 2D PCA cluster visualization and degradation profile breakdowns.
7. **System Information & Governance:** Technical documentation, citations, and ethical limitations.

---

## 9. Team Responsibilities & Contributions
The project was executed collaboratively by three undergraduate computer science students:
- **Student 1 (Data Engineering & EDA):** Handled dataset downloading, column verification, exploratory data analysis, constant-feature pruning, leak-free feature engineering (rolling windows, lag differences), and data loader testing.
- **Student 2 (Prognostics & Evaluation):** Designed and trained regression models (Linear, Decision Tree, Random Forest, GBR) and classification models (Logistic, Decision Tree, RF, SVC); implemented GroupKFold cross-validation; evaluated test sets; and generated metric tables.
- **Student 3 (Clustering & Deployment):** Implemented unsupervised K-Means clustering, Elbow and Silhouette analyses, 2D PCA projections, model interpretation, and the complete Streamlit web dashboard.

---

## 10. Conclusion, Limitations & Future Work

### 10.1 Conclusions
This project developed an explainable, end-to-end predictive maintenance system for turbofan engines. By combining leakage-safe temporal feature engineering with ensemble machine learning, our Random Forest Regressor predicted remaining useful life with an MAE of $12.02$ cycles, while our risk classifier achieved an $F_1$-score of $0.8936$. Furthermore, unsupervised K-Means clustering confirmed that physical sensor measurements form identifiable operational clusters that align with actual machine health.

### 10.2 Limitations
1. **Simulation-Based Environment:** C-MAPSS data is generated via mathematical simulation models. Physical turbofans encounter ambient atmospheric variations and flight maneuvers that induce additional operational noise.
2. **Single Operating Regime:** Subset FD001 assumes sea-level flight. Multi-regime datasets (FD002/FD004) require operational normalization across altitudes and Mach numbers.
3. **Decision-Support Scope:** The application serves as an educational decision-support tool and must not be used for autonomous flight-critical decisions.

### 10.3 Future Work
Future extensions include evaluating conformal prediction intervals for quantified uncertainty bounds and extending the architecture to multi-condition subsets (FD002, FD004).

---

## 11. References
1. **Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008).** Damage propagation modeling for aircraft engine run-to-failure simulation. *2008 IEEE International Conference on Prognostics and Health Management*, 1-9. DOI: [10.1109/PHM.2008.4578638](https://doi.org/10.1109/PHM.2008.4578638)
2. **Heimes, F. O. (2008).** Recurrent neural networks for remaining useful life estimation. *2008 International Conference on Prognostics and Health Management*, 1-6. DOI: [10.1109/PHM.2008.4578637](https://doi.org/10.1109/PHM.2008.4578637)
3. **Ramasso, E., & Saxena, A. (2014).** Performance benchmarking and analysis of prognostic methods for turbofan engines. *International Journal of Prognostics and Health Management*, 5(2), 1-15.
4. **Zheng, S., Ristovski, K., Farahat, A., & Gupta, C. (2017).** Long short-term memory network for remaining useful life estimation. *2017 IEEE International Conference on Prognostics and Health Management (ICPHM)*, 88-95. DOI: [10.1109/ICPHM.2017.7998311](https://doi.org/10.1109/ICPHM.2017.7998311)
5. **Li, X., Ding, Q., & Sun, J. Q. (2018).** Remaining useful life estimation in prognostics using deep convolution neural networks. *Reliability Engineering & System Safety*, 172, 1-11. DOI: [10.1016/j.ress.2017.11.021](https://doi.org/10.1016/j.ress.2017.11.021)
6. **Pedregosa, F., et al. (2011).** Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
