# Viva Voce & Oral Examination Preparation Guide
## Machine Health Intelligence: Predictive Maintenance System

This document equips the three students with concise, conceptually rigorous, and technically sound answers for their university project viva evaluation.

---

### 1. General & Problem Formulation Questions

#### Q1: Why is Predictive Maintenance (PdM) an important real-world problem?
**Answer:**
Industrial machinery, especially aircraft turbofan engines, experiences continuous thermal, mechanical, and fatigue degradation. Traditional maintenance strategies rely either on:
1. *Run-to-failure (reactive):* Waiting until breakdown, causing catastrophic downtime, safety risks, and expensive emergency overhauls.
2. *Time-based (preventive):* Overhauling components at fixed intervals, wasting useful equipment life and inflating operational costs.
Predictive maintenance uses real-time condition-monitoring sensor streams to forecast degradation, enabling maintenance teams to service equipment *just in time*—minimizing costs while maximizing safety and operational availability.

#### Q2: Why was NASA's C-MAPSS dataset selected for this project?
**Answer:**
The NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset is the globally recognized academic and industrial benchmark for predictive maintenance. It provides realistic multivariate thermodynamic run-to-failure sensor trajectories for turbofan engines with varying lifespans, sensor noise, and a single fault mode (High-Pressure Compressor degradation in subset FD001). It allows fair benchmarking against published literature without requiring proprietary industrial data or paid APIs.

#### Q3: How is Remaining Useful Life (RUL) calculated in this project?
**Answer:**
For each training engine $i$ that runs to failure at maximum cycle $T_i$, the raw RUL at cycle $t$ is:
$$\text{RUL}_{raw}(t) = T_i - t$$
We apply a **piece-wise linear cap** at $RUL_{cap} = 125$ cycles:
$$\text{RUL}(t) = \min(T_i - t, 125)$$
*Rationale:* During early operation, engines operate in pristine baseline conditions where wear is negligible. Capping the target prevents regression models from attempting to fit random sensor noise when no physical degradation has initiated.

#### Q4: Why does the project formulate both a Regression task and a Classification task?
**Answer:**
They serve distinct, complementary operational purposes:
- **Regression (Task A):** Provides fine-grained planning granularity by estimating the exact number of remaining flight cycles, allowing fleet managers to schedule long-term depot overhauls.
- **Classification (Task B):** Provides immediate tactical triage and binary alerting. By classifying whether an engine has entered the critical risk window ($RUL \le 30$ cycles), ground crews receive an unambiguous binary alert (Nominal vs. Critical Action Required) without needing to interpret continuous error margins.

---

### 2. Machine Learning Theory & Algorithms

#### Q5: What is the fundamental difference between Supervised and Unsupervised Learning?
**Answer:**
- **Supervised Learning:** The algorithm is provided with input features $X$ alongside ground-truth labels $y$ (e.g., actual RUL cycles in regression, or binary risk labels in classification) and optimizes a loss function to map $X \to y$.
- **Unsupervised Learning:** The algorithm receives only unlabeled sensor observations $X$ without target labels. Its objective is to discover underlying statistical structure, groupings, or clusters (e.g., K-Means grouping operational health stages based purely on sensor similarity).

#### Q6: What is the difference between Linear Regression and Logistic Regression?
**Answer:**
- **Linear Regression:** A continuous regression model that predicts an unbounded numeric quantity: $\hat{y} = w^T X + b$. It optimizes Mean Squared Error (MSE).
- **Logistic Regression:** A classification algorithm that predicts the probability of a discrete class (0 or 1). It passes a linear combination through the sigmoid activation function:
  $$P(y=1|X) = \sigma(w^T X + b) = \frac{1}{1 + e^{-(w^T X + b)}}$$
  It optimizes binary cross-entropy (log-loss), yielding calibrated probabilities between 0 and 1.

#### Q7: Why do Decision Trees and Random Forests perform differently?
**Answer:**
A single **Decision Tree** makes orthogonal partition splits based on feature thresholds. While intuitive, individual decision trees are prone to high variance and overfitting noisy sensor fluctuations.
A **Random Forest** is an ensemble method that trains hundreds of de-correlated decision trees using:
1. *Bootstrap aggregating (Bagging):* Training each tree on a random sample with replacement.
2. *Feature subspace sampling:* Selecting a random subset of features at each split.
By averaging the predictions of individual trees, the Random Forest drastically reduces variance without increasing bias, producing significantly lower test error ($\text{MAE} = 12.02$ vs. $13.44$ cycles).

#### Q8: Why does Support Vector Machine (SVM) require feature scaling while Decision Trees do not?
**Answer:**
- **SVM:** Relies on geometric distance calculations (Euclidean distance in dot products or RBF kernels: $\exp(-\gamma \|x_i - x_j\|^2)$). If one sensor has values around $9000$ (like core shaft speed `s_9`) and another has values around $0.05$ (like bypass ratio `s_15`), the large-magnitude sensor will completely dominate the distance metric. Therefore, scaling (`StandardScaler`) is mandatory.
- **Decision Trees:** Split nodes using monotonic orderings of individual features (e.g., $x_j \le \theta$). The scale or absolute numerical range of a feature does not affect where optimal splits occur, making tree-based models invariant to monotonic monotonic scaling.

#### Q9: How does K-Means clustering work, and how does the Elbow Method determine $k$?
**Answer:**
**K-Means** partitions $n$ observations into $k$ clusters by iteratively:
1. Assigning each data point to its nearest cluster centroid (minimizing Euclidean distance).
2. Updating cluster centroids to be the arithmetic mean of all points assigned to that cluster.
The process repeats until centroids converge.
The **Elbow Method** plots the cluster *Inertia* (Within-Cluster Sum of Squared Distances, WCSS) across different values of $k$. As $k$ increases, inertia naturally decreases. The optimal $k$ is selected at the "elbow" point—the point where adding another cluster yields diminishing returns in inertia reduction (in our project, $k=3$).

---

### 3. Data Leakage & Validation Methodology

#### Q10: Why is random train-test splitting dangerous for this dataset, and how did you prevent it?
**Answer:**
A random row-wise split would scatter observations from the *same engine unit* across both training and test sets. Since sensor degradation is a continuous time series, training on cycle $100$ of Engine #1 and testing on cycle $101$ of Engine #1 produces **catastrophic temporal leakage**, resulting in unrealistically inflated accuracy that collapses in production.
We prevented this by enforcing **Engine-Level Group Splitting**:
1. Cross-validation used `GroupKFold` partitioned strictly by `unit_nr`.
2. Final evaluation was performed exclusively on the 100 unseen test engine trajectories in `test_FD001.txt`, verified against official ground-truth labels in `RUL_FD001.txt`.

#### Q11: How did you ensure temporal features did not leak future information?
**Answer:**
All rolling window calculations (rolling mean over 5 and 15 cycles, rolling standard deviation, and lag-1 differences) were:
1. Partitioned strictly by engine ID (`groupby('unit_nr')`).
2. Calculated using only causal, backward-looking windows ($t-w \dots t$). At no point were future cycles ($t+1$) included in the rolling computation.

---

### 4. Metrics & Evaluation Interpretation

#### Q12: What do MAE, RMSE, and $R^2$ mean in the context of RUL prediction?
**Answer:**
- **MAE (Mean Absolute Error):** The average absolute difference between predicted and actual RUL:
  $$\text{MAE} = \frac{1}{n} \sum |\hat{y}_i - y_i|$$
  Our model's MAE of $12.02$ cycles means predictions deviate by approximately $12$ flight cycles on average.
- **RMSE (Root Mean Squared Error):** The square root of the mean squared error. Because errors are squared before averaging, RMSE penalizes large outlier errors more heavily than MAE. Our model achieved an RMSE of $16.53$ cycles.
- **$R^2$ (Coefficient of Determination):** The proportion of variance in true RUL explained by the model ($R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$). An $R^2$ of $0.8299$ indicates that our model explains $\sim 83\%$ of the variance in test engine degradation.

#### Q13: Why is high Accuracy misleading for maintenance-risk classification?
**Answer:**
In our dataset, healthy nominal observations comprise $85\%$ of the data, while critical risk observations ($RUL \le 30$) comprise only $15\%$. A naive dummy classifier that always predicts "Nominal" achieves $85\%$ accuracy while missing $100\%$ of failing engines!
Therefore, we prioritize **Recall** (detecting true failures) and **$F_1$-score** (harmonic mean of Precision and Recall) rather than raw accuracy.

#### Q14: In predictive maintenance, is a False Negative or a False Positive worse?
**Answer:**
A **False Negative** is dramatically worse. A False Negative means the model predicts an engine is safe when it is actually within $30$ cycles of failure, leading to potential in-flight shutdown or catastrophic mechanical breakdown. 
A **False Positive** merely triggers an inspection slightly earlier than necessary in the maintenance hangar. While incurring minor inspection overhead, it never compromises safety. Consequently, our models were trained with balanced class weights to maximize Recall ($92.00\%$ in Logistic Regression, $84.00\%$ in Random Forest).

---

### 5. Interpretability & System Deployment

#### Q15: Why does high feature importance not prove causal influence?
**Answer:**
Feature importance reflects **statistical correlation and predictive utility** within the model's split decisions, not direct physical causation. For example, high-pressure compressor outlet static pressure (`s_11`) and turbine outlet temperature (`s_4`) have the highest feature importances because their values drift systematically as damage accumulates. However, blade erosion and aerodynamic boundary layer separation are the actual physical root causes that causally drive the observed temperature and pressure spikes.

#### Q16: How does the application perform real-time inference without retraining?
**Answer:**
All model pipelines, preprocessing scalers, PCA transformers, and metadata are serialized to disk as `.joblib` and `.json` artifacts during the offline experimentation phase. When the Streamlit application starts, it loads these pre-trained artifacts into memory using `@st.cache_resource`. When a user selects an engine and cycle cutoff, the app extracts historical features on that specific window and performs sub-millisecond inference (`< 1` ms) without retraining.

#### Q17: What are the primary limitations of this project?
**Answer:**
1. *Synthetic Simulation Constraints:* The C-MAPSS dataset is generated by a computer simulation model and does not include real-world environmental disruptions like bird strikes, volcanic ash, or erratic flight maneuvers.
2. *Single Condition Focus:* Subset FD001 evaluates sea-level flight with a single fault mode. In commercial aviation, engines operate across diverse flight envelopes requiring altitude/Mach normalization.
3. *Advisory Decision Support:* This system is an educational prototype and decision-support assistant; it is not certified for autonomous safety-critical flight decisions.

---

### 6. Team Contributions

#### Student 1: Data Engineering, EDA & Preprocessing
- Led raw data inspection and integrity checks.
- Conducted exploratory data analysis, identified zero-variance sensors, and produced diagnostic plots.
- Engineered leakage-free temporal features (rolling means, rolling std, lag differences) partitioned by engine unit.
- Implemented and verified the `src/data_loader.py`, `src/preprocessing.py`, and `src/feature_engineering.py` modules.

#### Student 2: Model Development, Optimization & Benchmarking
- Designed and trained all four regression models (Linear, Decision Tree, Random Forest, Gradient Boosting).
- Designed and trained all four classification models (Logistic Regression, Decision Tree, Random Forest, SVC).
- Implemented GroupKFold cross-validation and hyperparameter tuning (`GridSearchCV`).
- Evaluated models on the official test set and produced benchmark comparison tables.

#### Student 3: Clustering, Interpretation & Deployment
- Implemented unsupervised K-Means clustering, Elbow inertia analysis, and Silhouette validation.
- Performed 2D PCA dimensionality reduction and mapped statistical clusters to engine degradation stages.
- Extracted and visualized tree feature importances and diagnostic coefficients.
- Built, tested, and deployed the full interactive Streamlit dashboard (`app.py`).
