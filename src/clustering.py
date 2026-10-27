"""
Unsupervised Clustering Module for Machine Health Intelligence.
Applies K-Means clustering to discover operating patterns from raw sensor features without target leakage.
Computes Elbow curve (Inertia), Silhouette scores, 2D PCA projection, and cluster degradation profiles.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.data_loader import load_raw_data, compute_train_rul
from src.visualization import plot_kmeans_clusters
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def run_clustering_experiments(k_range=(2, 8), selected_k: int = 3) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Executes K-Means clustering on informative raw sensor measurements.
    """
    print("=" * 65)
    print("STARTING TASK C: UNSUPERVISED SENSOR PATTERN CLUSTERING")
    print("=" * 65)

    # 1. Load Data
    train_raw, _, _ = load_raw_data()
    train_df = compute_train_rul(train_raw, rul_cap=config.RUL_CAP)

    # Informative sensor features (excluding constant ones, no engineered RUL targets)
    features_to_cluster = config.INFORMATIVE_SENSORS
    print(f"[1/5] Selected {len(features_to_cluster)} informative physical sensor features for clustering...")

    X_raw = train_df[features_to_cluster].copy()
    rul_actual = train_df["RUL_raw"].values

    # 2. Standardization
    print("[2/5] Standardizing sensor features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 3. Elbow Analysis & Silhouette Score
    print(f"[3/5] Evaluating K values from {k_range[0]} to {k_range[1]} for Elbow Method...")
    inertias = []
    silhouette_scores = []
    k_values = list(range(k_range[0], k_range[1] + 1))

    # Subsample for fast, stable silhouette computation
    subsample_idx = np.random.default_rng(config.RANDOM_SEED).choice(len(X_scaled), size=3000, replace=False)
    X_sub = X_scaled[subsample_idx]

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=config.RANDOM_SEED, n_init=10)
        km.fit(X_scaled)
        inertias.append(float(km.inertia_))
        
        # Silhouette on subsample
        km_sub_labels = km.predict(X_sub)
        sil = float(silhouette_score(X_sub, km_sub_labels))
        silhouette_scores.append(round(sil, 4))
        print(f"      k={k} | Inertia: {km.inertia_:10.2f} | Silhouette Score: {sil:.4f}")

    # Plot Elbow & Silhouette Curve
    fig, ax1 = plt.subplots(figsize=(7.5, 4.2))
    ax1.plot(k_values, inertias, "bo-", linewidth=2, label="Inertia (Sum of Squared Distances)")
    ax1.set_xlabel("Number of Clusters (k)", fontweight="bold")
    ax1.set_ylabel("Inertia", color="b", fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="b")

    ax2 = ax1.twinx()
    ax2.plot(k_values, silhouette_scores, "rs--", linewidth=2, label="Silhouette Score")
    ax2.set_ylabel("Silhouette Score", color="r", fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="r")

    plt.title("K-Means Cluster Optimization: Elbow Method & Silhouette Score", fontweight="bold")
    plt.tight_layout()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(config.FIGURES_DIR / "clustering_elbow_silhouette.png", bbox_inches="tight")
    plt.close()

    # 4. Final K-Means Fit with selected_k
    print(f"[4/5] Fitting final K-Means model with k={selected_k}...")
    final_kmeans = KMeans(n_clusters=selected_k, random_state=config.RANDOM_SEED, n_init=15)
    cluster_labels = final_kmeans.fit_predict(X_scaled)

    # 5. Dimensionality Reduction for Visualization (PCA 2D)
    pca = PCA(n_components=2, random_state=config.RANDOM_SEED)
    X_pca_2d = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_
    print(f"      PCA Variance Explained: PC1={var_explained[0]:.2%}, PC2={var_explained[1]:.2%}, Total={sum(var_explained):.2%}")

    plot_kmeans_clusters(
        X_pca_2d[subsample_idx],
        cluster_labels[subsample_idx],
        rul_actual[subsample_idx],
        save_path=config.FIGURES_DIR / "clustering_pca_visualization.png"
    )

    # Profile clusters by degradation stage and sensor readings
    train_clustered = train_df.copy()
    train_clustered["cluster_id"] = cluster_labels
    cluster_profiles = train_clustered.groupby("cluster_id").agg({
        "RUL_raw": ["mean", "median", "std", "count"],
        "time_cycles": ["mean", "max"],
        "s_2": "mean",
        "s_3": "mean",
        "s_4": "mean",
        "s_7": "mean",
        "s_11": "mean",
        "s_12": "mean"
    }).round(2)

    # Save artifacts
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_kmeans, config.MODELS_DIR / "kmeans_model.joblib")
    joblib.dump(scaler, config.MODELS_DIR / "kmeans_scaler.joblib")
    joblib.dump(pca, config.MODELS_DIR / "kmeans_pca.joblib")

    meta = {
        "k_selected": selected_k,
        "k_range": k_values,
        "inertias": inertias,
        "silhouette_scores": silhouette_scores,
        "features": features_to_cluster,
        "pca_variance_ratio": [float(v) for v in var_explained]
    }
    with open(config.RESULTS_DIR / "metrics" / "clustering_metadata.json", "w") as f:
        json.dump(meta, f, indent=4)

    cluster_profiles.to_csv(config.RESULTS_DIR / "metrics" / "cluster_profiles.csv")
    print("\nTask C Clustering experiments completed successfully!\n")
    return meta, cluster_profiles


if __name__ == "__main__":
    run_clustering_experiments(selected_k=3)
