"""
Visualization Module for Machine Health Intelligence.
Generates publication-quality charts for EDA, Model Benchmarks,
Feature Importance, and Unsupervised Clustering.
"""

from pathlib import Path
from typing import List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless script execution
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

# Clean scientific plotting aesthetics
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "figure.dpi": 200
})


def plot_engine_cycle_distribution(df: pd.DataFrame, save_path: Optional[Path] = None):
    """
    Plots the distribution of maximum operating cycles (engine lifespan) across all units.
    """
    max_cycles = df.groupby("unit_nr")["time_cycles"].max()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(max_cycles, kde=True, bins=15, color="#1f77b4", ax=ax, edgecolor="black")
    ax.axvline(max_cycles.median(), color="red", linestyle="--", label=f"Median: {max_cycles.median():.0f} cycles")
    ax.set_title("Distribution of Engine Lifespan (Total Cycles to Failure)", fontweight="bold")
    ax.set_xlabel("Operating Cycles at Failure")
    ax.set_ylabel("Engine Count")
    ax.legend()
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    return fig


def plot_sensor_distributions(df: pd.DataFrame, sensors: List[str], save_path: Optional[Path] = None):
    """
    Plots subplots of histograms and KDEs for key selected sensors.
    """
    n_sensors = len(sensors)
    cols = 3
    rows = int(np.ceil(n_sensors / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.2))
    axes = axes.flatten()

    for i, s in enumerate(sensors):
        ax = axes[i]
        sns.histplot(df[s], kde=True, bins=25, color="#2ca02c", ax=ax, edgecolor="none", alpha=0.7)
        desc = config.SENSOR_DESCRIPTIONS.get(s, s)
        ax.set_title(f"{s}: {desc}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Sensor Reading", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)

    # Hide extra axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Probability Distributions of Informative Sensors (FD001)", fontweight="bold", y=0.995)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, features: List[str], save_path: Optional[Path] = None):
    """
    Computes and plots Pearson correlation matrix for informative features and RUL.
    """
    cols = [c for c in features if c in df.columns]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax, annot=False)
    ax.set_title("Correlation Heatmap: Engine Sensors & Operational Settings", fontweight="bold", pad=12)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    return fig


def plot_sensor_trajectories(df: pd.DataFrame, engine_ids: List[int], sensors: List[str], save_path: Optional[Path] = None):
    """
    Plots the progression of physical sensor readings across operating cycles for representative engines.
    """
    n_sensors = len(sensors)
    fig, axes = plt.subplots(n_sensors, 1, figsize=(10, n_sensors * 2.8), sharex=True)
    if n_sensors == 1:
        axes = [axes]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for s_idx, s in enumerate(sensors):
        ax = axes[s_idx]
        for e_idx, eid in enumerate(engine_ids):
            subset = df[df["unit_nr"] == eid]
            ax.plot(subset["time_cycles"], subset[s], label=f"Engine {eid}",
                    color=colors[e_idx % len(colors)], linewidth=1.5, alpha=0.85)
        desc = config.SENSOR_DESCRIPTIONS.get(s, s)
        ax.set_ylabel(f"{s}\n({desc})", fontsize=8)
        if s_idx == 0:
            ax.legend(loc="upper right", ncol=len(engine_ids))
    axes[-1].set_xlabel("Operating Cycles (Run-to-Failure)", fontsize=10)
    plt.suptitle("Sensor Trajectories Over Life Cycles (Engine Degradation Dynamics)", fontweight="bold", y=0.995)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    return fig


def plot_early_vs_late_comparison(df: pd.DataFrame, sensors: List[str], save_path: Optional[Path] = None):
    """
    Compares sensor values during early healthy stage (RUL > 100) vs critical degradation stage (RUL <= 25).
    """
    data = df.copy()
    data["Stage"] = np.where(data["RUL"] <= 25, "Critical Late Stage (RUL ≤ 25)",
                    np.where(data["RUL"] > 100, "Healthy Early Stage (RUL > 100)", "Transition Phase"))
    filtered = data[data["Stage"].isin(["Healthy Early Stage (RUL > 100)", "Critical Late Stage (RUL ≤ 25)"])]

    n_sensors = len(sensors)
    cols = 3
    rows = int(np.ceil(n_sensors / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, rows * 3.0))
    axes = axes.flatten()

    for i, s in enumerate(sensors):
        ax = axes[i]
        sns.boxplot(x="Stage", y=s, data=filtered, ax=ax, palette=["#2ca02c", "#d62728"], width=0.5)
        ax.set_title(s, fontweight="bold", fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("Reading", fontsize=8)
        ax.set_xticklabels(["Healthy", "Critical"], fontsize=8)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Sensor Behavior Shift: Early Healthy vs Late Critical Engine Life", fontweight="bold", y=0.995)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    return fig


def plot_regression_comparison(metrics_df: pd.DataFrame, save_path: Optional[Path] = None):
    """
    Bar plots comparing regression models across Test MAE, RMSE, and R2.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    
    # MAE
    sns.barplot(x="model_name", y="test_mae", data=metrics_df, hue="model_name", ax=axes[0], palette="Blues_d", legend=False)
    axes[0].set_title("Test Mean Absolute Error (MAE)", fontweight="bold")
    axes[0].set_ylabel("Cycles (Lower is Better)")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=25)

    # RMSE
    sns.barplot(x="model_name", y="test_rmse", data=metrics_df, hue="model_name", ax=axes[1], palette="Purples_d", legend=False)
    axes[1].set_title("Test Root Mean Squared Error (RMSE)", fontweight="bold")
    axes[1].set_ylabel("Cycles (Lower is Better)")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=25)

    # R2
    sns.barplot(x="model_name", y="test_r2", data=metrics_df, hue="model_name", ax=axes[2], palette="Greens_d", legend=False)
    axes[2].set_title("Test R² Score", fontweight="bold")
    axes[2].set_ylabel("R² (Higher is Better)")
    axes[2].set_xlabel("")
    axes[2].tick_params(axis="x", rotation=25)

    plt.suptitle("Remaining Useful Life (RUL) Regression Model Benchmarks", fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    return fig


def plot_regression_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, save_path: Optional[Path] = None):
    """
    Scatter plot of True RUL vs Predicted RUL on the 100 test engines.
    """
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_true, y_pred, alpha=0.75, color="#1f77b4", edgecolors="black", linewidth=0.5, s=45)
    max_val = max(np.max(y_true), np.max(y_pred)) + 10
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Ideal 1:1 Identity Line")
    ax.set_title(f"True vs Predicted RUL: {model_name}", fontweight="bold")
    ax.set_xlabel("True Ground Truth RUL (Cycles)")
    ax.set_ylabel("Predicted RUL (Cycles)")
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.legend()
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    return fig


def plot_classification_confusion_matrices(cm_dict: dict, save_path: Optional[Path] = None):
    """
    Plots confusion matrices side-by-side for all evaluated classification models.
    """
    n_models = len(cm_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(n_models * 3.5, 3.2))
    if n_models == 1:
        axes = [axes]

    for idx, (m_name, cm) in enumerate(cm_dict.items()):
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Nominal", "Risk (≤30)"],
                    yticklabels=["Nominal", "Risk (≤30)"])
        ax.set_title(m_name, fontweight="bold", fontsize=10)
        ax.set_xlabel("Predicted Label", fontsize=9)
        if idx == 0:
            ax.set_ylabel("True Label", fontsize=9)

    plt.suptitle("Confusion Matrices: Maintenance Risk Classification (100 Test Engines)", fontweight="bold", y=1.05)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    return fig


def plot_feature_importances(feature_names: List[str], importances: np.ndarray, top_n: int = 15, title: str = "Feature Importances", save_path: Optional[Path] = None):
    """
    Horizontal bar chart of top N feature importances.
    """
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_scores = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=top_scores, y=top_features, hue=top_features, ax=ax, palette="mako", legend=False)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Relative Importance")
    ax.set_ylabel("Engine Feature")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    return fig


def plot_kmeans_clusters(pca_2d: np.ndarray, cluster_labels: np.ndarray, rul_values: np.ndarray, save_path: Optional[Path] = None):
    """
    Plots PCA 2D projection colored by K-Means cluster and by actual RUL.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Cluster assignments
    scatter1 = axes[0].scatter(pca_2d[:, 0], pca_2d[:, 1], c=cluster_labels, cmap="viridis", alpha=0.7, s=25)
    axes[0].set_title("K-Means Cluster Partitions (PCA 2D)", fontweight="bold")
    axes[0].set_xlabel("Principal Component 1")
    axes[0].set_ylabel("Principal Component 2")
    cbar1 = plt.colorbar(scatter1, ax=axes[0])
    cbar1.set_label("Cluster ID")

    # True RUL overlay to inspect degradation correlation
    scatter2 = axes[1].scatter(pca_2d[:, 0], pca_2d[:, 1], c=rul_values, cmap="plasma_r", alpha=0.7, s=25)
    axes[1].set_title("Operational State vs True RUL (PCA 2D)", fontweight="bold")
    axes[1].set_xlabel("Principal Component 1")
    axes[1].set_ylabel("Principal Component 2")
    cbar2 = plt.colorbar(scatter2, ax=axes[1])
    cbar2.set_label("RUL (Cycles Remaining)")

    plt.suptitle("Unsupervised Operating State Clustering & Degradation Mapping", fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    return fig
