"""
Script to generate and save all publication-quality EDA figures into results/figures/.
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from src.data_loader import load_raw_data, compute_train_rul
from src.visualization import (
    plot_engine_cycle_distribution,
    plot_sensor_distributions,
    plot_correlation_heatmap,
    plot_sensor_trajectories,
    plot_early_vs_late_comparison
)

def main():
    print("Generating comprehensive EDA figures...")
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    train_raw, _, _ = load_raw_data()
    train_df = compute_train_rul(train_raw, rul_cap=config.RUL_CAP)
    
    # 1. Lifespan Distribution
    plot_engine_cycle_distribution(
        train_df,
        save_path=config.FIGURES_DIR / "eda_engine_lifespan_dist.png"
    )
    print(" -> Saved eda_engine_lifespan_dist.png")

    # 2. Key Sensor Distributions
    key_sensors = ["s_2", "s_3", "s_4", "s_7", "s_8", "s_11", "s_12", "s_15", "s_21"]
    plot_sensor_distributions(
        train_df,
        sensors=key_sensors,
        save_path=config.FIGURES_DIR / "eda_sensor_distributions.png"
    )
    print(" -> Saved eda_sensor_distributions.png")

    # 3. Correlation Heatmap
    corr_cols = config.INFORMATIVE_RAW_FEATURES + ["RUL", "time_cycles"]
    plot_correlation_heatmap(
        train_df,
        features=corr_cols,
        save_path=config.FIGURES_DIR / "eda_correlation_heatmap.png"
    )
    print(" -> Saved eda_correlation_heatmap.png")

    # 4. Engine Degradation Trajectories
    rep_engines = [1, 2, 3, 4]
    trend_sensors = ["s_2", "s_3", "s_4", "s_11"]
    plot_sensor_trajectories(
        train_df,
        engine_ids=rep_engines,
        sensors=trend_sensors,
        save_path=config.FIGURES_DIR / "eda_sensor_trajectories.png"
    )
    print(" -> Saved eda_sensor_trajectories.png")

    # 5. Early vs Late Boxplots
    plot_early_vs_late_comparison(
        train_df,
        sensors=trend_sensors + ["s_12", "s_15"],
        save_path=config.FIGURES_DIR / "eda_early_vs_late_boxplots.png"
    )
    print(" -> Saved eda_early_vs_late_boxplots.png")
    
    print("All EDA figures successfully generated!")

if __name__ == "__main__":
    main()
