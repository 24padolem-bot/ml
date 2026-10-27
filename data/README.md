# C-MAPSS Turbofan Engine Degradation Dataset (FD001)

## Overview
This directory stores raw and processed data for the **Machine Health Intelligence** project.
The dataset is derived from the **NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)** dataset, widely acknowledged as the standard academic and industrial benchmark for predictive maintenance and Remaining Useful Life (RUL) estimation.

- **Primary Source:** NASA Prognostics Center of Excellence (PCoE)
- **NASA Open Data Portal:** [https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data](https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data)
- **Sub-dataset:** `FD001`
  - Operating conditions: 1 (Sea Level)
  - Fault condition: High Pressure Compressor (HPC) degradation
  - Training units: 100 engines (run-to-failure)
  - Test units: 100 engines (censored prior to failure)

## File Descriptions
- `data/raw/train_FD001.txt`:
  - Contains complete run-to-failure trajectories for 100 engines.
  - Rows: 20,631
  - Format: Space-delimited, no header.
- `data/raw/test_FD001.txt`:
  - Contains sensor trajectories for 100 test engines, truncated at some point prior to system failure.
  - Rows: 13,096
- `data/raw/RUL_FD001.txt`:
  - Contains the ground truth Remaining Useful Life (in cycles) for each of the 100 test engines corresponding to the last recorded cycle in `test_FD001.txt`.
  - Rows: 100

## Column Layout (26 Columns)
1. `unit_nr`: Engine unit identifier (1 to 100)
2. `time_cycles`: Operational cycle index
3. `setting_1`: Altitude and flight condition operational setting 1
4. `setting_2`: Mach number operational setting 2
5. `setting_3`: Throttle resolver angle operational setting 3
6. `s_1` through `s_21`: 21 physical engine sensor measurements (temperatures, pressures, fan and core shaft speeds, coolant bleed rates, bypass ratios).

## Constant Features in FD001
Because `FD001` operates at a single sea-level flight condition, several sensor measurements do not vary:
- `setting_3`, `s_1`, `s_5`, `s_10`, `s_16`, `s_18`, `s_19` (Standard deviation = 0.0)
These features are removed during preprocessing to prevent collinearity and matrix singularity.
