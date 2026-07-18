# Lithium-ion Battery State of Health (SoH) Estimation

This project provides a complete machine learning and statistical framework for predicting the **State of Health (SoH)** of Lithium-ion batteries — a critical metric for estimating remaining useful life (RUL) in Battery Management Systems (BMS).

The pipeline covers data preprocessing, outlier removal, and predictive modeling using **Linear Regression** (baseline) and **LSTM** (deep learning) on the NASA Prognostics battery dataset across 8 battery cells.

---

## Results

Models were evaluated at two lifecycle entry points: **50% cycle** and **70% cycle** milestones.

| Model | Split | Avg RMSE | Avg MAE |
|:---|:---:|:---:|:---:|
| Linear Regression | 50% cycle | 0.0434 | 0.0351 |
| Linear Regression | 70% cycle | 0.0462 | 0.0388 |
| **LSTM** | **50% cycle** | **0.0197** | **0.0142** |
| **LSTM** | **70% cycle** | **0.0225** | **0.0161** |

> LSTM achieves **~54% lower RMSE** than Linear Regression at the 50% cycle milestone.

---

## Project Structure & Workflow

```
SoH_estimation_of_Lithium-ion_battery/
├── 1_Calculation_and_Visulaliztion_of_SoH/   # Stage 1: SoH calculation from raw discharge data
├── 2_Elimination_of_outliers/                 # Stage 2: Quantile-based outlier removal
├── 3_Linear_Regresssion_with_SoH/            # Stage 3: Baseline linear regression model
├── 4_LSTM_with_SoH/                           # Stage 4: LSTM deep learning model
├── run_pipeline.py                            # End-to-end runner script
└── results_summary.csv                       # Per-battery RMSE/MAE results
```

The estimation pipeline runs in four sequential stages:

### 1. SoH Calculation and Visualization
Loads raw NASA discharge data (8 battery cells: B05, B07, B18, B33, B34, B46, B47, B48), aggregates per-cycle capacity measurements, computes the SoH ratio, and generates cycle degradation plots.

- **Notebook**: [Calculation_and_Visualization_of_SoH.ipynb](./1_Calculation_and_Visulaliztion_of_SoH/Calculation_and_Visualization_of_SoH.ipynb)

### 2. Outlier Elimination
Applies 5th–95th percentile quantile filtering to remove noise and measurement anomalies from the SoH curves, producing clean per-cycle refined datasets for modeling.

- **Notebook**: [Calculation_and_Visualization_of_refined_SoH.ipynb](./2_Elimination_of_outliers/Calculation_and_Visualization_of_refined_SoH.ipynb)

### 3. Linear Regression (Baseline)
Fits a linear degradation model on training cycles and extrapolates into the future. Evaluated at both 50% and 70% train/test split points.

- **Script**: [SoH_estimation_with_Linear_Regression.m](./3_Linear_Regresssion_with_SoH/SoH_estimation_with_Linear_Regression.m)

### 4. LSTM Deep Learning
Trains a 64-unit LSTM network (look-back window = 3 cycles, early stopping) to capture non-linear temporal degradation patterns. Significantly outperforms the linear baseline.

- **Notebook**: [SoH_estimation_with_LSTM.ipynb](./4_LSTM_with_SoH/SoH_estimation_with_LSTM.ipynb)

---

## Methodology

| Component | Detail |
|:---|:---|
| **Dataset** | NASA Prognostics – 8 Li-ion cells (B05–B48), 64–197 discharge cycles each |
| **Outlier Removal** | Quantile filtering (5th–95th percentile) |
| **Baseline Model** | Ordinary Least Squares Linear Regression |
| **Deep Learning Model** | LSTM (64 units, look-back=3, EarlyStopping, Adam optimizer) |
| **Evaluation Splits** | 50% and 70% train/test cycle milestones |
| **Metrics** | RMSE, MAE |

---

## Getting Started

### Prerequisites
- Python 3.8+
- MATLAB (for the Linear Regression `.m` script only)

### Install Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras
```

### Run the Full Pipeline
```bash
python run_pipeline.py
```

This runs all four stages end-to-end, saves all figures under each module's `fig/` folder, saves trained LSTM model weights, and outputs `results_summary.csv`.

---

## Dataset

NASA Prognostics Center of Excellence – Li-ion Battery Aging Datasets.

> B. Saha and K. Goebel (2007). "Battery Data Set", NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.  
> [https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
