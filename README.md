# Lithium-ion Battery State of Health (SoH) Estimation

This project provides a comprehensive machine learning and statistical framework to predict and monitor the **State of Health (SoH)** of Lithium-ion batteries. Estimating SoH is crucial for predicting the remaining useful life (RUL) and ensuring the safety and reliability of battery management systems (BMS).

This repository implements data pre-processing, outlier detection, and predictive modeling using both **Linear Regression** and **Long Short-Term Memory (LSTM)** networks.

---

## Project Structure & Workflow

The estimation pipeline is organized into four sequential stages:

### 1. Data Calculation and Visualization
Calculates the State of Health (SoH) across various battery cell cycles from the raw datasets. 
* **Key Steps**: Load raw capacity measurements, calculate SoH relative to nominal capacity, and generate cycle-by-cycle profiles.
* **Component Link**: [Calculation & Visualization Notebook](./1_Calculation_and_Visulaliztion_of_SoH/Calculation_and_Visualization_of_SoH.ipynb)

### 2. Outlier Elimination
Implements statistical filtering techniques using quantile methods to remove noise and experimental measurement anomalies from the computed SoH curves.
* **Key Steps**: Identify outlier boundaries, filter out transient noise spikes, and smooth the dataset for modeling.
* **Component Link**: [Outlier Elimination Notebook](./2_Elimination_of_outliers/Calculation_and_Visualization_of_refined_SoH.ipynb)

### 3. Linear Regression Modeling
Applies a baseline Linear Regression model to forecast the trend of SoH decay. Predictions are evaluated at different entry points in the battery lifecycle.
* **Configurations**: Models are trained and tested starting at **50%** and **70%** cycle milestones.
* **Component Link**: [Linear Regression Script](./3_Linear_Regresssion_with_SoH/SoH_estimation_with_Linear_Regression.m)

### 4. Recurrent Neural Network (LSTM) Modeling
Utilizes a deep learning approach using Long Short-Term Memory (LSTM) networks to capture non-linear degradation characteristics and long-term dependencies in battery aging.
* **Configurations**: Evaluated starting at **50%** and **70%** cycle milestones to simulate early-stage and mid-stage lifetime prediction.
* **Component Link**: [LSTM Notebook](./4_LSTM_with_SoH/SoH_estimation_with_LSTM.ipynb)

---

## Methodology Overview

| Methodology | Application Scenario | Features Covered |
| :--- | :--- | :--- |
| **Quantile Filtering** | Preprocessing | Noise reduction, anomaly detection |
| **Linear Regression** | Baseline Prediction | Linear trend fitting, computationally efficient |
| **LSTM Networks** | Advanced Prediction | Sequential dependencies, non-linear degradation modeling |

---

## Getting Started

### Prerequisites
* Python 3.8+
* Jupyter Notebook / JupyterLab
* MATLAB (for the baseline Linear Regression execution)

### Required Python Libraries
```bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras
```

---

## Dataset Reference
This project utilizes the battery aging datasets provided by the **NASA Prognostics Center of Excellence**.
* [NASA Randomized Battery Dataset](https://www.nasa.gov/) (Prognostic Data Repository)
