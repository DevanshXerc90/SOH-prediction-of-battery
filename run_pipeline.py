"""
run_pipeline.py
Full end-to-end SoH estimation pipeline.
Aggregates raw timestep data to per-cycle before modeling.
"""

import os, math, warnings, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
num      = ['B05', 'B07', 'B18', 'B33', 'B34', 'B46', 'B47', 'B48']

# ───────────────────────────────────────────────────────────────────────────────
# STAGE 1 : Calculation & Visualisation of SoH
# ───────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STAGE 1 - SoH Calculation & Visualisation")
print("="*70)

CALC_DIR = os.path.join(BASE_DIR, "1_Calculation_and_Visulaliztion_of_SoH")
FIG1     = os.path.join(CALC_DIR, "fig")
os.makedirs(FIG1, exist_ok=True)

data = {}   # per-cycle aggregated data

for n in num:
    path = os.path.join(CALC_DIR, "dataset", f"{n}_discharge_soh.csv")
    df   = pd.read_csv(path)
    # Aggregate to ONE ROW PER CYCLE (SOH is constant per cycle in the raw data)
    df_cyc = df.groupby('cycle').agg(
        capacity=('capacity', 'last'),
        SOH=('SOH', 'last')
    ).reset_index()
    data[n] = df_cyc
    print(f"  {n}: {len(df_cyc)} cycles  |  SoH [{df_cyc['SOH'].min():.4f} -> {df_cyc['SOH'].max():.4f}]")

# Capacity plots
for n in num:
    dff = data[n]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(dff['cycle'], dff['capacity'], s=20, color='steelblue')
    ax.set_ylabel('Capacity (Ah)'); ax.set_xlabel('Discharge Cycle')
    ax.set_title(f'Discharge Capacity - {n}')
    sns.despine(); fig.tight_layout()
    fig.savefig(os.path.join(FIG1, f"Capacity_{n}.jpg"), dpi=100)
    plt.close(fig)

# SoH plots
for n in num:
    dff = data[n]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(dff['cycle'], dff['SOH'], s=20, color='darkorange')
    ax.set_ylabel('State of Health (SoH)'); ax.set_xlabel('Discharge Cycle')
    ax.set_title(f'State of Health - {n}')
    sns.despine(); fig.tight_layout()
    fig.savefig(os.path.join(FIG1, f"SoH_{n}.jpg"), dpi=100)
    plt.close(fig)

print("  Stage 1 figures saved.")

# ───────────────────────────────────────────────────────────────────────────────
# STAGE 2 : Outlier Elimination
# ───────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STAGE 2 - Outlier Elimination (Quantile Filtering)")
print("="*70)

OUT_DIR = os.path.join(BASE_DIR, "2_Elimination_of_outliers")
FIG2    = os.path.join(OUT_DIR, "fig")
RDATA2  = os.path.join(OUT_DIR, "refined_dataset")
os.makedirs(FIG2, exist_ok=True)
os.makedirs(RDATA2, exist_ok=True)

refined = {}
for n in num:
    df = data[n].copy()
    q_low  = df['SOH'].quantile(0.05)
    q_high = df['SOH'].quantile(0.95)
    df_r   = df[(df['SOH'] >= q_low) & (df['SOH'] <= q_high)].reset_index(drop=True)
    refined[n] = df_r
    df_r.to_csv(os.path.join(RDATA2, f"{n}_discharge_soh.csv"), index=False)
    print(f"  {n}: {len(df)} -> {len(df_r)} cycles after filtering")

# Group plots
groups = {'A': ['B05','B07','B18'], 'B': ['B33','B34'], 'C': ['B46','B47','B48']}
for grp, members in groups.items():
    fig, ax = plt.subplots(figsize=(10, 6))
    for n in members:
        ax.scatter(refined[n]['cycle'], refined[n]['SOH'], s=20, label=n)
    ax.legend(fontsize=12); ax.set_ylabel('SoH'); ax.set_xlabel('Discharge Cycle')
    ax.set_title(f'SoH - Group {grp}')
    sns.despine(); fig.tight_layout()
    fig.savefig(os.path.join(FIG2, f"{grp}_group.jpg"), dpi=100)
    plt.close(fig)

for n in num:
    dff = refined[n]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(dff['cycle'], dff['SOH'], s=20)
    ax.set_ylabel('SoH'); ax.set_xlabel('Cycle')
    ax.set_title(f'Refined SoH - {n}')
    sns.despine(); fig.tight_layout()
    fig.savefig(os.path.join(FIG2, f"refined_SoH_{n}.jpg"), dpi=100)
    plt.close(fig)

print("  Stage 2 figures saved.")

# ───────────────────────────────────────────────────────────────────────────────
# STAGE 3 : Linear Regression
# ───────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STAGE 3 - Linear Regression (50% and 70% splits)")
print("="*70)

LR_DIR = os.path.join(BASE_DIR, "3_Linear_Regresssion_with_SoH")

def lin_reg_manual(x, y):
    m   = len(x)
    xs  = np.sum(x); ys = np.sum(y)
    x2s = np.sum(x**2); xys = np.sum(x * y)
    a1  = (m * xys - xs * ys) / (m * x2s - xs**2)
    a0  = (x2s * ys - xys * xs) / (m * x2s - xs**2)
    return a0, a1

lr_results = {}

for split_pct, split_name in [(0.5, '50%'), (0.7, '70%')]:
    fig_dir = os.path.join(LR_DIR, "fig", split_name)
    os.makedirs(fig_dir, exist_ok=True)
    split_results = {}

    for n in num:
        df    = refined[n]
        x_all = df['cycle'].values.astype(float)
        y_all = df['SOH'].values.astype(float)
        idx   = int(len(x_all) * split_pct)
        a0, a1 = lin_reg_manual(x_all[:idx], y_all[:idx])
        y_pred  = a1 * x_all[idx:] + a0
        rmse = math.sqrt(mean_squared_error(y_all[idx:], y_pred))
        mae  = mean_absolute_error(y_all[idx:], y_pred)
        split_results[n] = {'RMSE': rmse, 'MAE': mae}

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x_all, y_all, s=15, color='steelblue', label='Data', alpha=0.6)
        ax.axvline(x=x_all[idx], color='k', linestyle='--', linewidth=1, label=f'Train/Test split ({split_name})')
        ax.plot(x_all[idx:], y_pred, color='red', linewidth=2, label='Linear Regression')
        ax.legend(fontsize=11); ax.set_ylabel('SoH'); ax.set_xlabel('Discharge Cycle')
        ax.set_title(f'{n} - Linear Regression ({split_name} split)')
        sns.despine(); fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{n}_Linear.jpg"), dpi=100)
        plt.close(fig)

    lr_results[split_name] = split_results
    rmses = [r['RMSE'] for r in split_results.values()]
    maes  = [r['MAE']  for r in split_results.values()]
    print(f"\n  {split_name} split:")
    print(f"  {'Battery':<8} {'RMSE':>8} {'MAE':>8}")
    for n, r in split_results.items():
        print(f"  {n:<8} {r['RMSE']:>8.4f} {r['MAE']:>8.4f}")
    print(f"  {'MEAN':<8} {np.mean(rmses):>8.4f} {np.mean(maes):>8.4f}")

# ───────────────────────────────────────────────────────────────────────────────
# STAGE 4 : LSTM  (runs on per-cycle data: 168 cycles max -> fast!)
# ───────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STAGE 4 - LSTM (50% and 70% splits)")
print("="*70)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.callbacks import EarlyStopping

LSTM_DIR = os.path.join(BASE_DIR, "4_LSTM_with_SoH")
RDATA4   = os.path.join(LSTM_DIR, "refined_dataset")
os.makedirs(RDATA4, exist_ok=True)
for n in num:
    refined[n].to_csv(os.path.join(RDATA4, f"{n}_discharge_soh.csv"), index=False)

def make_sequences(arr, look_back):
    X, y = [], []
    for i in range(len(arr) - look_back):
        X.append(arr[i:i+look_back])
        y.append(arr[i+look_back])
    return np.array(X), np.array(y)

LOOK_BACK = 3
lstm_results = {}

for split_pct, split_name in [(0.5, '50%'), (0.7, '70%')]:
    fig_dir   = os.path.join(LSTM_DIR, split_name, "fig")
    model_dir = os.path.join(LSTM_DIR, split_name, "model")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    split_results = {}
    rmse_list, mae_list = [], []

    for n in num:
        df    = refined[n]
        soh   = df['SOH'].values.astype(float)
        cycle = df['cycle'].values
        idx   = int(len(soh) * split_pct)

        train_soh = soh[:idx]
        test_soh  = soh[idx:]

        if len(test_soh) <= LOOK_BACK + 2:
            print(f"  [{split_name}] {n}: Skipped (too few test samples: {len(test_soh)})")
            continue

        trainX, trainY = make_sequences(train_soh, LOOK_BACK)
        testX,  testY  = make_sequences(test_soh,  LOOK_BACK)

        # Shape: (samples, timesteps, features)
        trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], 1)
        testX  = testX.reshape(testX.shape[0],  testX.shape[1],  1)

        tf.random.set_seed(42)
        model = Sequential([
            Input(shape=(LOOK_BACK, 1)),
            LSTM(64),
            Dense(1)
        ])
        model.compile(loss='mae', optimizer='adam')

        es = EarlyStopping(patience=15, restore_best_weights=True, verbose=0)
        model.fit(
            trainX, trainY,
            epochs=200, batch_size=8,
            validation_data=(testX, testY),
            callbacks=[es],
            verbose=0
        )

        # Save model
        model_json = model.to_json()
        with open(os.path.join(model_dir, f"{n}_model.json"), 'w') as f:
            f.write(model_json)
        model.save_weights(os.path.join(model_dir, f"{n}_weights.weights.h5"))

        yhat = model.predict(testX, verbose=0).flatten()
        rmse = math.sqrt(mean_squared_error(testY, yhat))
        mae  = mean_absolute_error(testY, yhat)
        rmse_list.append(rmse); mae_list.append(mae)
        split_results[n] = {'RMSE': rmse, 'MAE': mae}
        print(f"  [{split_name}] {n}: RMSE={rmse:.4f}  MAE={mae:.4f}")

        # Plot
        c_train = cycle[LOOK_BACK:idx]
        c_test  = cycle[idx + LOOK_BACK: idx + LOOK_BACK + len(testY)]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(c_train[:len(trainX)], trainY, color='red',   linewidth=2, label='Train data')
        if len(c_test) == len(testY):
            ax.plot(c_test, testY, color='blue',  linewidth=2, label='Real data')
            ax.plot(c_test, yhat[:len(c_test)], color='green', linewidth=2, label='LSTM Prediction')
        ax.axvline(x=cycle[idx], color='k', linestyle='--', linewidth=1)
        ax.legend(fontsize=12); ax.set_ylabel('SoH'); ax.set_xlabel('Discharge Cycle')
        ax.set_title(f'{n} SoH Prediction - LSTM ({split_name} split)')
        sns.despine(); fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{n}_LSTM.jpg"), dpi=100)
        plt.close(fig)

    lstm_results[split_name] = split_results

    # Save metric text files
    with open(os.path.join(LSTM_DIR, split_name, f"{split_name}_rmse.txt"), 'w') as f:
        for n2, r in split_results.items():
            f.write(f"{n2}: {r['RMSE']:.6f}\n")
        if rmse_list:
            f.write(f"MEAN: {np.mean(rmse_list):.6f}\n")
    with open(os.path.join(LSTM_DIR, split_name, f"{split_name}_mae.txt"), 'w') as f:
        for n2, r in split_results.items():
            f.write(f"{n2}: {r['MAE']:.6f}\n")
        if mae_list:
            f.write(f"MEAN: {np.mean(mae_list):.6f}\n")

    if rmse_list:
        print(f"\n  {split_name} LSTM Avg -> RMSE: {np.mean(rmse_list):.4f}  MAE: {np.mean(mae_list):.4f}")

# ───────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ───────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("FINAL RESULTS SUMMARY")
print("="*70)

print("\n  LINEAR REGRESSION:")
for split, res in lr_results.items():
    rmses = [r['RMSE'] for r in res.values()]
    maes  = [r['MAE']  for r in res.values()]
    print(f"    {split} split  ->  Avg RMSE: {np.mean(rmses):.4f}   Avg MAE: {np.mean(maes):.4f}")

print("\n  LSTM:")
for split, res in lstm_results.items():
    if not res: continue
    rmses = [r['RMSE'] for r in res.values()]
    maes  = [r['MAE']  for r in res.values()]
    print(f"    {split} split  ->  Avg RMSE: {np.mean(rmses):.4f}   Avg MAE: {np.mean(maes):.4f}")

# Save results CSV
rows = []
for split in ['50%', '70%']:
    for n, r in lr_results.get(split, {}).items():
        rows.append({'Model': 'Linear Regression', 'Split': split, 'Battery': n,
                     'RMSE': round(r['RMSE'], 4), 'MAE': round(r['MAE'], 4)})
    for n, r in lstm_results.get(split, {}).items():
        rows.append({'Model': 'LSTM', 'Split': split, 'Battery': n,
                     'RMSE': round(r['RMSE'], 4), 'MAE': round(r['MAE'], 4)})

pd.DataFrame(rows).to_csv(os.path.join(BASE_DIR, "results_summary.csv"), index=False)
print(f"\n  Results saved -> results_summary.csv")
print("\nPipeline complete!")
