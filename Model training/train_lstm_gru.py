import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import time
import os
from scipy.stats import t

# Load sliding window features
data_path = os.path.join(os.path.dirname(__file__), 'sliding_windows.csv')
df = pd.read_csv(data_path, index_col=0)

# Auto-detect feature and target columns for robustness
feature_cols = [col for col in df.columns if not col.endswith('_t+1')]
target_cols = [col for col in df.columns if col.endswith('_t+1')]
X = df[feature_cols].values
Y = df[target_cols].values  # [cpu_rate_mean_t+1, canonical_memory_usage_mean_t+1]

# Normalize features and targets
X_scaler = MinMaxScaler()
Y_scaler = MinMaxScaler()
X_scaled = X_scaler.fit_transform(X)
Y_scaled = Y_scaler.fit_transform(Y)

# Reshape for GRU: (samples, timesteps, features)
window_size = 30  # keep consistent with feature engineering
num_features = int(X.shape[1] // window_size)
X_seq = X_scaled.reshape((X.shape[0], window_size, num_features))

# Train/test split (time-based, not random)
split = int(0.8 * X_seq.shape[0])
X_train, X_test = X_seq[:split], X_seq[split:]
Y_train, Y_test = Y_scaled[:split], Y_scaled[split:]

# Statistical significance: run with multiple seeds
SEEDS = list(range(10, 30))  # 20 seeds
metrics_all = []

for seed in SEEDS:
    tf.keras.utils.set_random_seed(seed)
    model = Sequential()
    model.add(Bidirectional(GRU(128, return_sequences=True), input_shape=(window_size, num_features)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(GRU(64, return_sequences=False)))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2))  # Multi-output: [cpu, mem]
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    model.fit(X_train, Y_train, epochs=200, batch_size=32, verbose=0, validation_split=0.1, callbacks=[early_stop])
    # Inference latency measurement
    start = time.time()
    Y_pred_scaled = model.predict(X_test, verbose=0)
    latency = (time.time() - start) / X_test.shape[0] * 1000  # ms per prediction
    # Invert normalization for metrics
    Y_pred = Y_scaler.inverse_transform(Y_pred_scaled)
    Y_true = Y_scaler.inverse_transform(Y_test)
    # Metrics
    mape_cpu = mean_absolute_percentage_error(Y_true[:, 0], Y_pred[:, 0])
    mape_mem = mean_absolute_percentage_error(Y_true[:, 1], Y_pred[:, 1])
    rmse_cpu = np.sqrt(mean_squared_error(Y_true[:, 0], Y_pred[:, 0]))
    rmse_mem = np.sqrt(mean_squared_error(Y_true[:, 1], Y_pred[:, 1]))
    mae_cpu = mean_absolute_error(Y_true[:, 0], Y_pred[:, 0])
    mae_mem = mean_absolute_error(Y_true[:, 1], Y_pred[:, 1])
    mask = Y_true[:, 0] > 0.01
    mape_cpu_filtered = mean_absolute_percentage_error(Y_true[mask, 0], Y_pred[mask, 0])
    metrics_all.append({
        'seed': seed,
        'rmse_cpu': rmse_cpu,
        'rmse_mem': rmse_mem,
        'mae_cpu': mae_cpu,
        'mae_mem': mae_mem,
        'mape_cpu': mape_cpu,
        'mape_mem': mape_mem,
        'mape_cpu_filtered': mape_cpu_filtered,
        'latency_ms': latency
    })

# Compute mean and 95% CI for each metric
print(f"\nGRU (bidirectional, deep) results across {len(SEEDS)} seeds:")
def print_ci(metric):
    vals = [m[metric] for m in metrics_all]
    mean = np.mean(vals)
    std = np.std(vals, ddof=1)
    ci95 = t.ppf(0.975, len(vals)-1) * std / np.sqrt(len(vals))
    print(f"{metric.upper()}: {mean:.6f} ± {ci95:.6f} (95% CI)")

for metric in ['rmse_cpu', 'rmse_mem', 'mae_cpu', 'mae_mem', 'mape_cpu', 'mape_mem', 'mape_cpu_filtered', 'latency_ms']:
    print_ci(metric)

# Save per-seed results
pd.DataFrame(metrics_all).to_csv('gru_metrics_seeds.csv', index=False)
print("Per-seed metrics saved to gru_metrics_seeds.csv")

# Optionally, save the last model
model.save(os.path.join(os.path.dirname(__file__), 'gru_deep_model.h5'))
print("Model saved as gru_deep_model.h5")
