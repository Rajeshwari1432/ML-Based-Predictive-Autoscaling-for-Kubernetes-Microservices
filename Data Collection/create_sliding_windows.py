import pandas as pd
import numpy as np
import os

# Parameters
WINDOW_SIZE = 30  # number of past minutes in each input window
HORIZONS = [1, 2, 4]  # predict 1, 2, 4 steps ahead (e.g., 30s, 60s, 120s if 30s step)

# Load per-minute aggregated data
csv_path = os.path.join(os.path.dirname(__file__), 'per_minute_agg.csv')
df = pd.read_csv(csv_path, header=[0, 1], index_col=0, parse_dates=True)

# Flatten multi-level columns
if isinstance(df.columns, pd.MultiIndex):
    df.columns = ['_'.join(col).strip() for col in df.columns.values]

# Feature engineering
features = ['cpu_rate_mean', 'canonical_memory_usage_mean']
data = df[features].copy()

# Add rolling mean and std features
for feat in features:
    data[f'{feat}_rollmean'] = data[feat].rolling(WINDOW_SIZE, min_periods=1).mean()
    data[f'{feat}_rollstd'] = data[feat].rolling(WINDOW_SIZE, min_periods=1).std().fillna(0)
    # Add EMA features
    data[f'{feat}_ema5'] = data[feat].ewm(span=5, adjust=False).mean()
    data[f'{feat}_ema10'] = data[feat].ewm(span=10, adjust=False).mean()
    data[f'{feat}_ema20'] = data[feat].ewm(span=20, adjust=False).mean()

# Add more lag features (every 5th minute up to 60 min)
for feat in features:
    for lag in range(5, 61, 5):
        data[f'{feat}_lag{lag}'] = data[feat].shift(lag)

# Add hour-of-day, day-of-week, and minute-of-hour features
if hasattr(data.index, 'hour'):
    data['hour'] = data.index.hour
    data['minute'] = data.index.minute
    data['dayofweek'] = data.index.dayofweek
else:
    dt = pd.to_datetime(data.index)
    data['hour'] = dt.hour
    data['minute'] = dt.minute
    data['dayofweek'] = dt.dayofweek

# Drop any NaNs
data = data.dropna()

# Create sliding windows
def make_windows(data, window_size, horizons):
    X, y = [], {h: [] for h in horizons}
    idx = []
    for i in range(len(data) - window_size - max(horizons)):
        window = data.iloc[i:i+window_size]
        # Flatten all features except targets
        X.append(window.values.flatten())
        for h in horizons:
            y[h].append(data.iloc[i+window_size+h-1][features].values)  # predict at horizon
        idx.append(data.index[i+window_size-1])
    X = np.array(X)
    y = {h: np.array(y[h]) for h in horizons}
    return X, y, idx

X, y, idx = make_windows(data, WINDOW_SIZE, HORIZONS)

# Save as DataFrame for the shortest horizon (e.g., 1 step ahead)
columns = [f'{col}_t-{WINDOW_SIZE-k}' for k in range(WINDOW_SIZE, 0, -1) for col in data.columns]
y_columns = [f'{feat}_t+{HORIZONS[0]}' for feat in features]
sliding_df = pd.DataFrame(X, columns=columns, index=idx)
for j, feat in enumerate(features):
    sliding_df[y_columns[j]] = y[HORIZONS[0]][:, j]

output_path = os.path.join(os.path.dirname(__file__), 'sliding_windows.csv')
sliding_df.to_csv(output_path)
print(f"Sliding window features saved to: {output_path}")
print(f"Shape: {sliding_df.shape}")
