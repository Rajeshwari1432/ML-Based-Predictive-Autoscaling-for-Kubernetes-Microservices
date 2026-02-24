import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the per-minute aggregated data
csv_path = os.path.join(os.path.dirname(__file__), 'per_minute_agg.csv')
df = pd.read_csv(csv_path, header=[0, 1], index_col=0, parse_dates=True)

# Flatten multi-level columns
if isinstance(df.columns, pd.MultiIndex):
    df.columns = ['_'.join(col).strip() for col in df.columns.values]

print("Summary statistics:")
print(df.describe())

# Plot CPU mean and max usage over time
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['cpu_rate_mean'], label='CPU Mean')
plt.plot(df.index, df['cpu_rate_max'], label='CPU Max', alpha=0.7)
plt.title('CPU Usage Over Time (Per Minute)')
plt.xlabel('Time')
plt.ylabel('CPU Rate (fraction of core)')
plt.legend()
plt.tight_layout()
plt.savefig('cpu_usage_over_time.png')
plt.close()

# Plot Memory usage over time
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['canonical_memory_usage_mean'], label='Memory Mean', color='green')
plt.title('Memory Usage Over Time (Per Minute)')
plt.xlabel('Time')
plt.ylabel('Memory Usage (bytes)')
plt.legend()
plt.tight_layout()
plt.savefig('memory_usage_over_time.png')
plt.close()

# Histograms
plt.figure(figsize=(8, 4))
sns.histplot(df['cpu_rate_mean'], bins=50, kde=True)
plt.title('Histogram of Per-Minute Mean CPU Rate')
plt.xlabel('CPU Rate (fraction of core)')
plt.tight_layout()
plt.savefig('cpu_rate_mean_hist.png')
plt.close()

plt.figure(figsize=(8, 4))
sns.histplot(df['canonical_memory_usage_mean'], bins=50, kde=True, color='green')
plt.title('Histogram of Per-Minute Mean Memory Usage')
plt.xlabel('Memory Usage (bytes)')
plt.tight_layout()
plt.savefig('memory_usage_mean_hist.png')
plt.close()

# Correlation
corr = df[['cpu_rate_mean', 'canonical_memory_usage_mean']].corr()
print("\nCorrelation between per-minute mean CPU and memory usage:")
print(corr)

# Optional: Day-of-week and hour-of-day patterns (seasonality)
df['hour'] = df.index.hour
df['minute'] = df.index.minute
plt.figure(figsize=(10, 4))
sns.boxplot(x='hour', y='cpu_rate_mean', data=df)
plt.title('CPU Rate Mean by Hour of Day')
plt.tight_layout()
plt.savefig('cpu_rate_by_hour.png')
plt.close()

# --- New: EDA for engineered features ---
# Load sliding window features for engineered feature EDA
sw_path = os.path.join(os.path.dirname(__file__), 'sliding_windows.csv')
sw_df = pd.read_csv(sw_path, index_col=0)

# Plot EMA features (first window)
ema_cols = [col for col in sw_df.columns if 'ema' in col]
if ema_cols:
    plt.figure(figsize=(12, 5))
    for col in ema_cols[:3]:  # plot up to 3 EMA features
        plt.plot(sw_df.index[:200], sw_df[col][:200], label=col)
    plt.title('Sample EMA Features (first 200 samples)')
    plt.xlabel('Sample Index')
    plt.ylabel('EMA Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig('ema_features_sample.png')
    plt.close()

# Plot lag features (first window)
lag_cols = [col for col in sw_df.columns if 'lag' in col]
if lag_cols:
    plt.figure(figsize=(12, 5))
    for col in lag_cols[:3]:  # plot up to 3 lag features
        plt.plot(sw_df.index[:200], sw_df[col][:200], label=col)
    plt.title('Sample Lag Features (first 200 samples)')
    plt.xlabel('Sample Index')
    plt.ylabel('Lag Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig('lag_features_sample.png')
    plt.close()

# Correlation heatmap for all features (first 30 for readability)
plt.figure(figsize=(16, 12))
sns.heatmap(sw_df.iloc[:, :30].corr(), cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap (first 30 features)')
plt.tight_layout()
plt.savefig('feature_corr_heatmap.png')
plt.close()

print("\nEDA plots for engineered features saved as PNG files in the current directory.")
