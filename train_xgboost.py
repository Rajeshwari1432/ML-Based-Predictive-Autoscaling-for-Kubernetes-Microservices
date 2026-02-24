import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, make_scorer
from sklearn.feature_selection import SelectFromModel
import os
import matplotlib.pyplot as plt
from scipy.stats import t

# Load sliding window features
data_path = os.path.join(os.path.dirname(__file__), 'sliding_windows.csv')
df = pd.read_csv(data_path, index_col=0)

# Feature groups for ablation
base_features = [col for col in df.columns if 'cpu_rate_mean_t-' in col or 'canonical_memory_usage_mean_t-' in col]
lag_features = [col for col in df.columns if 'lag' in col]
ema_features = [col for col in df.columns if 'ema' in col]
time_features = [col for col in df.columns if any(t in col for t in ['hour', 'minute', 'dayofweek'])]

# Target selection
target_col = [col for col in df.columns if col == 'cpu_rate_mean_t+1']
y_cpu = df[target_col[0]].values

# Ablation configs
ablation_configs = [
    ("Base", base_features),
    ("Base+Lags", base_features + lag_features),
    ("Base+Lags+EMA", base_features + lag_features + ema_features),
    ("All", base_features + lag_features + ema_features + time_features)
]

# Statistical significance: run with multiple seeds
SEEDS = list(range(10, 30))  # 20 seeds
metrics_all = {label: [] for label, _ in ablation_configs}

for label, feature_cols in ablation_configs:
    print(f"\n--- Running ablation: {label} ---")
    X = df[feature_cols].values
    # Feature selection
    base_selector = xgb.XGBRegressor(n_estimators=200, max_depth=6, random_state=42, tree_method='hist')
    base_selector.fit(X, y_cpu)
    selector = SelectFromModel(base_selector, threshold='median', prefit=True)
    X_selected = selector.transform(X)
    # Time series split
    tscv = TimeSeriesSplit(n_splits=5)
    param_dist = {
        'n_estimators': [300],
        'max_depth': [6],
        'learning_rate': [0.01],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'reg_alpha': [0.05],
        'reg_lambda': [0.7]
    }
    mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)
    for seed in SEEDS:
        search = RandomizedSearchCV(
            xgb.XGBRegressor(tree_method='hist', random_state=seed),
            param_distributions=param_dist,
            n_iter=1,
            scoring=mape_scorer,
            cv=tscv,
            verbose=0,
            n_jobs=-1
        )
        search.fit(X_selected, y_cpu)
        # Train/test split
        split = int(0.8 * X_selected.shape[0])
        X_train, X_test = X_selected[:split], X_selected[split:]
        y_train, y_test = y_cpu[:split], y_cpu[split:]
        best_model = search.best_estimator_
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mask = y_test > 0.01
        mape_filtered = mean_absolute_percentage_error(y_test[mask], y_pred[mask])
        metrics_all[label].append({
            'seed': seed,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'mape_filtered': mape_filtered
        })
    # Compute mean and 95% CI for each metric
    print(f"\n{label} (across {len(SEEDS)} seeds):")
    for metric in ['rmse', 'mae', 'mape', 'mape_filtered']:
        vals = [m[metric] for m in metrics_all[label]]
        mean = np.mean(vals)
        std = np.std(vals, ddof=1)
        ci95 = t.ppf(0.975, len(vals)-1) * std / np.sqrt(len(vals))
        print(f"{metric.upper()}: {mean:.6f} ± {ci95:.6f} (95% CI)")
    # Save per-seed results
    pd.DataFrame(metrics_all[label]).to_csv(f'xgb_{label}_metrics_seeds.csv', index=False)
    print(f"Per-seed metrics saved to xgb_{label}_metrics_seeds.csv")

# Optionally, run the original ablation for plots (single seed)
results = []
for label, feature_cols in ablation_configs:
    X = df[feature_cols].values
    base_selector = xgb.XGBRegressor(n_estimators=200, max_depth=6, random_state=42, tree_method='hist')
    base_selector.fit(X, y_cpu)
    selector = SelectFromModel(base_selector, threshold='median', prefit=True)
    X_selected = selector.transform(X)
    split = int(0.8 * X_selected.shape[0])
    X_train, X_test = X_selected[:split], X_selected[split:]
    y_train, y_test = y_cpu[:split], y_cpu[split:]
    best_model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.05, reg_lambda=0.7, tree_method='hist', random_state=42)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mask = y_test > 0.01
    mape_filtered = mean_absolute_percentage_error(y_test[mask], y_pred[mask])
    results.append((label, rmse, mae, mape, mape_filtered, y_test, y_pred))
    # Plot residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 4))
    plt.hist(residuals, bins=40, color='skyblue', edgecolor='black')
    plt.title(f'Residuals Histogram ({label})')
    plt.xlabel('Residual (True - Predicted)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'residuals_hist_{label}.png')
    plt.close()
    # Plot predictions vs. ground truth (first 200)
    plt.figure(figsize=(12, 5))
    plt.plot(y_test[:200], label='True', marker='o', markersize=2)
    plt.plot(y_pred[:200], label='Predicted', marker='x', markersize=2)
    plt.title(f'Predictions vs. True ({label}, first 200)')
    plt.xlabel('Sample Index')
    plt.ylabel('CPU Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'pred_vs_true_{label}.png')
    plt.close()

# Print summary table
print("\nAblation Study Results (single seed):")
print("Label\t\tRMSE\t\tMAE\t\tMAPE(all)\tMAPE(>0.01)")
for label, rmse, mae, mape, mape_filtered, _, _ in results:
    print(f"{label:12s}\t{rmse:.6f}\t{mae:.6f}\t{mape:.4f}\t\t{mape_filtered:.4f}")

# Final model: All features, plot importances
X = df[ablation_configs[-1][1]].values
base_selector = xgb.XGBRegressor(n_estimators=200, max_depth=6, random_state=42, tree_method='hist')
base_selector.fit(X, y_cpu)
selector = SelectFromModel(base_selector, threshold='median', prefit=True)
X_selected = selector.transform(X)
split = int(0.8 * X_selected.shape[0])
X_train, X_test = X_selected[:split], X_selected[split:]
y_train, y_test = y_cpu[:split], y_cpu[split:]
best_model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.05, reg_lambda=0.7, tree_method='hist', random_state=42)
best_model.fit(X_train, y_train)
plt.figure(figsize=(12, 6))
xgb.plot_importance(best_model, max_num_features=15, importance_type='gain', show_values=False)
plt.title('Top 15 XGBoost Feature Importances (Gain, All Features)')
plt.tight_layout()
plt.savefig('xgboost_feature_importance.png')
plt.close()
print("Feature importance plot saved as xgboost_feature_importance.png")
