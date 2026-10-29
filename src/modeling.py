"""
src/modeling.py
===============
Step 4 – Train, evaluate, and save three models per plant.

Pipeline
--------
1.  Load feature CSVs produced by features.py
2.  Drop rows with NaN lags (first hour + first day of each plant)
3.  TIME-BASED split: first 80% of calendar days → train, last 20% → test
    (never random, never shuffle — real forecasting scenario)
4.  Train three models on daytime rows only:
      a) Linear Regression  (baseline)
      b) Random Forest
      c) XGBoost
5.  Evaluate each model on daytime TEST rows:
      MAE, RMSE, R²
6.  Print a comparison table for both plants
7.  Save the best model (lowest RMSE) per plant as a joblib file
8.  Plot predicted vs actual AC_POWER for 3 consecutive test days → save PNG
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib

from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
from xgboost                 import XGBRegressor

warnings.filterwarnings('ignore')   # suppress XGBoost version warnings

# ── paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
MODEL_DIR  = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# columns used as model inputs
FEATURE_COLS = [
    'hour_sin', 'hour_cos',
    'minute_sin', 'minute_cos',
    'day_of_year', 'day_of_week',
    'IRRADIATION',
    'AMBIENT_TEMPERATURE',
    'MODULE_TEMPERATURE',
    'ac_lag_4',
    'ac_lag_96',
]
TARGET = 'AC_POWER'


# ── helpers ───────────────────────────────────────────────────────────────────
def load_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    f1 = pd.read_csv(os.path.join(OUTPUT_DIR, 'plant1_features.csv'),
                     parse_dates=['DATE_TIME'])
    f2 = pd.read_csv(os.path.join(OUTPUT_DIR, 'plant2_features.csv'),
                     parse_dates=['DATE_TIME'])
    return f1, f2


def time_split(df: pd.DataFrame, train_frac: float = 0.80):
    """
    Split by calendar date.
    First train_frac of unique dates → train, remainder → test.
    Never shuffles — preserves temporal order.
    """
    unique_days = sorted(df['DATE_TIME'].dt.date.unique())
    n_train     = int(len(unique_days) * train_frac)
    cutoff_date = unique_days[n_train]          # first test date

    train = df[df['DATE_TIME'].dt.date < cutoff_date].copy()
    test  = df[df['DATE_TIME'].dt.date >= cutoff_date].copy()
    return train, test, cutoff_date


def metrics(y_true, y_pred) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}


# ── model definitions ─────────────────────────────────────────────────────────
def get_models() -> dict:
    return {
        'LinearRegression': LinearRegression(),
        'RandomForest':     RandomForestRegressor(
                                n_estimators=200,
                                max_depth=12,
                                n_jobs=-1,
                                random_state=42
                            ),
        'XGBoost':          XGBRegressor(
                                n_estimators=400,
                                max_depth=6,
                                learning_rate=0.05,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                random_state=42,
                                verbosity=0,
                            ),
    }


# ── train + evaluate one plant ────────────────────────────────────────────────
def run_plant(df: pd.DataFrame, plant_name: str) -> dict:
    """
    Full training pipeline for one plant.
    Returns a results dict keyed by model name.
    """
    # drop rows where lag features are NaN (first 96 rows)
    df = df.dropna(subset=FEATURE_COLS).copy()

    train_all, test_all, cutoff = time_split(df)

    # only daytime rows for training and evaluation
    train = train_all[train_all['IS_DAYTIME']].copy()
    test  = test_all[test_all['IS_DAYTIME']].copy()

    X_train = train[FEATURE_COLS]
    y_train = train[TARGET]
    X_test  = test[FEATURE_COLS]
    y_test  = test[TARGET]

    print(f"\n  {plant_name}: cutoff date = {cutoff}")
    print(f"    Train daytime rows: {len(X_train)}  |  "
          f"Test daytime rows: {len(X_test)}")

    models   = get_models()
    results  = {}
    best_rmse = np.inf
    best_name = None
    best_model = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        preds = np.clip(preds, 0, None)         # power cannot be negative
        m = metrics(y_test, preds)
        results[name] = {**m, 'model': model, 'preds': preds,
                         'test_df': test}
        print(f"    {name:<20}  MAE={m['MAE']:>8.1f}  "
              f"RMSE={m['RMSE']:>8.1f}  R2={m['R2']:.4f}")

        if m['RMSE'] < best_rmse:
            best_rmse  = m['RMSE']
            best_name  = name
            best_model = model

    # save best model
    safe_plant = plant_name.replace(' ', '_').lower()
    model_path = os.path.join(MODEL_DIR, f'best_model_{safe_plant}.joblib')
    joblib.dump(best_model, model_path)
    print(f"\n    Best model: {best_name} (RMSE={best_rmse:.1f})")
    print(f"    Saved: {model_path}")

    results['_best'] = best_name
    results['_test_all'] = test_all    # full test set (day+night) for plotting

    return results


# ── comparison table ──────────────────────────────────────────────────────────
def print_comparison(res1: dict, res2: dict):
    models = ['LinearRegression', 'RandomForest', 'XGBoost']
    print("\n" + "="*70)
    print(f"{'MODEL':<22} {'PLANT':>8}  {'MAE':>9}  {'RMSE':>9}  {'R2':>7}")
    print("="*70)
    for name in models:
        for plant, res in [('Plant 1', res1), ('Plant 2', res2)]:
            m = res[name]
            best_tag = ' *' if res['_best'] == name else ''
            print(f"  {name:<20} {plant:>8}  "
                  f"{m['MAE']:>9.1f}  {m['RMSE']:>9.1f}  "
                  f"{m['R2']:>7.4f}{best_tag}")
        print("-"*70)
    print("  * = best model (saved)")


# ── predicted vs actual plot (3 test days) ───────────────────────────────────
def plot_pred_vs_actual(res: dict, plant_name: str):
    """
    Pick 3 consecutive daytime test days and plot predicted vs actual
    for the best model.
    """
    best_name = res['_best']
    test_df   = res[best_name]['test_df']   # daytime test rows
    preds_arr = res[best_name]['preds']

    # attach predictions to the test dataframe
    test_df = test_df.copy()
    test_df['PREDICTED'] = preds_arr

    # pick first 3 available test dates
    test_dates = sorted(test_df['DATE_TIME'].dt.date.unique())[:3]
    subset = test_df[test_df['DATE_TIME'].dt.date.isin(test_dates)]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    fig.suptitle(f'Predicted vs Actual AC Power – {plant_name} '
                 f'(Best model: {best_name})', fontsize=13)

    for ax, date in zip(axes, test_dates):
        day = subset[subset['DATE_TIME'].dt.date == date]
        ax.plot(day['DATE_TIME'], day[TARGET],    label='Actual',    lw=2)
        ax.plot(day['DATE_TIME'], day['PREDICTED'],
                label='Predicted', lw=2, linestyle='--')
        ax.set_title(str(date))
        ax.set_ylabel('AC Power (kW)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.legend(fontsize=9)

    fig.tight_layout()
    safe_plant = plant_name.replace(' ', '_').lower()
    path = os.path.join(OUTPUT_DIR, f'pred_vs_actual_{safe_plant}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Plot saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    feat1, feat2 = load_features()

    print("Training models ...")
    print("\n--- Plant 1 ---")
    res1 = run_plant(feat1, 'Plant 1')

    print("\n--- Plant 2 ---")
    res2 = run_plant(feat2, 'Plant 2')

    print_comparison(res1, res2)

    print("\nGenerating predicted vs actual plots ...")
    plot_pred_vs_actual(res1, 'Plant 1')
    plot_pred_vs_actual(res2, 'Plant 2')

    print("\nDone. Models saved in outputs/models/")
