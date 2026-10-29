"""
src/features.py
===============
Step 3 – Feature Engineering.

Features built
--------------
TIME FEATURES
  hour_sin, hour_cos   – cyclical encoding of hour (so hour 23 is close to hour 0)
  minute_sin, minute_cos – cyclical encoding of minute-within-hour
  day_of_year          – captures seasonal trend across the 34-day window
  day_of_week          – 0=Monday … 6=Sunday (mild weekly effect)

WEATHER FEATURES (passed in directly from the sensor data)
  IRRADIATION          – strongest predictor (kW/m²)
  AMBIENT_TEMPERATURE  – air temperature (°C)
  MODULE_TEMPERATURE   – panel surface temperature (°C)

LAG FEATURES  (shift on the per-plant sorted time-series)
  ac_lag_4             – AC_POWER 4 steps ago  = 1 hour ago  (4 × 15 min)
  ac_lag_96            – AC_POWER 96 steps ago = 24 hours ago (96 × 15 min)

TARGET
  AC_POWER             – plant-level AC output in kW

After building features the script:
  * Prints a feature correlation table (daytime rows only)
  * Saves outputs/plant1_features.csv and outputs/plant2_features.csv
"""

import os
import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')


# ── helpers ───────────────────────────────────────────────────────────────────
def load_clean() -> tuple[pd.DataFrame, pd.DataFrame]:
    p1 = pd.read_csv(os.path.join(OUTPUT_DIR, 'plant1_clean.csv'),
                     parse_dates=['DATE_TIME'])
    p2 = pd.read_csv(os.path.join(OUTPUT_DIR, 'plant2_clean.csv'),
                     parse_dates=['DATE_TIME'])
    return p1, p2


# ── feature builder ───────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a cleaned plant DataFrame and returns a new DataFrame with all
    engineered features plus the target column AC_POWER.

    The input df must be sorted by DATE_TIME (ascending) — the cleaning
    step already guarantees this.
    """
    out = df[['DATE_TIME', 'PLANT_ID', 'IS_DAYTIME', 'AC_POWER']].copy()

    # ── time features ──────────────────────────────────────────────────────
    hour        = df['DATE_TIME'].dt.hour
    minute      = df['DATE_TIME'].dt.minute

    # sin/cos encoding wraps the cycle: hour 23 becomes close to hour 0
    out['hour_sin']    = np.sin(2 * np.pi * hour   / 24)
    out['hour_cos']    = np.cos(2 * np.pi * hour   / 24)
    out['minute_sin']  = np.sin(2 * np.pi * minute / 60)
    out['minute_cos']  = np.cos(2 * np.pi * minute / 60)
    out['day_of_year'] = df['DATE_TIME'].dt.day_of_year
    out['day_of_week'] = df['DATE_TIME'].dt.day_of_week

    # ── weather features ───────────────────────────────────────────────────
    out['IRRADIATION']          = df['IRRADIATION']
    out['AMBIENT_TEMPERATURE']  = df['AMBIENT_TEMPERATURE']
    out['MODULE_TEMPERATURE']   = df['MODULE_TEMPERATURE']

    # ── lag features ──────────────────────────────────────────────────────
    # shift() works on row position; since the data is sorted and has
    # (mostly) uniform 15-min spacing, step 4 = 1 hour, step 96 = 24 hours.
    out['ac_lag_4']  = df['AC_POWER'].shift(4)    # 1 hour ago
    out['ac_lag_96'] = df['AC_POWER'].shift(96)   # 24 hours ago

    # Rows where lags are NaN (first hour / first day) will be dropped
    # downstream when we split train/test — we keep them here so the
    # DATE_TIME index stays intact.

    return out


# ── correlation report ────────────────────────────────────────────────────────
def print_correlations(df_feat: pd.DataFrame, plant_name: str):
    """Print Pearson correlation of each feature with AC_POWER (daytime only)."""
    feature_cols = [
        'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',
        'day_of_year', 'day_of_week',
        'IRRADIATION', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE',
        'ac_lag_4', 'ac_lag_96',
    ]
    daytime = df_feat[df_feat['IS_DAYTIME']].dropna(subset=feature_cols)
    corr = daytime[feature_cols + ['AC_POWER']].corr()['AC_POWER'].drop('AC_POWER')
    corr_sorted = corr.abs().sort_values(ascending=False)

    print(f"\n  Feature correlations with AC_POWER ({plant_name}, daytime rows):")
    print(f"  {'Feature':<25} {'Pearson r':>10}")
    print(f"  {'-'*37}")
    for feat in corr_sorted.index:
        print(f"  {feat:<25} {corr[feat]:>10.4f}")


# ── main ──────────────────────────────────────────────────────────────────────
def build_and_save() -> tuple[pd.DataFrame, pd.DataFrame]:
    p1, p2 = load_clean()

    feat1 = build_features(p1)
    feat2 = build_features(p2)

    # save for use by modeling script
    out1 = os.path.join(OUTPUT_DIR, 'plant1_features.csv')
    out2 = os.path.join(OUTPUT_DIR, 'plant2_features.csv')
    feat1.to_csv(out1, index=False)
    feat2.to_csv(out2, index=False)
    print(f"Saved: {out1}")
    print(f"Saved: {out2}")

    print(f"\nPlant 1 feature table shape: {feat1.shape}")
    print(f"Plant 2 feature table shape: {feat2.shape}")

    print_correlations(feat1, 'Plant 1')
    print_correlations(feat2, 'Plant 2')

    # show a few rows so we can visually verify the features
    print("\nPlant 1 – sample rows (daytime):")
    cols_show = ['DATE_TIME', 'hour_sin', 'hour_cos', 'day_of_year',
                 'IRRADIATION', 'ac_lag_4', 'ac_lag_96', 'AC_POWER']
    print(feat1[feat1['IS_DAYTIME']].head(4)[cols_show].to_string(index=False))

    return feat1, feat2


if __name__ == '__main__':
    build_and_save()
