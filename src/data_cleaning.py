"""
src/data_cleaning.py
====================
Step 1 – Load, inspect, clean, and merge the four CSV files.

Key decisions
-------------
* Plant 1 generation uses DD-MM-YYYY HH:MM  → explicit format '%d-%m-%Y %H:%M'
* All other files use YYYY-MM-DD HH:MM:SS   → explicit format '%Y-%m-%d %H:%M:%S'
* Generation data has one row per inverter (SOURCE_KEY).
  We aggregate (sum) DC_POWER and AC_POWER per (timestamp, plant) so the result
  can be joined 1-to-1 with the single-sensor weather data.
* Daytime is defined as rows where IRRADIATION > 0.
* A boolean column 'IS_DAYTIME' is added so downstream steps can filter easily.
"""

import os
import pandas as pd
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. load raw files ─────────────────────────────────────────────────────────
def load_raw():
    """Read all four CSVs and return them as a dict of DataFrames."""

    # Plant 1 generation: dates are  DD-MM-YYYY HH:MM  (no seconds)
    gen1 = pd.read_csv(os.path.join(DATA_DIR, 'Plant_1_Generation_Data.csv'))

    # Plant 2 generation: dates are  YYYY-MM-DD HH:MM:SS
    gen2 = pd.read_csv(os.path.join(DATA_DIR, 'Plant_2_Generation_Data.csv'))

    # Weather files both use YYYY-MM-DD HH:MM:SS
    wx1  = pd.read_csv(os.path.join(DATA_DIR, 'Plant_1_Weather_Sensor_Data.csv'))
    wx2  = pd.read_csv(os.path.join(DATA_DIR, 'Plant_2_Weather_Sensor_Data.csv'))

    return {'gen1': gen1, 'gen2': gen2, 'wx1': wx1, 'wx2': wx2}


# ── 2. inspect ────────────────────────────────────────────────────────────────
def inspect(raw: dict):
    """Print shape, columns, dtypes, and missing-value counts for each file."""
    labels = {
        'gen1': 'Plant 1 – Generation',
        'gen2': 'Plant 2 – Generation',
        'wx1' : 'Plant 1 – Weather',
        'wx2' : 'Plant 2 – Weather',
    }
    for key, df in raw.items():
        print(f"\n{'='*60}")
        print(f"  {labels[key]}")
        print(f"{'='*60}")
        print(f"  Shape  : {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  dtypes :\n{df.dtypes.to_string()}")
        missing = df.isnull().sum()
        if missing.any():
            print(f"  Missing values:\n{missing[missing > 0].to_string()}")
        else:
            print("  Missing values: none")
        print(f"  Sample DATE_TIME values: {df['DATE_TIME'].iloc[:3].tolist()}")


# ── 3. parse dates ────────────────────────────────────────────────────────────
def parse_dates(raw: dict) -> dict:
    """
    Convert DATE_TIME strings to pandas Timestamps.
    Plant 1 generation uses DD-MM-YYYY HH:MM; everything else uses
    YYYY-MM-DD HH:MM:SS.  We pass explicit format strings so there is
    no ambiguity and parsing is fast.
    """
    raw['gen1']['DATE_TIME'] = pd.to_datetime(
        raw['gen1']['DATE_TIME'], format='%d-%m-%Y %H:%M'
    )
    for key in ('gen2', 'wx1', 'wx2'):
        raw[key]['DATE_TIME'] = pd.to_datetime(
            raw[key]['DATE_TIME'], format='%Y-%m-%d %H:%M:%S'
        )
    print("\n[parse_dates] All DATE_TIME columns converted to datetime64.")
    return raw


# ── 4. aggregate generation ───────────────────────────────────────────────────
def aggregate_generation(gen_df: pd.DataFrame) -> pd.DataFrame:
    """
    Each timestamp has one row per inverter.
    Sum DC_POWER and AC_POWER across all inverters for each timestamp.
    Also keep DAILY_YIELD (max across inverters is meaningless; sum gives
    plant-level daily yield).
    """
    agg = (
        gen_df
        .groupby('DATE_TIME', as_index=False)
        .agg(
            DC_POWER    = ('DC_POWER',    'sum'),
            AC_POWER    = ('AC_POWER',    'sum'),
            DAILY_YIELD = ('DAILY_YIELD', 'sum'),
        )
    )
    return agg


# ── 5. merge generation + weather ────────────────────────────────────────────
def merge_plant(gen_df: pd.DataFrame,
                wx_df: pd.DataFrame,
                plant_id: int) -> pd.DataFrame:
    """
    Inner-join aggregated generation with weather on DATE_TIME.
    An inner join drops timestamps that exist in only one file
    (handles missing timestamps / sensor outages automatically).
    """
    # keep only the columns we need from the weather file
    wx_keep = ['DATE_TIME', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']
    merged = gen_df.merge(wx_df[wx_keep], on='DATE_TIME', how='inner')
    merged.insert(0, 'PLANT_ID', plant_id)
    merged = merged.sort_values('DATE_TIME').reset_index(drop=True)
    return merged


# ── 6. sensor-glitch / outlier handling ──────────────────────────────────────
def clean_glitches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple rule-based glitch removal:
    * Negative power → set to 0 (inverter fault codes sometimes log -ve values)
    * Irradiation > 1.5 kW/m² is physically implausible → cap at 1.5
    * MODULE_TEMPERATURE > 100 °C → replace with NaN, forward-fill
    After these fixes, forward-fill any remaining NaNs (up to 2 steps = 30 min)
    so short sensor outages don't create gaps in the time-series.
    """
    for col in ('DC_POWER', 'AC_POWER', 'DAILY_YIELD'):
        df[col] = df[col].clip(lower=0)

    df['IRRADIATION'] = df['IRRADIATION'].clip(upper=1.5)

    implausible_temp = df['MODULE_TEMPERATURE'] > 100
    if implausible_temp.any():
        print(f"  [clean] Replacing {implausible_temp.sum()} implausible MODULE_TEMPERATURE values.")
        df.loc[implausible_temp, 'MODULE_TEMPERATURE'] = np.nan

    # forward-fill then back-fill short gaps
    df = df.ffill(limit=2).bfill(limit=2)
    return df


# ── 7. add helper columns ─────────────────────────────────────────────────────
def add_helpers(df: pd.DataFrame) -> pd.DataFrame:
    """Add IS_DAYTIME flag (irradiation > 0) and hour column for convenience."""
    df['IS_DAYTIME'] = df['IRRADIATION'] > 0.0
    df['HOUR']       = df['DATE_TIME'].dt.hour
    return df


# ── 8. main pipeline ──────────────────────────────────────────────────────────
def build_clean_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full cleaning pipeline.
    Returns (plant1_df, plant2_df) — one merged DataFrame per plant,
    indexed by DATE_TIME, with generation + weather columns.
    """
    raw = load_raw()
    inspect(raw)
    raw = parse_dates(raw)

    # aggregate inverter-level → plant-level
    gen1_agg = aggregate_generation(raw['gen1'])
    gen2_agg = aggregate_generation(raw['gen2'])

    # merge with weather
    plant1 = merge_plant(gen1_agg, raw['wx1'], plant_id=1)
    plant2 = merge_plant(gen2_agg, raw['wx2'], plant_id=2)

    # clean glitches
    plant1 = clean_glitches(plant1)
    plant2 = clean_glitches(plant2)

    # add helper columns
    plant1 = add_helpers(plant1)
    plant2 = add_helpers(plant2)

    # save to outputs so other scripts can load the cleaned data
    plant1.to_csv(os.path.join(OUTPUT_DIR, 'plant1_clean.csv'), index=False)
    plant2.to_csv(os.path.join(OUTPUT_DIR, 'plant2_clean.csv'), index=False)

    print("\n[build_clean_data] Summary after cleaning:")
    for name, df in [('Plant 1', plant1), ('Plant 2', plant2)]:
        day = df[df['IS_DAYTIME']]
        print(f"  {name}: {len(df)} rows total, "
              f"{len(day)} daytime rows, "
              f"date range {df['DATE_TIME'].min().date()} to {df['DATE_TIME'].max().date()}")

    print(f"\n  Cleaned CSVs saved to {os.path.abspath(OUTPUT_DIR)}")
    return plant1, plant2


# ── run directly ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    p1, p2 = build_clean_data()
    print("\nPlant 1 – head:")
    print(p1.head(3).to_string())
    print("\nPlant 2 – head:")
    print(p2.head(3).to_string())
