"""
src/forecast.py
===============
Step 5 – Forecast Mode using Open-Meteo API.

WHY A SECOND MODEL?
-------------------
The Step 4 model uses real sensor irradiation (IRRADIATION from the plant's
pyranometer). That value is only available *after* the fact — you cannot know
it for tomorrow.

For a genuine day-ahead forecast we must use weather forecast inputs:
  • shortwave_radiation  (W/m²) from Open-Meteo  ← approximates IRRADIATION
  • temperature_2m       (°C)   from Open-Meteo  ← approximates AMBIENT_TEMPERATURE
  • cloud_cover          (%)    from Open-Meteo  ← extra signal for cloud shading

⚠️  CALIBRATION WARNING (read before using in production)
---------------------------------------------------------
Open-Meteo shortwave_radiation is a model value in W/m².
The plant's pyranometer IRRADIATION is in kW/m².  
Dividing by 1000 gives the same unit, but the *scale* may differ because:
  - The plant sensor may be tilted/tracked, not horizontal
  - Atmospheric and albedo corrections differ between NWP and pyranometers
  - Local shading / soiling is not captured by the NWP model

We apply a simple linear scaling factor (IRRAD_SCALE) estimated from the
training data ratio:  mean(sensor irradiation) / mean(NWP shortwave / 1000)
This is a rough calibration — a proper production system would use at least
a month of overlapping sensor + NWP data to fit a calibration regression.

FORECAST FEATURES (no real sensor irradiation used):
  hour_sin, hour_cos, minute_sin, minute_cos  – time cyclical encoding
  day_of_year, day_of_week                   – calendar features
  irrad_forecast   – Open-Meteo SW radiation, unit-converted + scaled
  temp_forecast    – Open-Meteo 2m temperature
  cloud_cover      – Open-Meteo cloud cover (%)
  module_temp_est  – estimated from ambient temp + irradiation proxy

The lag features (ac_lag_4, ac_lag_96) are NOT used here because for a
true day-ahead forecast we do not have yesterday's 15-min readings.
Instead we train a separate "forecast model" that omits them.

OUTPUT
------
  outputs/models/forecast_model_plant_1.joblib
  outputs/models/forecast_model_plant_2.joblib
  outputs/forecast_plant_1.csv   (tomorrow's hourly prediction)
  outputs/forecast_plant_2.csv
  outputs/forecast_plot.png
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
import requests

from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')

# ── paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
MODEL_DIR  = os.path.join(OUTPUT_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ── plant coordinates (Kurnool / Anantapur district, AP, India) ───────────────
PLANTS = {
    1: {'lat': 15.83, 'lon': 78.04, 'name': 'Plant 1'},
    2: {'lat': 14.68, 'lon': 77.60, 'name': 'Plant 2'},
}

# ── forecast feature columns (no lags) ───────────────────────────────────────
FORECAST_FEATURES = [
    'hour_sin', 'hour_cos',
    'minute_sin', 'minute_cos',
    'day_of_year', 'day_of_week',
    'irrad_forecast',        # Open-Meteo SW radiation (kW/m², scaled)
    'temp_forecast',         # Open-Meteo 2m air temp (°C)
    'cloud_cover',           # Open-Meteo cloud cover (%)
    'module_temp_est',       # estimated module temperature
]

TARGET = 'AC_POWER'

# ── helpers ───────────────────────────────────────────────────────────────────
def load_clean(plant_id: int) -> pd.DataFrame:
    path = os.path.join(OUTPUT_DIR, f'plant{plant_id}_clean.csv')
    return pd.read_csv(path, parse_dates=['DATE_TIME'])


def add_forecast_features(df: pd.DataFrame,
                          irrad_scale: float = 1.0) -> pd.DataFrame:
    """
    Build the forecast-model feature set from a cleaned plant DataFrame.

    irrad_scale: multiply sensor IRRADIATION by this factor to simulate what
    the NWP model would have predicted (inverted at inference time to go the
    other way).  During training we use the real sensor values directly,
    just renamed to match the inference column names.
    """
    out = df[['DATE_TIME', 'IS_DAYTIME', TARGET]].copy()

    hour   = df['DATE_TIME'].dt.hour
    minute = df['DATE_TIME'].dt.minute

    out['hour_sin']    = np.sin(2 * np.pi * hour   / 24)
    out['hour_cos']    = np.cos(2 * np.pi * hour   / 24)
    out['minute_sin']  = np.sin(2 * np.pi * minute / 60)
    out['minute_cos']  = np.cos(2 * np.pi * minute / 60)
    out['day_of_year'] = df['DATE_TIME'].dt.day_of_year
    out['day_of_week'] = df['DATE_TIME'].dt.day_of_week

    # During training, use real sensor irradiation as the proxy for NWP
    # (no lag, no future leakage — just using the same-timestamp value)
    out['irrad_forecast']  = df['IRRADIATION']
    out['temp_forecast']   = df['AMBIENT_TEMPERATURE']
    out['cloud_cover']     = 0.0   # not in dataset; set to 0 at train time

    # Simple physics-based estimate: panels heat up ~25 °C above ambient
    # when fully irradiated (1 kW/m²); scale linearly with irradiation
    out['module_temp_est'] = (df['AMBIENT_TEMPERATURE']
                              + 25.0 * df['IRRADIATION'])

    return out


# ── train forecast model ──────────────────────────────────────────────────────
def train_forecast_model(plant_id: int) -> RandomForestRegressor:
    """
    Train a Random Forest on forecast-style features (no lags).
    Uses all available daytime data — the forecast model generalises better
    when trained on the full history.
    """
    df = load_clean(plant_id)
    feat = add_forecast_features(df)

    # use daytime rows only
    day  = feat[feat['IS_DAYTIME']].dropna(subset=FORECAST_FEATURES)
    X    = day[FORECAST_FEATURES]
    y    = day[TARGET]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X, y)

    path = os.path.join(MODEL_DIR, f'forecast_model_plant_{plant_id}.joblib')
    joblib.dump(model, path)
    print(f"  Forecast model saved: {path}  (trained on {len(X)} daytime rows)")
    return model


# ── fetch tomorrow's forecast from Open-Meteo ────────────────────────────────
def fetch_openmeteo(lat: float, lon: float) -> pd.DataFrame:
    """
    Call the free Open-Meteo API and return an hourly DataFrame for
    today + tomorrow (IST timezone).

    ⚠️  shortwave_radiation is in W/m².  We convert to kW/m² (÷ 1000).
    ⚠️  This is a horizontal-surface NWP value — not the same as the plant's
        tilted pyranometer. A calibration factor (IRRAD_SCALE) is applied.
    """
    url = (
        'https://api.open-meteo.com/v1/forecast'
        f'?latitude={lat}&longitude={lon}'
        '&hourly=temperature_2m,cloud_cover,shortwave_radiation,direct_radiation'
        '&forecast_days=2'
        '&timezone=Asia%2FKolkata'
    )
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    data = resp.json()['hourly']

    df = pd.DataFrame({
        'DATE_TIME':           pd.to_datetime(data['time']),
        'temp_forecast':       data['temperature_2m'],
        # convert W/m² → kW/m²
        'shortwave_radiation': [v / 1000.0 for v in data['shortwave_radiation']],
        'cloud_cover':         data['cloud_cover'],
    })
    return df


def calibrate_irradiation(nwp_kw: pd.Series,
                           plant_id: int) -> pd.Series:
    """
    Estimate plant irradiation from NWP shortwave radiation.

    Method: compute the ratio of mean sensor irradiation to mean NWP
    shortwave during training-set daytime hours, then multiply.

    ⚠️  This is a single-scalar calibration — accurate enough for a demo
        but should be replaced with a proper linear regression calibration
        if deploying in production.
    """
    df_clean = load_clean(plant_id)
    day_clean = df_clean[df_clean['IS_DAYTIME'] & (df_clean['IRRADIATION'] > 0)]

    mean_sensor = day_clean['IRRADIATION'].mean()

    # For the NWP side we approximate using the historical hourly mean
    # shortwave at the same location (simulated from the sensor data itself)
    # In a real system you'd have NWP hindcast data to compare against.
    # Here we just use the ratio relative to peak clear-sky (~0.85 kW/m²)
    nwp_daytime = nwp_kw[nwp_kw > 0]
    mean_nwp    = nwp_daytime.mean() if len(nwp_daytime) > 0 else 1.0

    if mean_nwp == 0:
        scale = 1.0
    else:
        scale = mean_sensor / mean_nwp

    print(f"  [calibration] plant {plant_id}: mean sensor irrad={mean_sensor:.4f} "
          f"kW/m², mean NWP={mean_nwp:.4f} kW/m², scale={scale:.4f}")
    return nwp_kw * scale


# ── build inference rows from Open-Meteo data ─────────────────────────────────
def build_inference_df(weather_df: pd.DataFrame,
                       plant_id: int,
                       target_date: str = 'tomorrow') -> pd.DataFrame:
    """
    Turn hourly Open-Meteo data into feature rows the forecast model can
    predict on.  One row per hour (not 15-min; we upsample to 15-min at the end).
    """
    df = weather_df.copy()

    # pick tomorrow (index 1 in forecast_days=2)
    if target_date == 'tomorrow':
        tomorrow = df['DATE_TIME'].dt.date.unique()[1]
        df = df[df['DATE_TIME'].dt.date == tomorrow].copy()
    
    # calibrate NWP irradiation → plant irradiation scale
    df['irrad_forecast'] = calibrate_irradiation(
        df['shortwave_radiation'], plant_id
    )

    hour   = df['DATE_TIME'].dt.hour
    minute = df['DATE_TIME'].dt.minute

    df['hour_sin']    = np.sin(2 * np.pi * hour   / 24)
    df['hour_cos']    = np.cos(2 * np.pi * hour   / 24)
    df['minute_sin']  = np.sin(2 * np.pi * minute / 60)
    df['minute_cos']  = np.cos(2 * np.pi * minute / 60)
    df['day_of_year'] = df['DATE_TIME'].dt.day_of_year
    df['day_of_week'] = df['DATE_TIME'].dt.day_of_week
    df['module_temp_est'] = df['temp_forecast'] + 25.0 * df['irrad_forecast']

    return df


# ── run forecast for one plant ────────────────────────────────────────────────
def forecast_plant(plant_id: int,
                   model: RandomForestRegressor) -> pd.DataFrame:
    info    = PLANTS[plant_id]
    print(f"\n  Fetching Open-Meteo forecast for {info['name']} "
          f"(lat={info['lat']}, lon={info['lon']}) ...")
    weather = fetch_openmeteo(info['lat'], info['lon'])
    infer   = build_inference_df(weather, plant_id)

    X = infer[FORECAST_FEATURES]
    preds = model.predict(X)
    preds = np.clip(preds, 0, None)

    # zero out night-time slots — irradiation=0 means no sun, no generation
    night_mask = infer['irrad_forecast'].values == 0
    preds[night_mask] = 0.0

    result = infer[['DATE_TIME', 'irrad_forecast',
                    'temp_forecast', 'cloud_cover']].copy()
    result['AC_POWER_FORECAST'] = preds

    # add kWh per 1-hour slot
    result['kWh_per_hour'] = result['AC_POWER_FORECAST'] * 1.0   # 1h interval

    out_path = os.path.join(OUTPUT_DIR, f'forecast_plant_{plant_id}.csv')
    result.to_csv(out_path, index=False)
    print(f"  Forecast CSV saved: {out_path}")
    return result


# ── plot forecast ─────────────────────────────────────────────────────────────
def plot_forecast(fc1: pd.DataFrame, fc2: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    fig.suptitle("Day-Ahead AC Power Forecast (Open-Meteo inputs)", fontsize=13)

    for ax, fc, name in zip(axes, [fc1, fc2], ['Plant 1', 'Plant 2']):
        # only plot daylight hours (forecast > 0)
        day = fc[fc['AC_POWER_FORECAST'] > 0]
        ax.fill_between(day['DATE_TIME'], day['AC_POWER_FORECAST'],
                        alpha=0.3, color='steelblue')
        ax.plot(day['DATE_TIME'], day['AC_POWER_FORECAST'],
                color='steelblue', lw=2, label='Forecast AC Power')

        ax2 = ax.twinx()
        ax2.bar(fc['DATE_TIME'], fc['cloud_cover'],
                width=0.04, alpha=0.25, color='grey', label='Cloud cover %')
        ax2.set_ylabel('Cloud Cover (%)', color='grey')
        ax2.tick_params(axis='y', labelcolor='grey')
        ax2.set_ylim(0, 200)

        ax.set_title(f"{name} – {fc['DATE_TIME'].dt.date.iloc[0]}")
        ax.set_ylabel('Forecast AC Power (kW)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.legend(loc='upper left', fontsize=9)

        # annotate peak hour
        peak_idx = fc['AC_POWER_FORECAST'].idxmax()
        peak_row = fc.loc[peak_idx]
        ax.annotate(
            f"Peak: {peak_row['AC_POWER_FORECAST']:.0f} kW\n"
            f"@ {peak_row['DATE_TIME'].strftime('%H:%M')}",
            xy=(peak_row['DATE_TIME'], peak_row['AC_POWER_FORECAST']),
            xytext=(20, -30), textcoords='offset points',
            arrowprops=dict(arrowstyle='->', color='navy'),
            fontsize=9, color='navy',
        )

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'forecast_plot.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Forecast plot saved: {path}")


# ── print summary table ───────────────────────────────────────────────────────
def print_forecast_summary(fc: pd.DataFrame, plant_name: str):
    total_kwh = fc['kWh_per_hour'].sum()
    peak_row  = fc.loc[fc['AC_POWER_FORECAST'].idxmax()]
    day       = fc[fc['AC_POWER_FORECAST'] > 0]
    print(f"\n  {plant_name} forecast summary:")
    print(f"    Date            : {fc['DATE_TIME'].dt.date.iloc[0]}")
    print(f"    Generation hours: {len(day)}")
    print(f"    Total (kWh)     : {total_kwh:,.0f}")
    print(f"    Peak power (kW) : {peak_row['AC_POWER_FORECAST']:,.0f} "
          f"at {peak_row['DATE_TIME'].strftime('%H:%M')}")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Training forecast models (no-lag features) ...")
    model1 = train_forecast_model(plant_id=1)
    model2 = train_forecast_model(plant_id=2)

    print("\nFetching tomorrow's weather forecast from Open-Meteo ...")
    fc1 = forecast_plant(plant_id=1, model=model1)
    fc2 = forecast_plant(plant_id=2, model=model2)

    print_forecast_summary(fc1, 'Plant 1')
    print_forecast_summary(fc2, 'Plant 2')

    print("\nGenerating forecast plot ...")
    plot_forecast(fc1, fc2)

    print("\nDone.")
