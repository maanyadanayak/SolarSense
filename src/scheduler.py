"""
src/scheduler.py
================
Step 6 – Smart Appliance Scheduler.

How it works
------------
1. Take a day's hourly forecast (from forecast.py output CSV).
2. Find the top-3 peak solar hours (highest predicted AC_POWER_FORECAST).
3. Match heavy appliances to those hours based on their power draw and
   how long they run.
4. Estimate:
     - kWh consumed by each appliance
     - Rupees saved by running during solar peak vs. grid tariff
5. Print a schedule table and return it as a DataFrame.

Appliance definitions (can be overridden by caller):
  Each appliance has:
    name        : display name
    power_kw    : rated power in kW
    duration_h  : typical run time in hours
    priority    : 1=must run in peak, 2=run if enough solar, 3=flexible

Tariff:
  Default grid tariff = Rs 7 per kWh (adjustable via TARIFF_RS_PER_KWH).
  Savings = kWh_consumed × tariff  (i.e., we assume all consumption during
  peak solar is offset by solar generation — no export/import modelling).
"""

import os
import pandas as pd
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

# ── default tariff ────────────────────────────────────────────────────────────
TARIFF_RS_PER_KWH = 7.0   # Rs per kWh — change this to your local rate

# ── appliance catalogue ───────────────────────────────────────────────────────
DEFAULT_APPLIANCES = [
    # name                 power_kw  duration_h  priority
    {'name': 'Water Pump',       'power_kw': 1.5,  'duration_h': 2.0, 'priority': 1},
    {'name': 'Washing Machine',  'power_kw': 0.8,  'duration_h': 1.5, 'priority': 1},
    {'name': 'Air Conditioner',  'power_kw': 1.5,  'duration_h': 3.0, 'priority': 2},
    {'name': 'Iron',             'power_kw': 1.0,  'duration_h': 0.5, 'priority': 2},
    {'name': 'Dishwasher',       'power_kw': 1.2,  'duration_h': 1.0, 'priority': 3},
    {'name': 'EV Charger',       'power_kw': 3.3,  'duration_h': 2.0, 'priority': 3},
]

# ── NOTE: this scheduler operates at household / small-plant scale. ───────────
# The forecast CSVs contain plant-level kW (tens of thousands).  We use the
# forecast shape (which hours are sunniest) to define the peak window, but
# the kWh savings are calculated for a *single household* using its own
# appliance load — not the whole plant output.


def load_forecast(plant_id: int) -> pd.DataFrame:
    """Load the hourly forecast CSV produced by forecast.py."""
    path = os.path.join(OUTPUT_DIR, f'forecast_plant_{plant_id}.csv')
    df = pd.read_csv(path, parse_dates=['DATE_TIME'])
    return df


def get_peak_hours(forecast_df: pd.DataFrame,
                   n_peaks: int = 3) -> list[dict]:
    """
    Return the top-n hours by forecasted AC power.
    Only consider hours where AC_POWER_FORECAST > 0 (daytime).
    Returns a list of dicts: {hour, datetime, power_kw}
    """
    day = forecast_df[forecast_df['AC_POWER_FORECAST'] > 0].copy()
    day = day.sort_values('AC_POWER_FORECAST', ascending=False).head(n_peaks)
    day = day.sort_values('DATE_TIME')   # sort chronologically for readability

    peaks = []
    for _, row in day.iterrows():
        peaks.append({
            'hour':     row['DATE_TIME'].hour,
            'datetime': row['DATE_TIME'],
            'power_kw': row['AC_POWER_FORECAST'],
        })
    return peaks


def schedule_appliances(peaks: list[dict],
                        appliances: list[dict] = None,
                        tariff: float = TARIFF_RS_PER_KWH) -> pd.DataFrame:
    """
    Assign each appliance to one of the peak hours.

    Strategy:
    - Priority-1 appliances go to the single highest-power hour (most solar).
    - Priority-2 appliances go to the second-highest hour.
    - Priority-3 appliances go to the third-highest hour.
    - If fewer than 3 peak hours exist, they share the available peaks.

    Returns a DataFrame with columns:
      appliance, start_time, duration_h, power_kw, kwh_used,
      rupees_saved, peak_rank
    """
    if appliances is None:
        appliances = DEFAULT_APPLIANCES

    # sort peaks by power descending so rank 1 = strongest
    sorted_peaks = sorted(peaks, key=lambda x: x['power_kw'], reverse=True)
    n = len(sorted_peaks)

    rows = []
    for appl in appliances:
        # map priority → peak rank (clamp to available peaks)
        rank  = min(appl['priority'], n) - 1   # 0-indexed
        peak  = sorted_peaks[rank]

        kwh_used     = appl['power_kw'] * appl['duration_h']
        rupees_saved = kwh_used * tariff

        rows.append({
            'Appliance':      appl['name'],
            'Start Hour':     f"{peak['hour']:02d}:00",
            'Duration (hrs)': appl['duration_h'],
            'Power (kW)':     appl['power_kw'],
            'kWh Used':       round(kwh_used, 2),
            'Rs Saved':       round(rupees_saved, 2),
            'Peak Rank':      rank + 1,
        })

    return pd.DataFrame(rows)


def summarise(schedule_df: pd.DataFrame,
              tariff: float = TARIFF_RS_PER_KWH) -> dict:
    """Return total kWh shifted to solar and total rupees saved."""
    total_kwh   = schedule_df['kWh Used'].sum()
    total_saved = schedule_df['Rs Saved'].sum()
    return {
        'total_kwh_shifted': round(total_kwh, 2),
        'total_rs_saved':    round(total_saved, 2),
        'tariff_used':       tariff,
    }


def print_schedule(plant_id: int,
                   peaks: list[dict],
                   schedule_df: pd.DataFrame,
                   summary: dict):
    """Pretty-print the peak hours and appliance schedule."""
    print(f"\n{'='*62}")
    print(f"  Smart Appliance Schedule  –  Plant {plant_id} forecast")
    print(f"{'='*62}")

    print(f"\n  Top-3 Peak Solar Hours:")
    print(f"  {'Rank':<6} {'Time':<10} {'Forecast Power':>16}")
    print(f"  {'-'*34}")
    sorted_peaks = sorted(peaks, key=lambda x: x['power_kw'], reverse=True)
    for i, p in enumerate(sorted_peaks, 1):
        print(f"  #{i:<5} {p['datetime'].strftime('%H:%M'):<10} "
              f"{p['power_kw']:>13,.0f} kW")

    print(f"\n  Appliance Schedule  (tariff = Rs {summary['tariff_used']}/kWh):")
    print(f"  {'Appliance':<20} {'Start':<8} {'Hrs':>5} {'kW':>6} "
          f"{'kWh':>6} {'Rs Saved':>9}")
    print(f"  {'-'*58}")
    for _, row in schedule_df.iterrows():
        print(f"  {row['Appliance']:<20} {row['Start Hour']:<8} "
              f"{row['Duration (hrs)']:>5.1f} {row['Power (kW)']:>6.1f} "
              f"{row['kWh Used']:>6.2f} {row['Rs Saved']:>9.2f}")

    print(f"\n  {'-'*58}")
    print(f"  Total kWh shifted to solar : {summary['total_kwh_shifted']:.2f} kWh")
    print(f"  Total rupees saved per day : Rs {summary['total_rs_saved']:.2f}")
    print(f"  Monthly estimate (30 days) : Rs {summary['total_rs_saved']*30:.0f}")
    print(f"{'='*62}\n")


def run_scheduler(plant_id: int = 1,
                  n_peaks: int = 3,
                  tariff: float = TARIFF_RS_PER_KWH,
                  appliances: list[dict] = None) -> tuple[list, pd.DataFrame, dict]:
    """
    Full scheduler pipeline for one plant.
    Returns (peaks, schedule_df, summary).
    """
    forecast_df  = load_forecast(plant_id)
    peaks        = get_peak_hours(forecast_df, n_peaks=n_peaks)
    schedule_df  = schedule_appliances(peaks, appliances=appliances, tariff=tariff)
    summary      = summarise(schedule_df, tariff=tariff)
    print_schedule(plant_id, peaks, schedule_df, summary)

    # save schedule to CSV
    out_path = os.path.join(OUTPUT_DIR, f'schedule_plant_{plant_id}.csv')
    schedule_df.to_csv(out_path, index=False)
    print(f"  Schedule saved: {out_path}")

    return peaks, schedule_df, summary


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Running Smart Scheduler for both plants ...\n")

    peaks1, sched1, summ1 = run_scheduler(plant_id=1)
    peaks2, sched2, summ2 = run_scheduler(plant_id=2)

    print("\nNote: kWh savings are calculated for a single household.")
    print("      Adjust appliance list and tariff to match your setup.")
