"""
src/eda.py
==========
Step 2 – Exploratory Data Analysis.

Plots produced (saved to /outputs):
  1. eda_avg_power_by_hour.png   – mean AC_POWER by hour of day, both plants
  2. eda_irradiation_vs_power.png – scatter: irradiation vs AC_POWER (daytime only)
  3. eda_modtemp_vs_power.png    – scatter: module temperature vs AC_POWER (daytime only)
  4. eda_daily_yield_trend.png   – total daily AC energy over the 34-day period
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend – works on all machines
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# ── paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

# consistent style for all plots
sns.set_theme(style='whitegrid', palette='tab10')
FIGSIZE = (10, 5)


# ── helpers ───────────────────────────────────────────────────────────────────
def load_clean() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the cleaned CSVs produced by data_cleaning.py."""
    p1 = pd.read_csv(os.path.join(OUTPUT_DIR, 'plant1_clean.csv'), parse_dates=['DATE_TIME'])
    p2 = pd.read_csv(os.path.join(OUTPUT_DIR, 'plant2_clean.csv'), parse_dates=['DATE_TIME'])
    return p1, p2


def save(fig: plt.Figure, name: str):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ── plot 1 : average AC power by hour of day ──────────────────────────────────
def plot_avg_power_by_hour(p1: pd.DataFrame, p2: pd.DataFrame):
    """
    Group by hour, compute mean AC_POWER.
    We use all rows (including night = 0) so the curve naturally shows the
    bell-shaped solar generation profile.
    """
    h1 = p1.groupby('HOUR')['AC_POWER'].mean().reset_index()
    h2 = p2.groupby('HOUR')['AC_POWER'].mean().reset_index()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(h1['HOUR'], h1['AC_POWER'], marker='o', label='Plant 1')
    ax.plot(h2['HOUR'], h2['AC_POWER'], marker='s', label='Plant 2')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Mean AC Power (kW)')
    ax.set_title('Average AC Power by Hour of Day')
    ax.set_xticks(range(0, 24))
    ax.legend()
    fig.tight_layout()
    save(fig, 'eda_avg_power_by_hour.png')


# ── plot 2 : irradiation vs AC power (daytime scatter) ───────────────────────
def plot_irradiation_vs_power(p1: pd.DataFrame, p2: pd.DataFrame):
    """
    Only daytime rows (IS_DAYTIME=True).
    Each dot is one 15-minute observation.
    We sample up to 2 000 points per plant so the plot is not too heavy.
    """
    day1 = p1[p1['IS_DAYTIME']].sample(min(2000, len(p1[p1['IS_DAYTIME']])), random_state=42)
    day2 = p2[p2['IS_DAYTIME']].sample(min(2000, len(p2[p2['IS_DAYTIME']])), random_state=42)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for ax, df, title in zip(axes, [day1, day2], ['Plant 1', 'Plant 2']):
        ax.scatter(df['IRRADIATION'], df['AC_POWER'], alpha=0.35, s=15, color='steelblue')
        ax.set_xlabel('Irradiation (kW/m²)')
        ax.set_ylabel('AC Power (kW)')
        ax.set_title(f'Irradiation vs AC Power – {title}')

    fig.tight_layout()
    save(fig, 'eda_irradiation_vs_power.png')


# ── plot 3 : module temperature vs AC power (daytime scatter) ────────────────
def plot_modtemp_vs_power(p1: pd.DataFrame, p2: pd.DataFrame):
    """
    At higher module temperatures, panel efficiency drops slightly.
    This plot helps visualise that effect.
    """
    day1 = p1[p1['IS_DAYTIME']].sample(min(2000, len(p1[p1['IS_DAYTIME']])), random_state=1)
    day2 = p2[p2['IS_DAYTIME']].sample(min(2000, len(p2[p2['IS_DAYTIME']])), random_state=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for ax, df, title in zip(axes, [day1, day2], ['Plant 1', 'Plant 2']):
        sc = ax.scatter(df['MODULE_TEMPERATURE'], df['AC_POWER'],
                        c=df['IRRADIATION'], cmap='YlOrRd', alpha=0.45, s=15)
        fig.colorbar(sc, ax=ax, label='Irradiation (kW/m²)')
        ax.set_xlabel('Module Temperature (°C)')
        ax.set_ylabel('AC Power (kW)')
        ax.set_title(f'Module Temp vs AC Power – {title}')

    fig.tight_layout()
    save(fig, 'eda_modtemp_vs_power.png')


# ── plot 4 : daily total AC energy trend ─────────────────────────────────────
def plot_daily_yield_trend(p1: pd.DataFrame, p2: pd.DataFrame):
    """
    Resample 15-min AC_POWER to daily totals (kWh = kW × 0.25 h).
    Shows how generation varied across the ~34-day period.
    """
    def daily_kwh(df):
        return (
            df.set_index('DATE_TIME')['AC_POWER']
            .resample('D')
            .sum() * 0.25          # 15-min intervals → hours
        )

    d1 = daily_kwh(p1)
    d2 = daily_kwh(p2)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(d1.index, d1.values, width=0.4, align='edge',  label='Plant 1', alpha=0.8)
    ax.bar(d2.index, d2.values, width=-0.4, align='edge', label='Plant 2', alpha=0.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total AC Energy (kWh)')
    ax.set_title('Daily Solar Generation – Both Plants')
    ax.legend()
    fig.tight_layout()
    save(fig, 'eda_daily_yield_trend.png')


# ── print key stats ───────────────────────────────────────────────────────────
def print_stats(p1: pd.DataFrame, p2: pd.DataFrame):
    print("\n=== Key Statistics (daytime rows only) ===")
    for name, df in [('Plant 1', p1), ('Plant 2', p2)]:
        d = df[df['IS_DAYTIME']]
        print(f"\n  {name}:")
        print(f"    AC_POWER  – mean: {d['AC_POWER'].mean():.1f} kW, "
              f"max: {d['AC_POWER'].max():.1f} kW")
        print(f"    IRRADIATION – mean: {d['IRRADIATION'].mean():.4f}, "
              f"max: {d['IRRADIATION'].max():.4f} kW/m2")
        print(f"    MODULE_TEMP – mean: {d['MODULE_TEMPERATURE'].mean():.1f} C, "
              f"max: {d['MODULE_TEMPERATURE'].max():.1f} C")
        # total energy over the period
        total_kwh = d['AC_POWER'].sum() * 0.25
        print(f"    Total energy (period): {total_kwh:,.0f} kWh")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    p1, p2 = load_clean()

    print("Generating EDA plots ...")
    plot_avg_power_by_hour(p1, p2)
    plot_irradiation_vs_power(p1, p2)
    plot_modtemp_vs_power(p1, p2)
    plot_daily_yield_trend(p1, p2)
    print_stats(p1, p2)
    print("\nAll EDA plots saved to /outputs.")
