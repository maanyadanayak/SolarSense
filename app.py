"""
app.py  –  SolarSense Dashboard
================================
Modern solar-themed Streamlit UI.

Run:
    streamlit run app.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
import joblib

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from forecast  import (fetch_openmeteo, build_inference_df,
                        FORECAST_FEATURES, PLANTS)
from scheduler import (get_peak_hours, schedule_appliances, summarise)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
MODEL_DIR  = os.path.join(OUTPUT_DIR, 'models')

# ── global style injection ────────────────────────────────────────────────────
st.set_page_config(
    page_title='SolarSense',
    page_icon='☀️',
    layout='wide',
    initial_sidebar_state='expanded',
)

GLOBAL_CSS = """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(160deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%);
    color: #e8eaf6;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #0a1628 100%);
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] label {
    color: #a8c8e8 !important;
    font-weight: 500;
}
[data-testid="stSidebar"] .stSelectbox label {
    color: #a8c8e8 !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0f2a44 0%, #1a3a5c 100%);
    border: 1px solid #2563a8;
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 4px 20px rgba(37,99,168,0.25);
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f59e0b !important;
    font-size: 1.8rem !important;
    font-weight: 700;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: #93c5fd !important;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    color: #34d399 !important;
    font-size: 0.9rem;
}

/* ── Section headers ── */
h1 { color: #f59e0b !important; font-weight: 700; letter-spacing: -0.5px; }
h2 { color: #93c5fd !important; font-weight: 600; }
h3 { color: #7dd3fc !important; font-weight: 600; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    background: #0d1b2a !important;
    border-radius: 12px;
    overflow: hidden;
}

/* ── Divider ── */
hr { border-color: #1e3a5f !important; }

/* ── Info / warning boxes ── */
.stAlert {
    background: rgba(37,99,168,0.15) !important;
    border: 1px solid #2563a8 !important;
    border-radius: 10px;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
    color: #93c5fd !important;
    font-weight: 600;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #f59e0b !important;
    border-bottom: 3px solid #f59e0b !important;
}

/* ── Selectbox / slider ── */
.stSelectbox > div > div,
.stSlider > div > div {
    background: #0f2a44 !important;
    border-color: #2563a8 !important;
    color: #e8eaf6 !important;
    border-radius: 8px;
}
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_forecast_cached(plant_id: int) -> pd.DataFrame:
    from forecast import calibrate_irradiation
    info    = PLANTS[plant_id]
    weather = fetch_openmeteo(info['lat'], info['lon'])
    infer   = build_inference_df(weather, plant_id, target_date='tomorrow')
    model   = joblib.load(os.path.join(MODEL_DIR,
                          f'forecast_model_plant_{plant_id}.joblib'))
    X     = infer[FORECAST_FEATURES]
    preds = model.predict(X)
    preds = np.clip(preds, 0, None)
    preds[infer['irrad_forecast'].values == 0] = 0.0
    result = infer[['DATE_TIME','irrad_forecast','temp_forecast','cloud_cover']].copy()
    result['AC_POWER_FORECAST'] = preds
    result['kWh_per_hour']      = preds
    return result

@st.cache_data
def load_clean(plant_id: int) -> pd.DataFrame:
    return pd.read_csv(os.path.join(OUTPUT_DIR, f'plant{plant_id}_clean.csv'),
                       parse_dates=['DATE_TIME'])

# ── dark-themed matplotlib style ──────────────────────────────────────────────
def dark_style():
    plt.rcParams.update({
        'figure.facecolor':  '#0d1b2a',
        'axes.facecolor':    '#0f2a44',
        'axes.edgecolor':    '#1e3a5f',
        'axes.labelcolor':   '#93c5fd',
        'axes.titlecolor':   '#f59e0b',
        'xtick.color':       '#7dd3fc',
        'ytick.color':       '#7dd3fc',
        'grid.color':        '#1e3a5f',
        'grid.linestyle':    '--',
        'grid.alpha':        0.5,
        'text.color':        '#e8eaf6',
        'legend.facecolor':  '#0d1b2a',
        'legend.edgecolor':  '#2563a8',
        'legend.labelcolor': '#e8eaf6',
    })

dark_style()


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h2 style='color:#f59e0b;margin-bottom:0'>☀️ SolarSense</h2>"
        "<p style='color:#7dd3fc;font-size:0.78rem;margin-top:2px'>"
        "Solar Energy Predictor &amp; Smart Scheduler</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    page = st.radio(
        "**Navigate**",
        ['🔮 Forecast & Schedule', '📊 Historical EDA', '🤖 Model Performance'],
    )

    st.markdown("---")
    plant_id = st.selectbox(
        '**Select Plant**',
        [1, 2],
        format_func=lambda x: f'Plant {x}  •  {"Kurnool" if x==1 else "Anantapur"}, India',
    )

    tariff = st.slider(
        '**Grid Tariff (₹/kWh)**',
        min_value=3.0, max_value=15.0, value=7.0, step=0.5,
    )

    st.markdown("---")
    st.markdown(
        "<small style='color:#4a6fa5'>"
        "Data: Kaggle Solar Power Generation  \n"
        "Weather: Open-Meteo API  \n"
        "Model: Random Forest Regressor"
        "</small>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – Forecast & Schedule
# ═══════════════════════════════════════════════════════════════════════════════
if page == '🔮 Forecast & Schedule':

    # ── hero banner ───────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg,#0f2a44 0%,#1a3a5c 100%);
            border: 1px solid #2563a8;
            border-radius: 16px;
            padding: 28px 32px;
            margin-bottom: 24px;
        ">
          <h1 style="color:#f59e0b;margin:0;font-size:2rem">
            Day-Ahead Solar Forecast
          </h1>
          <p style="color:#93c5fd;margin:6px 0 0">
            Plant {plant_id} &nbsp;|&nbsp;
            Live weather from <strong style="color:#7dd3fc">Open-Meteo API</strong> &nbsp;|&nbsp;
            Random Forest model
          </p>
          <p style="color:#4a6fa5;font-size:0.75rem;margin:4px 0 0">
            ⚠️ NWP radiation values are approximations — calibrated against plant sensor mean.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── fetch forecast ────────────────────────────────────────────────────────
    with st.spinner('🌐 Fetching live weather forecast …'):
        fc    = get_forecast_cached(plant_id)
        peaks = get_peak_hours(fc, n_peaks=3)

    forecast_date = fc['DATE_TIME'].dt.date.iloc[0]

    # ── top KPI strip ─────────────────────────────────────────────────────────
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    day_fc   = fc[fc['AC_POWER_FORECAST'] > 0]
    peak_row = fc.loc[fc['AC_POWER_FORECAST'].idxmax()]
    sorted_peaks = sorted(peaks, key=lambda x: x['power_kw'], reverse=True)

    kpi1.metric('Forecast Date',     str(forecast_date))
    kpi2.metric('Peak Power',        f"{peak_row['AC_POWER_FORECAST']:,.0f} kW",
                f"@ {peak_row['DATE_TIME'].strftime('%H:%M')}")
    kpi3.metric('Total Generation',  f"{day_fc['kWh_per_hour'].sum():,.0f} kWh")
    kpi4.metric('Sunshine Hours',    f"{len(day_fc)} hrs")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── forecast chart ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(day_fc['DATE_TIME'], day_fc['AC_POWER_FORECAST'],
                    alpha=0.18, color='#f59e0b')
    ax.plot(day_fc['DATE_TIME'], day_fc['AC_POWER_FORECAST'],
            color='#f59e0b', lw=2.5, label='Forecast AC Power')

    peak_colors = ['#ef4444', '#f97316', '#22c55e']
    for i, p in enumerate(sorted_peaks):
        ax.axvline(p['datetime'], color=peak_colors[i], lw=1.8,
                   linestyle='--', alpha=0.9,
                   label=f"Peak #{i+1}  {p['datetime'].strftime('%H:%M')}  "
                         f"({p['power_kw']:,.0f} kW)")
        ax.annotate(
            f"#{i+1}",
            xy=(p['datetime'], p['power_kw'] * 0.92),
            ha='center', va='top',
            color=peak_colors[i], fontsize=10, fontweight='bold',
        )

    # cloud cover on twin axis
    ax2 = ax.twinx()
    ax2.bar(fc['DATE_TIME'], fc['cloud_cover'],
            width=0.035, alpha=0.18, color='#7dd3fc')
    ax2.set_ylabel('Cloud Cover (%)', color='#7dd3fc', fontsize=9)
    ax2.tick_params(axis='y', labelcolor='#7dd3fc')
    ax2.set_ylim(0, 250)
    ax2.set_facecolor('#0f2a44')

    ax.set_ylabel('AC Power (kW)', fontsize=10)
    ax.set_title(f'Plant {plant_id}  –  Day-Ahead Forecast  ({forecast_date})',
                 fontsize=12, pad=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.legend(loc='upper left', fontsize=8.5, framealpha=0.6)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── peak hours cards ──────────────────────────────────────────────────────
    st.markdown(
        "<h3 style='color:#f59e0b'>⚡ Top-3 Peak Solar Hours</h3>",
        unsafe_allow_html=True,
    )
    rank_cfg = [
        ('#ef4444', '#2d0a0a', '🔴', 'Highest Solar'),
        ('#f97316', '#2d1500', '🟠', '2nd Highest'),
        ('#22c55e', '#0a2d0e', '🟢', '3rd Highest'),
    ]
    col_a, col_b, col_c = st.columns(3)
    for col, p, (border, bg, emoji, label) in zip(
        [col_a, col_b, col_c], sorted_peaks, rank_cfg
    ):
        with col:
            st.markdown(
                f"""
                <div style="
                    background:{bg};
                    border:2px solid {border};
                    border-radius:14px;
                    padding:20px 16px;
                    text-align:center;
                ">
                  <div style="font-size:2rem">{emoji}</div>
                  <div style="color:#e8eaf6;font-size:0.78rem;
                              text-transform:uppercase;letter-spacing:.06em;
                              margin:4px 0">{label}</div>
                  <div style="color:#f59e0b;font-size:2.2rem;
                              font-weight:700;line-height:1.1">
                    {p['datetime'].strftime('%H:%M')}
                  </div>
                  <div style="color:#93c5fd;font-size:0.9rem;margin-top:4px">
                    {p['power_kw']:,.0f} kW
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # ── appliance schedule ────────────────────────────────────────────────────
    st.markdown(
        "<h3 style='color:#f59e0b'>🏠 Smart Appliance Schedule</h3>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Priority-1 appliances run at the highest solar hour, "
        "Priority-2 at the second peak, Priority-3 at the third. "
        "Savings = kWh × grid tariff."
    )

    schedule_df = schedule_appliances(peaks, tariff=tariff)
    summary     = summarise(schedule_df, tariff=tariff)

    # ── appliance cards ───────────────────────────────────────────────────────
    appliance_icons = {
        'Water Pump':      '💧',
        'Washing Machine': '🫧',
        'Air Conditioner': '❄️',
        'Iron':            '👔',
        'Dishwasher':      '🍽️',
        'EV Charger':      '🔋',
    }
    rank_border = {1: '#ef4444', 2: '#f97316', 3: '#22c55e'}
    rank_bg     = {1: '#1a0808', 2: '#1a0e00', 3: '#081a0a'}

    cols = st.columns(3)
    for idx, (_, row) in enumerate(schedule_df.iterrows()):
        r   = row['Peak Rank']
        col = cols[idx % 3]
        with col:
            st.markdown(
                f"""
                <div style="
                    background:{rank_bg[r]};
                    border:1px solid {rank_border[r]};
                    border-radius:12px;
                    padding:16px;
                    margin-bottom:12px;
                ">
                  <div style="font-size:1.6rem;margin-bottom:4px">
                    {appliance_icons.get(row['Appliance'],'⚡')}
                  </div>
                  <div style="color:#f59e0b;font-weight:700;font-size:1rem">
                    {row['Appliance']}
                  </div>
                  <div style="color:#93c5fd;font-size:0.8rem;margin:2px 0 8px">
                    Peak #{r} &nbsp;•&nbsp; Start at {row['Start Hour']}
                  </div>
                  <div style="display:flex;gap:8px;flex-wrap:wrap">
                    <span style="background:#0f2a44;border:1px solid #2563a8;
                                 border-radius:6px;padding:2px 8px;
                                 font-size:0.75rem;color:#7dd3fc">
                      {row['Duration (hrs)']} hrs
                    </span>
                    <span style="background:#0f2a44;border:1px solid #2563a8;
                                 border-radius:6px;padding:2px 8px;
                                 font-size:0.75rem;color:#7dd3fc">
                      {row['Power (kW)']} kW
                    </span>
                    <span style="background:#0f2a44;border:1px solid #2563a8;
                                 border-radius:6px;padding:2px 8px;
                                 font-size:0.75rem;color:#34d399">
                      ₹ {row['Rs Saved']:.1f} saved
                    </span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── savings banner ────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg,#1a3a1a 0%,#0a2a0a 100%);
            border: 2px solid #22c55e;
            border-radius: 16px;
            padding: 24px 32px;
            display: flex;
            gap: 40px;
            align-items: center;
            flex-wrap: wrap;
        ">
          <div>
            <div style="color:#86efac;font-size:0.75rem;
                        text-transform:uppercase;letter-spacing:.06em">
              kWh Shifted to Solar
            </div>
            <div style="color:#f59e0b;font-size:2.4rem;
                        font-weight:700;line-height:1.1">
              {summary['total_kwh_shifted']:.1f}
              <span style="font-size:1rem;color:#86efac">kWh</span>
            </div>
          </div>
          <div style="border-left:1px solid #22c55e;
                      padding-left:40px">
            <div style="color:#86efac;font-size:0.75rem;
                        text-transform:uppercase;letter-spacing:.06em">
              Saved Today
            </div>
            <div style="color:#f59e0b;font-size:2.4rem;
                        font-weight:700;line-height:1.1">
              ₹ {summary['total_rs_saved']:.0f}
            </div>
          </div>
          <div style="border-left:1px solid #22c55e;
                      padding-left:40px">
            <div style="color:#86efac;font-size:0.75rem;
                        text-transform:uppercase;letter-spacing:.06em">
              Monthly Estimate
            </div>
            <div style="color:#f59e0b;font-size:2.4rem;
                        font-weight:700;line-height:1.1">
              ₹ {summary['total_rs_saved']*30:,.0f}
            </div>
          </div>
          <div style="border-left:1px solid #22c55e;
                      padding-left:40px">
            <div style="color:#86efac;font-size:0.75rem;
                        text-transform:uppercase;letter-spacing:.06em">
              Tariff Used
            </div>
            <div style="color:#f59e0b;font-size:2.4rem;
                        font-weight:700;line-height:1.1">
              ₹ {tariff}
              <span style="font-size:1rem;color:#86efac">/kWh</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander('📋 View raw hourly forecast table'):
        show = day_fc[['DATE_TIME','irrad_forecast','temp_forecast',
                        'cloud_cover','AC_POWER_FORECAST','kWh_per_hour']].copy()
        show.columns = ['Time','Irrad (kW/m²)','Temp (°C)',
                        'Cloud (%)','Forecast Power (kW)','kWh']
        st.dataframe(show.set_index('Time'), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – Historical EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == '📊 Historical EDA':

    st.markdown(
        """
        <div style="
            background:linear-gradient(135deg,#0f2a44,#1a3a5c);
            border:1px solid #2563a8;border-radius:16px;
            padding:24px 32px;margin-bottom:24px
        ">
          <h1 style="color:#f59e0b;margin:0">Historical EDA</h1>
          <p style="color:#93c5fd;margin:6px 0 0">
            34 days of 15-minute sensor data · May – Jun 2020 · Two plants in India
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df  = load_clean(plant_id)
    day = df[df['IS_DAYTIME']]

    # ── KPIs ──────────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric('Mean Daytime Power', f"{day['AC_POWER'].mean():,.0f} kW")
    k2.metric('Peak AC Power',      f"{day['AC_POWER'].max():,.0f} kW")
    k3.metric('Max Irradiation',    f"{day['IRRADIATION'].max():.3f} kW/m²")
    k4.metric('Total Energy',       f"{day['AC_POWER'].sum()*0.25:,.0f} kWh")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── EDA plot selector ─────────────────────────────────────────────────────
    plots = {
        'Average Power by Hour':   'eda_avg_power_by_hour.png',
        'Irradiation vs AC Power': 'eda_irradiation_vs_power.png',
        'Module Temp vs AC Power': 'eda_modtemp_vs_power.png',
        'Daily Generation Trend':  'eda_daily_yield_trend.png',
    }
    chosen = st.selectbox('**Select EDA Plot**', list(plots.keys()))
    img_path = os.path.join(OUTPUT_DIR, plots[chosen])
    if os.path.exists(img_path):
        st.image(img_path, use_container_width=True)
    else:
        st.warning(f'Plot not found. Run `python src/eda.py` first.')

    st.divider()

    # ── single-day browser ────────────────────────────────────────────────────
    st.markdown(
        "<h3 style='color:#f59e0b'>🔍 Browse a Single Day</h3>",
        unsafe_allow_html=True,
    )
    dates = sorted(df['DATE_TIME'].dt.date.unique())
    sel   = st.date_input('Pick a date',
                          value=dates[0], min_value=dates[0], max_value=dates[-1])
    day_df = df[df['DATE_TIME'].dt.date == sel]

    if not day_df.empty:
        fig3, ax3 = plt.subplots(figsize=(11, 3.5))
        ax3.fill_between(day_df['DATE_TIME'], day_df['AC_POWER'],
                         alpha=0.2, color='#f59e0b')
        ax3.plot(day_df['DATE_TIME'], day_df['AC_POWER'],
                 color='#f59e0b', lw=2)
        ax3b = ax3.twinx()
        ax3b.fill_between(day_df['DATE_TIME'], day_df['IRRADIATION'],
                          alpha=0.15, color='#7dd3fc')
        ax3b.plot(day_df['DATE_TIME'], day_df['IRRADIATION'],
                  color='#7dd3fc', lw=1.4, linestyle='--', label='Irradiation')
        ax3b.set_ylabel('Irradiation (kW/m²)', color='#7dd3fc', fontsize=9)
        ax3b.tick_params(axis='y', labelcolor='#7dd3fc')
        ax3b.set_facecolor('#0f2a44')
        ax3.set_title(f'Plant {plant_id}  –  {sel}', fontsize=12)
        ax3.set_ylabel('AC Power (kW)')
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

        c1, c2, c3 = st.columns(3)
        c1.metric('Day Total Energy',
                  f"{day_df['AC_POWER'].sum()*0.25:,.0f} kWh")
        c2.metric('Peak Power',
                  f"{day_df['AC_POWER'].max():,.0f} kW")
        c3.metric('Max Irradiation',
                  f"{day_df['IRRADIATION'].max():.3f} kW/m²")
    else:
        st.info('No data for this date.')


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – Model Performance
# ═══════════════════════════════════════════════════════════════════════════════
elif page == '🤖 Model Performance':

    st.markdown(
        """
        <div style="
            background:linear-gradient(135deg,#0f2a44,#1a3a5c);
            border:1px solid #2563a8;border-radius:16px;
            padding:24px 32px;margin-bottom:24px
        ">
          <h1 style="color:#f59e0b;margin:0">Model Performance</h1>
          <p style="color:#93c5fd;margin:6px 0 0">
            Time-based split &nbsp;|&nbsp;
            Train: 15 May – 10 Jun 2020 &nbsp;|&nbsp;
            Test: 11 – 17 Jun 2020 &nbsp;|&nbsp;
            Evaluated on daytime rows only
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── metrics table ─────────────────────────────────────────────────────────
    st.markdown(
        "<h3 style='color:#f59e0b'>📐 Evaluation Metrics</h3>",
        unsafe_allow_html=True,
    )
    metrics_data = pd.DataFrame({
        'Model':     ['Linear Regression','Random Forest ★','XGBoost',
                      'Linear Regression','Random Forest ★','XGBoost'],
        'Plant':     ['Plant 1']*3 + ['Plant 2']*3,
        'MAE (kW)':  [561.5, 467.1, 569.4, 1974.9, 1321.2, 1346.9],
        'RMSE (kW)': [791.9, 730.6, 800.6, 2780.3, 2424.0, 2430.9],
        'R²':        [0.9892, 0.9908, 0.9889, 0.7236, 0.7899, 0.7887],
    })

    def highlight_best(row):
        if '★' in str(row['Model']):
            return ['background-color:#0d2b1a; color:#34d399']*len(row)
        return ['']*len(row)

    st.dataframe(
        metrics_data.style
            .apply(highlight_best, axis=1)
            .format({'MAE (kW)': '{:,.1f}', 'RMSE (kW)': '{:,.1f}', 'R²': '{:.4f}'}),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        '★ **Random Forest** wins on both plants (lowest RMSE). '
        'Plant 2 has lower R² due to abrupt inverter trips not predictable from weather alone.'
    )

    # ── model KPIs ────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    if plant_id == 1:
        m1.metric('Best RMSE', '730.6 kW')
        m2.metric('Best R²',   '0.9908')
        m3.metric('Best MAE',  '467.1 kW')
    else:
        m1.metric('Best RMSE', '2,424.0 kW')
        m2.metric('Best R²',   '0.7899')
        m3.metric('Best MAE',  '1,321.2 kW')

    st.divider()

    # ── pred vs actual ────────────────────────────────────────────────────────
    st.markdown(
        "<h3 style='color:#f59e0b'>📈 Predicted vs Actual (3 Test Days)</h3>",
        unsafe_allow_html=True,
    )
    tab1, tab2 = st.tabs(['Plant 1', 'Plant 2'])
    for tab, pid in zip([tab1, tab2], [1, 2]):
        with tab:
            img = os.path.join(OUTPUT_DIR, f'pred_vs_actual_plant_{pid}.png')
            if os.path.exists(img):
                st.image(img, use_container_width=True)
            else:
                st.warning('Run `python src/modeling.py` to generate this plot.')
