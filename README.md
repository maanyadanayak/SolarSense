# SolarSense ☀️
**Solar Energy Output Predictor with Smart Appliance Scheduler**

A college project built in Python that predicts solar power generation and 
intelligently schedules home appliances to maximise solar self-consumption.

---

## Project Structure

```
SolarSense/
├── data/                          # Raw Kaggle CSVs (4 files, 2 plants)
├── src/
│   ├── data_cleaning.py           # Step 1 – Load, clean, merge, save
│   ├── eda.py                     # Step 2 – EDA plots
│   ├── features.py                # Step 3 – Feature engineering
│   ├── modeling.py                # Step 4 – Train LR / RF / XGBoost
│   ├── forecast.py                # Step 5 – Day-ahead forecast (Open-Meteo)
│   └── scheduler.py               # Step 6 – Smart appliance scheduler
├── outputs/                       # Generated plots, CSVs, models
│   ├── models/                    # Saved joblib models
│   ├── plant1_clean.csv           # Cleaned + merged data
│   ├── plant2_clean.csv
│   ├── plant1_features.csv        # Engineered features
│   ├── plant2_features.csv
│   ├── forecast_plant_1.csv       # Today's forecast output
│   ├── forecast_plant_2.csv
│   ├── schedule_plant_1.csv       # Appliance schedule
│   └── schedule_plant_2.csv
├── app.py                         # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
python src/data_cleaning.py    # cleans data, saves to outputs/
python src/eda.py              # generates EDA plots
python src/features.py         # builds feature tables
python src/modeling.py         # trains models, saves best to outputs/models/
python src/forecast.py         # fetches Open-Meteo forecast, runs forecast model
python src/scheduler.py        # prints appliance schedule and savings
```

### 3. Launch the dashboard
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

---

## Data

**Source:** [Kaggle – Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)

Two solar plants in India (Telangana / Andhra Pradesh region), ~34 days of 
15-minute readings (May–Jun 2020).

| File | Rows | Description |
|---|---|---|
| Plant_1_Generation_Data.csv | 68,778 | Per-inverter DC/AC power, Plant 1 |
| Plant_1_Weather_Sensor_Data.csv | 3,182 | Irradiation, temp (Plant 1) |
| Plant_2_Generation_Data.csv | 67,698 | Per-inverter DC/AC power, Plant 2 |
| Plant_2_Weather_Sensor_Data.csv | 3,259 | Irradiation, temp (Plant 2) |

---

## Models

| Model | Plant 1 RMSE | Plant 1 R² | Plant 2 RMSE | Plant 2 R² |
|---|---|---|---|---|
| Linear Regression | 791.9 kW | 0.989 | 2780.3 kW | 0.724 |
| **Random Forest** ★ | **730.6 kW** | **0.991** | **2424.0 kW** | **0.790** |
| XGBoost | 800.6 kW | 0.989 | 2430.9 kW | 0.789 |

★ Best model — saved and used in the dashboard.

---

## Forecast Mode (Open-Meteo)

The day-ahead forecast uses the free [Open-Meteo API](https://open-meteo.com/):
- `shortwave_radiation` (W/m²) → converted to kW/m² and calibrated against plant sensor mean
- `temperature_2m` → ambient temperature
- `cloud_cover` → additional shading signal

> ⚠️ **Calibration note:** Open-Meteo radiation values are horizontal-surface NWP 
> estimates. Plant pyranometers may be tilted/tracked. A simple mean-ratio scaling 
> is applied. For production use, calibrate with ≥30 days of overlapping data.

---

## Smart Scheduler

Finds the top-3 peak solar hours from the day-ahead forecast and assigns 
appliances by priority:

| Priority | Appliances | Assigned to |
|---|---|---|
| 1 | Water Pump, Washing Machine | Peak #1 (most solar) |
| 2 | Air Conditioner, Iron | Peak #2 |
| 3 | Dishwasher, EV Charger | Peak #3 |

Default tariff: **Rs 7/kWh** (adjustable via dashboard slider).  
Typical daily savings: **Rs 119** → **Rs 3,570/month**.

---

## Dashboard Pages

| Page | What you see |
|---|---|
| **Forecast & Schedule** | Live forecast chart, top-3 peak hours, appliance schedule table, savings metrics |
| **Historical EDA** | EDA plots, per-day power browser |
| **Model Performance** | Metrics table, predicted vs actual plots |
