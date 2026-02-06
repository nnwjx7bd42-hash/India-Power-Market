# Historical Data Usage in Inference Pipeline

## Dataset Overview

### File Information

**File**: `data/processed/dataset_cleaned.parquet`  
**Format**: Parquet (columnar storage)  
**Index**: Datetime (IST timezone, +05:30)

### Date Range & Coverage

- **Start Date**: September 1, 2021, 00:00:00 IST
- **End Date**: June 24, 2025, 23:00:00 IST
- **Total Period**: ~3 years, 9 months
- **Total Hours**: ~33,000+ hourly records

### Time Granularity

- **Interval**: Hourly (1-hour resolution)
- **Timezone**: IST (Indian Standard Time, UTC+05:30)
- **Frequency**: One record per hour

### Data Sources

The dataset combines data from three primary sources:

1. **IEX Price Data** (`data/raw/price/`)
   - Source: Indian Energy Exchange (IEX) Day-Ahead Market
   - Columns: `P(T)` (hourly electricity price in ₹/MWh)
   - Period: Sep 2021 - Jun 2025

2. **NERLDC SCADA Data** (`data/raw/nerldc/`)
   - Source: Northern Regional Load Dispatch Centre monthly Excel files
   - Columns: `Demand`, `Thermal`, `Hydro`, `Gas`, `Nuclear`, `Wind`, `Solar`
   - Period: Sep 2021 - Jun 2025
   - Note: Hydro/Gas/Nuclear for 2024-2025 are **estimated during load** (`estimate_missing_generation()` in `load_nerdc_data.py`) using historical ratios from 2021-2023, so the unified and cleaned datasets have **complete values** (no NaN in these columns).

3. **Weather Data** (`data/processed/weather_national.parquet`)
   - Source: Open-Meteo API (aggregated from 5 cities: Delhi, Mumbai, Chennai, Kolkata, Guwahati)
   - Columns: Temperature, humidity, radiation, wind speed, cloud cover
   - Period: Aug 31, 2021 - Jun 24, 2025 (extended to cover Sep 1 start)
   - Aggregation: Demand-weighted national averages

### Data Transformations & Processing Pipeline

The dataset undergoes the following transformations:

1. **Unified Dataset Creation** (`unified_dataset.parquet`)
   - Merge IEX price data with NERLDC SCADA data (Hydro/Gas/Nuclear for 2024-2025 are already estimated in the NERLDC load step using historical ratios, so the unified dataset has complete values)
   - Align timestamps and handle missing periods

2. **Weather Data Merge** (`dataset_with_features.parquet`)
   - Merge national weather data using `merge_asof` (nearest match)
   - Handle missing weather values (backward fill for first 5 hours)
   - Align weather timestamps with hourly price/load data

3. **Feature Engineering** (`dataset_with_features.parquet`)
   - Calendar features: Hour, DayOfWeek, Month, Season, cyclic encodings
   - Weather features: CDH, HDH, Heat Index, deviations, proxies
   - Supply-side features: RE_Generation, RE_Penetration, Net_Load, Thermal_Share
   - Lag features: P_T-168, L_T-168, P_T-3, P_T-4, rolling statistics
   - Interaction features: Hour × CDH, Month × Hour, RE_Pen × Hour, Season × Temp (Hour × NetLoad and Temp × Hour are created but removed during cleaning)

4. **Data Cleaning** (`dataset_cleaned.parquet`)
   - Remove redundant/low-correlation features (see list below)
   - Handle missing lag values (fill with mean value for initial periods)
   - Check for any remaining missing SCADA values (typically none; Hydro/Gas/Nuclear are already estimated in the NERLDC load step, so the pipeline has complete values)
   - Final shape: 33,409 rows × 65 columns

   **Features removed during cleaning** (created in feature engineering but dropped as redundant): Heat_Index, Total_Generation, Hydro_Availability, Hour_x_NetLoad, Temp_x_Hour, IsFriday, WeekOfYear, DayOfYear, Quarter, shortwave_radiation_national (if present).

### What Is Lagged vs Not Lagged

**Lagged (explicit lag features):**
- **Price:** P(T-1), P(T-2), P(T-24), P(T-48), P_T-168, P_T-3, P_T-4, plus Price_MA_24h, Price_Std_24h.
- **Load:** L(T-1), L(T-2), L(T-24), L(T-48), L_T-168, plus Load_MA_24h, Load_Std_24h.

**Not lagged (used as current-hour / contemporaneous):**
- **Weather:** temperature, humidity, radiation, wind, cloud cover (and derived CDH, HDH, etc.) — all for the **same hour** as the target.
- **Supply:** Demand, Thermal, Hydro, Gas, Nuclear, Wind, Solar (and RE_Generation, RE_Penetration, Net_Load, etc.) — all for the **same hour**.
- **Calendar:** Hour, DayOfWeek, Month, Season, etc. — for the **same hour**.

**Exception:** `Solar_Ramp` = Solar(T) − Solar(T−1) (one-hour change, not a lag level).

**Why this design:** Price and load are strongly autocorrelated (e.g. P(T) with P(T-1) ~ 0.89), so their lags are the main autoregressive drivers. Demand, weather, and supply are used as **current-hour** drivers of price; adding many lags of them would increase multicollinearity and feature count. The current setup (price + load lagged; others contemporaneous) is standard for hourly price forecasting and is **good to use** as-is. You could experiment with 1–2 lags of demand or wind/solar if you want, but it is not required.

---

### Complete Column List (~65 Features)

#### Target Variable
- `P(T)` - Hourly electricity price (₹/MWh)

#### Price & Load Lags (Autoregressive Features)
- `P(T-1)`, `P(T-2)`, `P(T-24)`, `P(T-48)` - Price lags (1h, 2h, 24h, 48h ago)
- `P_T-168` - Price same hour last week (7 days ago)
- `P_T-3`, `P_T-4` - Price 3-4 hours ago
- `L(T-1)`, `L(T-2)`, `L(T-24)`, `L(T-48)` - Load lags
- `L_T-168` - Load same hour last week
- `Price_MA_24h`, `Price_Std_24h` - 24h rolling statistics
- `Load_MA_24h`, `Load_Std_24h` - 24h rolling statistics

#### Weather Features (Raw)
- `temperature_2m_national` - Temperature (°C)
- `relative_humidity_2m_national` - Humidity (%)
- `direct_radiation_national` - Direct solar radiation (W/m²)
- `diffuse_radiation_national` - Diffuse radiation (W/m²)
- `wind_speed_10m_national` - Wind speed (m/s)
- `cloud_cover_national` - Cloud cover (%)

#### Weather Features (Derived)
- `CDH` - Cooling Degree Hours (AC demand threshold)
- `HDH` - Heating Degree Hours (heating demand threshold)
- `Temp_Deviation` - Temperature deviation from 30-day rolling mean
- `Humidity_Deviation` - Humidity deviation from 30-day rolling mean
- `Solar_Irradiance_Effective` - Effective irradiance accounting for clouds
- `Wind_Power_Proxy` - Wind power proxy (wind_speed³)

#### Calendar Features
- `Hour` - Hour of day (0-23)
- `DayOfWeek` - Day of week (0=Monday, 6=Sunday)
- `DayOfMonth` - Day of month (1-31)
- `Month` - Month (1-12)
- `Day` - Weekend indicator (0=weekday, 1=weekend)
- `Season` - Season (0=Winter, 1=Summer/Spring, 2=Monsoon, 3=Post-Monsoon)
- `IsWeekend` - Boolean (1 if Saturday/Sunday)
- `IsMonday` - Boolean (1 if Monday)
- `Season_Month` - Season-month encoding (0=Winter, 1=Summer/Spring, 2=Monsoon, 3=Post-Monsoon)
- `TimeOfDay` - Time period (Morning/Afternoon/Evening/Night)
- `IsHoliday` - Holiday indicator (0=not holiday, 1=holiday)

#### Calendar Features (Cyclic Encodings)
- `Hour_Sin`, `Hour_Cos` - Hour cyclic encoding
- `Month_Sin`, `Month_Cos` - Month cyclic encoding
- `DayOfWeek_Sin`, `DayOfWeek_Cos` - Day of week cyclic encoding
- `DayOfYear_Sin`, `DayOfYear_Cos` - Day of year cyclic encoding (DayOfYear column itself is removed during cleaning)

#### Supply-Side Features (Raw)
- `Demand` - Total electricity demand (MW)
- `Thermal` - Thermal generation (MW)
- `Hydro` - Hydroelectric generation (MW)
- `Gas` - Gas generation (MW)
- `Nuclear` - Nuclear generation (MW)
- `Wind` - Wind generation (MW)
- `Solar` - Solar generation (MW)

#### Supply-Side Features (Derived)
- `RE_Generation` - Renewable generation (Wind + Solar)
- `RE_Penetration` - Renewable penetration (% of demand)
- `Net_Load` - Net load (Demand - RE_Generation)
- `Thermal_Share` - Thermal share of total generation
- `Solar_Ramp` - Solar generation change from previous hour
- `RE_Volatility` - Renewable generation volatility (24h rolling std)

#### Interaction Features
- `Hour_x_CDH` - Hour × Cooling Degree Hours
- `Month_x_Hour` - Month × Hour
- `RE_Pen_x_Hour` - Renewable Penetration × Hour
- `Season_x_Temp` - Season × Temperature

---

## Overview

The inference pipeline loads historical data from `data/processed/dataset_cleaned.parquet` to support forecast generation. This document details what historical data is used and how.

## Data Source

**File**: `data/processed/dataset_cleaned.parquet`  
**Loaded**: Last 168 hours (7 days) minimum  
**Purpose**: Extract patterns, learn relationships, and provide lag features

---

## Key Design Summary

The model uses **64 features + 1 target** P(T). Below is how each category behaves in time and at inference.

| Category | Count | Temporal Relationship | At Inference |
|----------|-------|------------------------|--------------|
| **Price Lags** | 7 | Past (T-1 to T-168) | Historical then Predictions |
| **Load Lags** | 5 | Past (T-1 to T-168) | Historical then Forecast |
| **Rolling Stats** | 4 | Past 24h window | Recalculated each step |
| **Weather (Raw)** | 6 | Same hour (T) | From weather forecast |
| **Weather (Derived)** | 6 | Same hour (T) | Calculated from forecast |
| **Calendar** | 11 | Same hour (T) | Deterministic |
| **Cyclic Encodings** | 8 | Same hour (T) | Deterministic |
| **Supply (Raw)** | 7 | Same hour (T) | Estimated from weather |
| **Supply (Derived)** | 6 | Same hour (T) | Calculated from estimates |
| **Interaction** | 4 | Same hour (T) | Calculated |

---

## Critical Point: Contemporaneous Supply Features

The model uses **same-hour supply data** (Demand, Solar, Wind, Thermal, etc.) as features. At **training** we have actual NERLDC values for hour T; at **inference** we do not know actual Solar(T), Demand(T), etc. yet, so the pipeline **estimates** them from weather forecast and daily peak loads. Inference accuracy depends heavily on these estimates.

| Feature | Training (Known) | Inference (Estimated) |
|---------|------------------|------------------------|
| `Solar` | NERLDC actual | direct_radiation × learned_ratio |
| `Wind` | NERLDC actual | wind_speed³ × learned_ratio |
| `Demand` | NERLDC actual | daily_peak × hourly_profile |
| `Thermal` | NERLDC actual | Demand - RE - Hydro - Gas - Nuclear (residual) |
| `Hydro`, `Gas`, `Nuclear` | NERLDC actual | Historical hourly averages |

---

## Inference Data Flow

The following diagram shows how inputs and historical data combine to produce P(T) for each hour.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                      INFERENCE FOR HOUR T                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  INPUTS YOU MUST PROVIDE                                                ││
│  ├─────────────────────────────────────────────────────────────────────────┤│
│  │                                                                         ││
│  │  1. Weather Forecast for hour T:                                        ││
│  │     temperature_2m, humidity, direct_radiation, wind_speed, cloud_cover ││
│  │                                                                         ││
│  │  2. Daily Peak Load Forecast:                                           ││
│  │     [Day1_peak, Day2_peak, ..., Day7_peak] in MW                        ││
│  │                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                      │                                       │
│                                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  DERIVED/ESTIMATED (Pipeline Calculates)                                ││
│  ├─────────────────────────────────────────────────────────────────────────┤│
│  │                                                                         ││
│  │  From Weather Forecast:                                                 ││
│  │  ├─ CDH = max(0, temp - 24)                                             ││
│  │  ├─ HDH = max(0, 18 - temp)                                             ││
│  │  ├─ Solar_Irradiance_Effective = direct_rad × (1 - cloud/100)           ││
│  │  ├─ Wind_Power_Proxy = wind_speed³                                      ││
│  │  ├─ Solar = direct_radiation × ratio[hour, month]  ← FROM HISTORY       ││
│  │  └─ Wind = wind_speed³ × ratio[hour]               ← FROM HISTORY        ││
│  │                                                                         ││
│  │  From Daily Peaks:                                                      ││
│  │  └─ Demand = daily_peak × profile[day_of_week, hour] ← FROM HISTORY     ││
│  │                                                                         ││
│  │  From Demand + RE:                                                      ││
│  │  ├─ RE_Generation = Solar + Wind                                        ││
│  │  ├─ RE_Penetration = RE_Generation / Demand                             ││
│  │  ├─ Net_Load = Demand - RE_Generation                                   ││
│  │  ├─ Hydro, Gas, Nuclear = hourly_avg[hour]         ← FROM HISTORY       ││
│  │  ├─ Thermal = Demand - RE - Hydro - Gas - Nuclear (residual)            ││
│  │  └─ Thermal_Share = Thermal / (Thermal + RE + Hydro + Gas + Nuclear)    ││
│  │                                                                         ││
│  │  Calendar (Deterministic):                                              ││
│  │  └─ Hour, DayOfWeek, Month, IsWeekend, cyclic encodings, etc.           ││
│  │                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                      │                                       │
│                                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  FROM HISTORICAL (Last 168h of Training Data)                            ││
│  ├─────────────────────────────────────────────────────────────────────────┤│
│  │                                                                         ││
│  │  Price Lags:                                                            ││
│  │  ├─ P(T-24), P(T-48), P(T-168) → Actual from history (if available)      ││
│  │  └─ P(T-1), P(T-2), P(T-3), P(T-4) → Actual (hour 1-4) OR Predicted     ││
│  │                                                                         ││
│  │  Load Lags:                                                             ││
│  │  ├─ L(T-24), L(T-48), L(T-168) → Actual from history                    ││
│  │  └─ L(T-1), L(T-2) → From shaped load forecast                          ││
│  │                                                                         ││
│  │  Rolling Stats:                                                         ││
│  │  └─ Price_MA_24h, Price_Std_24h, Load_MA_24h, Load_Std_24h             ││
│  │     → Initialized from history, updated with predictions               ││
│  │                                                                         ││
│  │  Pattern Learning (from ALL training data):                             ││
│  │  ├─ Solar ratios by [hour, month]                                       ││
│  │  ├─ Wind ratios by [hour]                                               ││
│  │  ├─ Load profiles by [day_of_week, hour]                                ││
│  │  └─ Conventional gen averages by [hour]                                 ││
│  │                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                      │                                       │
│                                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  ASSEMBLE 64 FEATURES → MODEL → PREDICT P(T)                            ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Minimum Inference Inputs

To run the inference pipeline you only need the following; everything else is derived.

| Input | Format | Example |
|-------|--------|---------|
| **Weather Forecast** | DataFrame (168 rows × 6 cols) | Open-Meteo 7-day hourly |
| **Daily Peak Loads** | List of 7 floats (MW) | [185000, 187000, 190000, 188000, 186000, 175000, 172000] |
| **Historical Context** | Auto-loaded from parquet | Last 168 hours before forecast start |

---

## Training vs Inference Split (Validation)

Training uses the full dataset (Sep 1, 2021 through Jun 24, 2025). For validation, a typical setup is to treat a recent period (e.g. Jun 18–24, 2025) as inference-only: predict P(T) for those hours, then compare to actuals. This validates the pipeline before deploying to truly unknown future dates.

---

## Historical Data Usage

### 1. Load Profile Pattern Learning

**Purpose**: Convert daily peak loads to hourly profiles

**Data Used**:
- Column: `L(T-1)` (hourly load)
- Minimum: 168+ hours (to get patterns for all days of week)
- Process:
  1. Groups historical load by day of week (Monday-Sunday)
  2. Calculates average hourly pattern for each day
  3. Normalizes patterns to peak = 1.0
  4. Stores 7 patterns (one per day of week)

**Output**: Hourly load factors by day of week used to scale daily peaks

**Example**:
- Monday pattern: [0.65, 0.60, ..., 1.0, ..., 0.70] (24 hourly factors)
- Applied: `hourly_load = pattern[hour] × daily_peak`

---

### 2. Renewable Generation Estimation

**Purpose**: Estimate Solar and Wind generation from weather forecasts

#### Solar Generation

**Historical Data Used**:
- Columns: `Solar`, `direct_radiation_national`, `Hour`, `Month`
- Process:
  1. Calculates ratio: `Solar / direct_radiation` for each hour-month combination
  2. Uses median ratio (more robust than mean)
  3. Caps ratios at reasonable maximum (0.5 MW per W/m²)
  4. Applies learned ratios to forecast weather

**Fallback**: If no historical data, uses simple model: `Solar = direct_radiation × 0.05` (only during daylight hours 6-18)

#### Wind Generation

**Historical Data Used**:
- Columns: `Wind`, `wind_speed_10m_national`, `Hour`
- Process:
  1. Calculates ratio: `Wind / (wind_speed^3)` for each hour
  2. Uses median ratio
  3. Caps ratios at reasonable maximum (100 MW per (m/s)³)
  4. Applies learned ratios to forecast weather

**Fallback**: If no historical data, uses simple model: `Wind = (wind_speed^3) × 0.5`

---

### 3. Conventional Generation Estimation

**Purpose**: Estimate Hydro, Gas, Nuclear (less variable sources)

**Historical Data Used**:
- Columns: `Hydro`, `Gas`, `Nuclear`, `Hour`
- Process:
  1. Calculates average generation by hour of day
  2. Uses these averages for forecast period
  3. Falls back to demand-based estimates if no historical data

**Default Estimates** (if no historical data):
- Hydro: ~10% of demand
- Gas: ~5% of demand
- Nuclear: ~3% of demand

---

### 4. Lag Features (Autoregressive)

**Purpose**: Provide historical context for price prediction

**Historical Data Used**:
- Columns: `P(T)` (price), `L(T-1)` (load)
- Minimum: 168 hours (for weekly lags)

**Lag Features Extracted**:

| Feature | Description | Historical Source |
|---------|-------------|------------------|
| `P(T-1)` | Price 1 hour ago | Last value from historical data |
| `P(T-2)` | Price 2 hours ago | Historical data |
| `P(T-24)` | Price same hour yesterday | Historical data (or prediction if available) |
| `P(T-48)` | Price same hour 2 days ago | Historical data (or prediction if available) |
| `P_T-168` | Price same hour last week | Historical data (or prediction if available) |
| `P_T-3`, `P_T-4` | Price 3-4 hours ago | Historical data (or predictions) |
| `L(T-2)` | Load 2 hours ago | Historical data |
| `L(T-24)` | Load same hour yesterday | Historical data |
| `L(T-48)` | Load same hour 2 days ago | Historical data |
| `L_T-168` | Load same hour last week | Historical data |

**Note**: For multi-step ahead predictions, previous predictions replace historical values as they become available.

---

### 5. Rolling Statistics

**Purpose**: Calculate moving averages and standard deviations

**Historical Data Used**:
- Columns: `P(T)` (price), `L(T-1)` (load)
- Window: Last 24 hours

**Statistics Calculated**:
- `Price_MA_24h`: 24-hour moving average of price
- `Price_Std_24h`: 24-hour standard deviation of price
- `Load_MA_24h`: 24-hour moving average of load
- `Load_Std_24h`: 24-hour standard deviation of load

**Process**:
1. Starts with last 24 hours of historical data
2. Appends predictions as they're made
3. Recalculates rolling stats each step

---

### 6. Feature Engineering Context

**Historical Data Used**:
- All columns from `dataset_cleaned.parquet` (65 features)
- Purpose: Ensure feature consistency between training and inference

**Process**:
1. Loads last 168 hours
2. Extracts exact feature list used during training
3. Ensures forecast features match training features exactly

---

## Data Loading Function

**Location**: `src/inference/predictor.py` → `load_historical_context()`

**Code**:
```python
def load_historical_context(historical_data_path, hours_needed=168):
    df = pd.read_parquet(historical_data_path)
    df = df.sort_index()
    df = df.iloc[-hours_needed:]  # Last 168 hours
    return df
```

**Parameters**:
- `historical_data_path`: Path to `dataset_cleaned.parquet` (default: `data/processed/dataset_cleaned.parquet`)
- `hours_needed`: Minimum hours needed (default: 168 for weekly lags)

---

## Summary Table

| Use Case | Historical Columns | Hours Needed | Purpose |
|----------|-------------------|--------------|---------|
| Load Patterns | `L(T-1)`, `Hour`, `DayOfWeek` | 168+ | Convert daily peaks to hourly |
| Solar Estimation | `Solar`, `direct_radiation_national`, `Hour`, `Month` | All available | Learn solar/radiation ratios |
| Wind Estimation | `Wind`, `wind_speed_10m_national`, `Hour` | All available | Learn wind/speed ratios |
| Conventional Gen | `Hydro`, `Gas`, `Nuclear`, `Hour` | All available | Average by hour |
| Price Lags | `P(T)` | 168+ | Autoregressive features |
| Load Lags | `L(T-1)` | 168+ | Autoregressive features |
| Rolling Stats | `P(T)`, `L(T-1)` | 24+ | Moving averages |
| Feature List | All columns | N/A | Ensure consistency |

---

## Key Points

1. **Minimum Requirement**: 168 hours (7 days) of historical data
2. **Actual Usage**: Last 168 hours from `dataset_cleaned.parquet`
3. **Primary Uses**:
   - Pattern learning (load profiles, renewable ratios)
   - Lag feature extraction
   - Rolling statistics initialization
4. **Data Source**: Same cleaned dataset used for training (ensures consistency)
5. **Separation**: Historical data is read-only, not modified during inference

---

## Data Flow

```
Historical Dataset (dataset_cleaned.parquet)
    ↓
Load last 168 hours
    ↓
┌─────────────────────────────────────┐
│ Extract/Use:                        │
│ 1. Load patterns (by day of week)   │
│ 2. Renewable ratios (solar/wind)    │
│ 3. Conventional averages (hydro/gas)│
│ 4. Lag features (P, L)              │
│ 5. Rolling stats (24h window)       │
│ 6. Feature list (for consistency)   │
└─────────────────────────────────────┘
    ↓
Used in forecast generation
```

---

## Notes

- Historical data is **read-only** - never modified during inference
- More historical data improves pattern learning (especially for renewable ratios)
- Lag features transition from historical → predictions as forecast progresses
- All historical data comes from the same cleaned dataset used for training
