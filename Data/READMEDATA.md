# Data Directory — IEX Price Forecasting VPP

## Project Goal

Forecast the **Market Clearing Price (MCP)** on the Indian Energy Exchange (IEX) across Day-Ahead (DAM), Green Day-Ahead (G-DAM), and Real-Time (RTM) electricity markets. The forecast target is `mcp_rs_mwh` at 15-minute granularity.

This document describes every dataset: source, acquisition method, cleaning process, schema, statistical profile, known limitations, and how each dataset relates to price formation. Written to enable feature engineering decisions.

---

## Final Cleaned Data Summary

```
Data/Cleaned/
├── price/       iex_prices_combined_filled.parquet      (6.6 MB,   394,848 rows)
├── bid_stack/   iex_aggregate_combined_filled.parquet    (36 MB,  4,689,282 rows)
├── grid/        nerldc_national_hourly.parquet           (2.7 MB,    33,432 rows)
└── weather/     weather_2022-04-01_to_2025-12-31.parquet (1.7 MB,   141,745 rows)
```

**Core overlapping period (all 4 datasets available): 2022-04-01 to 2025-06-24 (1,181 days)**

| Dataset | Resolution | Date Range | Rows | Nulls |
|---------|-----------|------------|------|-------|
| IEX Prices | 15-min (96 blocks/day) | 2022-04-01 to 2025-12-31 | 394,848 | 0 |
| IEX Bid Stack | 15-min x 12 price bands | 2022-04-01 to 2025-12-31 | 4,689,282 | 0 |
| NERLDC Grid | Hourly | 2021-09-01 to 2025-06-24 | 33,432 | 0 |
| Weather | Hourly x 5 cities | 2022-04-01 to 2025-06-25 | 141,745 | 0 |

All datasets are **zero-null, gap-filled, timezone-aligned (IST)**.

---

## 1. IEX Clearing Prices (The Forecast Target)

### What This Is

The Indian Energy Exchange runs three electricity markets. Each market conducts a **double-sided closed auction** for every 15-minute delivery block. Buyers submit purchase bids (MW at price), sellers submit sell offers. The intersection of the aggregate demand and supply curves determines the **Market Clearing Price (MCP)** and **Market Clearing Volume (MCV)**.

### Markets

| Market | Code | Auction | Delivery | Description |
|--------|------|---------|----------|-------------|
| Day-Ahead Market | `dam` | Daily at 10:00 AM, results by 13:00 | Next day, all 96 blocks | Main liquidity pool; sets the reference price |
| Green Day-Ahead Market | `gdam` | Daily, after DAM | Next day, all 96 blocks | Renewable energy certificates; tends to price higher than DAM |
| Real-Time Market | `rtm` | Every 15 min, gate closure 45 min before delivery | Same day | Intraday balancing; higher volatility |

### Source & Acquisition

- **Primary source**: IEX Official REST API (`/IEXPublish/AppServices.svc/IEXGetTradeData`) — returns clean JSON, 31-day max per request, 30-day chunking with 2s rate limit
- **Fallback source**: IEX website Next.js RSC endpoints (cookie-authenticated, fragile)
- **Scraping scripts**: `data_sourcing/scripts/fetch_iex_data.py` (API approach), with `--batch-all --skip-existing` for full historical pull
- **Raw location**: `Data/Raw/price/iex_{dam,gdam,rtm}_price/`

### Cleaning Pipeline

1. `data_sourcing/scripts/combine_iex_prices_to_cleaned.py` — Reads CSVs per market, standardizes to snake_case, adds `market` column, concatenates
2. `data_sourcing/scripts/fill_iex_prices_gaps.py` — Constructs full (date x market x time_block) grid, left-joins observed data, forward-fills then backward-fills MCP and volume per (market, time_block)

### Schema

**File**: `Cleaned/price/iex_prices_combined_filled.parquet` — 394,848 rows

| Column | Type | Null% | Description |
|--------|------|-------|-------------|
| `date` | datetime64 | 0% | Trading date |
| `market` | str | 0% | `dam`, `gdam`, or `rtm` |
| `time_block` | str | 0% | 15-min slot, e.g., `00:00-00:15`, `18:00-18:15` |
| `date_str` | str | 0% | Date as DD-MM-YYYY |
| `purchase_bid_mwh` | float64 | 0% | Total buy bid volume (MWh) |
| `sell_bid_mwh` | float64 | 0% | Total sell bid volume (MWh) |
| `mcv_mwh` | float64 | 0% | Market Clearing Volume — actual traded energy (MWh) |
| `final_scheduled_volume_mwh` | float64 | 0% | Final scheduled after transmission constraints |
| `mcp_rs_mwh` | float64 | 0% | **Market Clearing Price (Rs/MWh)** — THE FORECAST TARGET |
| `weighted_mcp_rs_mwh` | float64 | 0% | Volume-weighted MCP |
| `token` | float64 | 0% | Time block index (1–96 within the day) |
| `hour` | float64 | 0% | Hour of day (0–23) |
| `session_id` | float64 | — | RTM session ID (NaN for DAM/G-DAM) |

### Key Statistics

**131,616 rows per market** (1,371 days x 96 blocks)

#### MCP Distribution by Market (Rs/MWh)

| Market | Min | P5 | P25 | Median | P75 | P95 | Max | Mean | Std |
|--------|-----|-----|-----|--------|-----|-----|-----|------|-----|
| DAM | 49.5 | 1,870.9 | 2,999.3 | 3,809.4 | 6,000.0 | 10,000.0 | 20,000.0 | 4,839 | 2,757 |
| G-DAM | 198.8 | 2,075.3 | 3,585.2 | 4,480.0 | 6,500.3 | 11,000.8 | 20,000.0 | 5,284 | 2,914 |
| RTM | 0.0 | 1,755.5 | 2,979.2 | 3,750.5 | 5,499.3 | 10,000.0 | 16,744.2 | 4,967 | 3,088 |

- DAM ceiling is Rs 20,000/MWh (exchange-imposed cap)
- RTM has wider tails — it can go to 0 (oversupply) and has higher variance
- G-DAM consistently prices above DAM (green premium)

#### DAM MCP by Hour of Day (Rs/MWh, mean)

| Hour | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|------|---|---|---|---|---|---|---|---|---|---|----|----|
| Mean | 5,629 | 4,945 | 4,368 | 4,032 | 4,071 | 4,484 | 5,209 | 5,332 | 4,565 | 4,157 | 3,676 | 3,343 |

| Hour | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|------|----|----|----|----|----|----|----|----|----|----|----|----|
| Mean | 3,060 | 2,831 | 3,260 | 3,789 | 4,447 | 5,475 | 7,020 | **7,978** | 7,091 | 6,864 | 6,679 | 6,335 |

**Clear solar duck curve**: prices collapse 11:00–14:00 (peak solar generation), spike 18:00–20:00 (solar ramp-down + evening demand peak). The 19:00 hour is the most expensive at Rs 7,978/MWh average.

#### DAM MCP by Month (Rs/MWh, mean)

| Month | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| Mean | 5,483 | 5,324 | 4,601 | **6,449** | 5,225 | 5,294 | 4,783 | 5,087 | 4,908 | 4,218 | **3,737** | 4,400 |

April is most expensive (pre-monsoon heat, peak AC demand, low hydro). November cheapest (mild weather, post-monsoon hydro still flowing, low demand).

### Price Formation Context (Why These Features Matter)

MCP is determined by supply-demand intersection. Things that increase price:
- **Higher demand** (heat waves → AC load, industrial activity)
- **Lower renewable supply** (cloudy days → less solar, calm wind)
- **Thermal capacity constraints** (coal shortages, plant outages)
- **Low hydro** (pre-monsoon, drought years)

Things that decrease price:
- **Excess solar** (midday collapse, the "duck curve")
- **High wind** (especially in Tamil Nadu during monsoon)
- **Monsoon hydro flush** (Jul-Sep, cheap baseload)
- **Low demand** (mild weather, holidays, weekends)

---

## 2. IEX Bid Stack (Aggregate Demand-Supply Curves)

### What This Is

For every 15-minute time block, the IEX publishes the **aggregate bid stack**: how much buy demand (MW) and sell supply (MW) was offered at each price band. This is the aggregate order book — the crossing point of demand and supply curves is where MCP is set.

This data captures market **microstructure** — not just what price cleared, but the shape of demand and supply around that price. Key for understanding price sensitivity and predicting spikes.

### Source & Acquisition

- **Source**: IEX website Next.js RSC endpoint (no official API alternative for this data)
- **Scraping**: `data_sourcing/scripts/fetch_iex_aggregate.py` — 96 blocks/day x 3 markets, 32 parallel requests, per-day CSV saves
- **Raw location**: `Data/Raw/agg-demand-supply/iex_{dam,gdam,rtm}_aggregate/`
- **Coverage**: 91.9% DAM, 97.0% G-DAM, 97.2% RTM (remaining dates return HTTP 500 — unrecoverable)

### Cleaning Pipeline

1. `data_sourcing/scripts/fill_iex_aggregate_gaps.py` — Reads 3 merged CSVs (or combines from per-day files), builds full (date x market x time_block x price_band) grid, ffill/bfill per group

### Schema

**File**: `Cleaned/bid_stack/iex_aggregate_combined_filled.parquet` — 4,689,282 rows

| Column | Type | Null% | Description |
|--------|------|-------|-------------|
| `date` | datetime64 | 0% | Trading date |
| `date_str` | str | 0% | Date as DD-MM-YYYY |
| `market` | str | 0% | `dam`, `gdam`, or `rtm` |
| `time_block` | str | 0% | 15-min slot (95 unique blocks) |
| `price_band_rs_mwh` | str | 0% | Price band bucket (12 bands) |
| `buy_demand_mw` | float64 | 0% | Aggregate buy demand at this price band (MW) |
| `sell_supply_mw` | float64 | 0% | Aggregate sell supply at this price band (MW) |

### Price Bands (12 buckets, Rs/MWh)

`0-1000`, `1001-2000`, `2001-3000`, `3001-4000`, `4001-5000`, `5001-6000`, `6001-7000`, `7001-8000`, `8001-9000`, `9001-10000`, `10001-11000`, `11001-12000`

### Key Statistics

| Metric | buy_demand_mw | sell_supply_mw |
|--------|--------------|----------------|
| Min | 0.0 | 0.0 |
| Mean | 4,085 MW | 5,293 MW |
| Max | 62,187 MW | 51,152 MW |

### Grid Structure

1,371 days x 3 markets x ~12 price bands x ~95 time blocks = 3,420 unique (market, time_block, price_band) combos x 1,371 days = 4,689,282 rows. Observed: 3,912,796 (83.4%). Gap-filled: 776,486 (16.6%) via ffill/bfill.

### Feature Engineering Value

From each time block's bid stack you can derive:
- **Demand elasticity at MCP**: slope of demand curve near the clearing price — how much would demand drop per Rs increase?
- **Supply margin**: excess sell MW above MCP — how tight is the market?
- **Bid concentration** (Herfindahl index): are bids clustered in one band or spread?
- **Demand-supply crossover band**: which price band does the clearing happen in?
- **Total bid/offer ratio**: buy_demand / sell_supply across all bands — >1 means demand pressure
- **Steep vs flat supply curve**: predicts whether small demand changes cause large price swings

---

## 3. National Grid Data (NERLDC/POSOCO)

### What This Is

Real-time SCADA telemetry from India's national grid operator. Shows the physical reality of the power system: how much electricity is being demanded, how much each fuel type is generating, and the balance between them. This is the **fundamental supply-demand driver** behind IEX prices.

### Source & Acquisition

- **Source**: NERLDC/POSOCO SCADA system exports (Excel files)
- **Raw location**: `Data/Raw/nerldc/` — 29 Excel files (~713 MB total)
- **Three file formats**: 5-min SCADA (2021-2022), 10-sec SCADA (2022-2023), hourly clean (2024-2025)
- **SCADA column mapping**: Row 1 in SCADA files contains readable names (e.g., `TOTAL_THM_ONLY|P`), used to decode `SCADA/ANALOG/044MQ068/0` → `total_thermal_mw`

### Cleaning Pipeline

`data_sourcing/scripts/clean_nerldc_to_hourly.py` — Handles all 3 formats, resamples sub-hourly to hourly via mean, clips negative solar, derives `net_demand_mw`, imputes fuel breakdown for 2024+ via month x hour ratio model, adds `fuel_mix_imputed` flag.

### Schema

**File**: `Cleaned/grid/nerldc_national_hourly.parquet` — 33,432 rows

| Column | Type | Null% | Observed% | Description | Price Relevance |
|--------|------|-------|-----------|-------------|-----------------|
| `delivery_start_ist` | datetime64[IST] | 0% | 100% | IST timestamp (hourly) | Join key |
| `all_india_demand_mw` | float64 | 0% | 100% | All-India grid demand | **PRIMARY** — demand is the #1 price driver |
| `all_india_wind_mw` | float64 | 0% | 100% | All-India wind generation | **HIGH** — zero-cost supply, depresses prices |
| `all_india_solar_mw` | float64 | 0% | 100% | All-India solar generation | **HIGH** — creates the midday price collapse |
| `net_demand_mw` | float64 | 0% | 100% | demand − solar − wind | **PRIMARY** — "how much conventional gen must run", 100% observed |
| `total_thermal_mw` | float64 | 0% | 61% obs / 39% imputed | Total thermal generation | MODERATE — sets marginal cost when near capacity |
| `total_hydro_mw` | float64 | 0% | 61% obs / 39% imputed | Total hydro generation | MODERATE — triples in monsoon, seasonal price effect |
| `total_gas_mw` | float64 | 0% | 61% obs / 39% imputed | Total gas generation | LOW — only ~3 GW on 180 GW system |
| `total_nuclear_mw` | float64 | 0% | 61% obs / 39% imputed | Total nuclear generation | LOW — flat baseload, rarely varies |
| `total_generation_mw` | float64 | 0% | 100% | Sum of all fuel types | MODERATE — gen-demand gap signals price direction |
| `fuel_mix_imputed` | bool | 0% | 100% | True = Jan 2024+ (fuel split estimated) | Model can weight observed vs imputed differently |

### Key Statistics

| Column | Min | P5 | Mean | P95 | Max |
|--------|-----|-----|------|-----|-----|
| all_india_demand_mw | 0 | 137,712 | 180,705 | 215,725 | 248,688 |
| all_india_wind_mw | 0 | 1,790 | 8,171 | 19,025 | 38,701 |
| all_india_solar_mw | 0 | 0 | 12,269 | 44,686 | 64,701 |
| net_demand_mw | 0 | 127,095 | 160,265 | 193,139 | 218,516 |
| total_thermal_mw | 0 | 105,264 | 135,115 | 162,264 | 193,974 |
| total_hydro_mw | 0 | 6,402 | 18,486 | 33,769 | 44,423 |

### Fuel Mix Imputation Detail

The Jan 2024–Jun 2025 file only provides Demand/Wind/Solar/Total — no thermal/hydro/gas/nuclear breakdown. Imputed using **month x hour median share** of conventional generation (288-cell lookup from 20,000+ training observations).

**Holdout validation** (trained Sep 2021 – Sep 2023, tested Oct – Dec 2023):

| Fuel | MAPE | RMSE | Risk Level |
|------|------|------|------------|
| Thermal | 5.8% | 9,375 MW | Low — stable year-to-year |
| Nuclear | 12.2% | 794 MW | Low — small absolute error |
| Gas | 34.9% | 1,361 MW | Irrelevant — gas is <2% of system |
| Hydro | 69.0% | 8,219 MW | High — monsoon rainfall varies 20-30% annually |

**Recommendation**: Use `net_demand_mw` (100% observed) as the primary grid feature. Use thermal/hydro as supplementary. XGBoost can use `fuel_mix_imputed` to learn different weights.

---

## 4. Weather Data

### What This Is

Hourly weather observations for 5 Indian cities, each representing a regional electricity grid. Weather drives both demand (temperature → AC load) and supply (radiation → solar, wind speed → wind generation).

### Source & Acquisition

- **Source**: Open-Meteo Historical Weather API (free, no authentication)
- **Raw location**: `Data/Raw/weather/` — 5 Parquet files, one per city

### City-Grid Mapping & Weights

| City | Grid | Weight | Why |
|------|------|--------|-----|
| Delhi | NR (Northern) | 0.30 | Largest demand center; extreme summer heat (46 C peak) |
| Mumbai | WR (Western) | 0.28 | Industrial hub; monsoon-affected |
| Chennai | SR (Southern) | 0.25 | Wind corridor (Tamil Nadu); different monsoon timing |
| Kolkata | ER (Eastern) | 0.12 | Coal belt; monsoon-affected |
| Guwahati | NER (North Eastern) | 0.05 | Small grid; hydro-dominated |

Weights approximate each grid's share of national electricity demand.

### Cleaning Pipeline

`data_sourcing/scripts/export_weather_to_cleaned.py` — IST conversion, period filter, combines 5 cities.

### Schema

**File**: `Cleaned/weather/weather_2022-04-01_to_2025-12-31.parquet` — 141,745 rows (28,349 per city)

| Column | Type | Null% | Description | Price Relevance |
|--------|------|-------|-------------|-----------------|
| `delivery_start_ist` | datetime64[IST] | 0% | IST timestamp (hourly) | Join key |
| `timestamp` | datetime64[IST] | 0% | Same (legacy) | — |
| `city` | str | 0% | City name | Group key |
| `grid` | str | 0% | Grid region code | Group key |
| `weight` | float64 | 0% | Demand-weighted importance (0.05–0.30) | For weighted aggregation |
| `temperature_2m` | float64 | 0% | Temperature at 2m (Celsius) | **PRIMARY** — heat drives AC demand → prices |
| `relative_humidity_2m` | int64 | 0% | Relative humidity (%) | Heat index; high humidity + heat = more AC |
| `direct_radiation` | float64 | 0% | Direct solar radiation (W/m2) | Solar generation proxy |
| `diffuse_radiation` | float64 | 0% | Diffuse solar radiation (W/m2) | Cloudy-day solar proxy |
| `shortwave_radiation` | float64 | 0% | Total shortwave radiation (W/m2) | **HIGH** — best single solar gen predictor |
| `wind_speed_10m` | float64 | 0% | Wind speed at 10m (m/s) | **HIGH** — wind generation proxy |
| `cloud_cover` | int64 | 0% | Cloud cover (%) | Inverse solar proxy |

### Key Statistics

| Column | Min | Mean | Max |
|--------|-----|------|-----|
| temperature_2m (C) | 3.2 | 26.3 | 46.0 |
| relative_humidity_2m (%) | 4 | 72.7 | 100 |
| shortwave_radiation (W/m2) | 0 | 206.9 | 1,013 |
| wind_speed_10m (m/s) | 0.0 | 9.5 | 53.1 |
| cloud_cover (%) | 0 | 49.9 | 100 |

### Feature Engineering Value

- **Demand-weighted national temperature**: `sum(city_temp * weight)` — single best demand predictor
- **Cooling Degree Days**: `max(0, temp - 24)` per hour — nonlinear AC load relationship
- **Solar radiation forecast error**: actual vs forecast shortwave → surprise supply/demand
- **Wind speed at Chennai** (SR): Tamil Nadu has most wind capacity, Chennai wind is a strong price predictor
- **Temperature spread** (Delhi - Mumbai): captures heat wave localization

---

## 5. Holiday Calendar

### File

`Data/Raw/holiday calendar/indian_holidays.csv` — 108 rows

| Column | Type | Description |
|--------|------|-------------|
| `date` | str (YYYY-MM-DD) | Holiday date |
| `holiday_name` | str | Holiday name (Republic Day, Diwali, etc.) |

### Price Relevance

Holidays reduce industrial demand (factories closed) → lower prices. But some holidays increase residential demand (Diwali lights). The effect is most visible in DAM volumes and morning/evening price patterns.

---

## 6. Cross-Dataset Alignment

### Resolution Mismatch

| Dataset | Native Resolution | Notes |
|---------|-------------------|-------|
| IEX Prices | 15-min (96 blocks/day) | Primary resolution |
| Bid Stack | 15-min x 12 price bands | Same as prices |
| Grid (NERLDC) | Hourly | Must broadcast to 15-min or aggregate prices to hourly |
| Weather | Hourly x 5 cities | Same as grid; needs city aggregation for national features |

### Date Range Alignment

| Dataset | Start | End | Days |
|---------|-------|-----|------|
| IEX Prices | 2022-04-01 | 2025-12-31 | 1,371 |
| Bid Stack | 2022-04-01 | 2025-12-31 | 1,371 |
| NERLDC Grid | 2021-09-01 | 2025-06-24 | 1,393 |
| Weather | 2022-04-01 | 2025-06-25 | 1,182 |
| **All-4 Overlap** | **2022-04-01** | **2025-06-24** | **1,181** |

Prices and bid stack extend to 2025-12-31 but grid and weather end mid-2025. For training, use 2022-04-01 to 2025-06-24. For the Dec 2025 tail, only price and bid stack features are available.

### Join Keys

- **Price ↔ Bid Stack**: join on `(date, market, time_block)`
- **Price ↔ Grid**: join `date` + `hour` to grid's `delivery_start_ist` (after extracting hour from time_block)
- **Price ↔ Weather**: same hourly join; decide whether to use per-city or demand-weighted national average
- **Price ↔ Holidays**: join on `date`

---

## 7. Data Quality Notes

### Known Issues

1. **Price ceiling**: DAM and G-DAM have a Rs 20,000/MWh cap. RTM cap is Rs 16,744/MWh. These are censored observations — the true market-clearing price could be higher.

2. **RTM session structure**: RTM has 4 sessions per hour (4 x 15-min blocks). The `session_id` field identifies which session, but is NaN for DAM/G-DAM.

3. **Bid stack gap-fill**: 16.6% of bid stack rows are ffill/bfill imputed (from dates where the IEX RSC endpoint returned empty). These imputed rows carry the previous day's bid shape — reasonable for consecutive days but may be stale for the 31-day DAM gap in Mar-Apr 2024.

4. **Grid fuel imputation**: 39% of thermal/hydro/gas/nuclear values are estimated via month x hour ratios. The `fuel_mix_imputed` flag marks these rows. Hydro estimates have 69% MAPE — use with caution.

5. **Solar clipping**: Negative nighttime solar values from SCADA noise were clipped to 0. This is physically correct (no solar at night).

6. **Weather data ends Jun 2025**: Open-Meteo historical data lags by ~6 months. The weather parquet covers Apr 2022 to Jun 2025.

### Data Provenance

| Dataset | Source | Method | Authentication |
|---------|--------|--------|----------------|
| IEX Prices | IEX India (`iexindia.com`) | Official REST API (preferred) or RSC scraping (fallback) | API token or browser cookies |
| Bid Stack | IEX India website | RSC endpoint scraping only (no API alternative) | Browser cookies |
| NERLDC Grid | POSOCO/NERLDC SCADA exports | Manual download (Excel files) | None |
| Weather | Open-Meteo API | Free API, no auth | None |
| Holidays | Manual compilation | Manually curated | N/A |

---

## 8. Suggested Feature Engineering Directions

Based on the data available, high-value features for MCP forecasting likely include:

### From Prices (lagged / autoregressive)
- Lagged MCP: t-1, t-4 (1 hour ago), t-96 (same block yesterday), t-672 (same block last week)
- Rolling statistics: 24h rolling mean/std of MCP
- Same-block-yesterday and same-block-last-week MCP
- DAM-RTM spread (cross-market signal)
- DAM-GDAM spread (green premium)
- Purchase bid / sell bid ratio (demand pressure indicator)
- MCV trend (volume momentum)

### From Bid Stack (market microstructure)
- Demand elasticity at MCP: slope of buy demand curve near clearing price
- Supply margin: total sell supply above MCP band
- Bid concentration (Herfindahl across bands)
- Which price band the clearing happens in
- Total buy/sell ratio across all bands

### From Grid (fundamental supply-demand)
- `net_demand_mw` — the single best fundamental feature (100% observed)
- net_demand delta: change from previous hour
- Demand-generation gap: `all_india_demand_mw - total_generation_mw`
- Solar ramp rate: hour-over-hour change in solar MW (predicts evening price spike)
- Wind capacity factor: actual wind / installed capacity
- Thermal utilization: thermal_mw / thermal_capacity (when near 100%, prices spike)

### From Weather (demand + supply drivers)
- Demand-weighted national temperature: `sum(city_temp * weight)`
- Cooling Degree Hours: `max(0, temp - 24)` — nonlinear AC demand
- Heating Degree Hours: `max(0, 18 - temp)` — winter heating (minor in India)
- Shortwave radiation (solar gen proxy) — demand-weighted
- Wind speed at Chennai (largest wind corridor)
- Temperature x humidity interaction (heat index → AC load)
- Cloud cover change (solar ramp prediction)

### Calendar / Temporal
- Hour of day, day of week, month, quarter
- Is_holiday flag
- Is_weekend flag
- Days since/until nearest holiday
- Monsoon flag (Jun 15 – Sep 30)
- Festival-specific dummies (Diwali, Holi — anomalous demand patterns)

---

## 9. Raw Data Location (for reference)

```
Data/Raw/
├── price/                               (~701 MB)
│   ├── iex_dam_price/                   3 CSVs, 146 MB
│   ├── iex_gdam_price/                  2 CSVs, 138 MB
│   └── iex_rtm_price/                   9 CSVs, 416 MB
├── agg-demand-supply/                   (~419 MB)
│   ├── iex_dam_aggregate/               1,260 per-day CSVs + 1 merged (130 MB)
│   ├── iex_gdam_aggregate/              1,330 per-day CSVs + 1 merged (135 MB)
│   └── iex_rtm_aggregate/              1,332 per-day CSVs + 1 merged (154 MB)
├── nerldc/                              29 xlsx files (~713 MB)
├── weather/                             5 city parquets (~2.5 MB)
└── holiday calendar/
    └── indian_holidays.csv              108 holidays
```

---

## 10. Processing Scripts (in `data_sourcing/`)

| Script | Input | Output |
|--------|-------|--------|
| `scripts/fetch_iex_data.py` | IEX API | `Raw/price/*.csv` |
| `scripts/fetch_iex_aggregate.py` | IEX RSC endpoint | `Raw/agg-demand-supply/*.csv` |
| `scripts/combine_iex_prices_to_cleaned.py` | Raw price CSVs | intermediate parquet |
| `scripts/fill_iex_prices_gaps.py` | intermediate | `Cleaned/price/iex_prices_combined_filled.parquet` |
| `scripts/fill_iex_aggregate_gaps.py` | Raw merged CSVs | `Cleaned/bid_stack/iex_aggregate_combined_filled.parquet` |
| `scripts/clean_nerldc_to_hourly.py` | Raw nerldc xlsx | `Cleaned/grid/nerldc_national_hourly.parquet` |
| `scripts/export_weather_to_cleaned.py` | Raw weather parquets | `Cleaned/weather/weather_*.parquet` |
| `scripts/aggregate_gap_report.py` | Merged bid stack CSVs | Gap analysis + optional backfill |

All scripts live in `data_sourcing/` which is gitignored. Config files are in `data_sourcing/config/`.
