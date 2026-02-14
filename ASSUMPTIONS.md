# Assumptions & Limitations

This document transparently records every assumption embedded in the BESS arbitrage optimizer. Assumptions are grouped by subsystem. Each includes a brief rationale and, where applicable, the likely direction of bias.

---

## 1. Market Structure

| Assumption | Detail | Bias Direction |
| :--- | :--- | :--- |
| **IEX DAM clears D-1 10:00 IST** | DAM schedule must be committed before RTM prices are known. | Neutral — this is market reality. |
| **RTM clears in 15-min blocks on Day D** | RTM recourse modeled at hourly granularity (not 15-min). | Optimistic — hourly aggregation smooths intra-hour volatility. |
| **Price-taker** | 50 MW BESS has no market impact on MCP. | Realistic at 50 MW vs ~180 GW system demand. |
| **No ancillary services** | FCAS, SRAS, and secondary reserves are not modeled. | Conservative — real revenue would be higher with AS stacking. |
| **Exchange-only trading** | No bilateral contracts, no term-ahead. Pure IEX DAM + RTM. | Conservative — bilaterals could hedge downside. |
| **No market rule changes** | CERC 2024 DSM regulations assumed static throughout backtest. | Neutral for backtest; risk for forward projections. |

---

## 2. BESS Physical Parameters

| Assumption | Detail | Bias Direction |
| :--- | :--- | :--- |
| **50 MW / 200 MWh nameplate** | 4-hour duration system. | Neutral — standard utility-scale configuration. |
| **80% Depth of Discharge** | Usable capacity = 160 MWh (SoC range: 20–180 MWh). | Neutral — manufacturer recommendation for LFP. |
| **90% round-trip efficiency** | Applied as 94.87% per leg (charge and discharge). | Slightly optimistic — real RTE degrades with C-rate and temperature. |
| **Flat degradation cost ₹650/MWh** | No cycle-depth curve, no calendar aging, no temperature dependence. | Optimistic — real degradation is nonlinear and accelerates with deep cycles. |
| **No thermal derating** | Full 50 MW available in all hours regardless of ambient temperature. | Optimistic — Indian summers (45°C+) can derate by 10–20%. |
| **No auxiliary load** | HVAC, BMS, and balance-of-plant parasitic loads not modeled. | Optimistic — typically 1–3% of throughput. |
| **No self-discharge** | SoC is preserved perfectly between hours and overnight. | Minor — LFP self-discharge is <0.1%/day. |
| **Infinite calendar life** | No capacity fade over the 143-day backtest window. | Realistic for 143 days; invalid for multi-year projections. |

---

## 3. Cost Model

| Assumption | Detail | Bias Direction |
| :--- | :--- | :--- |
| **IEX transaction fee: ₹200/MWh per side** | CERC Regulation 23 cap applied uniformly. | Neutral — regulatory ceiling. |
| **Degradation: ₹650/MWh on discharge** | Flat rate, no cycle-counting methodology. | See §2 above. |
| **DSM penalty: CERC 2024 block-wise Normal Rate** | 3% physical error proxy × Normal Rate per block. | Approximate — actual DSM depends on real-time frequency and schedule adherence. |
| **ISTS waiver assumed** | No inter-state transmission charges. | Optimistic — waiver is policy-dependent and may expire. |
| **No working capital cost** | IEX settlement cycle cash drag not modeled. | Optimistic — typically 2–5 day settlement lag. |
| **No insurance, no O&M** | Operating expenditure beyond degradation not included. | Optimistic — real O&M adds ₹150–300/kW-yr. |
| **No land lease or grid connection costs** | Capital structure not modeled; analysis is pure operating P&L. | Intentional — this is a dispatch optimizer, not a project finance model. |

---

## 4. Forecasting

| Assumption | Detail | Bias Direction |
| :--- | :--- | :--- |
| **D-1 08:00 IST information snapshot** | All features use data available by 08:00 the day before delivery. No future leakage. | Conservative — validated via anti-leakage checks. |
| **LightGBM Quantile Regression** | 5 quantiles (q10, q25, q50, q75, q90) per market per hour. | Neutral — standard probabilistic forecasting approach. |
| **Training: Apr 2022 – Sep 2024** | 2.5 years of historical IEX MCP data. | Neutral — sufficient for seasonal patterns. |
| **Validation: Oct 2024 – Jan 2025** | 4 months out-of-sample for CQR recalibration. | Neutral. |
| **D+1 forecaster reuses DAM hyperparameters** | Not separately tuned — same LGBM architecture, shifted targets. | Slightly optimistic — separate tuning may improve D+1 accuracy. |
| **Climatological fallback for D+2..D+6** | Expanding-window median of quantile predictions used for days beyond D+1. | Conservative — produces mean-reverting, low-information scenarios. |
| **No exogenous shock modeling** | Grid failures, policy changes, sudden demand spikes not explicitly modeled. | Optimistic — tail events are underrepresented in scenarios. |
| **Gaussian Copula for dependency structure** | Assumes elliptical joint distribution after PIT transform. | Approximate — real DAM/RTM dependency may have asymmetric tail structure. |
| **Cross-day ρ = 0.241 (AR(1))** | Estimated from daily-average z-score autocorrelation. | Neutral — empirically measured from validation data. |

---

## 5. Optimization

| Assumption | Detail | Bias Direction |
| :--- | :--- | :--- |
| **Perfect DAM schedule execution** | No partial fills, no gate closure delays, no bid rejection. | Optimistic — IEX can reject bids in extreme conditions. |
| **Full RTM flexibility** | RTM recourse can use full 50 MW in any direction within SoC limits. | Slightly optimistic — RTM liquidity may constrain large positions. |
| **Continuous LP relaxation** | No integer variables — charge/discharge can take any value in [0, 50]. | Realistic — BESS dispatch is inherently continuous. |
| **200 scenarios per solve** | No convergence sensitivity analysis presented. | Approximate — 200 is standard but may under-sample tails. |
| **Initial SoC = 100 MWh on Day 1** | Arbitrary starting point; chains overnight thereafter. | Minor — washes out after first few days. |
| **Soft terminal continuation value** | `max(0, E[spread] × η_c × η_d − 2×IEX_fee − degradation − 2×DSM_friction)` | Approximate — heuristic estimate, not solved via backward induction. |
| **Single planning solve per day** | DAM schedule is committed at D-1 and not revised. No intraday re-optimization. | See §6 — backtest RTM evaluation uses realized prices, which partially compensates. |

---

## 6. Backtesting Methodology

| Assumption | Detail | Bias Direction |
| :--- | :--- | :--- |
| **Feb 1 – Jun 24, 2025 (143 days)** | Peak spread season (pre-monsoon heat). | **Optimistic for annualization** — monsoon/shoulder months have 40–50% narrower spreads. |
| **No transaction slippage** | Execution at exact MCP with no latency or spread. | Optimistic — real execution has bid-ask friction. |
| **RTM evaluation uses realized prices** | In the backtest, Stage 2 dispatch is optimized against actual RTM prices. In production, RTM decisions would be made against forecasted prices, reducing capture ratio. Intraday re-optimization with rolling forecasts is a future enhancement. | **Optimistic** — backtest overstates RTM capture vs what a live system would achieve. |
| **No market feedback** | BESS trading does not affect prices in the backtest. | Realistic at 50 MW. |
| **Walk-forward design** | Each day uses only information available at D-1 08:00. | Neutral — proper out-of-sample discipline. |

---

## 7. Known Limitations & Future Work

- **Seasonality**: The 143-day window is India's most profitable BESS season. A conservative full-year estimate applies ~50% seasonal discount to remaining 222 days.
- **Degradation modeling**: Moving from flat ₹/MWh to a cycle-counting + calendar aging model would reduce projected revenue by an estimated 3–8%.
- **Intraday re-optimization**: Current system commits at D-1 and evaluates RTM against realized prices. A production system would need a live RTM forecaster and rolling re-dispatch every 15 minutes — expected to capture less than the backtest shows.
- **15-minute granularity**: RTM operates at 15-min blocks; hourly modeling misses intra-hour volatility that a fast-cycling BESS could exploit.
- **Multi-asset / portfolio effects**: Optimizer runs a single BESS in isolation. Co-optimization with RE assets or multiple storage sites is not modeled.
- **Regulatory risk**: CERC DSM rules, IEX fee caps, and ISTS waivers are policy instruments that can change.
