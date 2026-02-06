"""Smoke tests for the data pipeline layer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ── estimate_missing_generation (vectorised) ─────────────────────────────

class TestEstimateMissingGeneration:
    """Verify the vectorised merge-based estimation produces sensible results."""

    @staticmethod
    def _make_sample_df() -> pd.DataFrame:
        """Build a small two-period DataFrame resembling NERLDC data."""
        rng = np.random.default_rng(0)

        # Period 1: complete data (2023-11-01 to 2023-12-31)
        idx1 = pd.date_range("2023-11-01", "2023-12-31 23:00", freq="h", tz="Asia/Kolkata")
        n1 = len(idx1)
        thermal = rng.uniform(50_000, 70_000, n1)
        hydro = rng.uniform(5_000, 10_000, n1)
        gas = rng.uniform(3_000, 6_000, n1)
        nuclear = rng.uniform(4_000, 7_000, n1)
        wind = rng.uniform(2_000, 8_000, n1)
        solar = rng.uniform(0, 15_000, n1)
        total_gen = thermal + hydro + gas + nuclear + wind + solar

        df1 = pd.DataFrame({
            "Thermal": thermal,
            "Hydro": hydro,
            "Gas": gas,
            "Nuclear": nuclear,
            "Wind": wind,
            "Solar": solar,
            "Total_Generation": total_gen,
        }, index=idx1)

        # Period 2: Hydro/Gas/Nuclear missing
        idx2 = pd.date_range("2024-01-01", "2024-01-14 23:00", freq="h", tz="Asia/Kolkata")
        n2 = len(idx2)
        df2 = pd.DataFrame({
            "Thermal": rng.uniform(50_000, 70_000, n2),
            "Hydro": np.nan,
            "Gas": np.nan,
            "Nuclear": np.nan,
            "Wind": rng.uniform(2_000, 8_000, n2),
            "Solar": rng.uniform(0, 15_000, n2),
            "Total_Generation": rng.uniform(80_000, 110_000, n2),
        }, index=idx2)

        return pd.concat([df1, df2])

    def test_fills_missing_values(self):
        from src.data_pipeline.load_nerdc_data import estimate_missing_generation

        df = self._make_sample_df()
        before_missing = df["Hydro"].isna().sum()
        assert before_missing > 0

        result = estimate_missing_generation(df)

        after_missing = result["Hydro"].isna().sum()
        assert after_missing < before_missing, "Vectorised fill should reduce NaN count"

    def test_estimated_values_are_positive(self):
        from src.data_pipeline.load_nerdc_data import estimate_missing_generation

        df = self._make_sample_df()
        result = estimate_missing_generation(df)

        for col in ("Hydro", "Gas", "Nuclear"):
            filled = result.loc["2024":, col].dropna()
            assert (filled >= 0).all(), f"Estimated {col} should be non-negative"


# ── aggregate_to_national (vectorised) ───────────────────────────────────

class TestAggregateToNational:
    """Verify the vectorised groupby-based weather aggregation."""

    @staticmethod
    def _make_sample_weather() -> dict:
        """Build a minimal 5-city weather dict."""
        rng = np.random.default_rng(1)
        cities = {
            "Delhi":    {"weight": 0.30, "region": "NR"},
            "Mumbai":   {"weight": 0.28, "region": "WR"},
            "Chennai":  {"weight": 0.25, "region": "SR"},
            "Kolkata":  {"weight": 0.12, "region": "ER"},
            "Guwahati": {"weight": 0.05, "region": "NER"},
        }
        idx = pd.date_range("2024-01-01", "2024-01-02 23:00", freq="h")
        result = {}
        for city, meta in cities.items():
            df = pd.DataFrame({
                "temperature_2m": rng.uniform(20, 40, len(idx)),
                "wind_speed_10m": rng.uniform(0, 15, len(idx)),
                "direct_radiation": rng.uniform(0, 800, len(idx)),
                "relative_humidity_2m": rng.uniform(30, 90, len(idx)),
                "diffuse_radiation": rng.uniform(0, 300, len(idx)),
                "shortwave_radiation": rng.uniform(0, 1000, len(idx)),
                "cloud_cover": rng.uniform(0, 100, len(idx)),
                "weight": meta["weight"],
            }, index=idx)
            result[city] = df
        return result

    def test_produces_national_columns(self):
        from src.data_pipeline.load_weather_data import aggregate_to_national

        data = self._make_sample_weather()
        df_nat = aggregate_to_national(data)

        assert "temperature_2m_national" in df_nat.columns
        assert "wind_speed_10m_national" in df_nat.columns
        assert len(df_nat) == 48  # 2 days × 24 hours

    def test_weighted_average_bounded(self):
        from src.data_pipeline.load_weather_data import aggregate_to_national

        data = self._make_sample_weather()
        df_nat = aggregate_to_national(data)

        temps = df_nat["temperature_2m_national"]
        assert temps.min() >= 15, "National avg should be within individual city ranges"
        assert temps.max() <= 45
