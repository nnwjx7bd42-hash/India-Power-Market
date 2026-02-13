#!/usr/bin/env python3
"""
Data alignment validation gate-check.

Loads all four cleaned parquet files and asserts structural integrity,
frequency regularity, null-freeness, and 1:1 hourly join coverage.

Exit 0 on pass, exit 1 on any failure.

Usage:
    python scripts/validate_alignment.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEANED = ROOT / "Data" / "Cleaned"

PRICE_PATH = CLEANED / "price" / "iex_prices_combined_filled.parquet"
BID_PATH = CLEANED / "bid_stack" / "iex_aggregate_combined_filled.parquet"
GRID_PATH = CLEANED / "grid" / "nerldc_national_hourly.parquet"
WEATHER_PATH = CLEANED / "weather" / "weather_2022-04-01_to_2025-12-31.parquet"

MARKETS = ["dam", "gdam", "rtm"]
EXPECTED_BLOCKS_PER_DAY = 96
EXPECTED_BANDS = 12

OVERLAP_START = pd.Timestamp("2022-04-01", tz="Asia/Kolkata")
OVERLAP_END = pd.Timestamp("2025-06-24 23:00:00", tz="Asia/Kolkata")
EXPECTED_OVERLAP_HOURS = 28_344


class ValidationError(Exception):
    pass


def _fail(msg: str):
    raise ValidationError(msg)


def validate_prices(price: pd.DataFrame) -> list[str]:
    """Validate price parquet. Returns list of pass messages."""
    results = []

    # Column check
    expected_cols = {
        "date", "market", "time_block", "hour", "delivery_start_ist",
        "purchase_bid_mwh", "sell_bid_mwh", "mcv_mwh",
        "final_scheduled_volume_mwh", "mcp_rs_mwh", "weighted_mcp_rs_mwh",
    }
    actual_cols = set(price.columns)
    if actual_cols != expected_cols:
        extra = actual_cols - expected_cols
        missing = expected_cols - actual_cols
        _fail(f"Price columns mismatch — extra: {extra}, missing: {missing}")
    results.append("Columns: 11 expected columns present")

    # Per-market block count
    for m in MARKETS:
        sub = price[price["market"] == m]
        bpd = sub.groupby("date")["time_block"].nunique()
        if (bpd != EXPECTED_BLOCKS_PER_DAY).any():
            bad = bpd[bpd != EXPECTED_BLOCKS_PER_DAY]
            _fail(f"Price {m}: {len(bad)} days with != {EXPECTED_BLOCKS_PER_DAY} blocks")
        results.append(f"{m}: {EXPECTED_BLOCKS_PER_DAY} blocks/day x {sub['date'].nunique()} days")

    # NaN in mcp_rs_mwh
    nan_count = price["mcp_rs_mwh"].isna().sum()
    if nan_count > 0:
        _fail(f"Price mcp_rs_mwh has {nan_count} NaN values")
    results.append("mcp_rs_mwh: zero NaN")

    # 15-min frequency
    for m in MARKETS:
        sub = price[price["market"] == m].sort_values("delivery_start_ist")
        deltas = sub["delivery_start_ist"].diff().dropna()
        non_15 = (deltas != pd.Timedelta(minutes=15)).sum()
        if non_15 > 0:
            _fail(f"Price {m}: {non_15} non-15min gaps in delivery_start_ist")
    results.append("All markets: 15-min frequency")

    return results


def validate_bid_stack(bid: pd.DataFrame) -> list[str]:
    """Validate bid stack parquet."""
    results = []

    # Per-market rows/day
    expected_rpd = EXPECTED_BLOCKS_PER_DAY * EXPECTED_BANDS  # 1152
    for m in MARKETS:
        sub = bid[bid["market"] == m]
        rpd = sub.groupby("date").size()
        if rpd.min() != rpd.max() or rpd.min() != expected_rpd:
            _fail(
                f"Bid {m}: rows/day not uniform — "
                f"min={rpd.min()}, max={rpd.max()}, expected={expected_rpd}"
            )
        results.append(f"{m}: {expected_rpd} rows/day x {sub['date'].nunique()} days")

    # 96 unique blocks
    unique_blocks = bid["time_block"].nunique()
    if unique_blocks != EXPECTED_BLOCKS_PER_DAY:
        _fail(f"Bid stack has {unique_blocks} unique blocks, expected {EXPECTED_BLOCKS_PER_DAY}")
    results.append(f"{EXPECTED_BLOCKS_PER_DAY} unique time blocks")

    # NaN in key columns
    for col in ["buy_demand_mw", "sell_supply_mw"]:
        nan_count = bid[col].isna().sum()
        if nan_count > 0:
            _fail(f"Bid {col} has {nan_count} NaN values")
    results.append("buy_demand_mw / sell_supply_mw: zero NaN")

    return results


def validate_grid(grid: pd.DataFrame) -> list[str]:
    """Validate grid parquet."""
    results = []

    ts = grid["delivery_start_ist"].sort_values()

    # Duplicates
    dupes = ts.duplicated().sum()
    if dupes > 0:
        _fail(f"Grid has {dupes} duplicate timestamps")
    results.append("Zero duplicate timestamps")

    # Hourly gaps
    deltas = ts.diff().dropna()
    non_1h = (deltas != pd.Timedelta(hours=1)).sum()
    if non_1h > 0:
        _fail(f"Grid has {non_1h} non-1h gaps")
    results.append("Hourly frequency, zero gaps")

    # NaN in numeric columns
    for col in grid.columns:
        if col in ("delivery_start_ist", "fuel_mix_imputed"):
            continue
        nan_count = grid[col].isna().sum()
        if nan_count > 0:
            _fail(f"Grid {col} has {nan_count} NaN values")
    results.append("All numeric columns: zero NaN")

    return results


def validate_weather(weather: pd.DataFrame) -> list[str]:
    """Validate weather parquet."""
    results = []

    cities = sorted(weather["city"].unique())
    results.append(f"Cities: {cities}")

    for city in cities:
        sub = weather[weather["city"] == city].sort_values("delivery_start_ist")
        ts = sub["delivery_start_ist"]

        # Timestamps at :00
        non_zero_min = (ts.dt.minute != 0).sum()
        if non_zero_min > 0:
            _fail(f"Weather {city}: {non_zero_min} timestamps not at :00")

        # Duplicates
        dupes = ts.duplicated().sum()
        if dupes > 0:
            _fail(f"Weather {city}: {dupes} duplicate timestamps")

        # Hourly gaps
        deltas = ts.diff().dropna()
        non_1h = (deltas != pd.Timedelta(hours=1)).sum()
        if non_1h > 0:
            _fail(f"Weather {city}: {non_1h} non-1h gaps")

        results.append(f"  {city}: {len(sub)} rows, hourly, no gaps, no dupes")

    # NaN in numeric columns
    for col in weather.columns:
        if col in ("delivery_start_ist", "city", "grid", "weight", "timestamp"):
            continue
        nan_count = weather[col].isna().sum()
        if nan_count > 0:
            _fail(f"Weather {col} has {nan_count} NaN values")
    results.append("All numeric columns: zero NaN")

    return results


def validate_hourly_join(
    price: pd.DataFrame, grid: pd.DataFrame, weather: pd.DataFrame
) -> list[str]:
    """Validate 1:1 hourly join across DAM, Grid, Weather."""
    results = []

    # DAM hourly timestamps
    dam = price[price["market"] == "dam"]
    dam_hours = set(dam["delivery_start_ist"].dt.floor("h").unique())

    # Grid hourly timestamps
    grid_hours = set(grid["delivery_start_ist"])

    # Weather hourly (Delhi as proxy — all cities share the same timestamps)
    delhi = weather[weather["city"] == "Delhi"]
    wx_hours = set(delhi["delivery_start_ist"])

    # Filter to overlap window
    dam_in = {ts for ts in dam_hours if OVERLAP_START <= ts <= OVERLAP_END}
    grid_in = {ts for ts in grid_hours if OVERLAP_START <= ts <= OVERLAP_END}
    wx_in = {ts for ts in wx_hours if OVERLAP_START <= ts <= OVERLAP_END}

    all3 = dam_in & grid_in & wx_in
    matched = len(all3)

    results.append(f"Overlap: {OVERLAP_START} to {OVERLAP_END}")
    results.append(f"DAM hours:     {len(dam_in)}")
    results.append(f"Grid hours:    {len(grid_in)}")
    results.append(f"Weather hours: {len(wx_in)}")
    results.append(f"All-3 matched: {matched}/{EXPECTED_OVERLAP_HOURS}")

    if matched != EXPECTED_OVERLAP_HOURS:
        _fail(
            f"Hourly join: {matched}/{EXPECTED_OVERLAP_HOURS} "
            f"({matched / EXPECTED_OVERLAP_HOURS * 100:.2f}%) — expected 100%"
        )
    results.append("Coverage: 100.00%")

    return results


def main():
    failed = False

    print("=" * 72)
    print("DATA ALIGNMENT VALIDATION")
    print("=" * 72)

    # Load all datasets
    print("\nLoading datasets...")
    try:
        price = pd.read_parquet(PRICE_PATH)
        bid = pd.read_parquet(BID_PATH)
        grid = pd.read_parquet(GRID_PATH)
        weather = pd.read_parquet(WEATHER_PATH)
    except FileNotFoundError as e:
        print(f"\nFAIL: {e}")
        sys.exit(1)

    checks = [
        ("PRICES", lambda: validate_prices(price)),
        ("BID STACK", lambda: validate_bid_stack(bid)),
        ("GRID", lambda: validate_grid(grid)),
        ("WEATHER", lambda: validate_weather(weather)),
        ("HOURLY JOIN", lambda: validate_hourly_join(price, grid, weather)),
    ]

    for name, check_fn in checks:
        print(f"\n--- {name} ---")
        try:
            results = check_fn()
            for r in results:
                print(f"  {r}")
            print(f"  PASS")
        except ValidationError as e:
            print(f"  FAIL: {e}")
            failed = True

    # Summary
    print("\n" + "=" * 72)
    if failed:
        print("RESULT: FAIL — one or more checks did not pass")
        sys.exit(1)
    else:
        print("RESULT: ALL CHECKS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
