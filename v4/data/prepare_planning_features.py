#!/usr/bin/env python3
"""
Prepare the v4 planning-feature dataset.

Loads dataset_cleaned.parquet, reuses existing calendar / weather / supply
features, adds long-memory price anchors, drops all near-term lags, and
writes planning_dataset.parquet.

Usage:
  python prepare_planning_features.py [--config ../config/planning_config.yaml]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

V4_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V4_ROOT / "data"))

from price_anchors import add_price_anchors


def load_config(config_path=None):
    if config_path is None:
        config_path = V4_ROOT / "config" / "planning_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Build v4 planning feature dataset")
    parser.add_argument("--config", default=None, help="Path to planning_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_path = (V4_ROOT / config["data"]["dataset"]).resolve()

    print("=" * 60)
    print("V4 — PLANNING FEATURE PREPARATION")
    print("=" * 60)

    # ------------------------------------------------------------------ load
    print(f"\n1. Loading dataset: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in df.columns:
            df = df.set_index("datetime")
    df = df.sort_index()
    print(f"   Rows: {len(df):,}   Columns: {len(df.columns)}")

    # ----------------------------------------- ensure Hour column exists
    target = config["data"]["target"]
    if "Hour" not in df.columns:
        df["Hour"] = df.index.hour

    # ----------------------------------------- add long-memory price anchors
    print("\n2. Adding long-memory price anchors ...")
    df = add_price_anchors(df, price_col=target, window_weeks=4)

    # ----------------------------------------- collect planning features
    feat_cfg = config["features"]
    feature_names = (
        feat_cfg["calendar"]
        + feat_cfg["weather"]
        + feat_cfg["load"]
        + feat_cfg["anchors"]
        + feat_cfg["interactions"]
    )

    # Verify all features present
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        print(f"   WARNING — missing features (will be dropped): {missing}")
        feature_names = [f for f in feature_names if f in df.columns]

    print(f"\n3. Selecting {len(feature_names)} planning features:")
    for i, f in enumerate(feature_names, 1):
        print(f"   {i:2d}. {f}")

    # ------------------------------------------------- build output frame
    out = df[[target] + feature_names].copy()

    # Drop rows with NaN in target or any feature
    n_before = len(out)
    out = out.dropna()
    n_after = len(out)
    print(f"\n4. Dropped {n_before - n_after:,} NaN rows → {n_after:,} remaining")
    print(f"   Date range: {out.index.min()} → {out.index.max()}")

    # ------------------------------------------------- save
    out_path = V4_ROOT / "data" / "planning_dataset.parquet"
    out.to_parquet(out_path)
    print(f"\n5. Saved: {out_path}")
    print(f"   Shape: {out.shape}")
    print("\nDone.")


if __name__ == "__main__":
    main()
