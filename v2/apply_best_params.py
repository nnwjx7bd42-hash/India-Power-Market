#!/usr/bin/env python3
"""
Apply Optuna best params from v2/results/optuna_lstm_best_params.yaml
into v2/lstm_config.yaml (merge sequence, model, training sections).
Run after: python tune_lstm_bayesian.py --n-trials 50
"""

import argparse
import sys
from pathlib import Path

import yaml


def get_v2_root():
    return Path(__file__).resolve().parent


def deep_merge(base, override):
    """Merge override into base in-place. Only keys present in override are updated."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value


def main():
    parser = argparse.ArgumentParser(description="Apply Optuna best params to lstm_config.yaml")
    parser.add_argument(
        "--best",
        default=None,
        help="Path to best params YAML (default: v2/results/optuna_lstm_best_params.yaml)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to lstm_config.yaml (default: v2/lstm_config.yaml)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print merged config, do not write")
    args = parser.parse_args()

    root = get_v2_root()
    best_path = Path(args.best) if args.best else root / "results" / "optuna_lstm_best_params.yaml"
    config_path = Path(args.config) if args.config else root / "lstm_config.yaml"

    if not best_path.exists():
        print(f"Best params not found: {best_path}", file=sys.stderr)
        print("Run: python v2/tune_lstm_bayesian.py --n-trials 50", file=sys.stderr)
        sys.exit(1)
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    with open(best_path, "r") as f:
        best = yaml.safe_load(f)

    for section in ("sequence", "model", "training"):
        if section in best and section in config:
            deep_merge(config[section], best[section])

    if args.dry_run:
        print(yaml.dump(config, default_flow_style=False, sort_keys=False))
        return

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Updated {config_path} with best params from {best_path}")


if __name__ == "__main__":
    main()
