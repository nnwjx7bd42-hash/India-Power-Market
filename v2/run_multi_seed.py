#!/usr/bin/env python3
"""
Run LSTM training with multiple random seeds and report mean ± std.
Use for honest multi-seed evaluation (Option A) or to find a good production seed (Option B).

Usage:
  python run_multi_seed.py                    # default seeds: 42, 123, 456, 789, 2024
  python run_multi_seed.py --seeds 0 7 42 123 2024
  python run_multi_seed.py --seeds 42 123 456 789 2024 --out results/multi_seed_results.csv
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def get_v2_root():
    return Path(__file__).resolve().parent


def run_one_seed(seed: int, quiet: bool = False) -> dict:
    """Run train_lstm.py with --seed and return test_mape, test_rmse from lstm_metrics.yaml."""
    root = get_v2_root()
    train_script = root / "train_lstm.py"
    metrics_path = root / "results" / "lstm_metrics.yaml"
    cmd = [sys.executable, str(train_script), "--seed", str(seed)]
    if quiet:
        out = subprocess.DEVNULL
        err = subprocess.DEVNULL
    else:
        out = None
        err = None
    result = subprocess.run(cmd, cwd=str(root), stdout=out, stderr=err)
    if result.returncode != 0:
        return {"seed": seed, "test_mape": None, "test_rmse": None, "error": True}
    if not metrics_path.exists():
        return {"seed": seed, "test_mape": None, "test_rmse": None, "error": True}
    with open(metrics_path, "r") as f:
        m = yaml.safe_load(f)
    return {
        "seed": seed,
        "test_mape": m.get("test_mape"),
        "test_rmse": m.get("test_rmse"),
        "error": False,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run LSTM training with multiple seeds and report mean ± std"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456, 789, 2024],
        help="Space-separated seeds (default: 42 123 456 789 2024)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress train_lstm.py stdout/stderr (only summary printed)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to save results CSV (e.g. results/multi_seed_results.csv)",
    )
    args = parser.parse_args()

    root = get_v2_root()
    (root / "results").mkdir(parents=True, exist_ok=True)

    results = []
    for i, seed in enumerate(args.seeds):
        if not args.quiet:
            print(f"\n--- Seed {seed} ({i + 1}/{len(args.seeds)}) ---")
        row = run_one_seed(seed, quiet=args.quiet)
        results.append(row)
        if row.get("error"):
            print(f"Seed {seed}: FAILED", file=sys.stderr)
        elif not args.quiet:
            print(f"Seed {seed}: MAPE={row['test_mape']:.2f}%, RMSE={row['test_rmse']:.2f}")

    # Summary table
    valid = [r for r in results if not r.get("error") and r.get("test_mape") is not None]
    if not valid:
        print("No successful runs.", file=sys.stderr)
        sys.exit(1)

    mapes = [r["test_mape"] for r in valid]
    rmses = [r["test_rmse"] for r in valid]
    mean_mape = sum(mapes) / len(mapes)
    mean_rmse = sum(rmses) / len(rmses)
    var_mape = sum((x - mean_mape) ** 2 for x in mapes) / len(mapes)
    var_rmse = sum((x - mean_rmse) ** 2 for x in rmses) / len(rmses)
    std_mape = var_mape ** 0.5
    std_rmse = var_rmse ** 0.5

    print("\n" + "=" * 60)
    print("MULTI-SEED LSTM RESULTS")
    print("=" * 60)
    print(f"{'Seed':>8}  {'MAPE %':>10}  {'RMSE':>10}")
    print("-" * 32)
    for r in results:
        if r.get("error") or r.get("test_mape") is None:
            print(f"{r['seed']:>8}  {'FAIL':>10}  {'—':>10}")
        else:
            print(f"{r['seed']:>8}  {r['test_mape']:>10.2f}  {r['test_rmse']:>10.2f}")
    print("-" * 32)
    print(f"Mean ± std:  MAPE {mean_mape:.2f}% ± {std_mape:.2f}%   RMSE {mean_rmse:.2f} ± {std_rmse:.2f}")
    best = min(valid, key=lambda r: r["test_mape"])
    print(f"Best seed:   {best['seed']} (MAPE {best['test_mape']:.2f}%)")
    print("=" * 60)

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = root / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write("seed,test_mape,test_rmse\n")
            for r in results:
                m = r.get("test_mape", "")
                rmse = r.get("test_rmse", "")
                f.write(f"{r['seed']},{m},{rmse}\n")
            f.write(f"mean_mape,{mean_mape}\n")
            f.write(f"std_mape,{std_mape}\n")
            f.write(f"mean_rmse,{mean_rmse}\n")
            f.write(f"std_rmse,{std_rmse}\n")
        print(f"Results saved: {out_path}")


if __name__ == "__main__":
    main()
