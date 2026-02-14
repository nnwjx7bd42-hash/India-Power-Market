#!/bin/bash
# finish_deployment.sh
# Run after backtests complete to regenerate charts and push to GitHub
# Usage: ./finish_deployment.sh
set -e

echo "============================================================"
echo "RUNNING FINAL DEPLOYMENT SEQUENCE"
echo "============================================================"

# Check if backtests are still running
if pgrep -f "scripts/run_cvar_sweep.py" > /dev/null; then
    echo "⚠️  WARNING: CVaR Sweep is still running. Charts will be incomplete."
    read -p "Do you want to proceed anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "1. Regenerating Charts from latest results..."
python scripts/visualize_results.py

echo "2. Adding updated results and charts to git..."
git add results/

echo "3. Committing changes..."
git commit -m "results: updated charts and backtest CSVs after full clean run"

echo "4. Pushing to origin main..."
git push origin main

echo "============================================================"
echo "✅ DEPLOYMENT COMPLETE"
echo "============================================================"
