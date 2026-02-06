#!/usr/bin/env python3
"""
Command-line interface for IEX price forecasting
Generates 7-day hourly price forecasts
"""

import argparse
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from inference.predictor import generate_forecast


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate IEX electricity price forecasts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using command-line arguments
  python predict.py --start-date 2024-01-15 --peaks 150000,152000,151000,153000,152500,151500,150500
  
  # Using CSV file
  python predict.py --start-date 2024-01-15 --peaks-file daily_peaks.csv
  
  # Using JSON input
  python predict.py --input forecast_input.json
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--peaks',
        type=str,
        help='Comma-separated list of daily peak loads (MW)'
    )
    input_group.add_argument(
        '--peaks-file',
        type=str,
        help='CSV file with daily peak loads (columns: date, peak_load_mw)'
    )
    input_group.add_argument(
        '--input',
        type=str,
        help='JSON input file with forecast parameters'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Forecast start date (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS+05:30)'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of days to forecast (default: 7)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='models/training/enhanced_model.pkl',
        help='Path to enhanced model (default: models/training/enhanced_model.pkl)'
    )
    
    parser.add_argument(
        '--quantile-model',
        type=str,
        default='models/training/quantile_models.pkl',
        help='Path to quantile models (default: models/training/quantile_models.pkl)'
    )
    
    parser.add_argument(
        '--historical-data',
        type=str,
        default='data/processed/dataset_cleaned.parquet',
        help='Path to historical dataset (default: data/processed/dataset_cleaned.parquet)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/inference',
        help='Output directory (default: results/inference)'
    )
    
    parser.add_argument(
        '--output-format',
        type=str,
        nargs='+',
        choices=['csv', 'parquet', 'json'],
        default=['csv', 'parquet'],
        help='Output formats (default: csv parquet)'
    )
    
    return parser.parse_args()


def load_peaks_from_csv(filepath):
    """Load daily peaks from CSV file"""
    df = pd.read_csv(filepath, parse_dates=['date'])
    peaks = df['peak_load_mw'].values.tolist()
    return peaks, df['date'].iloc[0]


def load_peaks_from_json(filepath):
    """Load forecast parameters from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    peaks = data.get('daily_peaks', [])
    start_date = data.get('start_date')
    
    return peaks, start_date


def main():
    """Main execution"""
    args = parse_arguments()
    
    print("="*80)
    print("IEX PRICE FORECAST - COMMAND LINE INTERFACE")
    print("="*80)
    
    # Parse input
    if args.input:
        # JSON input
        peaks, start_date = load_peaks_from_json(args.input)
        if not start_date:
            start_date = args.start_date
    elif args.peaks_file:
        # CSV file
        peaks, start_date = load_peaks_from_csv(args.peaks_file)
        if args.start_date:
            start_date = args.start_date
    else:
        # Command-line peaks
        peaks = [float(p.strip()) for p in args.peaks.split(',')]
        start_date = args.start_date
    
    if not start_date:
        raise ValueError("Start date must be provided (via --start-date or in input file)")
    
    # Validate peaks
    if len(peaks) != args.days:
        print(f"Warning: {len(peaks)} peaks provided, but forecasting {args.days} days")
        if len(peaks) < args.days:
            # Extend with last value
            peaks.extend([peaks[-1]] * (args.days - len(peaks)))
        else:
            # Truncate
            peaks = peaks[:args.days]
    
    print(f"\nInput Parameters:")
    print(f"  Start date: {start_date}")
    print(f"  Forecast days: {args.days}")
    print(f"  Daily peaks: {peaks}")
    
    # Generate forecast
    try:
        results = generate_forecast(
            start_date=start_date,
            daily_peak_loads=peaks,
            weather_forecasts=None,  # Will fetch from API
            model_path=args.model,
            quantile_model_path=args.quantile_model,
            historical_data_path=args.historical_data,
            forecast_days=args.days
        )
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"forecast_{timestamp_str}"
        
        print(f"\nSaving results to {output_dir}...")
        
        if 'csv' in args.output_format:
            csv_path = output_dir / f"{base_filename}.csv"
            results['predictions'].to_csv(csv_path)
            print(f"  ✓ Saved: {csv_path}")
        
        if 'parquet' in args.output_format:
            parquet_path = output_dir / f"{base_filename}.parquet"
            results['predictions'].to_parquet(parquet_path)
            print(f"  ✓ Saved: {parquet_path}")
        
        if 'json' in args.output_format:
            json_path = output_dir / f"{base_filename}.json"
            # Convert to JSON-serializable format
            json_data = {
                'metadata': results['metadata'],
                'predictions': results['predictions'].reset_index().to_dict('records')
            }
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            print(f"  ✓ Saved: {json_path}")
        
        # Save metadata
        metadata_path = output_dir / f"{base_filename}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(results['metadata'], f, indent=2, default=str)
        print(f"  ✓ Saved: {metadata_path}")
        
        print(f"\n✓ Forecast generation complete!")
        print(f"\nResults Summary:")
        print(f"  Forecast period: {results['metadata']['forecast_start']} to {results['metadata']['forecast_end']}")
        print(f"  Price range: ₹{results['metadata']['price_range']['min']:,.2f} - ₹{results['metadata']['price_range']['max']:,.2f}")
        print(f"  Average price: ₹{results['metadata']['price_range']['mean']:,.2f}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
