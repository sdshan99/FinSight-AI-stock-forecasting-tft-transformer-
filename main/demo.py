"""
Demo Script - Stock Forecasting with Transformer

Quick demo script to understand the project structure.

Usage:
    python demo.py              # Quick training demo
    python demo.py --data-only  # Just data processing (fast)
"""

import sys
sys.path.append('src')

import warnings
warnings.filterwarnings('ignore')

from main import StockForecastingPipeline

def run_quick_demo(symbol, model_name):
    print(f"Quick Demo - Stock Forecasting with {model_name.upper()} on {symbol}")
    print("="*60)
    
    demo_config = {
        'model': model_name,
        'symbol': symbol,
        'sequence_length': 30,
        'epochs': 10,
        'batch_size': 16,
        'model_config': {
            'd_model': 128,
            'num_heads': 4,
            'num_layers': 2,
            'dff': 256,
            'dropout_rate': 0.1
        }
    }
    
    pipeline = StockForecastingPipeline(demo_config)
    results = pipeline.run_complete_pipeline()
        
    print("\nDemo completed!")
    print("For full training: python main.py")
    print("Check README.md for detailed documentation")
    return results

def run_data_only_demo(symbol):

    """
    Demo that only shows data processing (very fast)
    """
    print("Data Processing Demo (No Training)")
    print("="*60)
    
    from data.data_fetcher import StockDataFetcher
    from data.stock_preprocessor import StockDataPreprocessor
    
    # 1. Fetch data
    print("\n1. Fetching AAPL data...")
    fetcher = StockDataFetcher()
    data = fetcher.fetch_stock_data(symbol, '2019-01-01', '2024-01-01')
    print(f"   Fetched {len(data)} records")
    
    # 2. Preprocess
    print("\n2. Preprocessing data...")
    preprocessor = StockDataPreprocessor(
        sequence_length=30,
        prediction_horizon=1,
        features=['Open', 'High', 'Low', 'Close', 'Volume'],
        scaler_type='standard'
    )
    processed = preprocessor.preprocess_data(
        data=data,
        target_column='Close'
    )
    
    print(f"   Created {len(processed['X_train'])} training samples")
    print(f"   Created {len(processed['X_val'])} validation samples")
    print(f"   Created {len(processed['X_test'])} test samples")
    
    print(f"\nData shapes:")
    print(f"  Training X: {processed['X_train'].shape}")
    print(f"  Training y: {processed['y_train'].shape}")
    print(f"  Features: {processed['X_train'].shape[2]}")
    
    print("\nData processing demo completed successfully!")
    print("To run with model training: python demo.py")
    
    return processed

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Demo Script')
    parser.add_argument('--model', type=str, default='transformer', choices=['transformer', 'tft'],
                        help='Model to use: transformer or tft')
    parser.add_argument('--symbol', type=str, default='AAPL',
                        help='Stock ticker symbol (e.g., AAPL, NVDA)')
    parser.add_argument('--data-only', action='store_true',
                        help='Run data processing demo only (no training)')
    args = parser.parse_args()

    
    if args.data_only:
        results = run_data_only_demo(args.symbol)
    else:
        results = run_quick_demo(args.symbol, args.model)

    
    print("\n" + "="*60)
    print("Demo completed. Thank you for trying our stock forecasting system!")
    print("For full documentation, see README.md")
    print("For the complete pipeline, run: python main.py") 