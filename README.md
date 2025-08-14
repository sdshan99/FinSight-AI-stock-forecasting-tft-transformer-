# Stock Price Forecasting with Transformers

A Transformer-based neural network for predicting stock prices using time series data.

## Quick Start

Get up and running in 3 simple steps:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the model on Apple stock
python3 main.py --symbol AAPL

# 3. Check results in the results/ folder
```

The model will automatically handle data fetching, training, and evaluation.

## What This Does

This system:
- **Downloads** historical stock data (or uses realistic mock data if APIs fail)
- **Processes** the data with technical indicators (moving averages, RSI, etc.)
- **Trains** a Transformer neural network to learn price patterns
- **Predicts** next-day stock prices
- **Evaluates** performance with standard metrics (MAE, RMSE, MAPE, R²)
- **Generates** visualizations and reports

## Installation

### Requirements
- Python 3.8+
- pip

### Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd final-project

# Install dependencies
pip install -r requirements.txt

# Test the installation
python3 demo.py
```

**Troubleshooting:**
- If `python3` doesn't work, try `python`
- If `pip` fails, try `pip3` or add `--user` flag
- On macOS, you might need: `xcode-select --install`

## Usage

### Basic Usage (Try with python/python3)
```bash
# Train on different stocks
python3 main.py --symbol AAPL    # Apple
python3 main.py --symbol TSLA    # Tesla
python3 main.py --symbol GOOGL   # Google

# Quick demo (2-3 minutes)
python3 demo.py
```
Full Exectution
python main.py --model transformer --symbol AAPL --sequence-length 60 --epochs 100 --batch-size 256 --learning-rate 3e-4
python main.py --model tft --symbol AAPL --sequence-length 60 --epochs 100 --batch-size 256 --learning-rate 3e-4

### Advanced Options
```bash
# Customize training
python3 main.py --symbol AAPL --epochs 200 --sequence-length 30
python3 main.py --symbol TSLA --batch-size 16 --learning-rate 0.001

# See all options
python3 main.py --help
```

### Output Files
After training, check these folders:
- `results/` - Performance metrics and reports
- `plots/` - Training curves and prediction visualizations  
- `models/` - Saved model weights

## Model Architecture

```
Input: [Batch, Sequence_Length, Features]
   ↓
Input Projection: Linear layer to model dimension
   ↓
Positional Encoding: Add position information
   ↓
Transformer Blocks (Multiple layers):
┌─────────────────────────────────────┐
│  Multi-Head Self-Attention          │
│           ↓                         │
│  Add & Layer Normalization          │
│           ↓                         │
│  Feed-Forward Network               │
│           ↓                         │
│  Add & Layer Normalization          │
└─────────────────────────────────────┘
   ↓
Global Average Pooling: Aggregate sequence information
   ↓
Dense Layer: Final processing
   ↓
Output: Single predicted price
```

**Transformer Components:**
- **Multi-Head Attention**: Focuses on important time periods
- **Positional Encoding**: Understands sequence order
- **Feed-Forward Networks**: Processes patterns
- **Layer Normalization**: Stabilizes training

**Default Configuration:**
- 802K parameters
- 60-day input sequences
- 4 transformer layers
- 8 attention heads

## Performance Metrics

The model reports these metrics:
- **MAE**: Mean Absolute Error (average prediction error)
- **RMSE**: Root Mean Square Error (penalizes large errors)
- **MAPE**: Mean Absolute Percentage Error (scale-independent)
- **R²**: Coefficient of determination (variance explained)

## Project Structure

```
final-project/
├── main.py              # Main pipeline
├── demo.py              # Quick demo
├── requirements.txt     # Dependencies
├── README               # This document
├── src/
│   ├── data/           # Data fetching & preprocessing
│   ├── models/         # Transformer implementation
│   └── evaluation/     # Metrics & visualization
├── results/            # Generated results
├── plots/              # Generated plots
└── models/             # Saved models
```

## Key Features

- **Robust Data Fetching**: Multiple fallback sources + mock data generation
- **Technical Indicators**: SMA, EMA, MACD, RSI, Bollinger Bands
- **Early Stopping**: Prevents overfitting
- **Comprehensive Metrics**: Financial and statistical evaluation
- **Professional Visualizations**: Training curves and prediction plots
- **Easy Configuration**: Command-line arguments or config modification

## Data Pipeline

1. **Fetch**: Downloads OHLCV data from Yahoo Finance (with fallbacks)
2. **Engineer**: Calculates 15+ technical indicators
3. **Process**: Scales, sequences, and splits data
4. **Train**: Transformer learns price patterns
5. **Evaluate**: Tests on unseen data
6. **Visualize**: Generates plots and reports

## Configuration

Edit `main.py` or use command-line arguments:

```python
# In main.py, modify _get_default_config()
config = {
    'symbol': 'AAPL',
    'sequence_length': 60,     # Days of history
    'epochs': 100,             # Training iterations
    'batch_size': 32,          # Training batch size
    'learning_rate': 1e-4,     # Learning rate
}
```

## Examples

**Train a quick model:**
```bash
python3 main.py --symbol AAPL --epochs 50
```

**Train with more data:**
```bash
python3 main.py --symbol TSLA --sequence-length 90
```

**Small model for testing:**
```bash
python3 main.py --symbol GOOGL --batch-size 16 --epochs 25
```

## Troubleshooting

**Common Issues:**
- **"No module named..."**: Run `pip install -r requirements.txt`
- **Data fetching fails**: The system automatically falls back to mock data
- **Out of memory**: Reduce `batch_size` to 16 or 8
- **Slow training**: Reduce `epochs` or `sequence_length`

## Understanding Results

**Good Performance Indicators:**
- MAE < $5 (for typical stock prices)
- MAPE < 20%
- R² > 0.3
- Decreasing training loss

**What the Plots Show:**
- `training_history.png`: Loss curves over epochs
- `predictions_vs_actual.png`: How well predictions match reality

---

**Note**: This implementation uses mock data when real APIs are unavailable, ensuring the pipeline always works for demonstration and development purposes.
