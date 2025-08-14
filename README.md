# Stock Price Forecasting with Transformer & Temporal Fusion Transformer (TFT)

This project implements **two state-of-the-art deep learning architectures** for short-term stock price forecasting:
1. **Transformer** (TensorFlow/Keras)
2. **Temporal Fusion Transformer (TFT)** (PyTorch)

It supports multiple stock symbols, customizable hyperparameters, automatic technical indicator generation, and detailed evaluation metrics.

---

## Features

- **Dual-Model Support** → Transformer (fast & accurate) and TFT (interpretable, multi-horizon forecasting)
- **Configurable Parameters** → Sequence length, features, batch size, learning rate, patience
- **Automatic Mixed Precision (AMP)** → Faster GPU training with reduced memory usage
- **Dynamic Close Index Detection** → No hardcoded target index
- **Rich Feature Engineering** → SMA, EMA, MACD, RSI, Bollinger Bands
- **Complete Evaluation** → MAE, RMSE, MAPE, R², Directional Accuracy, Sharpe Ratio, Total Return
- **Live Demo Script** (`demo.py`) → Quick training/inference without full training

---

## Installation

### Prerequisites
- Python **3.8+**
- pip

### Setup
```bash
# 1. Clone the repository
git clone <your-repo-url>
cd final-project-main

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage

### **Train & Evaluate**

#### Transformer (AAPL example)
```bash
python main.py --model transformer --symbol AAPL --sequence-length 60 --batch-size 256 --epochs 50 --learning-rate 1e-4 --patience 7
```

#### TFT (NVDA example)
```bash
python main.py --model tft --symbol NVDA --sequence-length 60 --batch-size 256 --epochs 50 --learning-rate 3e-4 --patience 7
```

**Key CLI Arguments**
| Argument | Description | Example |
|----------|-------------|---------|
| `--model` | Model type (`transformer` / `tft`) | `--model transformer` |
| `--symbol` | Stock ticker | `--symbol AAPL` |
| `--sequence-length` | Lookback window size | `--sequence-length 60` |
| `--prediction-horizon` | Days ahead to predict | `--prediction-horizon 1` |
| `--epochs` | Training epochs | `--epochs 50` |
| `--batch-size` | Training batch size | `--batch-size 256` |
| `--learning-rate` | Learning rate | `--learning-rate 1e-4` |
| `--patience` | Early stopping patience | `--patience 7` |

---

### **Quick Demo**
Run a short demo with reduced parameters:
```bash
python demo.py --model transformer --symbol AAPL
python demo.py --model tft --symbol NVDA
```

Run only data processing (no training):
```bash
python demo.py --data-only --symbol TSLA
```

---

## Final Results (Interim → Final)

| Model       | Symbol | R² Score (Interim) | R² Score (Final) | MAE Final | RMSE Final |
|-------------|--------|-------------------|------------------|-----------|------------|
| Transformer | AAPL   | -0.4468           | **0.9066**       | 0.079     | 0.101      | 
| TFT         | NVDA   | 0.0707            | 0.1257           | 0.602     | 0.754      | 

Transformer → **Major accuracy improvement** after efficiency optimizations
TFT → Minor gains, requires further hyperparameter tuning.

---

## Project Structure

```
final-project-main/
├── main.py               # Main training & evaluation pipeline
├── trainer.py            # PyTorch training loop for TFT
├── transformer.py        # Transformer model
├── tft.py                # Temporal Fusion Transformer model
├── stock_preprocessor.py # Data preprocessing & feature engineering
├── demo.py               # Quick demo script
├── requirements.txt      # Python dependencies
├── results/              # Metrics and reports
├── plots/                # Generated plots
└── models/               # Saved model checkpoints
```

---

## Model Architectures

**Transformer**
```
[Input Sequence] → Input Projection → Positional Encoding
→ Multi-Head Attention → Layer Norm → Feed-Forward → Layer Norm
→ Global Average Pooling → Dense → Output
```

**TFT**
```
Static Variable Encoder + Historical Encoder
→ Multi-Head Attention + LSTM Layers
→ Temporal Decoder → Output Layer
```

---

## License
MIT License – free to use, modify, and distribute.

---

## Acknowledgements
- Yahoo Finance for data
- PyTorch, TensorFlow/Keras for model implementation
- Original TFT paper: *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*
