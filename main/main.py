"""
Stock Price Forecasting Pipeline

Run either model:
- Transformer (TensorFlow/Keras):   python main.py --model transformer --symbol AAPL --sequence-length 60
- TFT (PyTorch):                    python main.py --model tft --symbol AAPL --sequence-length 60
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

# local imports
sys.path.append("src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from data.data_fetcher import StockDataFetcher
from data.stock_preprocessor import StockDataPreprocessor
from models.trainer import ModelTrainer
from evaluation.metrics import StockForecastingMetrics
from src.models.transformer import Transformer as TimeSeriesTransformer
from src.models.tft import TFTSingleStep


class StockForecastingPipeline:
    def __init__(self, config: dict):
        self.config = self._with_defaults(config)
        self.data_fetcher = StockDataFetcher()
        self.trainer = ModelTrainer(verbose=True)
        self.metrics = StockForecastingMetrics()
        os.makedirs("models", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        os.makedirs("plots", exist_ok=True)

        # model save path per model type
        if self.config["model"] == "transformer":
            self.model_path = "models/transformer_model.keras"
        else:
            self.model_path = "models/tft_model.pt"

    def _with_defaults(self, cfg):
        base = {
            "symbol": "AAPL",
            "start_date": "2019-01-01",
            "end_date": "2024-01-01",
            "sequence_length": 60,
            "prediction_horizon": 1,
            "scaler_type": "standard",
            "features": ["Open", "High", "Low", "Close", "Volume"],
            "model": "transformer",
            "model_config": {
                "d_model": 128,
                "num_heads": 8,
                "num_layers": 4,
                "dff": 512,
                "dropout_rate": 0.1,
            },
            "epochs": 100,
            "batch_size": 128,
            "learning_rate": 3e-4,
            "patience": 15,
        }
        base.update(cfg or {})
        return base

    def build_model(self, input_features: int, close_index: int = 3):
        mcfg = self.config["model_config"]
        if self.config["model"] == "transformer":
            return TimeSeriesTransformer(
                seq_len=self.config["sequence_length"],
                d_model=mcfg["d_model"],
                num_heads=mcfg["num_heads"],
                num_layers=mcfg["num_layers"],
                dff=mcfg["dff"],
                input_features=input_features,
                dropout_rate=mcfg["dropout_rate"],
                close_index=close_index
            )
        else:
            return TFTSingleStep(
                num_features=input_features,
                d_model=mcfg["d_model"],
                d_hidden=mcfg["dff"],
                n_heads=mcfg["num_heads"],
                dropout=mcfg["dropout_rate"],
                static_num_ids=None,
                static_embed_dim=16,
                lstm_hidden=mcfg["d_model"],
                lstm_layers=1,
            )

    def run_complete_pipeline(self):
        print("Starting Stock Forecasting Pipeline")
        print("=" * 60)

        # 1) Fetch data
        print("\nStep 1: Fetching Stock Data")
        raw = self.data_fetcher.fetch_stock_data(
            symbol=self.config["symbol"],
            start_date=self.config["start_date"],
            end_date=self.config["end_date"],
        )
        print(f"Fetched {len(raw)} records for {self.config['symbol']}")

        # 2) Preprocess
        print("\nStep 2: Preprocessing Data")
        pre = StockDataPreprocessor(
            sequence_length=self.config["sequence_length"],
            scaler_type=self.config["scaler_type"],
            features=self.config.get("features")
        ).preprocess_data(data=raw, target_column="Close")

        X_train, y_train = pre["X_train"], pre["y_train"]
        X_val, y_val = pre["X_val"], pre["y_val"]
        X_test, y_test = pre["X_test"], pre["y_test"]
        print("Preprocessed data:")
        print(f"   Training: {X_train.shape[0]} samples")
        print(f"   Validation: {X_val.shape[0]} samples")
        print(f"   Test: {X_test.shape[0]} samples")
        print(f"   Features: {X_train.shape[2]}")

        # 3) Build & Train
        print(f"\nStep 3: Training {self.config['model'].upper()} Model")
        feature_names = pre["feature_names"]
        close_index = next((i for i, n in enumerate(feature_names) if "close" in n.lower()), 3)

        # Build model
        model = self.build_model(
            input_features=X_train.shape[2],
            close_index=close_index
        )
        self.trainer.compile_model(model, learning_rate=self.config["learning_rate"])
        train_out = self.trainer.train(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            model_save_path=self.model_path,
        )
        print(f"Training completed in {train_out['training_time']:.2f}s")
        print(f"Best validation loss: {train_out['final_val_loss']:.6f}")

        # 4) Load best model & Evaluate
        print("\nStep 4: Model Evaluation")
        best_model = self.trainer.load_model(train_out["best_model_path"])
        y_pred = self.trainer.predict(best_model, X_test)
        base_metrics = self.trainer.evaluate(best_model, X_test, y_test)
        fin_metrics = self.metrics.calculate_all_metrics(
            y_test.flatten(), y_pred.flatten(), self.config["model"]
        )

        # Merge metrics and save json
        all_results = {
            "model": self.config["model"],
            "symbol": self.config["symbol"],
            "config": self.config,
            "metrics": {**base_metrics, **fin_metrics},
        }
        results_path = f"results/results_{self.config['model']}.json"
        self.trainer.save_results_json(results_path, all_results)
        print(f"Saved results to {results_path}")

        # 5) Save plots
        print("\nStep 5: Creating Visualizations")
        self._save_plots(
            model_name=self.config["model"],
            history=train_out["history"],
            y_true=y_test,
            y_pred=y_pred,
        )

        print("\nPipeline completed successfully!")
        return {
            "model": best_model,
            "results_json": results_path,
            "y_pred": y_pred,
            "y_true": y_test,
            "history": train_out["history"],
        }

    def _save_plots(self, model_name: str, history: dict, y_true: np.ndarray, y_pred: np.ndarray):
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        os.makedirs("plots", exist_ok=True)

        # 1️⃣ Loss curves
        plt.figure()
        plt.plot(history.get("loss", []), label="train_loss")
        plt.plot(history.get("val_loss", []), label="val_loss")
        plt.title(f"{model_name.upper()} - Training History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        path1 = f"plots/{model_name}_history.png"
        plt.savefig(path1, bbox_inches="tight")
        plt.close()

        # 2️⃣ Combined 4-panel plot
        residuals = y_true - y_pred

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        # Predictions vs Actual
        axs[0, 0].plot(y_true, label="Actual")
        axs[0, 0].plot(y_pred, label="Predicted")
        axs[0, 0].set_title(f"{model_name.upper()} - Predictions vs Actual")
        axs[0, 0].set_xlabel("Time")
        axs[0, 0].set_ylabel("Stock Price")
        axs[0, 0].legend()

        # Scatter plot
        axs[0, 1].scatter(y_true, y_pred, alpha=0.6)
        axs[0, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label="Perfect Prediction")
        axs[0, 1].set_title(f"{model_name.upper()} - Scatter Plot")
        axs[0, 1].set_xlabel("Actual Values")
        axs[0, 1].set_ylabel("Predicted Values")
        axs[0, 1].legend()

        # Residuals over time
        axs[1, 0].plot(residuals, label="Residuals")
        axs[1, 0].axhline(0, color='r', linestyle='--')
        axs[1, 0].set_title(f"{model_name.upper()} - Residuals")
        axs[1, 0].set_xlabel("Time")
        axs[1, 0].set_ylabel("Residuals")

        # Residuals distribution
        axs[1, 1].hist(residuals, bins=30)
        axs[1, 1].set_title(f"{model_name.upper()} - Residuals Distribution")
        axs[1, 1].set_xlabel("Residuals")
        axs[1, 1].set_ylabel("Frequency")

        plt.tight_layout()
        path2 = f"plots/{model_name}_analysis.png"
        plt.savefig(path2, bbox_inches="tight")
        plt.close()

        print(f"Saved plots:\n - {path1}\n - {path2}")


def parse_args():
    p = argparse.ArgumentParser(description="Stock Price Forecasting")
    p.add_argument("--model", type=str, default="transformer", choices=["transformer", "tft"])
    p.add_argument("--symbol", type=str, default="AAPL")
    p.add_argument("--sequence-length", type=int, default=60)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--scaler-type", type=str, default="standard", choices=["standard", "minmax"])
    p.add_argument("--features", type=str, default=None,
                help="Comma-separated features, e.g. 'Open,High,Low,Close,Volume,RSI'")
    return p.parse_args()


def main():
    args = parse_args()
    config = {
        "model": args.model,
        "symbol": args.symbol,
        "sequence_length": args.sequence_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "scaler_type": args.scaler_type,
        "features": [f.strip() for f in args.features.split(",")] if args.features else None,
    }
    pipe = StockForecastingPipeline(config)
    pipe.run_complete_pipeline()

import matplotlib.pyplot as plt
import numpy as np

def plot_model_results(y_true, y_pred, model_name="Model"):
    residuals = y_true - y_pred

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # 1. Predictions vs Actual
    axs[0, 0].plot(y_true, label="Actual")
    axs[0, 0].plot(y_pred, label="Predicted")
    axs[0, 0].set_title(f"{model_name} - Predictions vs Actual")
    axs[0, 0].set_xlabel("Time")
    axs[0, 0].set_ylabel("Stock Price")
    axs[0, 0].legend()

    # 2. Scatter Plot
    axs[0, 1].scatter(y_true, y_pred)
    axs[0, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label="Perfect Prediction")
    axs[0, 1].set_title(f"{model_name} - Scatter Plot")
    axs[0, 1].set_xlabel("Actual Values")
    axs[0, 1].set_ylabel("Predicted Values")
    axs[0, 1].legend()

    # 3. Residuals over Time
    axs[1, 0].plot(residuals)
    axs[1, 0].axhline(0, color="red", linestyle="--")
    axs[1, 0].set_title(f"{model_name} - Residuals")
    axs[1, 0].set_xlabel("Time")
    axs[1, 0].set_ylabel("Residuals")

    # 4. Residuals Distribution
    axs[1, 1].hist(residuals, bins=30)
    axs[1, 1].set_title(f"{model_name} - Residuals Distribution")
    axs[1, 1].set_xlabel("Residuals")
    axs[1, 1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
