"""
Stock Data Preprocessor for Transformer Models

This module handles preprocessing of stock data for time series forecasting,
including feature engineering, scaling, and sequence creation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class StockDataPreprocessor:
    """
    Comprehensive preprocessor for stock data to prepare for Transformer/TFT models
    """
    
    def __init__(self, 
                 sequence_length: int = 30,
                 prediction_horizon: int = 1,
                 features: List[str] = None,
                 scaler_type: str = 'standard'):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler_type = scaler_type
        
        # Default features
        self.features = features or ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Initialize scalers
        self.feature_scalers = {}
        self.target_scaler = None
        
        # Store processed data
        self.processed_data = None
        self.sequences = None
        self.targets = None

    # ----------------------------
    # MultiIndex and column safety
    # ----------------------------
    def _flatten_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Flatten MultiIndex columns from yfinance output."""
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [
                "_".join([str(c) for c in col if c]).strip("_") for col in data.columns
            ]
        return data

    def _safe_series(self, data: pd.DataFrame, colname: str) -> pd.Series:
        """Safely get a column even if Yahoo renames it (e.g., Close_AAPL)."""
        if colname in data.columns:
            return data[colname]
        matches = [c for c in data.columns if colname in c]
        if matches:
            return data[matches[0]]
        raise ValueError(f"Column '{colname}' not found in DataFrame")

    # ----------------------------
    # Technical Indicators
    # ----------------------------
    def _create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        data = self._flatten_columns(df.copy())

        # Moving averages
        data['SMA_5'] = self._safe_series(data, 'Close').rolling(window=5).mean()
        data['SMA_10'] = self._safe_series(data, 'Close').rolling(window=10).mean()
        data['SMA_20'] = self._safe_series(data, 'Close').rolling(window=20).mean()

        # Exponential moving averages
        data['EMA_12'] = self._safe_series(data, 'Close').ewm(span=12).mean()
        data['EMA_26'] = self._safe_series(data, 'Close').ewm(span=26).mean()

        # MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_signal'] = data['MACD'].ewm(span=9).mean()

        # RSI
        delta = self._safe_series(data, 'Close').diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands (fixed)
        bb_period = 20
        data['BB_middle'] = self._safe_series(data, 'Close').rolling(window=bb_period).mean()
        bb_std = self._safe_series(data, 'Close').rolling(window=bb_period).std().astype(float)
        data['BB_upper'] = data['BB_middle'] + (bb_std * 2)
        data['BB_lower'] = data['BB_middle'] - (bb_std * 2)
        data['BB_width'] = data['BB_upper'] - data['BB_lower']
        data['BB_position'] = (self._safe_series(data, 'Close') - data['BB_lower']) / data['BB_width']

        # Volatility
        data['Volatility'] = self._safe_series(data, 'Close').rolling(window=20).std()

        # Price returns
        data['Returns'] = self._safe_series(data, 'Close').pct_change()
        data['Log_Returns'] = np.log(self._safe_series(data, 'Close') / self._safe_series(data, 'Close').shift(1))

        # Volume indicators
        data['Volume_SMA'] = self._safe_series(data, 'Volume').rolling(window=20).mean()
        data['Volume_Ratio'] = self._safe_series(data, 'Volume') / data['Volume_SMA']

        # Price position
        data['High_Low_Ratio'] = self._safe_series(data, 'High') / self._safe_series(data, 'Low')
        data['Close_Open_Ratio'] = self._safe_series(data, 'Close') / self._safe_series(data, 'Open')

        return data

    # ----------------------------
    # Data Cleaning & Scaling
    # ----------------------------
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy().fillna(method='ffill').fillna(method='bfill')
        if data.isnull().sum().sum() > 0:
            print(f"Warning: Dropping {data.isnull().sum().sum()} NaN values")
            data = data.dropna()
        return data

    def _create_scalers(self, data: pd.DataFrame, target_column: str = 'Close'):
        ScalerClass = StandardScaler if self.scaler_type == 'standard' else MinMaxScaler
        for column in data.columns:
            if column != target_column:
                self.feature_scalers[column] = ScalerClass()
                self.feature_scalers[column].fit(data[column].values.reshape(-1, 1))
        self.target_scaler = ScalerClass()
        self.target_scaler.fit(data[target_column].values.reshape(-1, 1))

    def _scale_data(self, data: pd.DataFrame, target_column: str = 'Close') -> pd.DataFrame:
        scaled_data = data.copy()
        for column in data.columns:
            if column != target_column and column in self.feature_scalers:
                scaled_data[column] = self.feature_scalers[column].transform(
                    data[column].values.reshape(-1, 1)
                ).flatten()
        scaled_data[target_column] = self.target_scaler.transform(
            data[target_column].values.reshape(-1, 1)
        ).flatten()
        return scaled_data

    # ----------------------------
    # Sequence Creation
    # ----------------------------
    def _create_sequences(self, data: pd.DataFrame, target_column: str = 'Close') -> Tuple[np.ndarray, np.ndarray]:
        feature_columns = [col for col in data.columns if col != target_column]
        X_data = data[feature_columns].values
        y_data = data[target_column].values
        X, y = [], []
        for i in range(len(data) - self.sequence_length - self.prediction_horizon + 1):
            X.append(X_data[i:(i + self.sequence_length)])
            if self.prediction_horizon == 1:
                y.append(y_data[i + self.sequence_length])
            else:
                y.append(y_data[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon])
        return np.array(X), np.array(y)

    # ----------------------------
    # Main Preprocessing Pipeline
    # ----------------------------
    def preprocess_data(self, 
                    data: pd.DataFrame,
                    target_column: str = 'Close',
                    add_technical_indicators: bool = True,
                    test_size: float = 0.2,
                    validation_size: float = 0.1) -> Dict:
        """
        Complete preprocessing pipeline
        """
        print("Starting data preprocessing...")

        # Step 1: Add technical indicators
        if add_technical_indicators:
            print("Adding technical indicators...")
            processed_data = self._create_technical_indicators(data)
        else:
            processed_data = data.copy()

        # Step 2: Handle missing values
        print("Handling missing values...")
        processed_data = self._handle_missing_values(processed_data)

        # Step 3: Select features
        if self.features:
            # Ensure target column is included
            all_features = list(set(self.features + [target_column]))

            # Match columns by partial name (case-insensitive) to handle yfinance suffixes like "_AAPL"
            available_features = []
            for feat in all_features:
                matches = [col for col in processed_data.columns if feat.lower() in col.lower()]
                if matches:
                    available_features.append(matches[0])  # pick the first match

            processed_data = processed_data[available_features]
            print(f"Selected features: {available_features}")

            # Also make sure we update the actual target_column name if it has a suffix
            target_matches = [col for col in processed_data.columns if target_column.lower() in col.lower()]
            if target_matches:
                target_column = target_matches[0]

        # Step 4: Create and fit scalers
        print("Fitting scalers...")
        self._create_scalers(processed_data, target_column)

        # Step 5: Scale data
        print("Scaling data...")
        scaled_data = self._scale_data(processed_data, target_column)

        # Step 6: Create sequences
        print("Creating sequences...")
        X, y = self._create_sequences(scaled_data, target_column)

        # Step 7: Split data
        print("Splitting data...")
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )

        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, shuffle=False
        )

        self.processed_data = processed_data
        self.sequences = X
        self.targets = y

        result = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': [col for col in scaled_data.columns if col != target_column],
            'target_name': target_column,
            'scaler_info': {
                'feature_scalers': self.feature_scalers,
                'target_scaler': self.target_scaler
            }
        }

        print(f"Preprocessing complete!")
        print(f"Train set shape: X={X_train.shape}, y={y_train.shape}")
        print(f"Validation set shape: X={X_val.shape}, y={y_val.shape}")
        print(f"Test set shape: X={X_test.shape}, y={y_test.shape}")

        return result

    # ----------------------------
    # Utilities
    # ----------------------------
    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        if self.target_scaler is None:
            raise ValueError("Target scaler not fitted. Run preprocess_data first.")
        original_shape = predictions.shape
        predictions_reshaped = predictions.reshape(-1, 1)
        inverse_predictions = self.target_scaler.inverse_transform(predictions_reshaped)
        return inverse_predictions.reshape(original_shape)

    def get_feature_info(self) -> Dict:
        if self.processed_data is None:
            return {"error": "No data processed yet"}
        return {
            "sequence_length": self.sequence_length,
            "prediction_horizon": self.prediction_horizon,
            "total_features": len(self.features) if self.features else len(self.processed_data.columns) - 1,
            "feature_names": self.features or list(self.processed_data.columns),
            "data_shape": self.processed_data.shape,
            "scaler_type": self.scaler_type,
            "sequences_shape": self.sequences.shape if self.sequences is not None else None,
            "targets_shape": self.targets.shape if self.targets is not None else None
        }

# Quick helper
def preprocess_stock_data(data: pd.DataFrame, sequence_length: int = 30, prediction_horizon: int = 1, features: List[str] = None) -> Dict:
    return StockDataPreprocessor(sequence_length=sequence_length, prediction_horizon=prediction_horizon, features=features).preprocess_data(data)

