"""
Evaluation Metrics for Stock Price Forecasting

This module implements the evaluation metrics specified in the project proposal:
- MAE (Mean Absolute Error): Measures average absolute prediction error
- RMSE (Root Mean Square Error): Penalizes large deviations
- MAPE (Mean Absolute Percentage Error): Expresses error as a percentage
- R² Score: Indicates how well the model explains variance

Additional metrics for comprehensive evaluation are also included.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class StockForecastingMetrics:
    """
    Comprehensive evaluation metrics for stock price forecasting models
    """
    
    def __init__(self):
        """
        Initialize the metrics calculator
        """
        self.results = {}
        
    def calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error (MAE)
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MAE value
        """
        return mean_absolute_error(y_true, y_pred)
    
    def calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Root Mean Square Error (RMSE)
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            RMSE value
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
        """
        Calculate Mean Absolute Percentage Error (MAPE)
        
        Args:
            y_true: True values
            y_pred: Predicted values
            epsilon: Small value to avoid division by zero
            
        Returns:
            MAPE value (in percentage)
        """
        # Avoid division by zero
        denominator = np.maximum(np.abs(y_true), epsilon)
        mape = np.mean(np.abs((y_true - y_pred) / denominator)) * 100
        return mape
    
    def calculate_r2_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate R² Score (Coefficient of Determination)
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            R² score
        """
        return r2_score(y_true, y_pred)
    
    def calculate_smape(self, y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
        """
        Calculate Symmetric Mean Absolute Percentage Error (SMAPE)
        
        Args:
            y_true: True values
            y_pred: Predicted values
            epsilon: Small value to avoid division by zero
            
        Returns:
            SMAPE value (in percentage)
        """
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon
        smape = np.mean(np.abs(y_true - y_pred) / denominator) * 100
        return smape
    
    def calculate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate directional accuracy (trend prediction accuracy)
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Directional accuracy (0-1)
        """
        if len(y_true) <= 1:
            return 0.0
        
        # Calculate actual and predicted directions
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        
        # Calculate accuracy
        accuracy = np.mean(true_direction == pred_direction)
        return accuracy
    
    def calculate_max_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate maximum absolute error
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Maximum absolute error
        """
        return np.max(np.abs(y_true - y_pred))
    
    def calculate_profit_loss(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            initial_capital: float = 10000, 
                            transaction_cost: float = 0.001) -> Dict[str, float]:
        """
        Calculate profit/loss from trading decisions based on predictions
        
        Args:
            y_true: True stock prices
            y_pred: Predicted stock prices
            initial_capital: Starting capital
            transaction_cost: Transaction cost as percentage
            
        Returns:
            Dictionary with trading performance metrics
        """
        if len(y_true) <= 1:
            return {"total_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0}
        
        # Simple trading strategy: buy if price is predicted to go up, sell if down
        true_returns = np.diff(y_true) / y_true[:-1]
        pred_direction = np.sign(np.diff(y_pred))
        
        # Calculate strategy returns
        strategy_returns = []
        capital = initial_capital
        max_capital = initial_capital
        
        for i, direction in enumerate(pred_direction):
            if direction > 0:  # Buy signal
                return_val = true_returns[i] - transaction_cost
            elif direction < 0:  # Sell signal
                return_val = -true_returns[i] - transaction_cost
            else:  # Hold
                return_val = 0
            
            capital *= (1 + return_val)
            max_capital = max(max_capital, capital)
            strategy_returns.append(return_val)
        
        strategy_returns = np.array(strategy_returns)
        
        # Calculate metrics
        total_return = (capital - initial_capital) / initial_capital
        
        # Sharpe ratio (annualized)
        if np.std(strategy_returns) > 0:
            sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        max_drawdown = (max_capital - capital) / max_capital if max_capital > 0 else 0.0
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "final_capital": capital
        }
    
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            model_name: str = "Model") -> Dict[str, float]:
        """
        Calculate all evaluation metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model for identification
            
        Returns:
            Dictionary containing all metrics
        """
        # Flatten arrays if needed
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # Calculate core metrics
        mae = self.calculate_mae(y_true, y_pred)
        rmse = self.calculate_rmse(y_true, y_pred)
        mape = self.calculate_mape(y_true, y_pred)
        r2 = self.calculate_r2_score(y_true, y_pred)
        
        # Calculate additional metrics
        smape = self.calculate_smape(y_true, y_pred)
        directional_acc = self.calculate_directional_accuracy(y_true, y_pred)
        max_error = self.calculate_max_error(y_true, y_pred)
        
        # Calculate trading metrics
        trading_metrics = self.calculate_profit_loss(y_true, y_pred)
        
        # Combine all metrics
        metrics = {
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "R2_Score": r2,
            "SMAPE": smape,
            "Directional_Accuracy": directional_acc,
            "Max_Error": max_error,
            **{f"Trading_{k}": v for k, v in trading_metrics.items()}
        }
        
        # Store results
        self.results[model_name] = metrics
        
        return metrics
    
    def compare_models(self, models_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Compare multiple models' performance
        
        Args:
            models_results: Dictionary of model results
            
        Returns:
            Comparison DataFrame
        """
        comparison_df = pd.DataFrame(models_results).T
        
        # Round values for better readability
        comparison_df = comparison_df.round(4)
        
        # Rank models for each metric (lower is better for error metrics)
        error_metrics = ["MAE", "RMSE", "MAPE", "SMAPE", "Max_Error", "Trading_max_drawdown"]
        for metric in error_metrics:
            if metric in comparison_df.columns:
                comparison_df[f"{metric}_Rank"] = comparison_df[metric].rank(ascending=True)
        
        # Higher is better for these metrics
        performance_metrics = ["R2_Score", "Directional_Accuracy", "Trading_total_return", "Trading_sharpe_ratio"]
        for metric in performance_metrics:
            if metric in comparison_df.columns:
                comparison_df[f"{metric}_Rank"] = comparison_df[metric].rank(ascending=False)
        
        return comparison_df
    
    def plot_predictions_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  model_name: str = "Model", 
                                  save_path: Optional[str] = None):
        """
        Plot predictions vs actual values
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Model name for title
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        # Time series plot
        plt.subplot(2, 2, 1)
        plt.plot(y_true, label='Actual', linewidth=2)
        plt.plot(y_pred, label='Predicted', linewidth=2, alpha=0.8)
        plt.title(f'{model_name} - Predictions vs Actual')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Scatter plot
        plt.subplot(2, 2, 2)
        plt.scatter(y_true, y_pred, alpha=0.6)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{model_name} - Scatter Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Residuals plot
        plt.subplot(2, 2, 3)
        residuals = y_true - y_pred
        plt.plot(residuals, linewidth=1)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        plt.title(f'{model_name} - Residuals')
        plt.xlabel('Time')
        plt.ylabel('Residuals')
        plt.grid(True, alpha=0.3)
        
        # Residuals distribution
        plt.subplot(2, 2, 4)
        plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        plt.title(f'{model_name} - Residuals Distribution')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_metrics_comparison(self, models_results: Dict[str, Dict[str, float]], 
                               save_path: Optional[str] = None):
        """
        Plot comparison of different models' metrics
        
        Args:
            models_results: Dictionary of model results
            save_path: Path to save the plot
        """
        comparison_df = pd.DataFrame(models_results).T
        
        # Select core metrics for plotting
        core_metrics = ["MAE", "RMSE", "MAPE", "R2_Score"]
        available_metrics = [m for m in core_metrics if m in comparison_df.columns]
        
        if not available_metrics:
            print("No core metrics available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(available_metrics):
            if i < len(axes):
                ax = axes[i]
                values = comparison_df[metric]
                bars = ax.bar(values.index, values.values, alpha=0.7)
                
                # Color bars (green for good, red for bad)
                if metric in ["MAE", "RMSE", "MAPE"]:  # Lower is better
                    colors = ['green' if v == values.min() else 'lightcoral' for v in values.values]
                else:  # Higher is better
                    colors = ['green' if v == values.max() else 'lightcoral' for v in values.values]
                
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
                
                ax.set_title(f'{metric} Comparison')
                ax.set_ylabel(metric)
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.4f}', ha='center', va='bottom')
        
        # Remove empty subplots
        for i in range(len(available_metrics), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, models_results: Dict[str, Dict[str, float]], 
                       save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive evaluation report
        
        Args:
            models_results: Dictionary of model results
            save_path: Path to save the report
            
        Returns:
            Report string
        """
        report = []
        report.append("=" * 80)
        report.append("STOCK PRICE FORECASTING EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Models overview
        report.append(f"Number of models evaluated: {len(models_results)}")
        report.append(f"Models: {', '.join(models_results.keys())}")
        report.append("")
        
        # Detailed metrics for each model
        for model_name, metrics in models_results.items():
            report.append(f"--- {model_name.upper()} ---")
            report.append(f"MAE:                  {metrics.get('MAE', 'N/A'):.6f}")
            report.append(f"RMSE:                 {metrics.get('RMSE', 'N/A'):.6f}")
            report.append(f"MAPE:                 {metrics.get('MAPE', 'N/A'):.2f}%")
            report.append(f"R² Score:             {metrics.get('R2_Score', 'N/A'):.6f}")
            report.append(f"SMAPE:                {metrics.get('SMAPE', 'N/A'):.2f}%")
            report.append(f"Directional Accuracy: {metrics.get('Directional_Accuracy', 'N/A'):.4f}")
            report.append(f"Max Error:            {metrics.get('Max_Error', 'N/A'):.6f}")
            
            # Trading metrics
            if 'Trading_total_return' in metrics:
                report.append(f"Trading Return:       {metrics['Trading_total_return']:.2%}")
                report.append(f"Sharpe Ratio:         {metrics.get('Trading_sharpe_ratio', 'N/A'):.4f}")
                report.append(f"Max Drawdown:         {metrics.get('Trading_max_drawdown', 'N/A'):.2%}")
            
            report.append("")
        
        # Best model analysis
        comparison_df = self.compare_models(models_results)
        
        report.append("--- BEST MODELS BY METRIC ---")
        
        # Find best models for each metric
        metrics_to_check = ["MAE", "RMSE", "MAPE", "R2_Score", "Directional_Accuracy"]
        for metric in metrics_to_check:
            if metric in comparison_df.columns:
                if metric in ["MAE", "RMSE", "MAPE"]:  # Lower is better
                    best_model = comparison_df[metric].idxmin()
                    best_value = comparison_df[metric].min()
                else:  # Higher is better
                    best_model = comparison_df[metric].idxmax()
                    best_value = comparison_df[metric].max()
                
                report.append(f"{metric:20s}: {best_model} ({best_value:.6f})")
        
        report.append("")
        
        # Summary recommendations
        report.append("--- SUMMARY AND RECOMMENDATIONS ---")
        
        # Calculate overall performance score
        normalized_scores = {}
        for model in models_results.keys():
            score = 0
            count = 0
            
            # Error metrics (lower is better)
            for metric in ["MAE", "RMSE", "MAPE"]:
                if metric in comparison_df.columns:
                    # Normalize to 0-1 scale (inverted for error metrics)
                    max_val = comparison_df[metric].max()
                    min_val = comparison_df[metric].min()
                    if max_val != min_val:
                        normalized = 1 - (comparison_df.loc[model, metric] - min_val) / (max_val - min_val)
                        score += normalized
                        count += 1
            
            # Performance metrics (higher is better)
            for metric in ["R2_Score", "Directional_Accuracy"]:
                if metric in comparison_df.columns:
                    max_val = comparison_df[metric].max()
                    min_val = comparison_df[metric].min()
                    if max_val != min_val:
                        normalized = (comparison_df.loc[model, metric] - min_val) / (max_val - min_val)
                        score += normalized
                        count += 1
            
            normalized_scores[model] = score / count if count > 0 else 0
        
        # Best overall model
        best_overall = max(normalized_scores, key=normalized_scores.get)
        report.append(f"Best Overall Model: {best_overall} (Score: {normalized_scores[best_overall]:.4f})")
        report.append("")
        
        # Convert to string
        report_str = "\n".join(report)
        
        # Save to file if requested
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_str)
        
        return report_str

# Convenience functions
def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> Dict[str, float]:
    """
    Quick evaluation function for a single model
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Model name
        
    Returns:
        Dictionary of metrics
    """
    evaluator = StockForecastingMetrics()
    return evaluator.calculate_all_metrics(y_true, y_pred, model_name)

def compare_models_quick(results: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """
    Quick comparison of multiple models
    
    Args:
        results: Dictionary with model_name: (y_true, y_pred) pairs
        
    Returns:
        Comparison DataFrame
    """
    evaluator = StockForecastingMetrics()
    models_results = {}
    
    for model_name, (y_true, y_pred) in results.items():
        models_results[model_name] = evaluator.calculate_all_metrics(y_true, y_pred, model_name)
    
    return evaluator.compare_models(models_results)

if __name__ == "__main__":
    # Demo of evaluation metrics
    print("=== Stock Forecasting Metrics Demo ===")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    
    # True values (simulated stock prices)
    y_true = 100 + np.cumsum(np.random.normal(0, 1, n_samples))
    
    # Simulated predictions (with some error)
    y_pred_good = y_true + np.random.normal(0, 0.5, n_samples)
    y_pred_bad = y_true + np.random.normal(0, 2, n_samples)
    
    # Evaluate models
    evaluator = StockForecastingMetrics()
    
    good_metrics = evaluator.calculate_all_metrics(y_true, y_pred_good, "Good Model")
    bad_metrics = evaluator.calculate_all_metrics(y_true, y_pred_bad, "Bad Model")
    
    # Compare models
    comparison = evaluator.compare_models({"Good Model": good_metrics, "Bad Model": bad_metrics})
    print("\nModel Comparison:")
    print(comparison[["MAE", "RMSE", "MAPE", "R2_Score"]].round(4))
    
    # Generate report
    report = evaluator.generate_report({"Good Model": good_metrics, "Bad Model": bad_metrics})
    print("\n" + report) 