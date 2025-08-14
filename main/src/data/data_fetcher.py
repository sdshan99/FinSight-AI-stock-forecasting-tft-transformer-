"""
Stock Data Fetcher with SSL Issue Handling and Multiple Data Sources

This module provides robust data fetching capabilities with fallback options
for SSL certificate issues and alternative data sources.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import ssl
import requests
import urllib3
from datetime import datetime, timedelta
import warnings
import os

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')

class StockDataFetcher:
    def __init__(self):
        self.data_sources = ['yfinance', 'mock', 'csv']
        self.ssl_context = self._setup_ssl_context()
    
    def fetch_stock_data(self, symbol, start_date, end_date):
        """
        Simple interface for fetching stock data
        """
        return self.fetch_data_with_fallback([symbol], start_date, end_date)
    
    def _setup_ssl_context(self):
        """Setup SSL context to handle certificate issues"""
        try:
            # Create unverified SSL context
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            return ssl_context
        except Exception as e:
            print(f"SSL context setup failed: {e}")
            return None
    
    def fetch_stock_data_yfinance_fixed(self, tickers, start_date, end_date, retries=3):
        """
        Fetch stock data using yfinance without breaking its internal session handling.
        """
        for attempt in range(retries):
            try:
                print(f"Attempt {attempt + 1}: Fetching data with yfinance...")

                # Temporarily disable SSL verification if needed
                import ssl
                ssl._create_default_https_context = ssl._create_unverified_context

                # ✅ Let yfinance manage its own session
                data = yf.download(
                    tickers,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    threads=True
                )

                if not data.empty:
                    print("✅ Successfully fetched data with yfinance!")
                    return data

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    print("Trying again...")
                else:
                    print("❌ All yfinance attempts failed. Switching to alternative method.")

        return None

        
    def fetch_stock_data_alternative_api(self, tickers, start_date, end_date):
        """
        Alternative API approach using direct requests
        """
        try:
            print("Trying alternative API approach...")
            
            # For demonstration, we'll create a simple API call
            # In practice, you might use Alpha Vantage, Quandl, or other APIs
            base_url = "https://query1.finance.yahoo.com/v7/finance/download/"
            
            all_data = {}
            
            for ticker in tickers if isinstance(tickers, list) else [tickers]:
                try:
                    # Convert dates to timestamps
                    start_ts = int(pd.Timestamp(start_date).timestamp())
                    end_ts = int(pd.Timestamp(end_date).timestamp())
                    
                    url = f"{base_url}{ticker}"
                    params = {
                        'period1': start_ts,
                        'period2': end_ts,
                        'interval': '1d',
                        'events': 'history'
                    }
                    
                    # Make request with SSL verification disabled
                    response = requests.get(url, params=params, verify=False, timeout=30)
                    
                    if response.status_code == 200:
                        # Parse CSV data
                        data = pd.read_csv(pd.io.common.StringIO(response.text))
                        data['Date'] = pd.to_datetime(data['Date'])
                        data.set_index('Date', inplace=True)
                        all_data[ticker] = data
                        print(f"Successfully fetched {ticker}")
                    else:
                        print(f"Failed to fetch {ticker}: HTTP {response.status_code}")
                        
                except Exception as e:
                    print(f"Error fetching {ticker}: {e}")
            
            if all_data:
                # Combine data similar to yfinance format
                if len(all_data) == 1:
                    return list(all_data.values())[0]
                else:
                    # Multi-level columns like yfinance
                    combined = pd.concat(all_data, axis=1)
                    return combined
            
        except Exception as e:
            print(f"Alternative API failed: {e}")
        
        return None
    
    def generate_mock_data(self, tickers, start_date, end_date):
        """
        Generate realistic mock stock data for development and testing
        """
        print("Generating mock stock data for development...")
        
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        # Filter to weekdays only (trading days)
        trading_days = date_range[date_range.weekday < 5]
        
        if isinstance(tickers, str):
            tickers = [tickers]
        
        # Stock characteristics for realistic simulation
        stock_params = {
            'AAPL': {'start_price': 150, 'volatility': 0.25, 'trend': 0.0002},
            'GOOGL': {'start_price': 2500, 'volatility': 0.28, 'trend': 0.0001},
            'MSFT': {'start_price': 300, 'volatility': 0.22, 'trend': 0.0003},
            'AMZN': {'start_price': 3200, 'volatility': 0.30, 'trend': 0.0001},
            'TSLA': {'start_price': 800, 'volatility': 0.45, 'trend': 0.0005},
            'NVDA': {'start_price': 400, 'volatility': 0.35, 'trend': 0.0008},
            'META': {'start_price': 280, 'volatility': 0.32, 'trend': 0.0002},
            'NFLX': {'start_price': 450, 'volatility': 0.35, 'trend': 0.0001},
            'AMD': {'start_price': 90, 'volatility': 0.40, 'trend': 0.0006},
            'CRM': {'start_price': 200, 'volatility': 0.30, 'trend': 0.0003}
        }
        
        all_data = {}
        
        for ticker in tickers:
            params = stock_params.get(ticker, {'start_price': 100, 'volatility': 0.25, 'trend': 0.0002})
            
            # Generate price series using geometric Brownian motion
            np.random.seed(42 + hash(ticker) % 100)  # Reproducible but different for each stock
            
            n_days = len(trading_days)
            dt = 1.0 / 252  # Daily time step (252 trading days per year)
            
            # Generate returns
            returns = np.random.normal(
                params['trend'], 
                params['volatility'] * np.sqrt(dt), 
                n_days
            )
            
            # Generate price series
            prices = [params['start_price']]
            for i in range(1, n_days):
                next_price = prices[-1] * np.exp(returns[i])
                prices.append(next_price)
            
            prices = np.array(prices)
            
            # Generate OHLCV data
            data = pd.DataFrame(index=trading_days)
            
            # Close prices
            data['Close'] = prices
            
            # Generate Open, High, Low based on Close
            daily_noise = np.random.normal(0, 0.005, n_days)  # Small intraday movements
            
            data['Open'] = data['Close'].shift(1).fillna(params['start_price']) * (1 + daily_noise)
            
            # High and Low based on Open and Close
            oc_max = np.maximum(data['Open'], data['Close'])
            oc_min = np.minimum(data['Open'], data['Close'])
            
            high_noise = np.random.exponential(0.01, n_days)
            low_noise = np.random.exponential(0.01, n_days)
            
            data['High'] = oc_max * (1 + high_noise)
            data['Low'] = oc_min * (1 - low_noise)
            
            # Volume (realistic trading volume)
            base_volume = np.random.randint(50_000_000, 150_000_000)
            volume_noise = np.random.lognormal(0, 0.5, n_days)
            data['Volume'] = (base_volume * volume_noise).astype(int)
            
            # Adjusted Close (same as Close for simplicity)
            data['Adj Close'] = data['Close']
            
            all_data[ticker] = data
        
        # Format similar to yfinance
        if len(tickers) == 1:
            result = all_data[tickers[0]]
        else:
            # Create multi-level columns
            result = pd.concat(all_data, axis=1)
            result.columns = pd.MultiIndex.from_tuples(
                [(col, ticker) for ticker in tickers for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
            )
            # Reorder to match yfinance format
            result = result.reindex(columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], level=0)
        
        print(f"Generated mock data for {len(tickers)} stocks over {len(trading_days)} trading days")
        return result
    
    def fetch_data_with_fallback(self, tickers, start_date, end_date):
        """
        Main method that tries multiple approaches with fallback
        """
        print(f"Fetching stock data for: {tickers}")
        print(f"Date range: {start_date} to {end_date}")
        
        # Method 1: Try yfinance with SSL fixes
        print("\n--- Method 1: yfinance with SSL fixes ---")
        data = self.fetch_stock_data_yfinance_fixed(tickers, start_date, end_date)
        if data is not None and not data.empty:
            return data
        
        # Method 2: Try alternative API
        print("\n--- Method 2: Alternative API ---")
        data = self.fetch_stock_data_alternative_api(tickers, start_date, end_date)
        if data is not None and not data.empty:
            return data
        
        # Method 3: Generate mock data
        print("\n--- Method 3: Mock data generation ---")
        data = self.generate_mock_data(tickers, start_date, end_date)
        
        return data
    
    def save_data(self, data, filename, directory='../data/raw'):
        """Save data to CSV file"""
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)
        data.to_csv(filepath)
        print(f"💾 Data saved to: {filepath}")
        return filepath

# Convenience functions
def fetch_stock_data(tickers, start_date='2019-01-01', end_date='2024-01-01'):
    """
    Convenience function to fetch stock data with automatic fallback
    """
    fetcher = StockDataFetcher()
    return fetcher.fetch_data_with_fallback(tickers, start_date, end_date)

def demo_data_fetching():
    """
    Demonstration of data fetching capabilities
    """
    print("=== Stock Data Fetcher Demo ===\n")
    
    fetcher = StockDataFetcher()
    
    # Test single stock
    print("=== Single Stock Example ===")
    aapl_data = fetcher.fetch_data_with_fallback('AAPL', '2023-01-01', '2023-12-31')
    if aapl_data is not None:
        print(f"AAPL data shape: {aapl_data.shape}")
        print("First 5 rows:")
        print(aapl_data.head())
    
    print("\n" + "="*50 + "\n")
    
    # Test multiple stocks
    print("=== Multiple Stocks Example ===")
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    multi_data = fetcher.fetch_data_with_fallback(tickers, '2023-01-01', '2023-12-31')
    if multi_data is not None:
        print(f"Multi-stock data shape: {multi_data.shape}")
        print("Column structure:")
        print(multi_data.columns)
        
        # Save the data
        fetcher.save_data(multi_data, 'stock_data_demo.csv')

if __name__ == "__main__":
    demo_data_fetching() 