import yfinance as yf
import pandas as pd
import os

def fetch_portfolio_data(tickers, start_date, end_date, cache_dir="data"):
    """
    Fetches adjusted closing prices for given tickers from yfinance and caches to CSV.
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Simple hash-like filename based on tickers and dates
    ticker_str = "_".join(sorted(tickers))
    cache_file = os.path.join(cache_dir, f"{ticker_str}_{start_date}_{end_date}.csv")
    
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df
    
    print(f"Downloading data for: {tickers} from {start_date} to {end_date}...")
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    
    # If single ticker, yfinance returns Series; convert to DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
        
    data = data.dropna()
    data.to_csv(cache_file)
    return data

def calculate_returns(df):
    """Calculates daily log returns."""
    import numpy as np
    return np.log(df / df.shift(1)).dropna()
