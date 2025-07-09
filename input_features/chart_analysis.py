import yfinance as yf
import pandas as pd
import numpy as np
import os

from tools.yfinance_utils import get_stock_data, get_stock_info
from tools.datetime_utils import get_current_date

def obtain_chart_analysis(stock: str):
    '''
    Parent function that will wrap the sum of all the analysis functions 

    Args:

    Returns:
    - np.array: Array containing the analysis data
    '''
    analysis_data = np.array([])
    data = get_stock_data(stock, "1d")
    analysis_data = np.append(analysis_data, obv_analysis(data), axis=None)    
    # print(analysis_data)
    # print(obv_analysis(data))
    # analysis_data = np.append(analysis_data, volume_oscillator_analysis(data), axis=None)
    # analysis_data = np.append(analysis_data, price_momentum_analysis(data), axis=None)
    # analysis_data = np.append(analysis_data, moving_average_analysis(data), axis=None)
    return analysis_data

def obv_analysis(data):
    if 'Close' not in data.columns or 'Volume' not in data.columns:
        raise ValueError("Data must contain 'Close' and 'Volume' columns.")
    if data.empty:
        raise ValueError("Data is empty.")

    close_diff = data['Close'].diff()  # Calculate difference between consecutive rows
    obv = np.where(
        close_diff > 0, data['Volume'], 
        np.where(close_diff < 0, -data['Volume'], 0)
    )
    obv = np.cumsum(obv)  # Cumulative sum for OBV
    return obv

def volume_oscillator_analysis(data, short_window=5, long_window=20):
    data['Short_EMA'] = data['Volume'].ewm(span=short_window, adjust=False).mean()
    data['Long_EMA'] = data['Volume'].ewm(span=long_window, adjust=False).mean()
    return ((data['Short_EMA'] - data['Long_EMA']) / data['Long_EMA']) * 100

def price_momentum_analysis(data, window=14):
    """
    Calculate the Relative Strength Index (RSI) and momentum indicator.
    
    Args:
    - data (pd.DataFrame): DataFrame containing a 'Close' column with price data.
    - window (int): Look-back period for RSI calculation (default: 14).
    
    Returns:
    - pd.Series: RSI values.
    """
    # Calculate price changes
    delta = data['Close'].diff()
    
    # Separate gains and losses
    gain = (delta.where(delta > 0, 0))  # Positive changes (gains)
    loss = (-delta.where(delta < 0, 0))  # Negative changes (losses)
    
    # Calculate EMA of gains and losses
    avg_gain = gain.ewm(span=window, adjust=False).mean()
    avg_loss = loss.ewm(span=window, adjust=False).mean()
    
    # Calculate Relative Strength (RS)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi

    data["Momentum"] = data['Close'].diff(window)
    return data

def moving_average_analysis():

    pass

def macd_analysis():
    pass

def bollinger_bands_analysis():
    pass

def adx_analysis():
    pass


if __name__ == "__main__":
    print(obtain_chart_analysis("AAPL"))