import yfinance as yf
import pandas as pd
from typing import Dict, Optional
import time
from datetime import datetime, timedelta

class StockAnalyzer:
    def analyze_stock(self, symbol: str) -> None:
        """
        Analyzes a stock using only the most reliable yfinance methods
        """
        try:
            print(f"\nAnalyzing {symbol}...")
            
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Get historical data (one of the most reliable endpoints)
            hist = ticker.history(period="1mo")
            if hist.empty:
                print(f"No price data available for {symbol}")
                return
                
            # Calculate current price and basic metrics
            current_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2]
            daily_change = ((current_price - prev_price) / prev_price) * 100
            volume = hist['Volume'].iloc[-1]
            
            # Calculate some basic technical indicators
            ma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
            ma_200 = hist['Close'].rolling(window=200).mean().iloc[-1]
            
            # Monthly performance
            monthly_return = ((current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
            
            # Volatility (standard deviation of returns)
            daily_returns = hist['Close'].pct_change()
            volatility = daily_returns.std() * 100
            
            # Print analysis
            print(f"\nPrice Information ({symbol}):")
            print(f"Current Price: ${current_price:.2f}")
            print(f"Daily Change: {daily_change:.2f}%")
            print(f"Monthly Return: {monthly_return:.2f}%")
            print(f"Daily Volume: {volume:,.0f}")
            
            print("\nTechnical Indicators:")
            if not pd.isna(ma_50):
                print(f"50-day MA: ${ma_50:.2f}")
            if not pd.isna(ma_200):
                print(f"200-day MA: ${ma_200:.2f}")
            print(f"30-day Volatility: {volatility:.2f}%")
            
            # Trading signals
            print("\nTrading Signals:")
            if ma_50 > ma_200:
                print("✅ Price above 200-day moving average (bullish)")
            else:
                print("⚠️ Price below 200-day moving average (bearish)")
                
            if volatility > 2:
                print("⚠️ High volatility detected")
            else:
                print("✅ Normal volatility levels")
                
            if current_price > ma_50:
                print("✅ Price above 50-day moving average (short-term bullish)")
            else:
                print("⚠️ Price below 50-day moving average (short-term bearish)")
            
            # Volume analysis
            avg_volume = hist['Volume'].mean()
            if volume > avg_volume * 1.5:
                print("⚠️ Unusually high volume")
            elif volume < avg_volume * 0.5:
                print("⚠️ Unusually low volume")
            else:
                print("✅ Normal trading volume")
                
        except Exception as e:
            print(f"Error analyzing {symbol}: {str(e)}")

def main():
    analyzer = StockAnalyzer()
    
    # Example usage with different types of securities
    symbols = [
        "AAPL",       # US Stock
        "TSLA",       # US Stock
        "SPY",        # S&P 500 ETF
        "^GSPC",      # S&P 500 Index
        "BARC.L",     # London Stock Exchange
        "7203.T"      # Tokyo Stock Exchange
    ]
    
    for symbol in symbols:
        analyzer.analyze_stock(symbol)
        time.sleep(2)  # Delay between analyses

if __name__ == "__main__":
    main()