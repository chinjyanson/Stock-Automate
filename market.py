"""
Market Data Provider - Pure data retrieval class for market metrics
Provides market index data and performance calculations without analysis logic
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional


class Market:
    """
    A class to analyze market performance using various metrics and indicators.

    Attributes:
        ticker (str): Market index ticker symbol (e.g., '^GSPC' for S&P 500)
        index (yf.Ticker): yfinance Ticker object for the market index
        info (dict): Market index information from yfinance
    """

    def __init__(self, ticker: str = "^GSPC"):
        """
        Initialize Market object with an index ticker.

        Args:
            ticker (str): Market index ticker (default: '^GSPC' for S&P 500)
                         Common indices: '^GSPC' (S&P 500), '^DJI' (Dow Jones),
                         '^IXIC' (NASDAQ), '^VIX' (Volatility Index)
        """
        self.ticker = ticker.upper()
        self.index = yf.Ticker(self.ticker)
        self.info = self.index.info

    def get_historical_data(self, period: str = "1y") -> Optional[pd.DataFrame]:
        """
        Get historical market data.

        Args:
            period (str): Time period ('1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max')

        Returns:
            Optional[pd.DataFrame]: Historical price data or None if unavailable
        """
        try:
            hist = self.index.history(period=period)
            return hist if not hist.empty else None
        except Exception as e:
            print(f"Error getting historical data: {e}")
            return None

    def calculate_returns(self, period: str = "1y") -> Optional[float]:
        """
        Calculate total return over a specified period.

        Args:
            period (str): Time period for return calculation

        Returns:
            Optional[float]: Total return as a percentage, or None if unavailable
        """
        try:
            hist = self.get_historical_data(period)
            if hist is None or len(hist) < 2:
                return None

            start_price = float(hist['Close'].iloc[0])
            end_price = float(hist['Close'].iloc[-1])

            total_return = ((end_price - start_price) / start_price) * 100
            return round(total_return, 2)
        except Exception as e:
            print(f"Error calculating returns: {e}")
            return None

    def calculate_volatility(self, period: str = "1y", annualize: bool = True) -> Optional[float]:
        """
        Calculate market volatility (standard deviation of returns).

        Args:
            period (str): Time period for volatility calculation
            annualize (bool): Whether to annualize the volatility

        Returns:
            Optional[float]: Volatility as a percentage, or None if unavailable
        """
        try:
            hist = self.get_historical_data(period)
            if hist is None or len(hist) < 2:
                return None

            # Calculate daily returns
            daily_returns = hist['Close'].pct_change().dropna()

            # Calculate standard deviation
            volatility = float(daily_returns.std())

            # Annualize if requested (assuming 252 trading days)
            if annualize:
                volatility = volatility * np.sqrt(252)

            return round(volatility * 100, 2)
        except Exception as e:
            print(f"Error calculating volatility: {e}")
            return None

    def calculate_sharpe_ratio(self, period: str = "1y", risk_free_rate: float = 0.04) -> Optional[float]:
        """
        Calculate Sharpe Ratio (risk-adjusted return).

        Args:
            period (str): Time period for calculation
            risk_free_rate (float): Annual risk-free rate (default: 0.04 or 4%)

        Returns:
            Optional[float]: Sharpe ratio, or None if unavailable
        """
        try:
            hist = self.get_historical_data(period)
            if hist is None or len(hist) < 2:
                return None

            # Calculate daily returns
            daily_returns = hist['Close'].pct_change().dropna()

            # Annualize returns and volatility
            annual_return = float(daily_returns.mean() * 252)
            annual_volatility = float(daily_returns.std() * np.sqrt(252))

            if annual_volatility == 0:
                return None

            # Calculate Sharpe ratio
            sharpe = (annual_return - risk_free_rate) / annual_volatility
            return round(sharpe, 2)
        except Exception as e:
            print(f"Error calculating Sharpe ratio: {e}")
            return None

    def calculate_max_drawdown(self, period: str = "1y") -> Optional[float]:
        """
        Calculate maximum drawdown (largest peak-to-trough decline).

        Args:
            period (str): Time period for calculation

        Returns:
            Optional[float]: Maximum drawdown as a percentage, or None if unavailable
        """
        try:
            hist = self.get_historical_data(period)
            if hist is None or len(hist) < 2:
                return None

            # Calculate cumulative returns
            cumulative = (1 + hist['Close'].pct_change()).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max

            max_drawdown = float(drawdown.min() * 100)
            return round(max_drawdown, 2)
        except Exception as e:
            print(f"Error calculating max drawdown: {e}")
            return None

    def calculate_moving_averages(self, period: str = "1y") -> Optional[Dict[str, float]]:
        """
        Calculate various moving averages.

        Args:
            period (str): Time period for calculation

        Returns:
            Optional[Dict]: Dictionary with moving averages, or None if unavailable
        """
        try:
            hist = self.get_historical_data(period)
            if hist is None or len(hist) < 200:
                return None

            current_price = float(hist['Close'].iloc[-1])

            moving_averages = {
                'current_price': round(current_price, 2),
                'ma_50': round(float(hist['Close'].rolling(window=50).mean().iloc[-1]), 2),
                'ma_100': round(float(hist['Close'].rolling(window=100).mean().iloc[-1]), 2),
                'ma_200': round(float(hist['Close'].rolling(window=200).mean().iloc[-1]), 2),
            }

            # Calculate distance from moving averages
            moving_averages['distance_from_ma_50'] = round(
                ((current_price - moving_averages['ma_50']) / moving_averages['ma_50']) * 100, 2
            )
            moving_averages['distance_from_ma_200'] = round(
                ((current_price - moving_averages['ma_200']) / moving_averages['ma_200']) * 100, 2
            )

            return moving_averages
        except Exception as e:
            print(f"Error calculating moving averages: {e}")
            return None

    def calculate_rsi(self, period: str = "3mo", window: int = 14) -> Optional[float]:
        """
        Calculate Relative Strength Index (RSI).

        Args:
            period (str): Time period for data
            window (int): RSI calculation window (default: 14 days)

        Returns:
            Optional[float]: RSI value (0-100), or None if unavailable
        """
        try:
            hist = self.get_historical_data(period)
            if hist is None or len(hist) < window + 1:
                return None

            # Calculate price changes
            delta = hist['Close'].diff()

            # Separate gains and losses
            gains = delta.where(delta > 0, 0.0)  # type: ignore
            losses = -delta.where(delta < 0, 0.0)  # type: ignore

            # Calculate average gains and losses
            avg_gains = gains.rolling(window=window).mean()
            avg_losses = losses.rolling(window=window).mean()

            # Calculate RS and RSI
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))

            return round(float(rsi.iloc[-1]), 2)
        except Exception as e:
            print(f"Error calculating RSI: {e}")
            return None

    def get_all_metrics(self) -> Dict:
        """
        Get all market metrics in one comprehensive dictionary.

        Returns:
            Dict: All market metrics organized by category
        """
        return {
            'ticker': self.ticker,
            'name': self.info.get('longName', 'Unknown'),
            'returns': {
                'returns_1m': self.calculate_returns("1mo"),
                'returns_3m': self.calculate_returns("3mo"),
                'returns_1y': self.calculate_returns("1y")
            },
            'risk': {
                'volatility_1y': self.calculate_volatility("1y"),
                'sharpe_ratio': self.calculate_sharpe_ratio("1y"),
                'max_drawdown_1y': self.calculate_max_drawdown("1y")
            },
            'technical': {
                'rsi': self.calculate_rsi(),
                'moving_averages': self.calculate_moving_averages("1y")
            }
        }


if __name__ == "__main__":
    # Example usage - just data retrieval
    print("Fetching S&P 500 data...\n")

    try:
        sp500 = Market("^GSPC")
        metrics = sp500.get_all_metrics()
        print(f"Index: {metrics['ticker']} - {metrics['name']}")
        print(f"1Y Return: {metrics['returns']['returns_1y']}%")
        print(f"Volatility: {metrics['risk']['volatility_1y']}%")
    except Exception as e:
        print(f"Error: {e}")
