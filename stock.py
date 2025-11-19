"""
Stock Data Provider - Pure data retrieval class for stock metrics
Provides all financial data needed for analysis without analysis logic
"""

import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Optional


class Stock:
    """
    A class to retrieve stock data and financial metrics.
    
    Attributes:
        ticker (str): Stock ticker symbol
        stock (yf.Ticker): yfinance Ticker object
        info (dict): Stock information from yfinance
    """
    
    def __init__(self, ticker: str):
        """
        Initialize Stock object with ticker symbol.
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        """
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)
        self.info = self.stock.info
        
    def get_revenue_growth(self) -> Optional[float]:
        """
        Calculate annual revenue growth rate over the past year.
        
        Returns:
            Optional[float]: Revenue growth rate as a percentage, or None if data unavailable
        """
        try:
            financials = self.stock.financials
            if financials is None or financials.empty:
                return None
            
            # Get revenue data (Total Revenue row)
            if 'Total Revenue' in financials.index:
                revenue_data = financials.loc['Total Revenue']
            else:
                return None
            
            # Sort by date to get most recent and previous year
            revenue_data = revenue_data.sort_index(ascending=False)
            
            if len(revenue_data) < 2:
                return None

            # Extract scalar values from Series and ensure they're numeric
            latest_revenue = float(revenue_data.iloc[0])  # type: ignore
            previous_revenue = float(revenue_data.iloc[1])  # type: ignore

            # Calculate growth rate
            growth_rate = ((latest_revenue - previous_revenue) / previous_revenue) * 100
            return round(growth_rate, 2)
            
        except Exception as e:
            print(f"Error getting revenue growth: {e}")
            return None
    
    def get_pe_ratio(self) -> Optional[float]:
        """
        Get the Price-to-Earnings (P/E) ratio.
        
        Returns:
            Optional[float]: P/E ratio, or None if unavailable
        """
        try:
            pe_ratio = self.info.get('trailingPE') or self.info.get('forwardPE')
            return round(pe_ratio, 2) if pe_ratio else None
        except Exception as e:
            print(f"Error getting P/E ratio: {e}")
            return None
    
    def get_peg_ratio(self) -> Optional[float]:
        """
        Get the Price/Earnings-to-Growth (PEG) ratio.
        
        Returns:
            Optional[float]: PEG ratio, or None if unavailable
        """
        try:
            peg_ratio = self.info.get('pegRatio')
            return round(peg_ratio, 2) if peg_ratio else None
        except Exception as e:
            print(f"Error getting PEG ratio: {e}")
            return None
    
    def get_roe_history(self) -> Optional[float]:
        """
        Calculate average Return on Equity (ROE) over the last 5 years.
        
        Returns:
            Optional[float]: Average ROE as a percentage, or None if unavailable
        """
        try:
            # Get ROE from info (this is typically trailing twelve months)
            current_roe = self.info.get('returnOnEquity')
            
            if current_roe is None:
                return None
            
            # Convert to percentage
            roe_percentage = current_roe * 100
            return round(roe_percentage, 2)
            
        except Exception as e:
            print(f"Error getting ROE: {e}")
            return None
    
    def get_quick_ratio(self) -> Optional[float]:
        """
        Calculate the Quick Ratio (Acid-Test Ratio).
        Quick Ratio = (Current Assets - Inventory) / Current Liabilities

        Returns:
            Optional[float]: Quick ratio, or None if unavailable
        """
        try:
            balance_sheet = self.stock.balance_sheet

            if balance_sheet is None or balance_sheet.empty:
                # Try to get from info
                quick_ratio = self.info.get('quickRatio')
                return round(quick_ratio, 2) if quick_ratio else None

            # Get most recent data
            latest = balance_sheet.iloc[:, 0]

            current_assets = latest.get('Current Assets', 0)
            inventory = latest.get('Inventory', 0)
            current_liabilities = latest.get('Current Liabilities', 0)

            if current_liabilities == 0:
                return None

            quick_ratio = (current_assets - inventory) / current_liabilities
            return round(quick_ratio, 2)

        except Exception as e:
            print(f"Error getting quick ratio: {e}")
            return None

    def get_current_ratio(self) -> Optional[float]:
        """
        Calculate the Current Ratio.
        Current Ratio = Current Assets / Current Liabilities

        Returns:
            Optional[float]: Current ratio, or None if unavailable
        """
        try:
            current_ratio = self.info.get('currentRatio')
            return round(current_ratio, 2) if current_ratio else None
        except Exception as e:
            print(f"Error getting current ratio: {e}")
            return None

    def get_debt_to_equity(self) -> Optional[float]:
        """
        Get the Debt-to-Equity ratio.

        Returns:
            Optional[float]: Debt-to-Equity ratio, or None if unavailable
        """
        try:
            debt_to_equity = self.info.get('debtToEquity')
            return round(debt_to_equity, 2) if debt_to_equity else None
        except Exception as e:
            print(f"Error getting debt-to-equity: {e}")
            return None

    def get_roa(self) -> Optional[float]:
        """
        Get Return on Assets (ROA).

        Returns:
            Optional[float]: ROA as a percentage, or None if unavailable
        """
        try:
            roa = self.info.get('returnOnAssets')
            if roa is None:
                return None
            return round(roa * 100, 2)
        except Exception as e:
            print(f"Error getting ROA: {e}")
            return None

    def get_profit_margins(self) -> Dict[str, Optional[float]]:
        """
        Get various profit margins.

        Returns:
            Dict: Dictionary with gross, operating, and net profit margins
        """
        try:
            margins = {
                'gross_margin': self.info.get('grossMargins'),
                'operating_margin': self.info.get('operatingMargins'),
                'net_margin': self.info.get('profitMargins')
            }

            # Convert to percentages and round
            for key in margins:
                if margins[key] is not None:
                    margins[key] = round(float(margins[key]) * 100, 2)  # type: ignore

            return margins
        except Exception as e:
            print(f"Error getting profit margins: {e}")
            return {'gross_margin': None, 'operating_margin': None, 'net_margin': None}

    def get_valuation_metrics(self) -> Dict[str, Optional[float]]:
        """
        Get various valuation metrics.

        Returns:
            Dict: Dictionary with valuation metrics
        """
        try:
            metrics = {
                'market_cap': self.info.get('marketCap'),
                'enterprise_value': self.info.get('enterpriseValue'),
                'price_to_sales': self.info.get('priceToSalesTrailing12Months'),
                'price_to_book': self.info.get('priceToBook'),
                'ev_to_revenue': self.info.get('enterpriseToRevenue'),
                'ev_to_ebitda': self.info.get('enterpriseToEbitda')
            }

            # Round numeric values
            for key in metrics:
                if metrics[key] is not None and key not in ['market_cap', 'enterprise_value']:
                    metrics[key] = round(float(metrics[key]), 2)  # type: ignore

            return metrics
        except Exception as e:
            print(f"Error getting valuation metrics: {e}")
            return {}

    def get_dividend_metrics(self) -> Dict[str, Optional[float]]:
        """
        Get dividend-related metrics.

        Returns:
            Dict: Dictionary with dividend metrics
        """
        try:
            dividend_yield = self.info.get('dividendYield')
            payout_ratio = self.info.get('payoutRatio')

            metrics = {
                'dividend_yield': round(dividend_yield * 100, 2) if dividend_yield else None,
                'dividend_rate': self.info.get('dividendRate'),
                'payout_ratio': round(payout_ratio * 100, 2) if payout_ratio else None,
                'five_year_avg_yield': self.info.get('fiveYearAvgDividendYield')
            }

            return metrics
        except Exception as e:
            print(f"Error getting dividend metrics: {e}")
            return {}

    def get_growth_metrics(self) -> Dict[str, Optional[float]]:
        """
        Get growth-related metrics.

        Returns:
            Dict: Dictionary with growth metrics
        """
        try:
            earnings_growth = self.info.get('earningsGrowth')
            revenue_growth = self.info.get('revenueGrowth')

            metrics = {
                'earnings_growth': round(earnings_growth * 100, 2) if earnings_growth else None,
                'revenue_growth_ttm': round(revenue_growth * 100, 2) if revenue_growth else None,
                'earnings_quarterly_growth': round(self.info.get('earningsQuarterlyGrowth', 0) * 100, 2) if self.info.get('earningsQuarterlyGrowth') else None
            }

            return metrics
        except Exception as e:
            print(f"Error getting growth metrics: {e}")
            return {}

    def get_cash_flow_metrics(self) -> Dict[str, Optional[float]]:
        """
        Get cash flow metrics.

        Returns:
            Dict: Dictionary with cash flow metrics
        """
        try:
            metrics = {
                'operating_cash_flow': self.info.get('operatingCashflow'),
                'free_cash_flow': self.info.get('freeCashflow'),
                'operating_margin': self.info.get('operatingMargins')
            }

            # Convert operating margin to percentage
            if metrics['operating_margin'] is not None:
                metrics['operating_margin'] = round(metrics['operating_margin'] * 100, 2)

            return metrics
        except Exception as e:
            print(f"Error getting cash flow metrics: {e}")
            return {}

    def get_ownership_data(self) -> Dict[str, Optional[float]]:
        """
        Get insider and institutional ownership data.

        Returns:
            Dict: Dictionary with ownership percentages
        """
        try:
            insider_pct = self.info.get('heldPercentInsiders')
            institutional_pct = self.info.get('heldPercentInstitutions')

            metrics = {
                'insider_ownership': round(insider_pct * 100, 2) if insider_pct else None,
                'institutional_ownership': round(institutional_pct * 100, 2) if institutional_pct else None
            }

            return metrics
        except Exception as e:
            print(f"Error getting ownership data: {e}")
            return {}

    def get_comprehensive_metrics(self) -> Dict:
        """
        Get all accounting and financial metrics in one place.

        Returns:
            Dict: Comprehensive dictionary with all metrics
        """
        return {
            'basic_metrics': {
                'current_price': self.info.get('currentPrice'),
                'pe_ratio': self.get_pe_ratio(),
                'peg_ratio': self.get_peg_ratio(),
                'revenue_growth': self.get_revenue_growth()
            },
            'profitability': {
                'roe': self.get_roe_history(),
                'roa': self.get_roa(),
                'profit_margins': self.get_profit_margins()
            },
            'liquidity': {
                'quick_ratio': self.get_quick_ratio(),
                'current_ratio': self.get_current_ratio()
            },
            'leverage': {
                'debt_to_equity': self.get_debt_to_equity()
            },
            'valuation': self.get_valuation_metrics(),
            'dividends': self.get_dividend_metrics(),
            'growth': self.get_growth_metrics(),
            'cash_flow': self.get_cash_flow_metrics(),
            'ownership': self.get_ownership_data()
        }



if __name__ == "__main__":
    # Example usage - just data retrieval
    example_ticker = "AAPL"
    print(f"Fetching data for {example_ticker}\n")

    try:
        stock = Stock(example_ticker)
        metrics = stock.get_comprehensive_metrics()
        print(f"Company: {stock.info.get('longName', 'Unknown')}")
        print(f"Current Price: ${metrics['basic_metrics']['current_price']}")
        print(f"P/E Ratio: {metrics['basic_metrics']['pe_ratio']}")
    except Exception as e:
        print(f"Error: {e}")