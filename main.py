import yfinance as yf

class StockAnalyzer:
    def __init__(self, ticker):
        stock = yf.Ticker(ticker)
    def get_info(self):
        pass
    def plot_stock(self):
        pass
    def analyze_stock(self):
        try:
            info = self.stock.info

            # Fetch Key Metrics
            price = info.get("currentPrice", "N/A")
            eps = info.get("trailingEps", "N/A")
            pe_ratio = price / eps if isinstance(price, (int, float)) and isinstance(eps, (int, float)) and eps != 0 else "N/A"
            forward_pe = info.get("forwardPE", "N/A")
            
            # Growth Metrics
            eps_growth_5y = info.get("earningsGrowth", "N/A")
            revenue_growth = info.get("revenueGrowth", "N/A")

            # Debt Metrics
            debt_to_equity = info.get("debtToEquity", "N/A")

            # Profitability
            roe = info.get("returnOnEquity", "N/A")
            roa = info.get("returnOnAssets", "N/A")

            # Compare with Industry
            industry_pe = info.get("industryPE", "N/A")

            # Insider & Institutional Ownership
            insider_ownership = info.get("heldPercentInsiders", "N/A")
            institutional_ownership = info.get("heldPercentInstitutions", "N/A")

            # Check for Valuation Trap Signals
            valuation_trap = False
            reasons = []

            if isinstance(pe_ratio, (int, float)) and isinstance(forward_pe, (int, float)):
                if pe_ratio < 10 and forward_pe > pe_ratio:
                    valuation_trap = True
                    reasons.append("Low current P/E but higher forward P/E (declining earnings).")

            if isinstance(eps_growth_5y, (int, float)) and eps_growth_5y < 0:
                valuation_trap = True
                reasons.append("Negative earnings growth over the past 5 years.")

            if isinstance(revenue_growth, (int, float)) and revenue_growth < 0:
                valuation_trap = True
                reasons.append("Negative revenue growth, indicating potential demand decline.")

            if isinstance(debt_to_equity, (int, float)) and debt_to_equity > 200:
                valuation_trap = True
                reasons.append("High debt-to-equity ratio (>200), indicating financial risk.")

            if isinstance(roe, (int, float)) and roe < 5:
                valuation_trap = True
                reasons.append("Low return on equity (ROE < 5%), indicating weak profitability.")

            if isinstance(industry_pe, (int, float)) and isinstance(pe_ratio, (int, float)) and pe_ratio < industry_pe * 0.5:
                valuation_trap = True
                reasons.append("Stock is significantly cheaper than industry peers, which may indicate hidden risks.")

            # Print Summary
            print(f"Stock Analysis for {self.ticker}")
            print(f"Current Price: {price}")
            print(f"P/E Ratio: {pe_ratio}, Forward P/E: {forward_pe}")
            print(f"EPS Growth (5Y): {eps_growth_5y}, Revenue Growth: {revenue_growth}")
            print(f"Debt-to-Equity: {debt_to_equity}")
            print(f"ROE: {roe}, ROA: {roa}")
            print(f"Industry Avg P/E: {industry_pe}")
            print(f"Insider Ownership: {insider_ownership}, Institutional Ownership: {institutional_ownership}")
            
            if valuation_trap:
                print("\n⚠️ Potential Valuation Trap Detected! ⚠️")
                for reason in reasons:
                    print(f" - {reason}")
            else:
                print("\n✅ No obvious valuation traps detected.")

        except Exception as e:
            print(f"Error retrieving data for {self.ticker}: {e}")

# Example Usage:
StockAnalyzer("AAPL").analyze_stock()  # Apple Inc. on NASDAQ
