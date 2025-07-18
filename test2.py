import yfinance as yf

stock = yf.Ticker("AAPL")  # Example ticker
info = stock.info
print(info)