import yfinance as yf

def get_stock_data(stock: str, start_date: str, end_date: str, interval: str):
    data = yf.download(stock, start=start_date, end=end_date, interval=interval)
    return data

def get_stock_info(stock: str, info: str):
    stock_info = yf.Ticker(stock)
    return stock_info[info]




if __name__ == "__main__":
    print(get_stock_data("AAPL", "2022-01-01", "2022-01-31", "1d")["Volume"])