import yfinance as yf

from .datetime_utils import get_current_date, time_difference_in_days, get_60days_ago

def get_hundred_datapoints(stock: str, interval: str):
    if interval == "1m":
        pass
    

def get_stock_data(stock: str, interval: str, start_date=get_60days_ago(), end_date=get_current_date()):
    if not check_data(stock, start_date, end_date, interval):
        raise ValueError("Data not valid")
    try:
        data = yf.download(stock, start=start_date, end=end_date, interval=interval)
        if data.empty:
            raise ValueError("No data found, interval not supported")
    except Exception as e:
        print(e)
        return None
    return data

def check_data(stock: str, start_date: str, end_date: str, interval: str):
    """
    Checks if the interval is not too long or if it is incompatible with the stock
    """
    # Check how many datapoint is within 60 days of the current date and that end date is not in the future
    if time_difference_in_days(get_current_date(), start_date) > 60 and interval in ["1m", "2m", "5m", "15m", "30m", "1hr"]:
        raise ValueError("The start date is too far in the past")
    if end_date > get_current_date():
        raise ValueError("The end date is in the future")
    
    if interval == "1m":
        if time_difference_in_days(start_date, end_date) > 7:
            raise ValueError("The interval is too long")
    elif interval in ["2m", "5m", "15m", "30m", "60m"]:
        if time_difference_in_days(start_date, end_date) > 60:
            raise ValueError("The interval is too long")
    elif interval == "1h":
        if time_difference_in_days(start_date, end_date) > 730:
            raise ValueError("The interval is too long")
    elif interval == "1d":
        if time_difference_in_days(start_date, end_date) > 365*40:
            raise ValueError("The interval is too long")
    elif interval == "1wk":
        if time_difference_in_days(start_date, end_date) > 365*100:
            raise ValueError("The interval is too long")
    else:
        raise ValueError("Interval not supported")

    return True

    
    
def get_stock_info(stock: str, info: str):
    stock_info = yf.Ticker(stock)
    return stock_info[info]

if __name__ == "__main__":
    print(get_stock_data("AAPL", "15m"))