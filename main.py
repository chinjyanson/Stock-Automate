import alpaca_trade_api as tradeapi
import time
from dotenv import load_dotenv
import os

# Replace these with your Alpaca API credentials
BASE_URL = 'https://paper-api.alpaca.markets'

# Initialize the Alpaca API
api = tradeapi.REST(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET_KEY"), BASE_URL, api_version='v2')

# Function to get the current price of VOO
def get_current_price(symbol):
    barset = api.get_barset(symbol, 'minute', 1)
    return barset[symbol][0].c

# Function to place a market order
def place_order(symbol, qty, side, order_type='market', time_in_force='gtc'):
    api.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        type=order_type,
        time_in_force=time_in_force
    )

# Monitor and trade VOO
def monitor_and_trade(symbol):
    while True:
        current_price = get_current_price(symbol)
        print(f"Current price of {symbol}: {current_price}")

        # Example trading logic: Buy if price is below a threshold, sell if above
        if current_price < 350:
            print("Placing a buy order for VOO")
            place_order(symbol, 1, 'buy')
        elif current_price > 400:
            print("Placing a sell order for VOO")
            place_order(symbol, 1, 'sell')

        # Wait for a minute before checking the price again
        time.sleep(60)

if __name__ == "__main__":
    monitor_and_trade('VOO')