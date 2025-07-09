import time
import pandas as pd
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import numpy as np

import tools.selenium_utils as su

def get_sentiment(stock: str):
    driver = su.new_driver(f"https://stocktwits.com/symbol/{stock}/sentiment")
    time.sleep(2)
    try:
        element = driver.find_element(By.CSS_SELECTOR, ".gauge_gagueNumber__Dr41m")
        gauge_value = element.text
        print(gauge_value)
    except Exception as e:
        print(e)
    return gauge_value

if __name__ == "__main__":
    get_sentiment("AAPL")