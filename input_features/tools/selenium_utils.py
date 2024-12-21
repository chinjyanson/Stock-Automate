import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

def new_driver(start_url: str):
    CHROME_DRIVER_PATH = os.environ.get("CHROMEDRIVER_PATH")

    # Set the options for the web browser, using Chrome For Testing
    options = webdriver.ChromeOptions()
    options.page_load_strategy = 'eager'

    # options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-pdf-viewer")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--disable-extensions")
    options.add_experimental_option("prefs", {
        "profile.default_content_settings.popups": 0,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "plugins.always_open_pdf_externally": True,
    })

    # Set the service for the web browser
    service = Service(executable_path=CHROME_DRIVER_PATH)

    # Create a new web browser
    driver = webdriver.Chrome(service = service, options=options)

    driver.delete_all_cookies()
    driver.get(start_url)
    driver.maximize_window()

    # Return the new web browser
    return driver

def retrieve_articles():
    pass

def click(driver, by, element_id: str):
    wait = WebDriverWait(driver, 10)
    if by == "ID":
        button = wait.until(EC.element_to_be_clickable((By.ID, element_id)))
    elif by == "XPATH":
        button = wait.until(EC.element_to_be_clickable((By.XPATH, element_id)))
    elif by == "CLASS_NAME":
        button = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, element_id)))
    elif by == "CSS_SELECTOR":
        button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, element_id)))
    else:
        raise ValueError("Invalid 'by' parameter. Must be one of ['ID', 'XPATH', 'CLASS_NAME', 'CSS_SELECTOR']")
    
    # Scroll to the button (optional, in case it's not in view)
    driver.execute_script("arguments[0].scrollIntoView(true);", button)

    # Click the button
    button.click()
    print("Clicked the 'Go to end' button successfully!")

