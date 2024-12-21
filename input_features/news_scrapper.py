import requests
import time
from bs4 import BeautifulSoup
import pandas as pd
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch

import tools.selenium_utils as su

def get_news(stock: str):
    driver = su.new_driver(f"https://finance.yahoo.com/quote/{stock}/")
    articles_array = []
    time.sleep(2)
    try:
        su.click(driver, "ID", "scroll-down-btn")
        print("Clicked the 'Go to end' button successfully!")
        time.sleep(5)
        su.click(driver, "XPATH", "//button[@type='submit' and @name='reject' and @value='reject']")
        print("Clicked the 'Reject all' button successfully!")
    except Exception as e:
        print(e)

    try:
        driver.execute_script("window.scrollBy(0, 800);")
        time.sleep(2)
        articles = driver.find_elements(By.CSS_SELECTOR, "div.content.yf-18q3fnf a.subtle-link")
        article_links = [article.get_attribute("href") for article in articles if article.get_attribute("href")]

        print(f"Found {len(article_links)} articles.")
        
        # Visit each article and collect content
        for i, link in enumerate(article_links):
            try:
                print(f"Opening article {i + 1}: {link}")
                driver.get(link)
                time.sleep(2)  # Allow the page to load 

                # Extract article content (adjust selectors based on the website structure)
                paragraphs = driver.find_elements(By.TAG_NAME, "p")  # Assuming paragraphs are in <p> tags
                article_text = "\n".join([p.text.replace('\n', '') for p in paragraphs])
                articles_array.append(article_text)
                print(f"Collected content for article {i + 1}:\n{article_text[:1000]}...\n")  # Preview content
            except Exception as e:
                print(f"Error with article {i + 1}: {e}")
                continue

            time.sleep(2)

    except Exception as e:
        print(f"Error: {e}")

    return articles_array

def split_text(text, max_tokens=512):
    tokens = text.split()
    chunks = [" ".join(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)]
    return chunks

def get_embeddings(text):
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModel.from_pretrained("yiyanghkust/finbert-tone")

    # Tokenize the input with truncation and max_length
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    # Forward pass through the model
    outputs = model(**inputs)

    # Extract embeddings
    token_embeddings = outputs.last_hidden_state
    cls_embedding = token_embeddings[:, 0, :]  # [CLS] token embedding

    return token_embeddings.detach().numpy(), cls_embedding.detach().numpy()

def get_articles_embeddings(articles):
    articles_embeddings = []
    for i, article in enumerate(articles):
        try:
            cls_embedding = get_embeddings(article)[1]
            articles_embeddings.append(cls_embedding)
        except Exception as e:
            print(f"Error processing article {i + 1}: {e}")
            continue

    if not articles_embeddings:
        print("No valid embeddings found!")
        return np.array([])

    # Stack embeddings into a single numpy array
    return np.vstack(articles_embeddings)

def news_to_embeddings(stock: str):
    articles = get_news(stock)
    return get_articles_embeddings(articles)

if __name__ == "__main__":
    print(news_to_embeddings("AAPL"))