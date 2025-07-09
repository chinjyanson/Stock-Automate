import requests
import os
from dotenv import load_dotenv

load_dotenv()
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

def get_gnews_articles(query, from_date, to_date, api_key):
    url = "https://gnews.io/api/v4/search"
    params = {
        "q": query,
        "from": from_date,
        "to": to_date,
        "lang": "en",
        "token": api_key,
        "max": 10 
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        return data["articles"] 
    else:
        print("Error:", response.status_code, response.text)
        return []

if __name__ == "__main__":
    articles = get_gnews_articles("apple stocks", "2024-12-01", "2024-12-05", GNEWS_API_KEY)

    for article in articles:
        print(f"Title: {article['title']}")
        print(f"Article Description: {article['description']}")
        print(f"Article Content: {article['content']}")
        print(f"Published: {article['publishedAt']}")
        print(f"URL: {article['url']}\n")
 