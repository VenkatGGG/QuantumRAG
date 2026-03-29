#!/usr/bin/env python3
"""Script to scrape Wikipedia articles about Quantum Cryptography."""
import sys
sys.path.insert(0, '/Users/sri/Desktop/silly_experiments/Droid_Project')

from src.wikipedia_scraper import fetch_articles, save_articles


def main():
    """Fetch and save Wikipedia articles."""
    print("Fetching Wikipedia articles about Quantum Cryptography...")
    articles = fetch_articles("Quantum Cryptography", limit=10)
    print(f"Fetched {len(articles)} articles")
    
    for article in articles:
        print(f"  - {article['title']}: {len(article['text'])} chars")
    
    save_articles(articles)
    print(f"Saved to data/raw_articles.json")
    return len(articles)


if __name__ == "__main__":
    count = main()
    if count != 10:
        print(f"ERROR: Expected 10 articles, got {count}")
        sys.exit(1)
    print("SUCCESS: All 10 articles fetched and saved")
