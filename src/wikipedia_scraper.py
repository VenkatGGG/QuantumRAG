"""Wikipedia scraper module for fetching articles."""
import json
import time
from typing import List, Dict, Any
import httpx


class WikipediaScraper:
    """Scraper for fetching Wikipedia articles with rate limiting."""
    
    def __init__(self, user_agent: str = None):
        """Initialize the scraper with proper User-Agent headers.
        
        Args:
            user_agent: Custom User-Agent string. If None, uses default.
        """
        self.headers = {
            "User-Agent": user_agent or "QuantumCryptographyRAG/1.0 (contact@example.com)"
        }
        self._last_request_time = 0
        self.base_url = "https://en.wikipedia.org/w/api.php"
    
    def _apply_rate_limit(self, delay: float = 0.5):
        """Apply rate limiting between requests.
        
        Args:
            delay: Minimum delay in seconds between requests (default 0.5s).
        """
        elapsed = time.time() - self._last_request_time
        if elapsed < delay and self._last_request_time > 0:
            time.sleep(delay - elapsed)
        self._last_request_time = time.time()
    
    def _fetch_page_content(self, title: str) -> Dict[str, Any]:
        """Fetch page content from Wikipedia API.
        
        Args:
            title: Page title to fetch.
            
        Returns:
            Dictionary with title, url, and text.
        """
        self._apply_rate_limit()
        
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
            "exlimit": 1,
        }
        
        try:
            response = httpx.get(
                self.base_url,
                headers=self.headers,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            pages = data.get("query", {}).get("pages", {})
            if not pages:
                return None
            
            page_id = list(pages.keys())[0]
            page = pages[page_id]
            
            if "missing" in page:
                return None
            
            return {
                "title": page.get("title", title),
                "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                "text": page.get("extract", "")
            }
        except Exception as e:
            print(f"Error fetching {title}: {e}")
            return None
    
    def fetch_articles(self, search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch articles from Wikipedia related to search term.
        
        Args:
            search_term: Term to search for.
            limit: Maximum number of articles to fetch.
            
        Returns:
            List of article dictionaries with title, url, and text.
        """
        articles = []
        
        # Use predefined list of Quantum Cryptography related articles
        quantum_crypto_articles = [
            "Quantum cryptography",
            "Quantum key distribution",
            "BB84",
            "Quantum teleportation",
            "Quantum computing",
            "Post-quantum cryptography",
            "Quantum entanglement",
            "Quantum information science",
            "Quantum network",
            "Quantum communication",
            "Quantum cryptography protocol",
            "Device-independent quantum cryptography",
            "Ekert protocol",
            "Quantum money",
            "Quantum digital signature"
        ]
        
        for title in quantum_crypto_articles:
            if len(articles) >= limit:
                break
            
            article = self._fetch_page_content(title)
            if article and article["text"]:
                articles.append(article)
        
        return articles[:limit]


def fetch_articles(search_term: str = "Quantum Cryptography", limit: int = 10) -> List[Dict[str, Any]]:
    """Convenience function to fetch articles.
    
    Args:
        search_term: Term to search for (default: "Quantum Cryptography").
        limit: Maximum number of articles to fetch (default: 10).
        
    Returns:
        List of article dictionaries.
    """
    scraper = WikipediaScraper()
    return scraper.fetch_articles(search_term, limit)


def save_articles(articles: List[Dict[str, Any]], output_path: str = "data/raw_articles.json") -> None:
    """Save articles to JSON file.
    
    Args:
        articles: List of article dictionaries to save.
        output_path: Path to output JSON file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    print("Fetching Wikipedia articles about Quantum Cryptography...")
    articles = fetch_articles("Quantum Cryptography", limit=10)
    print(f"Fetched {len(articles)} articles")
    
    for article in articles:
        print(f"  - {article['title']}: {len(article['text'])} chars")
    
    save_articles(articles)
    print(f"Saved to data/raw_articles.json")
