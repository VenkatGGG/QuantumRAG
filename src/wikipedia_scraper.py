"""Wikipedia scraper module for fetching articles."""
import json
import time
from typing import List, Dict, Any, Optional
import httpx


class WikipediaAPIError(Exception):
    """Custom exception for Wikipedia API errors.
    
    Preserves the original exception context for better error handling.
    """
    
    def __init__(self, message: str, original_error: Exception = None):
        """Initialize the error with message and optional original exception.
        
        Args:
            message: Human-readable error message.
            original_error: The original exception that caused this error, if any.
        """
        super().__init__(message)
        self.original_error = original_error
        self.message = message
    
    def __str__(self) -> str:
        """Return string representation of the error."""
        if self.original_error:
            return f"{self.message} (caused by {type(self.original_error).__name__}: {self.original_error})"
        return self.message


class WikipediaScraper:
    """Scraper for fetching Wikipedia articles with rate limiting."""
    
    # Quantum cryptography relevance keywords for filtering
    QUANTUM_KEYWORDS = [
        'quantum', 'qkd', 'bb84', 'ekert', 'b92', 'sarg04', 'cOW',
        'quantum key distribution', 'quantum communication',
        'quantum network', 'quantum channel', 'quantum entanglement',
        'quantum superposition', 'quantum measurement', 'quantum bit',
        'qubit', 'photon', 'photonic', 'optical fiber', 'single-photon',
        'eavesdropping', 'intercept-resend', 'decoy state', 'device-independent',
        'measurement-device-independent', 'mdiqkd', 'continuous-variable',
        'discrete-variable', 'prepare-and-measure', 'entanglement-based',
        'post-quantum', 'quantum-resistant', 'quantum-safe', 'quantum-secure',
        'quantum random', 'quantum noise', 'quantum repeaters', 'quantum relay',
        'quantum memory', 'quantum teleportation', 'quantum information',
        'quantum security', 'quantum privacy', 'quantum secrecy',
        'harvest now decrypt later', 'harvest now, decrypt later'
    ]
    
    # Generic cryptography terms that indicate NON-quantum articles
    GENERIC_CRYPTO_EXCLUSIONS = [
        'cryptography (general)', 'history of cryptography',
        'symmetric-key', 'public-key', 'elliptic-curve',
        'rsa', 'aes', 'des', '3des', 'blowfish', 'twofish',
        'diffie-hellman', 'dsa', 'ecdsa', 'ecdh',
        'block cipher', 'stream cipher', 'hash function',
        'message authentication', 'digital signature',
        'key exchange', 'key agreement', 'key management',
        'certificate', 'pki', 'ssl', 'tls', 'https'
    ]
    
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
    
    def _is_quantum_relevant(self, title: str) -> bool:
        """Check if an article title is relevant to quantum cryptography.
        
        Uses keyword matching to determine topical relevance. Articles must
        contain quantum-specific terms to be included in the corpus.
        
        Args:
            title: Article title to check.
            
        Returns:
            True if the article is quantum-relevant, False otherwise.
        """
        title_lower = title.lower()
        
        # Check for quantum keywords
        has_quantum_keyword = any(
            keyword.lower() in title_lower 
            for keyword in self.QUANTUM_KEYWORDS
        )
        
        # Check for generic-only cryptography terms (indicates non-quantum focus)
        generic_only = any(
            exclusion.lower() in title_lower
            for exclusion in self.GENERIC_CRYPTO_EXCLUSIONS
        ) and not has_quantum_keyword
        
        # Reject if it's purely generic cryptography
        if generic_only:
            return False
        
        # Accept if it has quantum keywords
        return has_quantum_keyword
    
    def _filter_quantum_relevant(self, titles: List[str]) -> List[str]:
        """Filter titles to only include quantum cryptography relevant articles.
        
        Args:
            titles: List of article titles from search results.
            
        Returns:
            Filtered list containing only quantum-relevant titles.
        """
        filtered = []
        rejected = []
        
        for title in titles:
            if self._is_quantum_relevant(title):
                filtered.append(title)
            else:
                rejected.append(title)
        
        if rejected:
            print(f"  Filtered out {len(rejected)} non-quantum articles: {rejected}")
        
        return filtered
    
    def _fetch_page_content(self, title: str) -> Optional[Dict[str, Any]]:
        """Fetch page content from Wikipedia API.
        
        Args:
            title: Page title to fetch.
            
        Returns:
            Dictionary with title, url, and text, or None if page is missing.
            
        Raises:
            WikipediaAPIError: If the API request fails.
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
            raise WikipediaAPIError(f"Failed to fetch page content for '{title}'", e)
    
    def _search_articles(self, search_term: str, limit: int = 10) -> List[str]:
        """Search Wikipedia for articles related to a search term.
        
        Uses the Wikipedia API's search action to find relevant articles.
        
        Args:
            search_term: Term to search for.
            limit: Maximum number of results to return.
            
        Returns:
            List of article titles matching the search.
            
        Raises:
            WikipediaAPIError: If the API request fails.
        """
        self._apply_rate_limit()
        
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": search_term,
            "srlimit": limit,
            "srprop": "",
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
            
            search_results = data.get("query", {}).get("search", [])
            titles = [result["title"] for result in search_results]
            return titles
        except Exception as e:
            raise WikipediaAPIError(f"Failed to search for '{search_term}'", e)
    
    def fetch_articles(self, search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch articles from Wikipedia related to search term.
        
        Uses Wikipedia's search API to find relevant articles dynamically
        based on the provided search_term, then fetches their content.
        Filters results to ensure only quantum cryptography relevant
        articles are included in the corpus.
        
        Args:
            search_term: Term to search for.
            limit: Maximum number of articles to fetch.
            
        Returns:
            List of article dictionaries with title, url, and text.
        """
        # First, search for articles related to the search term
        # Request more results to account for filtering
        search_limit = limit * 3  # Request 3x to have enough after filtering
        titles = self._search_articles(search_term, limit=search_limit)
        
        if not titles:
            print(f"No search results found for '{search_term}'")
            return []
        
        # Filter for quantum relevance
        print(f"Filtering {len(titles)} search results for quantum cryptography relevance...")
        filtered_titles = self._filter_quantum_relevant(titles)
        
        if len(filtered_titles) < limit:
            print(f"Warning: Only {len(filtered_titles)} quantum-relevant articles found (requested {limit})")
        
        articles = []
        
        for title in filtered_titles:
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
