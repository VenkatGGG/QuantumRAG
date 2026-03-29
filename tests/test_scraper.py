"""Tests for Wikipedia scraper module."""
import json
import time
from unittest.mock import Mock, patch
import pytest

from src.wikipedia_scraper import WikipediaScraper, fetch_articles, save_articles


class TestWikipediaScraper:
    """Test cases for WikipediaScraper class."""

    def test_user_agent_present(self):
        """Verify scraper has proper User-Agent header configured."""
        scraper = WikipediaScraper()
        assert "User-Agent" in scraper.headers
        assert "QuantumCryptographyRAG" in scraper.headers["User-Agent"]
        assert "contact@example.com" in scraper.headers["User-Agent"]

    def test_rate_limiting(self):
        """Verify rate limiting delays requests by at least 0.5s."""
        scraper = WikipediaScraper()
        # First call sets the timestamp but doesn't delay (no previous request)
        scraper._apply_rate_limit()
        # Second call should delay since we had a previous request
        start_time = time.time()
        scraper._apply_rate_limit()
        elapsed = time.time() - start_time
        assert elapsed >= 0.5, f"Rate limit delay too short: {elapsed}s"

    @patch("src.wikipedia_scraper.httpx.get")
    def test_fetch_articles_count(self, mock_get):
        """Verify scraper fetches exactly 10 articles."""
        # Setup mock
        def mock_response(url, **kwargs):
            mock_resp = Mock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status = Mock()
            # Extract title from URL params
            mock_resp.json.return_value = {
                "query": {
                    "pages": {
                        "123": {
                            "title": "Mock Article",
                            "extract": "This is the full text content for the article. " * 50
                        }
                    }
                }
            }
            return mock_resp
        
        mock_get.side_effect = mock_response
        
        # Execute
        scraper = WikipediaScraper()
        articles = scraper.fetch_articles("Quantum Cryptography", limit=10)
        
        # Verify
        assert len(articles) == 10, f"Expected 10 articles, got {len(articles)}"
        for article in articles:
            assert "title" in article
            assert "url" in article
            assert "text" in article

    @patch("src.wikipedia_scraper.httpx.get")
    def test_handles_nonexistent_pages(self, mock_get):
        """Verify scraper gracefully handles non-existent pages."""
        call_count = [0]
        
        def mock_response(url, **kwargs):
            call_count[0] += 1
            mock_resp = Mock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status = Mock()
            
            # Return missing for some pages
            if call_count[0] % 2 == 0:
                mock_resp.json.return_value = {
                    "query": {
                        "pages": {
                            "456": {
                                "title": "Missing",
                                "missing": True
                            }
                        }
                    }
                }
            else:
                mock_resp.json.return_value = {
                    "query": {
                        "pages": {
                            "789": {
                                "title": f"Exists {call_count[0]}",
                                "extract": f"Content for article {call_count[0]}"
                            }
                        }
                    }
                }
            return mock_resp
        
        mock_get.side_effect = mock_response
        
        scraper = WikipediaScraper()
        articles = scraper.fetch_articles("Quantum Cryptography", limit=3)
        
        # Should only return existing pages with content
        assert len(articles) > 0
        assert all(a["title"] != "Missing" for a in articles)

    def test_save_articles_creates_valid_json(self, tmp_path):
        """Verify articles are saved as valid JSON."""
        articles = [
            {"title": "Test Article", "url": "https://example.com", "text": "Test content"}
        ]
        
        output_path = tmp_path / "test_articles.json"
        save_articles(articles, str(output_path))
        
        # Verify file was created and is valid JSON
        assert output_path.exists()
        with open(output_path) as f:
            loaded = json.load(f)
        
        assert len(loaded) == 1
        assert loaded[0]["title"] == "Test Article"
