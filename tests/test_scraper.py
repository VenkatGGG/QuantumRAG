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
        """Verify scraper fetches exactly 10 articles using search API."""
        call_count = [0]
        
        def mock_response(url, **kwargs):
            call_count[0] += 1
            mock_resp = Mock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status = Mock()
            
            # Check if this is a search request (has 'list' param)
            params = kwargs.get('params', {})
            if params.get('list') == 'search':
                # Return search results
                mock_resp.json.return_value = {
                    "query": {
                        "search": [
                            {"title": f"Article {i}"} for i in range(1, 11)
                        ]
                    }
                }
            else:
                # This is a content fetch request
                title = params.get('titles', 'Unknown')
                mock_resp.json.return_value = {
                    "query": {
                        "pages": {
                            str(call_count[0]): {
                                "title": title,
                                "extract": f"This is the full text content for {title}. " * 50
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
            
            params = kwargs.get('params', {})
            if params.get('list') == 'search':
                # Return search results
                mock_resp.json.return_value = {
                    "query": {
                        "search": [
                            {"title": "Exists 1"},
                            {"title": "Missing"},
                            {"title": "Exists 2"}
                        ]
                    }
                }
            else:
                # This is a content fetch request
                title = params.get('titles', '')
                if title == "Missing":
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
                                    "title": title,
                                    "extract": f"Content for {title}"
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

    @patch("src.wikipedia_scraper.httpx.get")
    def test_search_uses_search_term(self, mock_get):
        """Verify scraper uses search_term parameter to find articles."""
        call_count = [0]
        
        def mock_response(url, **kwargs):
            call_count[0] += 1
            mock_resp = Mock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status = Mock()
            
            params = kwargs.get('params', {})
            if params.get('list') == 'search':
                search_term = params.get('srsearch', '')
                # Return different results based on search term
                if search_term == "Quantum Cryptography":
                    mock_resp.json.return_value = {
                        "query": {
                            "search": [
                                {"title": "Quantum cryptography"},
                                {"title": "Quantum key distribution"}
                            ]
                        }
                    }
                elif search_term == "Machine Learning":
                    mock_resp.json.return_value = {
                        "query": {
                            "search": [
                                {"title": "Machine learning"},
                                {"title": "Deep learning"}
                            ]
                        }
                    }
                else:
                    mock_resp.json.return_value = {"query": {"search": []}}
            else:
                title = params.get('titles', 'Unknown')
                mock_resp.json.return_value = {
                    "query": {
                        "pages": {
                            "123": {
                                "title": title,
                                "extract": f"Content for {title}"
                            }
                        }
                    }
                }
            return mock_resp
        
        mock_get.side_effect = mock_response
        
        scraper = WikipediaScraper()
        
        # Test with Quantum Cryptography
        articles1 = scraper.fetch_articles("Quantum Cryptography", limit=2)
        titles1 = [a["title"] for a in articles1]
        assert "Quantum cryptography" in titles1 or "Quantum key distribution" in titles1
        
        # Test with Machine Learning - should get different results
        articles2 = scraper.fetch_articles("Machine Learning", limit=2)
        titles2 = [a["title"] for a in articles2]
        assert "Machine learning" in titles2 or "Deep learning" in titles2
        
        # Results should be different
        assert titles1 != titles2

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
