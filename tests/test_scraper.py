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
                # Return search results with quantum-relevant titles
                mock_resp.json.return_value = {
                    "query": {
                        "search": [
                            {"title": "Quantum cryptography"},
                            {"title": "Quantum key distribution"},
                            {"title": "Post-quantum cryptography"},
                            {"title": "Quantum computing"},
                            {"title": "Quantum network"},
                            {"title": "Quantum entanglement"},
                            {"title": "Quantum information science"},
                            {"title": "Relativistic quantum cryptography"},
                            {"title": "Quantum channel"},
                            {"title": "Quantum communication"}
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
                # Return search results with quantum-relevant titles
                mock_resp.json.return_value = {
                    "query": {
                        "search": [
                            {"title": "Quantum cryptography"},
                            {"title": "Quantum key distribution"},
                            {"title": "Quantum computing"}
                        ]
                    }
                }
            else:
                # This is a content fetch request
                title = params.get('titles', '')
                # Simulate that "Quantum key distribution" is missing
                if title == "Quantum key distribution":
                    mock_resp.json.return_value = {
                        "query": {
                            "pages": {
                                "456": {
                                    "title": "Quantum key distribution",
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
        assert all(a["title"] != "Quantum key distribution" for a in articles)

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
                    # These will be filtered out as non-quantum
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
        
        # Test with Quantum Cryptography - should get quantum-relevant results
        articles1 = scraper.fetch_articles("Quantum Cryptography", limit=2)
        titles1 = [a["title"] for a in articles1]
        assert "Quantum cryptography" in titles1 or "Quantum key distribution" in titles1
        
        # Test with Machine Learning - results will be filtered out (non-quantum)
        articles2 = scraper.fetch_articles("Machine Learning", limit=2)
        # Machine learning articles are filtered as non-quantum
        assert len(articles2) == 0

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


class TestQuantumRelevanceFiltering:
    """Test cases for quantum cryptography relevance filtering."""

    def test_quantum_relevant_titles_accepted(self):
        """Verify quantum-relevant article titles are accepted."""
        scraper = WikipediaScraper()
        
        quantum_titles = [
            "Quantum cryptography",
            "Quantum key distribution",
            "Post-quantum cryptography",
            "Quantum computing",
            "Quantum entanglement",
            "BB84 protocol",
            "Quantum network",
            "QKD",
            "Relativistic quantum cryptography"
        ]
        
        for title in quantum_titles:
            assert scraper._is_quantum_relevant(title), f"'{title}' should be quantum-relevant"

    def test_generic_crypto_titles_rejected(self):
        """Verify generic cryptography titles are rejected."""
        scraper = WikipediaScraper()
        
        generic_titles = [
            "Cryptography",
            "Key (cryptography)",
            "Public-key cryptography",
            "Symmetric-key algorithm",
            "Elliptic-curve cryptography",
            "RSA (cryptosystem)",
            "AES",
            "Digital signature"
        ]
        
        for title in generic_titles:
            assert not scraper._is_quantum_relevant(title), f"'{title}' should be rejected as non-quantum"

    def test_filter_quantum_relevant_removes_non_quantum(self):
        """Verify filtering removes non-quantum articles."""
        scraper = WikipediaScraper()
        
        mixed_titles = [
            "Quantum cryptography",
            "Cryptography",
            "Quantum key distribution",
            "Key (cryptography)",
            "Quantum computing"
        ]
        
        filtered = scraper._filter_quantum_relevant(mixed_titles)
        
        # Should keep quantum-relevant titles
        assert "Quantum cryptography" in filtered
        assert "Quantum key distribution" in filtered
        assert "Quantum computing" in filtered
        
        # Should remove generic cryptography titles
        assert "Cryptography" not in filtered
        assert "Key (cryptography)" not in filtered
        
        # Should have exactly 3 results
        assert len(filtered) == 3

    def test_fetch_articles_returns_quantum_specific(self):
        """Verify fetch_articles returns quantum-specific articles."""
        scraper = WikipediaScraper()
        
        # Test with mock search results
        titles = scraper._search_articles("Quantum Cryptography", limit=15)
        filtered = scraper._filter_quantum_relevant(titles)
        
        # All filtered results should contain quantum keywords
        for title in filtered:
            assert scraper._is_quantum_relevant(title), f"'{title}' should be quantum-relevant"
        
        # Should not contain generic cryptography articles
        generic_crypto = [
            "Cryptography",
            "Key (cryptography)",
            "Public-key cryptography",
            "Symmetric-key algorithm"
        ]
        
        for generic in generic_crypto:
            assert generic not in filtered, f"Generic crypto article '{generic}' should be filtered out"
