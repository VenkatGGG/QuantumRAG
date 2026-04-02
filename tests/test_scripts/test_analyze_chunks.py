"""Tests for scripts/analysis/analyze_chunks.py."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.analysis.analyze_chunks import load_articles, analyze_corpus_from_articles


class TestAnalyzeChunks(unittest.TestCase):
    """Test cases for analyze_chunks script."""

    def test_load_articles_success(self) -> None:
        """Test loading articles from valid JSON file."""
        test_data = [
            {"title": "Test 1", "text": "This is test content."},
            {"title": "Test 2", "text": "More test content here."}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            articles = load_articles(temp_path)
            self.assertEqual(len(articles), 2)
            self.assertEqual(articles[0]['title'], "Test 1")
        finally:
            os.unlink(temp_path)

    def test_load_articles_file_not_found(self) -> None:
        """Test loading from non-existent file raises error."""
        with self.assertRaises(FileNotFoundError):
            load_articles("/nonexistent/path/articles.json")

    @patch('scripts.analysis.analyze_chunks.load_articles')
    @patch('builtins.print')
    def test_analyze_corpus_from_articles(self, mock_print: MagicMock, mock_load: MagicMock) -> None:
        """Test analyze_corpus_from_articles processes chunks correctly."""
        # Mock articles with enough text to create chunks
        long_text = " ".join([f"This is sentence number {i} for testing chunking." for i in range(100)])
        mock_load.return_value = [
            {"title": "Test Article", "text": long_text}
        ]
        
        chunks, overlaps = analyze_corpus_from_articles()
        
        self.assertIsInstance(chunks, list)
        self.assertIsInstance(overlaps, list)
        mock_load.assert_called_once()


if __name__ == '__main__':
    unittest.main()
