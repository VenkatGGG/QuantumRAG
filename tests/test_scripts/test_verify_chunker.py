"""Tests for scripts/manual_tests/verify_chunker.py."""

import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.manual_tests.verify_chunker import main


class TestVerifyChunker(unittest.TestCase):
    """Test cases for verify_chunker script."""

    @patch('scripts.manual_tests.verify_chunker.fetch_articles')
    @patch('scripts.manual_tests.verify_chunker.save_articles')
    @patch('builtins.print')
    def test_verify_chunker_runs(self, mock_print: MagicMock, mock_save: MagicMock, mock_fetch: MagicMock) -> None:
        """Test verify chunker runs with mocked articles."""
        mock_fetch.return_value = [
            {
                'title': 'Quantum Cryptography',
                'url': 'https://en.wikipedia.org/wiki/Quantum_cryptography',
                'text': 'This is a test article. It has multiple sentences. Each sentence is short. ' * 50
            }
        ]
        
        result = main()
        self.assertEqual(result, 0)
        mock_fetch.assert_called_once()
        mock_save.assert_called_once()


if __name__ == '__main__':
    unittest.main()
