"""Tests for scripts/manual_tests/manual_vector_store.py."""

import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.manual_tests.manual_vector_store import main, test_vector_store


class TestManualVectorStore(unittest.TestCase):
    """Test cases for manual_vector_store script."""

    @patch('builtins.print')
    def test_main_runs_successfully(self, mock_print: MagicMock) -> None:
        """Test manual vector store main runs without errors."""
        result = main()
        self.assertEqual(result, 0)

    @patch('builtins.print')
    def test_vector_store_outputs_success(self, mock_print: MagicMock) -> None:
        """Test vector store tests produce success output."""
        test_vector_store()
        
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(
            any('passed' in call.lower() or 'All manual vector store tests passed' in call for call in print_calls),
            "Expected success message"
        )


if __name__ == '__main__':
    unittest.main()
