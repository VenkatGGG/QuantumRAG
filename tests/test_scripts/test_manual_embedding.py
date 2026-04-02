"""Tests for scripts/manual_tests/manual_embedding.py."""

import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.manual_tests.manual_embedding import main, test_embedding_pipeline


class TestManualEmbedding(unittest.TestCase):
    """Test cases for manual_embedding script."""

    @patch('builtins.print')
    def test_main_runs_successfully(self, mock_print: MagicMock) -> None:
        """Test manual embedding main runs without errors."""
        result = main()
        self.assertEqual(result, 0)

    @patch('builtins.print')
    def test_pipeline_outputs_success(self, mock_print: MagicMock) -> None:
        """Test pipeline produces success output."""
        test_embedding_pipeline()
        
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(
            any('passed' in call.lower() or 'All manual embedding tests passed' in call for call in print_calls),
            "Expected success message"
        )


if __name__ == '__main__':
    unittest.main()
