"""Tests for scripts/manual_tests/manual_chunker.py."""

import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.manual_tests.manual_chunker import main


class TestManualChunker(unittest.TestCase):
    """Test cases for manual_chunker script."""

    @patch('builtins.print')
    def test_manual_chunker_all_tests_pass(self, mock_print: MagicMock) -> None:
        """Test manual chunker runs all tests successfully."""
        result = main()
        # main() may return None or 0 on success
        self.assertIn(result, [0, None])
        
        # Check that success message was printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(
            any('PASSED' in call or 'ALL TESTS PASSED' in call for call in print_calls),
            "Expected test success message"
        )


if __name__ == '__main__':
    unittest.main()
