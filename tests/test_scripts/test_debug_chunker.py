"""Tests for scripts/analysis/debug_chunker.py."""

import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.analysis.debug_chunker import main


class TestDebugChunker(unittest.TestCase):
    """Test cases for debug_chunker script."""

    @patch('builtins.print')
    def test_debug_chunker_runs(self, mock_print: MagicMock) -> None:
        """Test debug chunker runs without errors."""
        result = main()
        self.assertEqual(result, 0)
        # Should have printed chunk analysis output
        mock_print.assert_called()


if __name__ == '__main__':
    unittest.main()
