"""Tests for scripts/pipeline/run_full_pipeline.py."""

import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.pipeline.run_full_pipeline import main


class TestRunFullPipeline(unittest.TestCase):
    """Test cases for run_full_pipeline script."""

    @patch('scripts.pipeline.run_full_pipeline.fetch_articles')
    @patch('scripts.pipeline.run_full_pipeline.save_articles')
    @patch('scripts.pipeline.run_full_pipeline.HeuristicChunker')
    @patch('scripts.pipeline.run_full_pipeline.EmbeddingPipeline')
    @patch('scripts.pipeline.run_full_pipeline.VectorStore')
    @patch('builtins.print')
    def test_pipeline_runs_successfully(
        self, mock_print: MagicMock, mock_store: MagicMock,
        mock_pipeline: MagicMock, mock_chunker: MagicMock,
        mock_save: MagicMock, mock_fetch: MagicMock
    ) -> None:
        """Test full pipeline runs with mocked dependencies."""
        # Mock fetch_articles
        mock_fetch.return_value = [
            {
                'title': 'Test Article',
                'url': 'https://test.com',
                'text': 'This is a test sentence. ' * 20
            }
        ]
        
        # Mock chunker
        mock_chunker_instance = MagicMock()
        mock_chunker_instance.chunk.return_value = ['chunk1', 'chunk2']
        mock_chunker.return_value = mock_chunker_instance
        
        # Mock pipeline
        mock_pipeline_instance = MagicMock()
        import numpy as np
        mock_pipeline_instance.embed_batch.return_value = np.array([[1.0]*384, [2.0]*384], dtype=np.float32)
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Mock store
        mock_store_instance = MagicMock()
        mock_store_instance.size = 2
        mock_store_instance.dimension = 384
        mock_store.return_value = mock_store_instance
        
        result = main()
        self.assertEqual(result, 0)
        mock_fetch.assert_called_once()


if __name__ == '__main__':
    unittest.main()
