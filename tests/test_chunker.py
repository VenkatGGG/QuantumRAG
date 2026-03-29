"""Tests for heuristic text chunker module."""
import pytest
from unittest.mock import Mock, patch

from src.heuristic_chunker import HeuristicChunker, chunk_text


class TestHeuristicChunker:
    """Test cases for HeuristicChunker class."""

    def test_chunk_respects_sentence_boundaries(self):
        """VAL-DATA-003: No chunk ends mid-sentence - respects sentence boundaries."""
        # Create text with multiple sentences
        text = "First sentence here. Second sentence is here. Third sentence comes next. Fourth sentence is last."
        
        chunker = HeuristicChunker()
        chunks = chunker.chunk(text)
        
        # Each chunk should end at a sentence boundary
        for chunk in chunks:
            # Chunk should not end mid-sentence (should end with period, question mark, or exclamation)
            stripped = chunk.strip()
            if stripped:  # Skip empty chunks
                assert stripped[-1] in '.!?', f"Chunk does not end at sentence boundary: '{stripped[-30:]}'"

    def test_chunk_size_500_tokens(self):
        """VAL-DATA-004: Each chunk contains at most 500 tokens."""
        # Create a long text with many sentences
        sentences = [f"This is sentence number {i} with some additional words to make it longer." for i in range(100)]
        text = " ".join(sentences)
        
        chunker = HeuristicChunker()
        chunks = chunker.chunk(text)
        
        # Each chunk should have at most 500 tokens
        for i, chunk in enumerate(chunks):
            token_count = chunker.count_tokens(chunk)
            assert token_count <= 500, f"Chunk {i} has {token_count} tokens, exceeds 500 limit"

    def test_overlap_50_tokens(self):
        """VAL-DATA-005: Consecutive chunks share exactly 50 tokens of overlap."""
        # Create a long text with many sentences
        sentences = [f"This is sentence number {i} with sufficient words for overlap testing purposes." for i in range(100)]
        text = " ".join(sentences)
        
        chunker = HeuristicChunker()
        chunks = chunker.chunk(text)
        
        # Skip if only one chunk
        if len(chunks) < 2:
            pytest.skip("Need at least 2 chunks to test overlap")
        
        # Check overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            chunk1 = chunks[i]
            chunk2 = chunks[i + 1]
            
            # Find the overlap by checking how much of chunk1's end appears in chunk2's start
            overlap_text = ""
            max_overlap = min(len(chunk1), len(chunk2))
            for j in range(max_overlap, 0, -1):
                if chunk2.startswith(chunk1[-j:]):
                    overlap_text = chunk1[-j:]
                    break
            
            overlap_tokens = chunker.count_tokens(overlap_text) if overlap_text else 0
            
            assert abs(overlap_tokens - 50) <= 5, f"Overlap between chunk {i} and {i+1} is {overlap_tokens} tokens, expected 50"

    def test_empty_text(self):
        """Verify empty input returns empty list."""
        chunker = HeuristicChunker()
        chunks = chunker.chunk("")
        assert chunks == [], "Empty text should return empty list"

    def test_single_sentence(self):
        """Verify single sentence smaller than chunk size returns as one chunk."""
        text = "This is a single short sentence."
        
        chunker = HeuristicChunker()
        chunks = chunker.chunk(text)
        
        assert len(chunks) == 1, "Single short sentence should produce exactly one chunk"
        assert chunks[0] == text, "Chunk should contain the full text"

    def test_exact_boundary(self):
        """Verify sentence ending exactly at 500 tokens is handled correctly."""
        # Create sentences that might align with 500-token boundary
        # Using a sentence that's roughly 100 tokens
        long_sentence = "The quick brown fox jumps over the lazy dog while the sun shines brightly on the green grass in the beautiful meadow where flowers bloom and birds sing their melodious songs creating a peaceful atmosphere. "
        
        # Repeat to create text around 500 tokens
        text = long_sentence * 6  # Should be around 600 tokens
        
        chunker = HeuristicChunker()
        chunks = chunker.chunk(text)
        
        # Should produce at least one chunk
        assert len(chunks) >= 1, "Should produce at least one chunk"
        
        # First chunk should be at most 500 tokens
        first_chunk_tokens = chunker.count_tokens(chunks[0])
        assert first_chunk_tokens <= 500, f"First chunk has {first_chunk_tokens} tokens, exceeds 500"

    def test_uses_same_tokenizer_as_embedding_model(self):
        """Verify chunker uses the same tokenizer as the embedding model for consistency."""
        from transformers import AutoTokenizer
        
        # The embedding model uses sentence-transformers/all-MiniLM-L6-v2
        expected_tokenizer = "sentence-transformers/all-MiniLM-L6-v2"
        
        chunker = HeuristicChunker()
        
        # Verify the tokenizer is loaded from the expected model
        assert chunker.tokenizer is not None, "Tokenizer should be loaded"
        
        # Count tokens using chunker's tokenizer
        test_text = "This is a test sentence for token counting."
        chunker_count = chunker.count_tokens(test_text)
        
        # Count tokens using the expected model's tokenizer directly
        expected_tokenizer_obj = AutoTokenizer.from_pretrained(expected_tokenizer)
        expected_count = len(expected_tokenizer_obj.encode(test_text))
        
        assert chunker_count == expected_count, f"Chunker uses different tokenizer: {chunker_count} vs {expected_count}"


class TestChunkTextFunction:
    """Test cases for the chunk_text convenience function."""

    def test_chunk_text_function(self):
        """Verify chunk_text convenience function works correctly."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunk_text(text)
        
        assert isinstance(chunks, list), "chunk_text should return a list"
        assert len(chunks) >= 1, "Should produce at least one chunk"
        
        # All chunks should end at sentence boundaries
        for chunk in chunks:
            stripped = chunk.strip()
            if stripped:
                assert stripped[-1] in '.!?'
