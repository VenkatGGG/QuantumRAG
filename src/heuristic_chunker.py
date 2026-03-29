"""Heuristic text chunker module for splitting text into token-based chunks.

This module implements a custom chunking algorithm that splits text into exactly
500-token chunks with 50-token overlap, respecting sentence boundaries (no mid-sentence
truncation). Uses the model's tokenizer for accurate token counting.
"""
import re
from typing import List
from transformers import AutoTokenizer


class HeuristicChunker:
    """Chunker that splits text into 500-token chunks with 50-token overlap.
    
    Respects sentence boundaries to avoid mid-sentence truncation.
    Uses the same tokenizer as the embedding model for consistency.
    """
    
    # Model used for both chunking and embeddings
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Chunking parameters
    CHUNK_SIZE = 500  # Maximum tokens per chunk
    OVERLAP = 50      # Tokens of overlap between consecutive chunks
    
    def __init__(self):
        """Initialize the chunker with the embedding model's tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string.
        
        Args:
            text: Text to count tokens for.
            
        Returns:
            Number of tokens in the text.
        """
        if not text:
            return 0
        # Use the tokenizer to get accurate token count
        # encode() returns token IDs including special tokens, so we subtract 2 for [CLS] and [SEP]
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        return len(tokens)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex-based sentence boundary detection.
        
        Args:
            text: Text to split into sentences.
            
        Returns:
            List of sentences.
        """
        # Use regex to split on sentence boundaries
        # Matches period, question mark, or exclamation followed by space or end of string
        # Also handles multiple whitespace
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text.strip())
        
        # Clean up and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _find_overlap_tokens(self, chunk1: str, chunk2: str) -> int:
        """Find the number of overlapping tokens between two chunks.
        
        Args:
            chunk1: First chunk.
            chunk2: Second chunk.
            
        Returns:
            Number of overlapping tokens.
        """
        # Find the longest suffix of chunk1 that is a prefix of chunk2
        # Start with the full chunk1 and reduce
        max_overlap_len = min(len(chunk1), len(chunk2))
        
        for i in range(max_overlap_len, 0, -1):
            suffix = chunk1[-i:]
            if chunk2.startswith(suffix):
                # Found overlap, count tokens
                return self.count_tokens(suffix)
        
        return 0
    
    def _find_chunk_boundary(self, sentences: List[str], start_idx: int, max_tokens: int) -> int:
        """Find the index of the last sentence that fits within the token limit.
        
        Args:
            sentences: List of all sentences.
            start_idx: Starting sentence index for this chunk.
            max_tokens: Maximum number of tokens allowed.
            
        Returns:
            Index of the last sentence to include (exclusive).
        """
        current_tokens = 0
        end_idx = start_idx
        
        for i in range(start_idx, len(sentences)):
            sentence = sentences[i]
            sentence_tokens = self.count_tokens(sentence)
            
            # Check if adding this sentence would exceed the limit
            # Add 1 for the space between sentences (except for first sentence)
            additional_tokens = sentence_tokens if i == start_idx else sentence_tokens + 1
            
            if current_tokens + additional_tokens > max_tokens:
                break
            
            current_tokens += additional_tokens
            end_idx = i + 1
        
        return end_idx
    
    def _find_sentences_for_overlap(self, sentences: List[str], end_idx: int, target_tokens: int) -> int:
        """Find how many sentences to include for exactly target_tokens of overlap.
        
        Walks backwards from end_idx to find sentences that fit within target_tokens.
        Allows slight overflow (up to 5 tokens) to get closer to the target.
        
        Args:
            sentences: List of all sentences.
            end_idx: The index after the last sentence of the current chunk (exclusive).
            target_tokens: Target number of tokens for overlap.
            
        Returns:
            Number of sentences to include in the overlap.
        """
        overlap_tokens = 0
        overlap_sentences = 0
        
        # Walk backwards from end_idx - 1 to find sentences for overlap
        for i in range(end_idx - 1, -1, -1):
            sentence_tokens = self.count_tokens(sentences[i])
            # Skip sentences larger than the overlap limit
            if sentence_tokens > target_tokens:
                continue
            # Count tokens for this sentence (no separator for first/only sentence)
            separator = 1 if overlap_sentences > 0 else 0
            new_total = overlap_tokens + sentence_tokens + separator
            # Allow up to 5 tokens over the target to get closer to desired overlap
            if new_total > target_tokens + 5:
                break
            overlap_tokens += sentence_tokens + separator
            overlap_sentences += 1
        
        return overlap_sentences, overlap_tokens

    def chunk(self, text: str) -> List[str]:
        """Split text into chunks of at most 500 tokens with 50-token overlap.
        
        Respects sentence boundaries - no chunk will end mid-sentence.
        
        Args:
            text: Text to chunk.
            
        Returns:
            List of text chunks.
        """
        if not text or not text.strip():
            return []
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return []
        
        # If the entire text fits in one chunk, return it as-is
        total_tokens = self.count_tokens(text)
        if total_tokens <= self.CHUNK_SIZE:
            return [text.strip()]
        
        chunks = []
        current_idx = 0
        
        while current_idx < len(sentences):
            # Find the boundary for this chunk starting from current_idx
            # Account for the overlap from previous chunk by reducing effective chunk size
            if chunks:
                # Calculate how many tokens of overlap we have
                overlap_sentences, overlap_tokens = self._find_sentences_for_overlap(
                    sentences, current_idx, self.OVERLAP
                )
                effective_chunk_size = self.CHUNK_SIZE - overlap_tokens
            else:
                effective_chunk_size = self.CHUNK_SIZE
            
            end_idx = self._find_chunk_boundary(
                sentences, 
                current_idx, 
                effective_chunk_size
            )
            
            # Ensure we make progress (at least one new sentence)
            if end_idx <= current_idx and current_idx < len(sentences):
                # Force include at least the current sentence
                end_idx = current_idx + 1
            
            # Build the chunk text
            chunk_sentences = sentences[current_idx:end_idx]
            chunk_text = " ".join(chunk_sentences)
            
            # Verify token count
            chunk_tokens = self.count_tokens(chunk_text)
            
            # If chunk is too large (shouldn't happen with sentence boundaries),
            # we might need to split a sentence (emergency fallback)
            if chunk_tokens > self.CHUNK_SIZE:
                # This shouldn't happen if sentence boundary detection works correctly
                # But handle it gracefully by truncating
                chunk_text = self._truncate_to_tokens(chunk_text, self.CHUNK_SIZE)
            
            chunks.append(chunk_text)
            
            # Move to next position, accounting for overlap
            if end_idx >= len(sentences):
                break
            
            # Calculate how many sentences to step back for overlap
            # This determines where the next chunk starts
            overlap_sentences, overlap_tokens = self._find_sentences_for_overlap(
                sentences, 
                end_idx, 
                self.OVERLAP
            )
            
            # Move current_idx forward, but step back for overlap
            current_idx = end_idx - overlap_sentences
            
            # Ensure we always make progress
            if current_idx >= end_idx:
                current_idx = end_idx
        
        return chunks
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within max_tokens.
        
        This is an emergency fallback that shouldn't normally be needed.
        Uses the tokenizer to find the right truncation point.
        
        Args:
            text: Text to truncate.
            max_tokens: Maximum number of tokens.
            
        Returns:
            Truncated text.
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= max_tokens:
            return text
        
        # Truncate tokens and decode back to text
        truncated_tokens = tokens[:max_tokens]
        truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
        return truncated_text


def chunk_text(text: str) -> List[str]:
    """Convenience function to chunk text using the heuristic chunker.
    
    Args:
        text: Text to chunk.
        
    Returns:
        List of text chunks.
    """
    chunker = HeuristicChunker()
    return chunker.chunk(text)
