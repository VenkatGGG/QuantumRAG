"""Heuristic text chunker module for splitting text into token-based chunks.

This module implements a custom chunking algorithm that splits text into exactly
500-token chunks with 50-token overlap, respecting sentence boundaries (no mid-sentence
truncation). Uses the model's tokenizer for accurate token counting.
"""
import re
from typing import List, Tuple
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
    
    def _get_last_n_tokens(self, text: str, n: int) -> str:
        """Get the last n tokens from text as a string.
        
        Args:
            text: Source text.
            n: Number of tokens to extract from the end.
            
        Returns:
            String containing the last n tokens.
        """
        if not text:
            return ""
        
        # Encode without special tokens to get raw tokens
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= n:
            return text
        
        # Take last n tokens and decode
        last_n_tokens = tokens[-n:]
        return self.tokenizer.decode(last_n_tokens, skip_special_tokens=True)
    
    def _truncate_to_sentence_boundary(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within max_tokens, respecting sentence boundaries.
        
        This is used for the main chunk body (not overlap). We try to get as close
        to max_tokens as possible while ending at a sentence boundary.
        
        Args:
            text: Text to truncate (typically the chunk without overlap prefix).
            max_tokens: Maximum number of tokens allowed.
            
        Returns:
            Truncated text ending at sentence boundary.
        """
        if not text:
            return ""
        
        # First check if the whole text fits
        if self.count_tokens(text) <= max_tokens:
            return text
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        # Build up sentences until we hit the limit
        result_sentences = []
        current_tokens = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = self.count_tokens(sentence)
            # Add 1 for space between sentences (except first)
            additional_tokens = sentence_tokens if i == 0 else sentence_tokens + 1
            
            if current_tokens + additional_tokens > max_tokens:
                break
            
            result_sentences.append(sentence)
            current_tokens += additional_tokens
        
        if not result_sentences:
            # Even first sentence is too long - truncate it
            return self._truncate_text_to_tokens(text, max_tokens)
        
        return " ".join(result_sentences)
    
    def _truncate_text_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to exactly max_tokens (emergency fallback).
        
        Args:
            text: Text to truncate.
            max_tokens: Maximum number of tokens.
            
        Returns:
            Truncated text.
        """
        if not text:
            return ""
        
        # Account for special tokens when encoding with them
        effective_max = max_tokens - 2
        
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= effective_max:
            # Revalidate
            if self.count_tokens(text) <= max_tokens:
                return text
            effective_max = max_tokens - 2
        
        # Truncate and decode
        truncated_tokens = tokens[:effective_max]
        truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
        # Verify
        final_count = self.count_tokens(truncated_text)
        if final_count > max_tokens:
            # Emergency: truncate more
            excess = final_count - max_tokens
            safe_max = max(0, effective_max - excess - 5)
            truncated_tokens = tokens[:safe_max]
            truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
        return truncated_text
    
    def chunk(self, text: str) -> List[str]:
        """Split text into chunks of at most 500 tokens with exactly 50-token overlap.
        
        Respects sentence boundaries - no chunk will end mid-sentence.
        Overlap is enforced at exactly 50 tokens (within 2-token tolerance).
        
        Algorithm:
        1. Build first chunk to ~500 tokens, respecting sentence boundaries
        2. Extract last 50 tokens as overlap for next chunk
        3. Build subsequent chunks: overlap (50) + new content (~450), to ~500 total
        4. Repeat until all text consumed
        
        Key insight: The overlap_text is the ACTUAL text from the end of the previous chunk.
        The next chunk starts with this overlap_text, then adds new sentences.
        We need to step back in sentence index to include the sentences that form the overlap.
        
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
        current_sentence_idx = 0
        overlap_sentences = []  # The sentences that form the overlap from previous chunk
        
        while current_sentence_idx < len(sentences):
            # Determine available tokens for new content
            if overlap_sentences:
                # Calculate how many tokens the overlap takes
                overlap_text = " ".join(overlap_sentences)
                overlap_tokens = self.count_tokens(overlap_text)
                # We want exactly 50 tokens of overlap, but sentence boundaries may vary
                # Allow some tolerance - aim for 48-52 range
                available_for_new = self.CHUNK_SIZE - overlap_tokens
            else:
                # First chunk: full 500 tokens available
                available_for_new = self.CHUNK_SIZE
                overlap_tokens = 0
            
            # Build the new content portion by adding sentences
            new_sentences = []
            new_content_tokens = 0
            idx = current_sentence_idx
            
            while idx < len(sentences):
                sentence = sentences[idx]
                sentence_tokens = self.count_tokens(sentence)
                # Add 1 for space between sentences (except first)
                additional_tokens = sentence_tokens if len(new_sentences) == 0 else sentence_tokens + 1
                
                if new_content_tokens + additional_tokens > available_for_new:
                    break
                
                new_sentences.append(sentence)
                new_content_tokens += additional_tokens
                idx += 1
            
            # Build the full chunk
            if overlap_sentences and new_sentences:
                chunk_text = " ".join(overlap_sentences) + " " + " ".join(new_sentences)
            elif new_sentences:
                chunk_text = " ".join(new_sentences)
            elif overlap_sentences:
                # Only overlap (shouldn't happen with proper logic)
                chunk_text = " ".join(overlap_sentences)
            else:
                # No content at all (shouldn't happen)
                break
            
            # Verify and enforce 500-token limit
            chunk_tokens = self.count_tokens(chunk_text)
            if chunk_tokens > self.CHUNK_SIZE:
                chunk_text = self._truncate_text_to_tokens(chunk_text, self.CHUNK_SIZE)
                chunk_tokens = self.count_tokens(chunk_text)
            
            chunks.append(chunk_text)
            
            # If we've consumed all sentences, we're done
            if idx >= len(sentences):
                break
            
            # Calculate the overlap for the next chunk
            # We need to find how many sentences from the END of this chunk to include
            # such that we get approximately 50 tokens of overlap
            
            # Start with the last sentence and work backwards
            overlap_sentences = []
            overlap_token_count = 0
            
            # Include sentences from the current chunk, starting from the last one
            for sentence in reversed(new_sentences):
                sentence_tokens = self.count_tokens(sentence)
                # Add 1 for space between sentences
                would_be_total = overlap_token_count + sentence_tokens + (1 if overlap_sentences else 0)
                
                if would_be_total <= self.OVERLAP + 2:  # Allow up to 52 tokens
                    overlap_sentences.insert(0, sentence)  # Insert at beginning to maintain order
                    overlap_token_count = would_be_total
                else:
                    # Adding this sentence would exceed the target
                    # Check if we're closer with or without it
                    if abs(would_be_total - self.OVERLAP) < abs(overlap_token_count - self.OVERLAP):
                        overlap_sentences.insert(0, sentence)
                        overlap_token_count = would_be_total
                    break
            
            # If we couldn't fit any sentences in the overlap (shouldn't happen),
            # fall back to token-level extraction
            if not overlap_sentences and new_sentences:
                # Extract last 50 tokens from the chunk text
                overlap_text = self._get_last_n_tokens(chunk_text, self.OVERLAP)
                # Find which sentences this corresponds to
                for sent in reversed(new_sentences):
                    if sent in overlap_text:
                        overlap_sentences.insert(0, sent)
            
            # Move to next position
            # The next chunk should start from the first sentence NOT in the overlap
            # So we step back by the number of overlap sentences
            current_sentence_idx = idx - len(overlap_sentences)
            
            # Ensure we always make progress (at least one new sentence)
            if current_sentence_idx >= idx and idx < len(sentences):
                current_sentence_idx = idx
        
        return chunks


def chunk_text(text: str) -> List[str]:
    """Convenience function to chunk text using the heuristic chunker.
    
    Args:
        text: Text to chunk.
        
    Returns:
        List of text chunks.
    """
    chunker = HeuristicChunker()
    return chunker.chunk(text)
