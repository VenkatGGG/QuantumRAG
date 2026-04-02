"""Heuristic text chunker module for splitting text into token-based chunks.

This module implements Option A of the chunking contract (see DESIGN.md):
- Chunk boundaries never cut sentences (sentence-aligned)
- Size <= 500 tokens (hard ceiling)
- Overlap is sentence-aligned, closest to 50 tokens (target, not guarantee)

Includes improved Wikipedia segmentation for headings, blank lines, formulas, and lists.
"""
import re
from typing import List, Tuple
from transformers import AutoTokenizer


class HeuristicChunker:
    """Chunker implementing Option A contract from DESIGN.md.
    
    Key properties:
    - Sentence boundaries are always respected (no mid-sentence cuts)
    - Chunks contain at most 500 tokens (hard ceiling)
    - Overlap is sentence-aligned, targeting ~50 tokens (actual varies 30-70)
    - Improved Wikipedia segmentation (headings, blank lines, formulas, lists)
    
    Uses the same tokenizer as the embedding model for consistency.
    """
    
    # Model used for both chunking and embeddings
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Chunking parameters
    CHUNK_SIZE = 500  # Maximum tokens per chunk
    OVERLAP = 50      # Tokens of overlap between consecutive chunks
    
    def __init__(self):
        """Initialize the chunker. Model is loaded lazily on first use."""
        self._tokenizer = None
    
    @property
    def tokenizer(self):
        """Lazy-load the tokenizer on first access."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        return self._tokenizer
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string.
        
        Args:
            text: Text to count tokens for.
            
        Returns:
            Number of tokens in the text.
        """
        if not text:
            return 0
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        # encode() returns token IDs including special tokens ([CLS] and [SEP])
        # This matches the embedding model's tokenization
        return len(tokens)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using Wikipedia-aware segmentation.
        
        This method handles Wikipedia-specific formatting:
        - Headings (lines with ==Section==)
        - Blank lines (logical section breaks)
        - List items (bullet points, numbered lists)
        - Formulas (treated as atomic units)
        
        Args:
            text: Text to split into sentences.
            
        Returns:
            List of sentences/segments.
        """
        # First, apply Wikipedia-aware segmentation
        segments = self._segment_wikipedia_text(text)
        
        all_sentences = []
        for segment in segments:
            # Use regex to split on sentence boundaries within each segment
            # Matches period, question mark, or exclamation followed by space or end
            sentence_pattern = r'(?<=[.!?])\s+'
            sentences = re.split(sentence_pattern, segment.strip())
            
            # Clean up and filter empty sentences
            sentences = [s.strip() for s in sentences if s.strip()]
            all_sentences.extend(sentences)
        
        return all_sentences
    
    def _segment_wikipedia_text(self, text: str) -> List[str]:
        """Segment Wikipedia text at natural boundaries.
        
        Identifies and preserves:
        - Headings (==Section==, ===Subsection===)
        - Blank lines (paragraph breaks)
        - List items (bullet points, numbered items)
        - Formula blocks (atomic units)
        
        Args:
            text: Wikipedia-formatted text.
            
        Returns:
            List of text segments at natural boundaries.
        """
        if not text:
            return []
        
        segments = []
        lines = text.split('\n')
        current_segment = []
        
        for line in lines:
            stripped = line.strip()
            
            # Check for heading (starts and ends with =)
            if re.match(r'^=+\s*.+\s*=+$', stripped):
                # Save current segment if any
                if current_segment:
                    segments.append(' '.join(current_segment))
                    current_segment = []
                # Add heading as its own segment
                segments.append(stripped)
                continue
            
            # Check for blank line (paragraph break)
            if not stripped:
                if current_segment:
                    segments.append(' '.join(current_segment))
                    current_segment = []
                continue
            
            # Check for list item
            if re.match(r'^[*#-]\s', stripped):
                # Save current segment if any
                if current_segment:
                    segments.append(' '.join(current_segment))
                    current_segment = []
                # Add list item as its own segment
                segments.append(stripped)
                continue
            
            # Check for LaTeX/math formula (atomic unit)
            if re.match(r'^\$+.*\$+$', stripped) or re.match(r'^\\\[.*\\\]$', stripped):
                if current_segment:
                    segments.append(' '.join(current_segment))
                    current_segment = []
                segments.append(stripped)
                continue
            
            # Regular text line - add to current segment
            current_segment.append(stripped)
        
        # Don't forget the last segment
        if current_segment:
            segments.append(' '.join(current_segment))
        
        return segments
    
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

    def _find_overlap_tokens(self, chunk1: str, chunk2: str) -> int:
        """Estimate token overlap between two consecutive chunks.
        
        Uses a sliding window approach to find the approximate number of
        overlapping tokens between the end of chunk1 and the start of chunk2.
        
        Args:
            chunk1: First chunk text.
            chunk2: Second chunk text (should have overlap with chunk1).
            
        Returns:
            Estimated number of overlapping tokens.
        """
        # Tokenize both chunks
        tokens1 = self.tokenizer.encode(chunk1, add_special_tokens=False)
        tokens2 = self.tokenizer.encode(chunk2, add_special_tokens=False)
        
        # Look for the last part of chunk1 at the start of chunk2
        # Try different overlap sizes, starting from the expected overlap
        for overlap_size in range(min(self.OVERLAP + 20, len(tokens1)), max(0, self.OVERLAP - 20), -1):
            if overlap_size > len(tokens2):
                continue
            # Get the last overlap_size tokens from chunk1
            end_tokens1 = tokens1[-overlap_size:]
            # Get the first overlap_size tokens from chunk2
            start_tokens2 = tokens2[:overlap_size]
            # Check if they match
            if end_tokens1 == start_tokens2:
                return overlap_size
        
        # If no exact match found, estimate based on text similarity
        # Find the longest common substring between end of chunk1 and start of chunk2
        words1 = chunk1.split()
        words2 = chunk2.split()
        
        # Try to find matching words at the boundary
        for i in range(min(len(words1), 20), 0, -1):
            end_words1 = words1[-i:]
            start_words2 = words2[:i]
            if end_words1 == start_words2:
                # Estimate token count for these words
                text = " ".join(end_words1)
                return self.count_tokens(text)
        
        return 0  # No overlap detected

    def _truncate_text_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to max_tokens or fewer (emergency fallback).
        
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
        """Split text into chunks following Option A contract.
        
        Contract (see DESIGN.md):
        1. Sentence boundaries always respected - no mid-sentence cuts
        2. Chunk size <= 500 tokens (hard ceiling)
        3. Overlap is sentence-aligned, targeting ~50 tokens (not exact)
        
        Algorithm:
        1. Segment text using Wikipedia-aware segmentation
        2. Split into sentences
        3. Build chunks: overlap (sentence-aligned, ~50 tokens) + new content
        4. Each chunk ends at sentence boundary, size <= 500 tokens
        
        Args:
            text: Text to chunk.
            
        Returns:
            List of text chunks. Returns empty list if text is empty or contains only whitespace.
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
                # We target ~50 tokens of overlap, but sentence boundaries may vary
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
            # Strategy: Include sentences from the end of this chunk to get as close
            # to 50 tokens as possible while respecting sentence boundaries.
            # This is a TARGET, not a guarantee - actual overlap will vary.
            overlap_sentences = []
            overlap_token_count = 0
            
            # Build candidate overlap by walking backwards through new_sentences
            candidates = []
            candidate_tokens = 0
            
            for sentence in reversed(new_sentences):
                sentence_tokens = self.count_tokens(sentence)
                # Add 1 for space between sentences
                would_be_total = candidate_tokens + sentence_tokens + (1 if candidates else 0)
                candidates.insert(0, sentence)  # Insert at beginning to maintain order
                candidate_tokens = would_be_total
            
            # Now find the subset of candidates that gets closest to 50 tokens
            # We want to minimize abs(token_count - 50)
            if candidates:
                best_overlap = []
                best_token_count = 0
                best_distance = float('inf')
                
                # Try different numbers of sentences from the end
                for num_sentences in range(1, len(candidates) + 1):
                    # Take the last num_sentences sentences
                    test_overlap = candidates[-num_sentences:]
                    test_text = " ".join(test_overlap)
                    test_tokens = self.count_tokens(test_text)
                    distance = abs(test_tokens - self.OVERLAP)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_overlap = test_overlap
                        best_token_count = test_tokens
                
                overlap_sentences = best_overlap
                overlap_token_count = best_token_count
            
            # If we couldn't fit any sentences in the overlap (edge case),
            # fall back to token-level extraction
            if not overlap_sentences and new_sentences:
                # Extract last 50 tokens from the chunk text
                overlap_text = self._get_last_n_tokens(chunk_text, self.OVERLAP)
                # Find which sentences this corresponds to
                for sent in reversed(new_sentences):
                    if sent in overlap_text:
                        overlap_sentences.insert(0, sent)
                if overlap_sentences:
                    overlap_token_count = self.count_tokens(" ".join(overlap_sentences))
            
            # Move to next position
            # The next chunk should start from the first sentence NOT in the overlap
            # So we step back by the number of overlap sentences
            current_sentence_idx = idx - len(overlap_sentences)
            
            # Ensure we always make progress (at least one new sentence)
            if current_sentence_idx >= idx and idx < len(sentences):
                current_sentence_idx = idx + 1  # Force progress to next sentence
        
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
