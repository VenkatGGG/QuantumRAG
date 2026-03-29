#!/usr/bin/env python3
"""Simple debug script for overlap."""
import sys
sys.path.insert(0, '.')

from src.heuristic_chunker import HeuristicChunker

chunker = HeuristicChunker()

# Create test text
sentences = [f'Sentence number {i} has exactly ten tokens in it.' for i in range(50)]
text = ' '.join(sentences)

# Split into sentences
sentences = chunker._split_into_sentences(text)

print(f"Total sentences: {len(sentences)}")
print(f"Sentence 0: '{sentences[0]}'")
print(f"Sentence 0 tokens: {chunker.count_tokens(sentences[0])}")

# Check individual sentence tokens
for i in range(min(5, len(sentences))):
    print(f"Sentence {i}: {chunker.count_tokens(sentences[i])} tokens")

# Now chunk
chunks = chunker.chunk(text)
print(f"\nNumber of chunks: {len(chunks)}")

# Check overlap between first two chunks
if len(chunks) >= 2:
    chunk1, chunk2 = chunks[0], chunks[1]
    overlap_tokens = chunker._find_overlap_tokens(chunk1, chunk2)
    print(f"\nChunk 1 tokens: {chunker.count_tokens(chunk1)}")
    print(f"Chunk 2 tokens: {chunker.count_tokens(chunk2)}")
    print(f"Overlap tokens: {overlap_tokens}")
    print(f"Expected: 50, diff: {abs(overlap_tokens - 50)}")
    
    # Show what the overlap is
    max_len = min(len(chunk1), len(chunk2))
    for i in range(max_len, 0, -1):
        if chunk2.startswith(chunk1[-i:]):
            overlap_text = chunk1[-i:]
            print(f"Overlap text: '{overlap_text[:100]}...'")
            break
