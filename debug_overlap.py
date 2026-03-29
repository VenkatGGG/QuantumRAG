#!/usr/bin/env python3
"""Debug script to understand overlap calculation."""
import sys
sys.path.insert(0, '.')

from src.heuristic_chunker import HeuristicChunker

chunker = HeuristicChunker()

# Create a long text with many sentences
sentences = [f'This is sentence number {i} with sufficient words for overlap testing purposes.' for i in range(100)]
text = ' '.join(sentences)

# Split into sentences
sentences = chunker._split_into_sentences(text)

print(f"Total sentences: {len(sentences)}")
print()

# Check token counts for individual sentences
for i in range(3):
    tokens = chunker.count_tokens(sentences[i])
    print(f"Sentence {i}: {tokens} tokens")
    print(f"  Text: {sentences[i]}")
    print()

# Check combined sentences
print("Combined sentences:")
for n in range(1, 6):
    combined = " ".join(sentences[:n])
    tokens = chunker.count_tokens(combined)
    print(f"  {n} sentences: {tokens} tokens")

print()

# Now trace through the chunking algorithm
CHUNK_SIZE = 500
OVERLAP = 50

# First chunk
print("=" * 60)
print("CHUNKING TRACE")
print("=" * 60)

end_idx_1 = chunker._find_chunk_boundary(sentences, 0, CHUNK_SIZE)
print(f"Chunk 1: sentences 0 to {end_idx_1-1} ({end_idx_1} sentences)")
chunk1_text = " ".join(sentences[0:end_idx_1])
chunk1_tokens = chunker.count_tokens(chunk1_text)
print(f"Chunk 1 tokens: {chunk1_tokens}")

# Calculate overlap sentences
overlap_sentences = chunker._find_sentences_for_overlap(sentences, end_idx_1, OVERLAP)
print(f"\nOverlap sentences calculated: {overlap_sentences}")

# Check what those sentences actually contribute
if overlap_sentences > 0:
    overlap_text = " ".join(sentences[end_idx_1 - overlap_sentences:end_idx_1])
    overlap_tokens = chunker.count_tokens(overlap_text)
    print(f"Overlap text tokens: {overlap_tokens}")
    print(f"Overlap text: {overlap_text[:100]}...")

# Next chunk starts at
next_start = end_idx_1 - overlap_sentences
print(f"\nChunk 2 starts at sentence: {next_start}")

# Build chunk 2
end_idx_2 = chunker._find_chunk_boundary(sentences, next_start, CHUNK_SIZE)
print(f"Chunk 2: sentences {next_start} to {end_idx_2-1}")
chunk2_text = " ".join(sentences[next_start:end_idx_2])
chunk2_tokens = chunker.count_tokens(chunk2_text)
print(f"Chunk 2 tokens: {chunk2_tokens}")

# Now calculate actual overlap between chunk1 and chunk2
print("\n" + "=" * 60)
print("ACTUAL OVERLAP CALCULATION")
print("=" * 60)

# Find longest suffix of chunk1 that's prefix of chunk2
max_len = min(len(chunk1_text), len(chunk2_text))
actual_overlap_text = ""
for i in range(max_len, 0, -1):
    if chunk2_text.startswith(chunk1_text[-i:]):
        actual_overlap_text = chunk1_text[-i:]
        break

if actual_overlap_text:
    actual_overlap_tokens = chunker.count_tokens(actual_overlap_text)
    print(f"Actual overlap tokens: {actual_overlap_tokens}")
    print(f"Expected: 50 (within 5 token tolerance = 45-55)")
    print(f"Difference: {abs(actual_overlap_tokens - 50)}")
    print(f"Within tolerance: {abs(actual_overlap_tokens - 50) <= 5}")
    print(f"Overlap text: {actual_overlap_text[:100]}...")
else:
    print("No overlap found!")
