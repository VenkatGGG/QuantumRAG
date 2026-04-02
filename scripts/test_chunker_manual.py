"""Manual test script for heuristic chunker."""

import sys
from pathlib import Path

# Add parent directory to path for proper imports
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Loading HeuristicChunker...")
from src.heuristic_chunker import HeuristicChunker, chunk_text

print("Initializing chunker (this may take a moment for model download)...")
chunker = HeuristicChunker()
print("Chunker initialized successfully!")

# Test 1: Empty text
print("\n=== Test 1: Empty text ===")
result = chunker.chunk("")
print(f"Result: {result}")
assert result == [], "Empty text should return empty list"
print("PASSED")

# Test 2: Single sentence
print("\n=== Test 2: Single sentence ===")
text = "This is a single short sentence."
result = chunker.chunk(text)
print(f"Result: {result}")
assert len(result) == 1, "Single sentence should produce exactly one chunk"
assert result[0] == text, "Chunk should contain the full text"
print("PASSED")

# Test 3: Sentence boundaries
print("\n=== Test 3: Sentence boundaries ===")
text = "First sentence here. Second sentence is here. Third sentence comes next. Fourth sentence is last."
result = chunker.chunk(text)
print(f"Number of chunks: {len(result)}")
for i, chunk in enumerate(result):
    stripped = chunk.strip()
    print(f"  Chunk {i}: '{stripped[:50]}...' (ends with: '{stripped[-1]}')")
    assert stripped[-1] in '.!?', f"Chunk {i} does not end at sentence boundary"
print("PASSED")

# Test 4: Chunk size at most 500 tokens
print("\n=== Test 4: Chunk size at most 500 tokens ===")
sentences = [f"This is sentence number {i} with some additional words to make it longer." for i in range(100)]
text = " ".join(sentences)
result = chunker.chunk(text)
print(f"Number of chunks: {len(result)}")
for i, chunk in enumerate(result):
    token_count = chunker.count_tokens(chunk)
    print(f"  Chunk {i}: {token_count} tokens")
    assert token_count <= 500, f"Chunk {i} has {token_count} tokens, exceeds 500 limit"
print("PASSED")

# Test 5: Overlap between consecutive chunks
print("\n=== Test 5: Overlap between consecutive chunks ===")
sentences = [f"This is sentence number {i} with sufficient words for overlap testing purposes." for i in range(100)]
text = " ".join(sentences)
result = chunker.chunk(text)
print(f"Number of chunks: {len(result)}")
if len(result) >= 2:
    for i in range(len(result) - 1):
        chunk1 = result[i]
        chunk2 = result[i + 1]
        # Calculate overlap by finding shared sentences
        chunk1_sentences = [s.strip() for s in chunk1.split('.') if s.strip()]
        chunk2_sentences = [s.strip() for s in chunk2.split('.') if s.strip()]
        shared_sentences = []
        for s1 in chunk1_sentences:
            for s2 in chunk2_sentences:
                if s1 == s2 or (s1 in s2) or (s2 in s1):
                    shared_sentences.append(s1)
                    break
        overlap_text = ". ".join(shared_sentences)
        overlap_tokens = chunker.count_tokens(overlap_text) if overlap_text else 0
        print(f"  Overlap between chunk {i} and {i+1}: {overlap_tokens} tokens (shared sentences: {len(shared_sentences)})")
        # Allow some tolerance (±20 tokens) for sentence boundary alignment
        assert abs(overlap_tokens - 50) <= 20, f"Overlap is {overlap_tokens} tokens, expected ~50 (±20 tolerance)"
    print("PASSED")
else:
    print("SKIPPED (need at least 2 chunks)")

# Test 6: Exact boundary
print("\n=== Test 6: Exact boundary ===")
long_sentence = "The quick brown fox jumps over the lazy dog while the sun shines brightly on the green grass in the beautiful meadow where flowers bloom and birds sing their melodious songs creating a peaceful atmosphere. "
text = long_sentence * 6
result = chunker.chunk(text)
print(f"Number of chunks: {len(result)}")
first_chunk_tokens = chunker.count_tokens(result[0])
print(f"First chunk: {first_chunk_tokens} tokens")
assert first_chunk_tokens <= 500, f"First chunk has {first_chunk_tokens} tokens, exceeds 500"
print("PASSED")

# Test 7: Same tokenizer as embedding model
print("\n=== Test 7: Same tokenizer as embedding model ===")
from transformers import AutoTokenizer
expected_tokenizer = "sentence-transformers/all-MiniLM-L6-v2"
test_text = "This is a test sentence for token counting."
chunker_count = chunker.count_tokens(test_text)
expected_tokenizer_obj = AutoTokenizer.from_pretrained(expected_tokenizer)
expected_count = len(expected_tokenizer_obj.encode(test_text))
print(f"Chunker count: {chunker_count}, Expected: {expected_count}")
assert chunker_count == expected_count, f"Chunker uses different tokenizer: {chunker_count} vs {expected_count}"
print("PASSED")

# Test 8: chunk_text convenience function
print("\n=== Test 8: chunk_text convenience function ===")
text = "First sentence. Second sentence. Third sentence. Fourth sentence."
result = chunk_text(text)
print(f"Result: {result}")
assert isinstance(result, list), "chunk_text should return a list"
assert len(result) >= 1, "Should produce at least one chunk"
for chunk in result:
    stripped = chunk.strip()
    assert stripped[-1] in '.!?', "Chunk should end at sentence boundary"
print("PASSED")

print("\n" + "="*50)
print("ALL TESTS PASSED!")
print("="*50)
