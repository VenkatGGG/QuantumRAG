"""Debug script for overlap calculation."""
import sys
sys.path.insert(0, '/Users/sri/Desktop/silly_experiments/Droid_Project')

from src.heuristic_chunker import HeuristicChunker

# Create test text
sentences = [f"This is sentence number {i} with sufficient words for overlap testing purposes." for i in range(100)]
text = " ".join(sentences)

chunker = HeuristicChunker()

# First, let's verify individual sentence token counts
print("=== Individual sentence token counts ===")
sentences_list = chunker._split_into_sentences(text)
for i in range(28, 33):
    tokens = chunker.count_tokens(sentences_list[i])
    print(f"Sentence {i}: {tokens} tokens - '{sentences_list[i]}'")

# Check what a single sentence tokenizes to
single_sent = "This is sentence number 28 with sufficient words for overlap testing purposes."
print(f"\nSingle sentence 28: {chunker.count_tokens(single_sent)} tokens")

# Check two sentences joined
two_sents = "This is sentence number 28 with sufficient words for overlap testing purposes. This is sentence number 29 with sufficient words for overlap testing purposes."
print(f"Two sentences (28, 29): {chunker.count_tokens(two_sents)} tokens")

# Check three sentences joined
three_sents = "This is sentence number 28 with sufficient words for overlap testing purposes. This is sentence number 29 with sufficient words for overlap testing purposes. This is sentence number 30 with sufficient words for overlap testing purposes."
print(f"Three sentences (28, 29, 30): {chunker.count_tokens(three_sents)} tokens")

# Manual calculation
sent_28_tokens = chunker.count_tokens(sentences_list[28])
sent_29_tokens = chunker.count_tokens(sentences_list[29])
sent_30_tokens = chunker.count_tokens(sentences_list[30])
print(f"\nManual calculation:")
print(f"  Sent 28: {sent_28_tokens}")
print(f"  Sent 29: {sent_29_tokens} + 1 separator = {sent_29_tokens + 1}")
print(f"  Sent 30: {sent_30_tokens} + 1 separator = {sent_30_tokens + 1}")
print(f"  Total: {sent_28_tokens + sent_29_tokens + 1 + sent_30_tokens + 1}")

# Run chunking
print("\n\n=== Chunking results ===")
chunks = chunker.chunk(text)
print(f"Number of chunks: {len(chunks)}")

if len(chunks) >= 2:
    chunk1 = chunks[0]
    chunk2 = chunks[1]
    
    # Find the actual overlap by string matching
    max_overlap_len = min(len(chunk1), len(chunk2))
    actual_overlap_text = ""
    for i in range(max_overlap_len, 0, -1):
        if chunk1[-i:] == chunk2[:i]:
            actual_overlap_text = chunk2[:i]
            break
    
    print(f"\nActual overlap text: '{actual_overlap_text}'")
    print(f"Actual overlap tokens: {chunker.count_tokens(actual_overlap_text)}")
    
    # Let's manually check what the overlap calculation found
    print("\n=== _find_sentences_for_overlap calculation ===")
    overlap_sents, overlap_toks = chunker._find_sentences_for_overlap(sentences_list, 31, 50)
    print(f"Overlap sentences found: {overlap_sents}")
    print(f"Overlap tokens found: {overlap_toks}")
    print(f"Sentence indices: {list(range(31 - overlap_sents, 31))}")
