"""Debug script for overlap calculation."""
import sys
sys.path.insert(0, '/Users/sri/Desktop/silly_experiments/Droid_Project')

from src.heuristic_chunker import HeuristicChunker

# Create test text
sentences = [f"This is sentence number {i} with sufficient words for overlap testing purposes." for i in range(100)]
text = " ".join(sentences)

chunker = HeuristicChunker()

# Run chunking
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
    
    print(f"\n=== Chunk 0 (last 200 chars) ===")
    print(f"'{chunk1[-200:]}'")
    
    print(f"\n=== Chunk 1 (first 200 chars) ===")
    print(f"'{chunk2[:200]}'")
    
    print(f"\n=== Actual Overlap Text ===")
    print(f"'{actual_overlap_text}'")
    print(f"Length: {len(actual_overlap_text)} chars")
    
    overlap_tokens = chunker.count_tokens(actual_overlap_text)
    print(f"Token count: {overlap_tokens}")
    
    # Now let's see what sentences are in the overlap
    sentences_in_overlap = chunker._split_into_sentences(actual_overlap_text)
    print(f"\nSentences in overlap: {len(sentences_in_overlap)}")
    for i, sent in enumerate(sentences_in_overlap):
        tokens = chunker.count_tokens(sent)
        print(f"  Sentence {i}: {tokens} tokens - '{sent[:60]}...'")
    
    # Calculate expected tokens in overlap
    expected_tokens = sum(chunker.count_tokens(s) for s in sentences_in_overlap)
    expected_tokens += len(sentences_in_overlap) - 1  # Add separators
    print(f"\nExpected tokens (with separators): {expected_tokens}")
    
    # Let's manually check what the overlap calculation found
    print("\n=== Manual overlap calculation ===")
    sentences_list = chunker._split_into_sentences(text)
    # First chunk ends at sentence 30 (index 31)
    overlap_sents, overlap_toks = chunker._find_sentences_for_overlap(sentences_list, 31, 50)
    print(f"Overlap sentences found: {overlap_sents}")
    print(f"Overlap tokens found: {overlap_toks}")
    print(f"Sentence indices: {list(range(31 - overlap_sents, 31))}")
    
    # Build what the overlap text SHOULD be
    expected_overlap_sents = sentences_list[31-overlap_sents:31]
    expected_overlap_text = " ".join(expected_overlap_sents)
    print(f"\nExpected overlap text: '{expected_overlap_text}'")
    print(f"Expected overlap tokens: {chunker.count_tokens(expected_overlap_text)}")
