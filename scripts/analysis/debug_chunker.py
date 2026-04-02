#!/usr/bin/env python3
"""Debug chunking to understand overlap issues."""

import sys

from src.heuristic_chunker import HeuristicChunker


def main() -> int:
    """Run the debug chunking analysis."""
    # Simple test case
    chunker = HeuristicChunker()

    # Create text with many similar sentences
    sentences = [f"This is sentence number {i} with sufficient words for overlap testing purposes." for i in range(30)]
    text = " ".join(sentences)

    print(f"Total text tokens: {chunker.count_tokens(text)}")
    print()

    chunks = chunker.chunk(text)
    print(f"Number of chunks: {len(chunks)}")
    print()

    for i, chunk in enumerate(chunks):
        tokens = chunker.count_tokens(chunk)
        print(f"Chunk {i}: {tokens} tokens")
        print(f"  Start: {chunk[:80]}...")
        print(f"  End: ...{chunk[-80:]}")
        print()

    # Check overlaps
    print("=" * 60)
    print("OVERLAP ANALYSIS")
    print("=" * 60)
    for i in range(len(chunks) - 1):
        chunk1 = chunks[i]
        chunk2 = chunks[i + 1]
        
        # Find overlap
        overlap_text = ""
        max_overlap = min(len(chunk1), len(chunk2))
        for j in range(max_overlap, 0, -1):
            if chunk2.startswith(chunk1[-j:]):
                overlap_text = chunk1[-j:]
                break
        
        overlap_tokens = chunker.count_tokens(overlap_text) if overlap_text else 0
        print(f"Overlap {i}->{i+1}: {overlap_tokens} tokens")
        if overlap_text:
            print(f"  Text: '{overlap_text[:60]}...'")
        else:
            print(f"  No overlap found!")
            # Show end of chunk1 and start of chunk2
            print(f"  Chunk {i} end: ...{chunk1[-60:]}")
            print(f"  Chunk {i+1} start: {chunk2[:60]}...")
        print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
