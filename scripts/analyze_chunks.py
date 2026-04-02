#!/usr/bin/env python3
"""Analyze chunk statistics in the vector store or from chunking directly."""

import sys
from pathlib import Path

# Add parent directory to path for proper imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import h5py
import numpy as np
from typing import List, Dict, Any, Tuple
from src.heuristic_chunker import HeuristicChunker


def load_articles(path: str = "data/raw_articles.json") -> List[Dict[str, Any]]:
    """Load articles from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_corpus_from_articles() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Analyze chunk statistics by re-chunking the raw articles."""
    print("=" * 60)
    print("CHUNK ANALYSIS - Re-chunking raw articles")
    print("=" * 60)
    
    # Load articles
    articles = load_articles()
    print(f"\nLoaded {len(articles)} articles")
    
    # Chunk all articles
    chunker = HeuristicChunker()
    all_chunks = []
    all_overlaps = []
    
    for article in articles:
        chunks = chunker.chunk(article['text'])
        
        # Calculate chunk sizes
        for chunk in chunks:
            token_count = chunker.count_tokens(chunk)
            all_chunks.append({
                'text': chunk,
                'tokens': token_count,
                'article': article['title']
            })
        
        # Calculate overlaps between consecutive chunks in this article
        for i in range(len(chunks) - 1):
            # Find overlap by checking suffix of chunk1 against prefix of chunk2
            chunk1, chunk2 = chunks[i], chunks[i + 1]
            overlap_text = ""
            
            # Find the longest suffix of chunk1 that matches prefix of chunk2
            max_overlap = min(len(chunk1), len(chunk2))
            for j in range(max_overlap, 0, -1):
                if chunk2.startswith(chunk1[-j:]):
                    overlap_text = chunk1[-j:]
                    break
            
            overlap_tokens = chunker.count_tokens(overlap_text) if overlap_text else 0
            all_overlaps.append({
                'chunk1_idx': len(all_chunks) - len(chunks) + i,
                'chunk2_idx': len(all_chunks) - len(chunks) + i + 1,
                'overlap_tokens': overlap_tokens,
                'article': article['title'],
                'overlap_text': overlap_text[:100] if overlap_text else ""
            })
    
    # Statistics
    token_counts = [c['tokens'] for c in all_chunks]
    overlap_tokens = [o['overlap_tokens'] for o in all_overlaps]
    
    print(f"\n{'='*60}")
    print("CHUNK SIZE STATISTICS")
    print(f"{'='*60}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Min tokens: {min(token_counts)}")
    print(f"Max tokens: {max(token_counts)}")
    print(f"Avg tokens: {sum(token_counts)/len(token_counts):.1f}")
    print(f"Median tokens: {sorted(token_counts)[len(token_counts)//2]}")
    
    # Distribution
    ranges = [
        (0, 100, "0-100"),
        (100, 200, "100-200"),
        (200, 300, "200-300"),
        (300, 400, "300-400"),
        (400, 450, "400-450"),
        (450, 480, "450-480"),
        (480, 500, "480-500"),
        (500, 600, "500+"),
    ]
    
    print(f"\nToken count distribution:")
    for min_t, max_t, label in ranges:
        count = sum(1 for t in token_counts if min_t <= t < max_t)
        pct = count / len(token_counts) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    print(f"\n{'='*60}")
    print("OVERLAP STATISTICS")
    print(f"{'='*60}")
    if overlap_tokens:
        print(f"Total overlaps: {len(overlap_tokens)}")
        print(f"Min overlap: {min(overlap_tokens)}")
        print(f"Max overlap: {max(overlap_tokens)}")
        print(f"Avg overlap: {sum(overlap_tokens)/len(overlap_tokens):.1f}")
        print(f"Median overlap: {sorted(overlap_tokens)[len(overlap_tokens)//2]}")
        
        # Check for 0-token overlaps
        zero_overlaps = [o for o in all_overlaps if o['overlap_tokens'] == 0]
        if zero_overlaps:
            print(f"\n⚠️  Found {len(zero_overlaps)} chunks with 0-token overlap!")
            for o in zero_overlaps[:5]:
                print(f"    Article: {o['article']}")
        
        # Check overlaps outside 48-52 range
        bad_overlaps = [o for o in all_overlaps if not (48 <= o['overlap_tokens'] <= 52)]
        if bad_overlaps:
            print(f"\n⚠️  Found {len(bad_overlaps)} overlaps outside 48-52 range")
            print(f"    Sample: {[o['overlap_tokens'] for o in bad_overlaps[:10]]}")
    
    # Show smallest chunks
    print(f"\n{'='*60}")
    print("SMALLEST CHUNKS (under 450 tokens)")
    print(f"{'='*60}")
    small_chunks = [c for c in all_chunks if c['tokens'] < 450]
    for c in sorted(small_chunks, key=lambda x: x['tokens'])[:10]:
        print(f"  {c['tokens']} tokens - {c['article']}")
        print(f"    Text: {c['text'][:100]}...")
    
    return all_chunks, all_overlaps


def main() -> int:
    """Run the chunk analysis."""
    analyze_corpus_from_articles()
    return 0


if __name__ == "__main__":
    sys.exit(main())
