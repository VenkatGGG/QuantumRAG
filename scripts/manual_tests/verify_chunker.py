"""Manual verification script for chunker with real articles."""

import sys

from src.heuristic_chunker import HeuristicChunker
from src.wikipedia_scraper import fetch_articles, save_articles


def main() -> int:
    """Run the chunker verification with Wikipedia articles."""
    print("=" * 60)
    print("MANUAL VERIFICATION: Heuristic Chunker with Wikipedia Articles")
    print("=" * 60)

    # Fetch real articles
    print("\nFetching Wikipedia articles...")
    articles = fetch_articles("Quantum Cryptography", limit=3)
    print(f"Fetched {len(articles)} articles")

    # Save articles for future use
    save_articles(articles)
    print(f"Saved articles to data/raw_articles.json")

    # Initialize chunker
    print("\nInitializing chunker...")
    chunker = HeuristicChunker()
    print("Chunker ready!")

    # Process each article
    for article in articles:
        print(f"\n{'='*60}")
        print(f"Article: {article['title']}")
        print(f"URL: {article['url']}")
        print(f"Text length: {len(article['text'])} characters")
        
        # Count total tokens
        total_tokens = chunker.count_tokens(article['text'])
        print(f"Total tokens: {total_tokens}")
        
        # Chunk the text
        chunks = chunker.chunk(article['text'])
        print(f"Number of chunks: {len(chunks)}")
        
        # Analyze each chunk
        for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
            token_count = chunker.count_tokens(chunk)
            char_count = len(chunk)
            # Check sentence boundary
            ends_with_punctuation = chunk.strip()[-1] in '.!?' if chunk.strip() else False
            print(f"  Chunk {i+1}: {token_count} tokens, {char_count} chars, ends with punctuation: {ends_with_punctuation}")
            if i == 4 and len(chunks) > 5:
                print(f"  ... and {len(chunks) - 5} more chunks")
                break
        
        # Check overlap between consecutive chunks
        if len(chunks) >= 2:
            print(f"\n  Overlap analysis:")
            for i in range(min(3, len(chunks) - 1)):
                chunk1 = chunks[i]
                chunk2 = chunks[i + 1]
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
                print(f"    Between chunk {i+1} and {i+2}: ~{overlap_tokens} tokens overlap")

    print(f"\n{'='*60}")
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
