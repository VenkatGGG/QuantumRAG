#!/usr/bin/env python3
"""Run the full data pipeline: scrape, chunk, embed, and save to HDF5."""

import sys

from src.wikipedia_scraper import fetch_articles, save_articles
from src.heuristic_chunker import HeuristicChunker
from src.embedding import EmbeddingPipeline
from src.vector_store import VectorStore


def main():
    """Run full data pipeline."""
    print("=" * 60)
    print("FULL DATA PIPELINE")
    print("=" * 60)
    
    # Step 1: Scrape articles
    print("\n[1/4] Scraping Wikipedia articles...")
    articles = fetch_articles("Quantum Cryptography", limit=10)
    save_articles(articles)
    print(f"  -> Fetched and saved {len(articles)} articles")
    
    # Step 2: Chunk articles
    print("\n[2/4] Chunking articles...")
    chunker = HeuristicChunker()
    all_chunks = []
    
    for article in articles:
        chunks = chunker.chunk(article['text'])
        for chunk_text in chunks:
            all_chunks.append({
                'text': chunk_text,
                'article_title': article['title'],
                'article_url': article['url']
            })
    
    print(f"  -> Created {len(all_chunks)} chunks")
    
    # Step 3: Generate embeddings
    print("\n[3/4] Generating embeddings...")
    pipeline = EmbeddingPipeline()
    
    texts = [c['text'] for c in all_chunks]
    embeddings = pipeline.embed_batch(texts)
    
    print(f"  -> Generated {len(embeddings)} embeddings (dim={len(embeddings[0])})")
    
    # Step 4: Save to vector store
    print("\n[4/4] Saving to vector store...")
    vector_store = VectorStore(dimension=384)
    
    for i, chunk in enumerate(all_chunks):
        vector_store.add(embeddings[i], chunk['text'])
    
    # Save to HDF5
    hdf5_path = "data/vector_store.h5"
    vector_store.save_to_hdf5(hdf5_path)
    
    print(f"  -> Saved {vector_store.size} vectors to {hdf5_path}")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Articles: {len(articles)}")
    print(f"Chunks: {len(all_chunks)}")
    print(f"Vectors: {vector_store.size}")
    print(f"Dimensions: {vector_store.dimension}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
