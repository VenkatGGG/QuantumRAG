"""Manual test script for embedding pipeline.

This script tests the embedding pipeline with actual model loading
to verify output shape and basic functionality.
"""

import sys

from src.embedding import EmbeddingPipeline
import numpy as np

def test_embedding_pipeline():
    """Test the embedding pipeline with actual model."""
    print("Loading embedding pipeline (this may take a moment on first run)...")
    
    # Initialize pipeline (downloads model if needed)
    pipeline = EmbeddingPipeline()
    print(f"✓ Model loaded: {pipeline.MODEL_NAME}")
    print(f"✓ Embedding dimension: {pipeline.EMBEDDING_DIM}")
    
    # Test single embedding
    test_text = "Quantum cryptography uses quantum mechanics to secure communications."
    print(f"\nEmbedding text: '{test_text[:50]}...'")
    
    embedding = pipeline.embed(test_text)
    
    print(f"✓ Output shape: {embedding.shape}")
    assert embedding.shape == (384,), f"Expected shape (384,), got {embedding.shape}"
    
    # Check output is float32 numpy array
    print(f"✓ Output dtype: {embedding.dtype}")
    assert embedding.dtype == np.float32, f"Expected float32, got {embedding.dtype}"
    
    # Test deterministic output (with tolerance for float32 inference drift)
    embedding2 = pipeline.embed(test_text)
    assert np.allclose(embedding, embedding2, rtol=1e-4, atol=1e-5), "Same input should produce same embedding"
    print("✓ Output is deterministic")
    
    # Test batch embedding
    texts = [
        "Quantum entanglement enables secure key distribution.",
        "BB84 protocol uses four quantum states.",
        "Eavesdropping can be detected in quantum channels."
    ]
    print(f"\nBatch embedding {len(texts)} texts...")
    batch_embeddings = pipeline.embed_batch(texts)
    
    print(f"✓ Batch output shape: {batch_embeddings.shape}")
    assert batch_embeddings.shape == (3, 384), f"Expected shape (3, 384), got {batch_embeddings.shape}"
    
    # Verify batch embeddings match individual embeddings (with tolerance for float32 inference drift)
    for i, text in enumerate(texts):
        individual = pipeline.embed(text)
        assert np.allclose(batch_embeddings[i], individual, rtol=1e-4, atol=1e-5), f"Batch embedding {i} should match individual"
    print("✓ Batch embeddings match individual embeddings")
    
    print("\n" + "="*50)
    print("All manual embedding tests passed!")
    print("="*50)

if __name__ == "__main__":
    test_embedding_pipeline()
