"""Manual test script for vector store.

This script tests the vector store with actual embeddings
to verify search, save/load, and HDF5 persistence.
"""

import sys

import tempfile
import os
import numpy as np
from src.vector_store import VectorStore, cosine_similarity


def test_vector_store() -> None:
    """Test the vector store with actual data."""

    print("Testing Vector Store...")
    
    # Create store
    store = VectorStore(dimension=384)
    print(f"✓ Created vector store with dimension {store.dimension}")
    
    # Add some vectors
    np.random.seed(42)
    num_vectors = 10
    texts = [f"Document about quantum cryptography topic {i}" for i in range(num_vectors)]
    vectors = []
    
    for i in range(num_vectors):
        vec = np.random.randn(384).astype(np.float32)
        vectors.append(vec)
        store.add(vec, texts[i])
    
    print(f"✓ Added {store.size} vectors to store")
    
    # Test search
    query = vectors[0]  # Search for first vector
    print(f"\nSearching with query vector (should find most similar to first document)...")
    results = store.search(query, k=3)
    
    print(f"✓ Found {len(results)} results")
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    
    # First result should be the exact match
    assert results[0]['text'] == texts[0], f"Expected '{texts[0]}', got '{results[0]['text']}'"
    assert np.isclose(results[0]['similarity'], 1.0, atol=1e-5), f"Expected similarity ~1.0, got {results[0]['similarity']}"
    print(f"✓ Top result is correct match with similarity {results[0]['similarity']:.6f}")
    
    # Results should be sorted
    similarities = [r['similarity'] for r in results]
    assert all(similarities[i] >= similarities[i+1] for i in range(len(similarities)-1)), "Results should be sorted by similarity"
    print("✓ Results are sorted by similarity (descending)")
    
    # Test HDF5 save/load
    print("\nTesting HDF5 persistence...")
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        temp_path = f.name
    
    try:
        # Save
        store.save_to_hdf5(temp_path)
        print(f"✓ Saved to {temp_path}")
        
        # Load into new store
        new_store = VectorStore(dimension=384)
        new_store.load_from_hdf5(temp_path)
        print(f"✓ Loaded into new store with {new_store.size} vectors")
        
        assert new_store.size == store.size, f"Expected {store.size} vectors, got {new_store.size}"
        
        # Verify search works the same
        new_results = new_store.search(query, k=3)
        assert len(new_results) == len(results), "Should return same number of results"
        
        for old, new in zip(results, new_results):
            assert old['text'] == new['text'], f"Text mismatch: {old['text']} vs {new['text']}"
            assert np.isclose(old['similarity'], new['similarity'], atol=1e-6), f"Similarity mismatch for {old['text']}"
        
        print("✓ Loaded store produces identical search results")
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    # Test cosine similarity directly
    print("\nTesting cosine similarity...")
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    c = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    d = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    
    sim_aa = cosine_similarity(a, a)
    sim_ab = cosine_similarity(a, b)
    sim_ac = cosine_similarity(a, c)
    sim_ad = cosine_similarity(a, d)
    
    assert np.isclose(sim_aa, 1.0), f"Self-similarity should be 1.0, got {sim_aa}"
    assert np.isclose(sim_ab, 1.0), f"Identical vectors should have similarity 1.0, got {sim_ab}"
    assert np.isclose(sim_ac, 0.0), f"Orthogonal vectors should have similarity 0.0, got {sim_ac}"
    assert np.isclose(sim_ad, -1.0), f"Opposite vectors should have similarity -1.0, got {sim_ad}"
    
    print(f"✓ Cosine similarity tests passed:")
    print(f"  - Self-similarity: {sim_aa:.6f}")
    print(f"  - Identical: {sim_ab:.6f}")
    print(f"  - Orthogonal: {sim_ac:.6f}")
    print(f"  - Opposite: {sim_ad:.6f}")
    
    print("\n" + "="*50)
    print("All manual vector store tests passed!")
    print("="*50)


def main() -> int:
    """Run all manual vector store tests."""
    test_vector_store()
    return 0


if __name__ == "__main__":
    sys.exit(main())
