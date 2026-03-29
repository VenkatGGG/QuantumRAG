"""Tests for the NumPy vector store.

This module tests the vector store implementation including:
- Cosine similarity formula: similarity = dot(A, B) / (||A|| * ||B||)
- Top-k retrieval with correct sorting
- HDF5 serialization and deserialization
- Edge cases: empty store, dimension mismatch
"""

import pytest
import numpy as np
import os
import tempfile
from src.vector_store import VectorStore, cosine_similarity


class TestCosineSimilarity:
    """Tests for the cosine similarity implementation."""
    
    def test_cosine_similarity_formula(self):
        """Test that cosine similarity follows dot(A, B) / (norm(A) * norm(B))."""
        # Test with simple vectors
        a = np.array([1.0, 0.0, 0.0])  # Unit vector along x
        b = np.array([1.0, 0.0, 0.0])  # Same as a
        
        # Similarity should be 1.0 (identical vectors)
        result = cosine_similarity(a, b)
        expected = 1.0
        
        assert np.isclose(result, expected, atol=1e-6)
    
    def test_similarity_range(self):
        """Test that similarity scores are in [-1, 1]."""
        # Test identical vectors (should be 1)
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        sim = cosine_similarity(a, b)
        assert np.isclose(sim, 1.0, atol=1e-6)
        
        # Test opposite vectors (should be -1)
        c = np.array([-1.0, -2.0, -3.0])
        sim_neg = cosine_similarity(a, c)
        assert np.isclose(sim_neg, -1.0, atol=1e-6)
        
        # Test orthogonal vectors (should be 0)
        d = np.array([0.0, 0.0, 1.0])
        e = np.array([1.0, 0.0, 0.0])
        sim_orth = cosine_similarity(d, e)
        assert np.isclose(sim_orth, 0.0, atol=1e-6)
        
        # Test random vectors are within [-1, 1]
        np.random.seed(42)
        for _ in range(100):
            v1 = np.random.randn(384)
            v2 = np.random.randn(384)
            sim = cosine_similarity(v1, v2)
            assert -1.0 - 1e-6 <= sim <= 1.0 + 1e-6
    
    def test_similarity_with_normalized_vectors(self):
        """Test similarity with pre-normalized vectors."""
        # Normalized vectors should still give correct similarity
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        
        sim = cosine_similarity(a, b)
        assert np.isclose(sim, 0.0, atol=1e-6)  # Orthogonal


class TestVectorStore:
    """Tests for the VectorStore class."""
    
    def test_empty_store_search(self):
        """Test that empty store returns empty results."""
        store = VectorStore()
        
        # Search in empty store
        query = np.random.randn(384)
        results = store.search(query, k=5)
        
        assert len(results) == 0
    
    def test_vector_dimension_mismatch(self):
        """Test that dimension mismatch raises error."""
        store = VectorStore()
        
        # Add a 384-dimensional vector
        vector = np.random.randn(384)
        store.add(vector, "test text")
        
        # Try to search with wrong dimension
        wrong_query = np.random.randn(100)
        
        with pytest.raises(ValueError):
            store.search(wrong_query, k=5)
    
    def test_top_k_retrieval(self):
        """Test that top-k search returns exactly k results sorted by similarity."""
        store = VectorStore()
        
        # Add 10 vectors with known similarity pattern
        # We'll use simple vectors where we can predict similarity
        np.random.seed(42)
        
        # Create a query vector
        query = np.random.randn(384)
        query = query / np.linalg.norm(query)  # Normalize
        
        # Add vectors with varying similarity to query
        vectors = []
        texts = []
        for i in range(10):
            # Create vector with controlled similarity
            # Mix query with random orthogonal component
            noise = np.random.randn(384)
            noise = noise - np.dot(noise, query) * query  # Make orthogonal to query
            noise = noise / np.linalg.norm(noise) if np.linalg.norm(noise) > 0 else noise
            
            # Similarity decreases as i increases
            similarity_weight = 1.0 - (i * 0.1)  # 1.0, 0.9, 0.8, ...
            vector = similarity_weight * query + (1 - similarity_weight) * noise
            vector = vector / np.linalg.norm(vector)
            
            vectors.append(vector)
            texts.append(f"text_{i}")
        
        # Add all vectors to store
        for vec, text in zip(vectors, texts):
            store.add(vec, text)
        
        # Search for top 5
        results = store.search(query, k=5)
        
        # Should return exactly 5 results
        assert len(results) == 5
        
        # Results should be sorted by similarity (highest first)
        similarities = [r['similarity'] for r in results]
        assert all(similarities[i] >= similarities[i+1] for i in range(len(similarities)-1))
        
        # First result should be most similar (text_0)
        assert results[0]['text'] == 'text_0'
        
        # All similarities should be in valid range (with small tolerance for floating point)
        assert all(-1.0001 <= s <= 1.0001 for s in similarities)
    
    def test_hdf5_save_load(self):
        """Test that vectors and texts persist through save/load."""
        store = VectorStore()
        
        # Add some vectors
        np.random.seed(42)
        original_vectors = []
        original_texts = []
        for i in range(5):
            vec = np.random.randn(384)
            vec = vec / np.linalg.norm(vec)
            original_vectors.append(vec)
            original_texts.append(f"text_{i}")
            store.add(vec, f"text_{i}")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_path = f.name
        
        try:
            store.save_to_hdf5(temp_path)
            
            # Load into new store
            new_store = VectorStore()
            new_store.load_from_hdf5(temp_path)
            
            # Verify all vectors and texts are preserved
            assert new_store.size == store.size
            
            # Verify we can search and get same results
            query = original_vectors[0]
            original_results = store.search(query, k=3)
            new_results = new_store.search(query, k=3)
            
            assert len(original_results) == len(new_results)
            
            for orig, new in zip(original_results, new_results):
                assert orig['text'] == new['text']
                assert np.isclose(orig['similarity'], new['similarity'], atol=1e-6)
        
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_add_and_size(self):
        """Test adding vectors and checking store size."""
        store = VectorStore()
        
        assert store.size == 0
        
        # Add vectors
        for i in range(5):
            vec = np.random.randn(384)
            store.add(vec, f"text_{i}")
            assert store.size == i + 1
    
    def test_search_with_k_larger_than_store(self):
        """Test search when k > number of stored vectors."""
        store = VectorStore()
        
        # Add 3 vectors
        for i in range(3):
            vec = np.random.randn(384)
            store.add(vec, f"text_{i}")
        
        # Search for k=5 (more than we have)
        query = np.random.randn(384)
        results = store.search(query, k=5)
        
        # Should return all 3 available
        assert len(results) == 3
    
    def test_batch_add(self):
        """Test adding multiple vectors at once."""
        store = VectorStore()
        
        # Create batch of vectors
        np.random.seed(42)
        vectors = [np.random.randn(384) for _ in range(5)]
        texts = [f"text_{i}" for i in range(5)]
        
        # Add batch
        store.add_batch(vectors, texts)
        
        assert store.size == 5
        
        # Verify search works
        query = vectors[0]
        results = store.search(query, k=3)
        assert len(results) == 3
