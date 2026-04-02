"""NumPy vector store module for storing and searching embeddings.

This module implements an in-memory vector store using pure NumPy arrays.
Includes manual cosine similarity search and HDF5 serialization for persistence.
"""

import numpy as np
import h5py
from typing import List, Dict, Optional
from pathlib import Path


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors.
    
    Implements the formula: similarity = dot(A, B) / (||A|| * ||B||)
    where ||A|| is the L2 norm of A.
    
    Args:
        a: First vector. Shape: (dim,)
        b: Second vector. Shape: (dim,)
    
    Returns:
        Cosine similarity score in range [-1, 1].
        1.0 = identical direction
        0.0 = orthogonal
        -1.0 = opposite direction
    """
    # Calculate dot product: dot(A, B)
    dot_product = np.dot(a, b)
    
    # Calculate L2 norms: ||A|| and ||B||
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    # Prevent division by zero
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    # Cosine similarity: dot(A, B) / (||A|| * ||B||)
    similarity = dot_product / (norm_a * norm_b)
    
    return float(similarity)


class VectorStore:
    """In-memory vector store using pure NumPy arrays.
    
    Stores embedding vectors and their associated text content.
    Implements manual cosine similarity search for top-k retrieval.
    Supports HDF5 serialization for persistence.
    
    Attributes:
        dimension: Expected dimension of vectors (default 384 for all-MiniLM-L6-v2)
    """
    
    def __init__(self, dimension: int = 384):
        """Initialize an empty vector store.
        
        Args:
            dimension: Expected dimension of vectors. Default 384 for all-MiniLM-L6-v2.
        """
        self.dimension = dimension
        self.vectors: List[np.ndarray] = []
        self.texts: List[str] = []
    
    @property
    def size(self) -> int:
        """Return the number of vectors in the store."""
        return len(self.vectors)
    
    def add(self, vector: np.ndarray, text: str) -> None:
        """Add a vector and its associated text to the store.
        
        Args:
            vector: Embedding vector. Shape must match dimension.
            text: Associated text content.
        
        Raises:
            ValueError: If vector dimension doesn't match store dimension.
        """
        if vector.shape != (self.dimension,):
            raise ValueError(
                f"Vector dimension {vector.shape} doesn't match store dimension {(self.dimension,)}"
            )
        
        self.vectors.append(vector)
        self.texts.append(text)
    
    def add_batch(self, vectors: List[np.ndarray], texts: List[str]) -> None:
        """Add multiple vectors and texts to the store.
        
        Args:
            vectors: List of embedding vectors.
            texts: List of associated text content.
        
        Raises:
            ValueError: If dimensions don't match or lists have different lengths.
        """
        if len(vectors) != len(texts):
            raise ValueError("Vectors and texts must have the same length")
        
        for vec, text in zip(vectors, texts):
            self.add(vec, text)
    
    def search(self, query: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for top-k most similar vectors.
        
        Uses cosine similarity to find the k nearest neighbors.
        Results are sorted by similarity (highest first).
        
        Args:
            query: Query embedding vector.
            k: Number of results to return. Default 5.
        
        Returns:
            List of dicts with 'text' and 'similarity' keys.
            Sorted by similarity descending.
            Returns empty list if store contains no vectors.
        
        Raises:
            ValueError: If query dimension doesn't match store dimension.
        """
        if query.shape != (self.dimension,):
            raise ValueError(
                f"Query dimension {query.shape} doesn't match store dimension {(self.dimension,)}"
            )
        
        # Handle empty store
        if self.size == 0:
            return []
        
        # Calculate similarities with all vectors
        similarities = []
        for i, vector in enumerate(self.vectors):
            sim = cosine_similarity(query, vector)
            similarities.append((i, sim))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Take top-k (or all if fewer than k)
        top_k = similarities[:min(k, len(similarities))]
        
        # Format results
        results = [
            {
                'text': self.texts[idx],
                'similarity': float(sim)
            }
            for idx, sim in top_k
        ]
        
        return results
    
    def clear(self) -> None:
        """Clear all vectors and texts from the store."""
        self.vectors = []
        self.texts = []
    
    def save_to_hdf5(self, filepath: str) -> None:
        """Save vectors and texts to HDF5 file.
        
        Args:
            filepath: Path to save the HDF5 file.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(filepath, 'w') as f:
            # Store metadata
            f.attrs['dimension'] = self.dimension
            f.attrs['size'] = self.size
            
            # Store vectors as a dataset
            if self.size > 0:
                vectors_array = np.array(self.vectors)
                f.create_dataset('vectors', data=vectors_array)
                
                # Store texts as a dataset (variable-length strings)
                dt = h5py.string_dtype(encoding='utf-8')
                text_dataset = f.create_dataset('texts', (self.size,), dtype=dt)
                for i, text in enumerate(self.texts):
                    text_dataset[i] = text
    
    def load_from_hdf5(self, filepath: str) -> None:
        """Load vectors and texts from HDF5 file.
        
        Args:
            filepath: Path to the HDF5 file.
        
        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If dimension doesn't match store dimension.
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"HDF5 file not found: {filepath}")
        
        with h5py.File(filepath, 'r') as f:
            # Verify dimension
            file_dimension = f.attrs['dimension']
            if file_dimension != self.dimension:
                raise ValueError(
                    f"File dimension {file_dimension} doesn't match store dimension {self.dimension}"
                )
            
            # Load size
            size = f.attrs['size']
            
            if size > 0:
                # Load vectors
                vectors_array = f['vectors'][:]
                self.vectors = [vectors_array[i] for i in range(size)]
                
                # Load texts
                self.texts = [f['texts'][i].decode('utf-8') for i in range(size)]
            else:
                self.vectors = []
                self.texts = []
