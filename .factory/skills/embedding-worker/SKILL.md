---
name: embedding-worker
description: Worker for local transformer embeddings and vector store implementation
---

# Embedding Worker

Handles loading transformer models, implementing manual mean pooling, and building NumPy-based vector stores.

## When to Use This Skill

Use for features involving:
- Loading transformer models with AutoModel/AutoTokenizer
- Manual mean pooling implementation
- NumPy vector storage and cosine similarity search
- HDF5 serialization

## Required Skills

None

## Work Procedure

1. **Write tests first (TDD)**
   - Create test file with pytest
   - Mock model outputs for deterministic testing
   - Test mean pooling formula: `E = sum(T_i * M_i) / max(sum(M_i), epsilon)`
   - Test cosine similarity: `similarity = (A · B) / (||A|| ||B||)`

2. **Implement the embedding pipeline**
   - Load `sentence-transformers/all-MiniLM-L6-v2` using transformers.AutoModel
   - Implement mean pooling with attention mask (NO pooler_output)
   - Use torch.clamp with min=1e-9 for epsilon
   - Output dimension must be 384

3. **Implement vector store**
   - Pure NumPy arrays for storage
   - Manual cosine similarity implementation
   - Top-k retrieval function
   - HDF5 serialization/deserialization

4. **Manual verification**
   - Test embedding a sample sentence
   - Verify output shape is (384,)
   - Test similarity search returns expected neighbors
   - Verify HDF5 save/load preserves data

5. **Run validators**
   - `python -m pytest tests/ -v`
   - Check type consistency

6. **Commit work**

## Example Handoff

```json
{
  "salientSummary": "Implemented local embedding pipeline with all-MiniLM-L6-v2 model, manual mean pooling with attention mask, and NumPy vector store with cosine similarity search and HDF5 persistence.",
  "whatWasImplemented": "Created embedding_pipeline.py loading sentence-transformers/all-MiniLM-L6-v2 via transformers.AutoModel/AutoTokenizer. Implemented mean_pooling() following formula E = sum(T_i * M_i) / max(sum(M_i), 1e-9). Created vector_store.py with pure NumPy arrays, manual cosine_similarity() using dot product and norms, top_k_search() returning k nearest neighbors. Added HDF5 serialization via save_to_hdf5() and load_from_hdf5().",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {"command": "python -m pytest tests/test_embedding.py -v", "exitCode": 0, "observation": "5 tests passed including mean pooling and embedding shape verification"},
      {"command": "python -m pytest tests/test_vector_store.py -v", "exitCode": 0, "observation": "6 tests passed for cosine similarity, top-k search, and HDF5 persistence"},
      {"command": "python scripts/test_embedding_manual.py", "exitCode": 0, "observation": "Embedded 'Quantum cryptography uses quantum mechanics' -> shape (384,), norm=1.0 (normalized)"},
      {"command": "python scripts/test_vector_store_manual.py", "exitCode": 0, "observation": "Created store with 100 vectors, search returned correct top-5 neighbors, save/load to HDF5 verified"}
    ],
    "interactiveChecks": []
  },
  "tests": {
    "added": [
      {"file": "tests/test_embedding.py", "cases": [
        {"name": "test_mean_pooling_formula", "verifies": "Mean pooling follows E = sum(T_i * M_i) / max(sum(M_i), epsilon)"},
        {"name": "test_embedding_shape", "verifies": "Output is 384-dimensional"},
        {"name": "test_attention_mask_applied", "verifies": "Padding tokens are ignored in pooling"},
        {"name": "test_model_loading", "verifies": "Model loads without external API calls"},
        {"name": "test_deterministic_output", "verifies": "Same input produces same embedding"}
      ]},
      {"file": "tests/test_vector_store.py", "cases": [
        {"name": "test_cosine_similarity_formula", "verifies": "Cosine similarity = dot(A,B) / (norm(A)*norm(B))"},
        {"name": "test_similarity_range", "verifies": "Similarity scores are in [-1, 1]"},
        {"name": "test_top_k_retrieval", "verifies": "Returns exactly k results sorted by similarity"},
        {"name": "test_hdf5_save_load", "verifies": "Vectors and texts persist through save/load"},
        {"name": "test_empty_store_search", "verifies": "Empty store returns empty results"},
        {"name": "test_vector_dimension_mismatch", "verifies": "Raises error on dimension mismatch"}
      ]}
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- Model download fails or is too large for available disk space
- CUDA/GPU issues preventing model loading
- Memory errors during embedding generation
- HDF5 library installation issues
