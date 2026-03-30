# Architecture

How the RAG system works - components, relationships, data flows, invariants.

**What belongs here:** High-level system design, component interactions, data flow diagrams, key invariants.

---

## System Overview

The RAG application consists of four main components:
1. **Data Pipeline** - Wikipedia scraping and text chunking
2. **Embedding Pipeline** - Local transformer-based embeddings
3. **Vector Store** - NumPy-based storage and similarity search
4. **API Layer** - FastAPI backend with vanilla frontend

## Component Diagram

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Wikipedia API  │────▶│  Data Pipeline  │────▶│  Chunked Texts  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Query Results  │◀────│   API Layer     │◀────│  Vector Store   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        ▲                                               ▲
        │                                               │
        └───────────────────────────────────────────────┘
                          Embedding Pipeline
```

## Data Flow

### Ingestion Flow
1. **Scraper** fetches 10 Wikipedia articles about Quantum Cryptography
2. **Chunker** splits articles into 500-token chunks with sentence-aligned overlap closest to 50 tokens
3. **Embedding Pipeline** generates 384-dimensional vectors for each chunk
4. **Vector Store** saves vectors and texts to HDF5 file

### Query Flow
1. **Frontend** sends POST /query with user query
2. **Backend** embeds query using same embedding pipeline
3. **Vector Store** performs cosine similarity search
4. **Backend** returns top-k results with text and similarity scores
5. **Frontend** displays results to user

## Key Invariants

1. **Sentence Boundaries**: No chunk may end mid-sentence
2. **Token Count**: Chunks contain at most 500 tokens
3. **Overlap**: Consecutive chunks share sentence-aligned overlap closest to 50 tokens (target, not guarantee)
4. **Embedding Dimension**: All vectors are exactly 384-dimensional
5. **Deterministic Output**: Same input always produces same embedding
6. **Persistence**: Vector store survives container restarts

## Component Details

### Data Pipeline
- **Scraper**: Uses wikipedia-api library with rate limiting
- **Chunker**: Uses model tokenizer for accurate token counting
- **Sentence Detection**: NLTK sent_tokenize or regex-based

### Embedding Pipeline
- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Method**: Manual mean pooling over last_hidden_state
- **Formula**: E = sum(T_i * M_i) / max(sum(M_i), 1e-9)
- **Output**: L2-normalized 384-dimensional vectors

### Vector Store
- **Storage**: NumPy array of shape (n_chunks, 384)
- **Similarity**: Cosine similarity = dot(A, B) / (||A|| * ||B||)
- **Search**: Full scan with NumPy operations (sufficient for <10k chunks)
- **Persistence**: HDF5 file format

### API Layer
- **Framework**: FastAPI with uvicorn
- **CORS**: Configured for frontend communication
- **Endpoints**:
  - GET /status - Returns chunk count and dimensions
  - POST /query - Accepts query, returns top-k results
- **Frontend**: Vanilla HTML/JS with Fetch API
