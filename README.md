# Quantum Retrieval - RAG Application

A production-ready Retrieval-Augmented Generation (RAG) application for querying Wikipedia articles about Quantum Cryptography. Built with FastAPI, sentence-transformers, and a custom heuristic chunking algorithm.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)

## Overview

This application implements a complete RAG pipeline:

1. **Data Ingestion**: Fetches Wikipedia articles on Quantum Cryptography
2. **Text Chunking**: Splits articles into sentence-aligned chunks (max 500 tokens, ~50 token overlap)
3. **Embeddings**: Generates 384-dimensional vectors using `sentence-transformers/all-MiniLM-L6-v2`
4. **Vector Store**: Fast cosine similarity search with HDF5 persistence
5. **REST API**: FastAPI backend with query and status endpoints
6. **Frontend**: Chat interface for interactive querying

## Features

- **Sentence-Aware Chunking**: Never splits sentences mid-way; overlap is sentence-aligned
- **Local Embeddings**: No external API calls; runs entirely on local hardware
- **Docker Support**: Containerized deployment with volume-mounted data persistence
- **Comprehensive Testing**: 20+ validation contracts covering data ingestion, embeddings, API, and Docker
- **CORS Enabled**: Frontend can communicate with backend from any origin

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Wikipedia  │────▶│   Chunker    │────▶│  Embedding  │
│   Scraper   │     │ (500 tokens) │     │  Pipeline   │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                                                ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Static    │◀────│   FastAPI    │◀────│   Vector    │
│   Frontend  │     │   Backend    │     │   Store     │
└─────────────┘     └──────────────┘     └─────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │   HDF5 File  │
                    │  (Persistent)│
                    └──────────────┘
```

## Project Structure

```
.
├── src/
│   ├── main.py                 # FastAPI application
│   ├── wikipedia_scraper.py    # Wikipedia article fetcher
│   ├── heuristic_chunker.py    # Sentence-aware text chunker
│   ├── embedding.py            # Embedding pipeline (MiniLM-L6-v2)
│   └── vector_store.py         # Cosine similarity vector store
├── tests/
│   ├── test_api.py             # API endpoint tests
│   ├── test_chunker.py         # Chunking algorithm tests
│   ├── test_embedding.py       # Embedding pipeline tests
│   ├── test_scraper.py         # Wikipedia scraper tests
│   ├── test_vector_store.py    # Vector store tests
│   └── test_scripts/           # Additional test scripts
├── scripts/
│   ├── analysis/               # Analysis scripts
│   ├── manual_tests/           # Manual testing utilities
│   ├── pipeline/               # Pipeline automation scripts
│   └── download_model.py       # Model pre-download utility
├── static/
│   └── index.html              # Frontend chat interface
├── data/
│   └── vector_store.h5         # HDF5 persistence (created at runtime)
├── Dockerfile                  # Docker image definition
├── docker-compose.yml          # Docker Compose configuration
├── setup.py                    # Package installation
├── requirements.txt            # Python dependencies
├── DESIGN.md                   # Chunking contract (Option A)
└── validation-contract.md      # 20+ validation test specifications
```

## Quick Start

### Prerequisites

- Python 3.10+
- Docker (optional, for containerized deployment)

### Local Development

1. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the embedding model (optional, for faster startup):**
   ```bash
   python scripts/download_model.py
   ```

4. **Run the FastAPI server:**
   ```bash
   uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Access the application:**
   - API Docs: http://localhost:8000/docs
   - Frontend: http://localhost:8000/
   - Status: http://localhost:8000/status

### Docker Deployment

1. **Build and start the container:**
   ```bash
   docker-compose up --build
   ```

2. **Access the application:**
   - API: http://localhost:8000
   - The HDF5 data persists in `./data/` via volume mount

3. **Stop the container:**
   ```bash
   docker-compose down
   ```

## API Endpoints

### GET /status

Returns the current state of the vector store.

**Response:**
```json
{
  "chunk_count": 150,
  "vector_dimensions": 384
}
```

### POST /query

Search for relevant context chunks based on a query string.

**Request:**
```json
{
  "query": "What is quantum key distribution?",
  "k": 5
}
```

**Response:**
```json
{
  "results": [
    {
      "text": "Quantum key distribution (QKD) is a secure communication method...",
      "similarity": 0.95
    },
    ...
  ]
}
```

### GET /health

Health check endpoint for container orchestration.

## Chunking Algorithm

The application implements **Option A** of the chunking contract (see `DESIGN.md`):

1. **Sentence Boundary Respect**: Chunk boundaries never cut sentences
2. **Maximum Chunk Size**: 500 tokens (hard ceiling)
3. **Sentence-Aligned Overlap**: Target ~50 tokens, but varies (30-70) based on sentence lengths

### Key Features

- **Wikipedia Segmentation**: Handles headings, blank lines, formulas, and lists
- **Tokenizer Consistency**: Uses the same tokenizer as the embedding model
- **Emergency Truncation**: Handles pathologically long sentences

## Testing

Run the complete test suite:

```bash
pytest tests/ -v
```

Run specific test categories:

```bash
pytest tests/test_chunker.py -v      # Chunking tests
pytest tests/test_embedding.py -v      # Embedding tests
pytest tests/test_api.py -v            # API tests
pytest tests/test_vector_store.py -v   # Vector store tests
pytest tests/test_scraper.py -v        # Scraper tests
```

## Validation Contracts

The application is validated against 20+ contracts covering:

| Area | Contracts | Description |
|------|-----------|-------------|
| Data Ingestion | VAL-DATA-001 to 005 | Wikipedia scraping, chunking correctness |
| Embeddings | VAL-EMBED-001 to 007 | Mean pooling, cosine similarity, HDF5 persistence |
| API & Frontend | VAL-API-001 to 008 | REST endpoints, CORS, chat interface |
| Docker | VAL-DOCKER-001 to 003 | Container build, startup, data persistence |
| Cross-Area | VAL-CROSS-001 to 003 | End-to-end flows, mathematical correctness |

See `validation-contract.md` for full specifications.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTHONPATH` | `/app` | Python module search path |
| `PYTHONUNBUFFERED` | `1` | Disable Python output buffering |

### Chunking Parameters

Edit `src/heuristic_chunker.py`:

```python
CHUNK_SIZE = 500  # Maximum tokens per chunk
OVERLAP = 50      # Target overlap tokens
```

### Model

Default: `sentence-transformers/all-MiniLM-L6-v2`

Produces 384-dimensional embeddings. To change the model, update `MODEL_NAME` in:
- `src/heuristic_chunker.py`
- `src/embedding.py`

## Dependencies

Core dependencies:

- `fastapi` + `uvicorn` - Web framework and server
- `transformers` + `torch` - Embedding model
- `wikipedia-api` + `httpx` - Wikipedia scraping
- `numpy` - Numerical operations
- `h5py` - HDF5 persistence
- `pytest` - Testing framework

See `requirements.txt` for complete list.

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- Embeddings powered by [sentence-transformers](https://www.sbert.net/)
- Wikipedia data via [Wikipedia API](https://pypi.org/project/wikipedia-api/)
- Built with [FastAPI](https://fastapi.tiangolo.com/)
