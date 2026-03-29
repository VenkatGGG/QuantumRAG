# Environment

Environment variables, external dependencies, and setup notes.

**What belongs here:** Required env vars, external API keys/services, dependency quirks, platform-specific notes.
**What does NOT belong here:** Service ports/commands (use `.factory/services.yaml`).

---

## External Dependencies

### Wikipedia API
- Used for: Scraping Quantum Cryptography articles
- Rate limiting: 0.5s delay between requests required
- User-Agent: Required header with project name and contact
- No API key needed

### Hugging Face Transformers
- Used for: Loading sentence-transformers/all-MiniLM-L6-v2
- Model size: ~80MB download on first run
- Local inference only - no external API calls
- CPU inference (no GPU required)

## Python Dependencies

Core dependencies (from requirements.txt):
- transformers>=4.30.0
- torch>=2.0.0
- numpy>=1.24.0
- fastapi>=0.100.0
- uvicorn>=0.23.0
- h5py>=3.9.0
- pytest>=7.4.0
- wikipedia-api>=0.6.0
- httpx>=0.24.0  # For FastAPI TestClient

## Data Storage

- HDF5 file at `data/vector_store.h5`
- Contains: vectors (NumPy array), texts (list), metadata
- Persisted across container restarts via volume mount
- Gitignored - not committed to repository

## Model Information

- Model: sentence-transformers/all-MiniLM-L6-v2
- Output dimension: 384
- Max sequence length: 256 tokens
- Pooling: Mean pooling with attention mask (manual implementation)

## Notes

- PyTorch CPU version is used to minimize Docker image size
- First model download may take 1-2 minutes depending on connection
- HDF5 requires system library libhdf5-dev on Linux
