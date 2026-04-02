# Quantum Cryptography RAG

Local Retrieval-Augmented Generation application focused on quantum cryptography. The project scrapes Wikipedia content, chunks it with sentence-aware heuristics, generates embeddings locally with raw `transformers` + `torch`, stores vectors in pure NumPy with HDF5 persistence, and serves retrieval through a FastAPI backend plus a vanilla HTML/JS chat UI.

This repo intentionally avoids orchestration and vector database frameworks such as LangChain, LlamaIndex, FAISS, ChromaDB, and `sentence-transformers`.

## What It Does

- Scrapes the top Wikipedia search hits for `Quantum Cryptography`
- Filters search results toward quantum-cryptography relevance
- Splits article text into sentence-aligned chunks with a hard `<= 500` token cap
- Uses sentence-aligned overlap targeted toward `50` tokens
- Loads `sentence-transformers/all-MiniLM-L6-v2` locally with raw `transformers`
- Implements manual mean pooling over `last_hidden_state`
- Stores embeddings and texts in an in-memory NumPy-backed vector store
- Persists vectors and texts to HDF5
- Exposes `/status`, `/query`, `/health`, and `/`
- Serves a vanilla chat-style frontend

## Current Repository Snapshot

At the time of this README update, the committed artifacts in [`data/raw_articles.json`](/Users/sri/Desktop/silly_experiments/Droid_Project/data/raw_articles.json) and [`data/vector_store.h5`](/Users/sri/Desktop/silly_experiments/Droid_Project/data/vector_store.h5) contain:

- `10` scraped articles
- `190` persisted chunks / vectors
- embedding dimension `384`

The committed article titles are:

- `Quantum cryptography`
- `Post-quantum cryptography`
- `Quantum computing`
- `NIST Post-Quantum Cryptography Standardization`
- `Quantum key distribution`
- `Relativistic quantum cryptography`
- `Quantum network`
- `Harvest now, decrypt later`
- `Quantum information science`
- `Quantum entanglement`

## Architecture

```text
Wikipedia API
  -> scraper + relevance filter
  -> sentence-aware chunker
  -> local embedding pipeline
  -> NumPy vector store
  -> HDF5 persistence
  -> FastAPI backend
  -> vanilla HTML/JS chat frontend
```

Core implementation files:

- [`src/wikipedia_scraper.py`](/Users/sri/Desktop/silly_experiments/Droid_Project/src/wikipedia_scraper.py): Wikipedia search, rate limiting, topical filtering, article fetch, JSON save
- [`src/heuristic_chunker.py`](/Users/sri/Desktop/silly_experiments/Droid_Project/src/heuristic_chunker.py): sentence-aware chunking with Option A contract
- [`src/embedding.py`](/Users/sri/Desktop/silly_experiments/Droid_Project/src/embedding.py): raw model loading, manual mean pooling, NumPy conversion
- [`src/vector_store.py`](/Users/sri/Desktop/silly_experiments/Droid_Project/src/vector_store.py): cosine similarity search and HDF5 save/load
- [`src/main.py`](/Users/sri/Desktop/silly_experiments/Droid_Project/src/main.py): FastAPI app, lifespan loading, lazy embedding initialization, static frontend serving
- [`static/index.html`](/Users/sri/Desktop/silly_experiments/Droid_Project/static/index.html): chat transcript UI with async `/query` calls

## Chunking Contract

This project follows the Option A contract described in [`validation-contract.md`](/Users/sri/Desktop/silly_experiments/Droid_Project/validation-contract.md):

- chunk boundaries do not cut sentences
- each chunk is at most `500` tokenizer tokens
- overlap is sentence-aligned and chosen to be as close as practical to `50` tokens
- `50` is a target, not a mathematical guarantee

This is important because exact token overlap and strict sentence-boundary preservation are not always simultaneously achievable when sentence lengths vary.

## Embedding and Retrieval Details

Model:

- `sentence-transformers/all-MiniLM-L6-v2`
- loaded via `AutoTokenizer` and `AutoModel`
- loaded with `local_files_only=True`

Mean pooling is implemented manually in [`src/embedding.py`](/Users/sri/Desktop/silly_experiments/Droid_Project/src/embedding.py) using:

```text
E = sum(T_i * M_i) / max(sum(M_i), 1e-9)
```

Cosine similarity is implemented manually in [`src/vector_store.py`](/Users/sri/Desktop/silly_experiments/Droid_Project/src/vector_store.py) using:

```text
similarity = dot(A, B) / (||A|| * ||B||)
```

Embeddings are converted with `detach().cpu().numpy()` before storage/search.

## Repository Layout

```text
.
├── src/
├── static/
├── data/
├── scripts/
│   ├── analysis/
│   ├── manual_tests/
│   └── pipeline/
├── tests/
│   └── test_scripts/
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
├── requirements.txt
├── setup.py
├── DESIGN.md
├── validation-contract.md
└── README.md
```

Useful entrypoints:

- [`scripts/pipeline/scrape_articles.py`](/Users/sri/Desktop/silly_experiments/Droid_Project/scripts/pipeline/scrape_articles.py)
- [`scripts/pipeline/run_full_pipeline.py`](/Users/sri/Desktop/silly_experiments/Droid_Project/scripts/pipeline/run_full_pipeline.py)
- [`scripts/download_model.py`](/Users/sri/Desktop/silly_experiments/Droid_Project/scripts/download_model.py)
- [`scripts/manual_tests/manual_chunker.py`](/Users/sri/Desktop/silly_experiments/Droid_Project/scripts/manual_tests/manual_chunker.py)
- [`scripts/manual_tests/manual_embedding.py`](/Users/sri/Desktop/silly_experiments/Droid_Project/scripts/manual_tests/manual_embedding.py)
- [`scripts/manual_tests/manual_vector_store.py`](/Users/sri/Desktop/silly_experiments/Droid_Project/scripts/manual_tests/manual_vector_store.py)

## Local Setup

### Prerequisites

- Python `3.10+`
- `venv` or equivalent virtual environment tooling

### Install

From the repository root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The editable install matters because many helper scripts import `src.*` directly.

### Download the Embedding Model

The embedding pipeline runs with `local_files_only=True`, so the model must already exist in the Hugging Face cache before local inference:

```bash
venv/bin/python scripts/download_model.py
```

If the model is not cached, local embedding and query calls will fail instead of downloading automatically.

### Run the API

If you want to use the committed `data/vector_store.h5` as-is:

```bash
venv/bin/uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

Then open:

- [http://localhost:8000](http://localhost:8000)
- [http://localhost:8000/docs](http://localhost:8000/docs)
- [http://localhost:8000/status](http://localhost:8000/status)

### Rebuild the Corpus

To regenerate the dataset from scratch:

```bash
venv/bin/python scripts/pipeline/run_full_pipeline.py
```

This runs:

1. Wikipedia search + scrape
2. sentence-aware chunking
3. local embedding generation
4. HDF5 persistence to [`data/vector_store.h5`](/Users/sri/Desktop/silly_experiments/Droid_Project/data/vector_store.h5)

If you only want the raw scraped articles:

```bash
venv/bin/python scripts/pipeline/scrape_articles.py
```

## Docker

The Docker path is designed to be self-contained:

- installs CPU PyTorch
- pre-downloads the Hugging Face model at build time via [`scripts/download_model.py`](/Users/sri/Desktop/silly_experiments/Droid_Project/scripts/download_model.py)
- sets `HF_HUB_OFFLINE=1` for runtime
- mounts `./data` into `/app/data` for persistence

Build and run:

```bash
docker compose up --build
```

Detached mode:

```bash
docker compose up --build -d
docker compose ps
```

Stop:

```bash
docker compose down
```

The compose service is defined in [`docker-compose.yml`](/Users/sri/Desktop/silly_experiments/Droid_Project/docker-compose.yml) and exposes port `8000`.

## API Reference

### `GET /status`

Returns vector store metadata.

Example response:

```json
{
  "chunk_count": 190,
  "vector_dimensions": 384
}
```

### `POST /query`

Embeds the incoming query, performs NumPy cosine similarity search, and returns the top `k` chunks.

Request:

```json
{
  "query": "What is BB84?",
  "k": 5
}
```

Response:

```json
{
  "results": [
    {
      "text": "Quantum key distribution ...",
      "similarity": 0.91
    }
  ]
}
```

Behavior:

- empty `query` returns `400`
- omitted `k` defaults to `5`
- `k < 1` returns `400`
- empty vector store returns `{"results": []}`

### `GET /health`

Container health endpoint.

Example response:

```json
{
  "status": "healthy",
  "chunks": 190
}
```

### `GET /`

Serves the vanilla HTML/JS chat interface from [`static/index.html`](/Users/sri/Desktop/silly_experiments/Droid_Project/static/index.html).

## Frontend

The frontend is intentionally framework-free:

- plain HTML/CSS/JS
- chat transcript with user and assistant messages
- async `fetch()` calls to `/query`
- welcome prompts and example questions
- connection status from `/status`
- retrieved-context rendering with ranked snippets and similarity scores

## Testing

Pytest discovery is restricted by [`pytest.ini`](/Users/sri/Desktop/silly_experiments/Droid_Project/pytest.ini) to the [`tests/`](/Users/sri/Desktop/silly_experiments/Droid_Project/tests) tree, so helper scripts under [`scripts/manual_tests/`](/Users/sri/Desktop/silly_experiments/Droid_Project/scripts/manual_tests) are not collected as part of the normal automated suite.

Run the main automated suite:

```bash
venv/bin/python -m pytest -q
```

Run targeted suites:

```bash
venv/bin/python -m pytest -q tests/test_scraper.py
venv/bin/python -m pytest -q tests/test_chunker.py
venv/bin/python -m pytest -q tests/test_embedding.py
venv/bin/python -m pytest -q tests/test_vector_store.py
venv/bin/python -m pytest -q tests/test_api.py
```

Run manual validators:

```bash
PYTHONPATH=. venv/bin/python scripts/manual_tests/manual_chunker.py
PYTHONPATH=. venv/bin/python scripts/manual_tests/manual_embedding.py
PYTHONPATH=. venv/bin/python scripts/manual_tests/manual_vector_store.py
```

## Known Notes

- Local scripts under [`scripts/`](/Users/sri/Desktop/silly_experiments/Droid_Project/scripts) assume the project package is importable. Use `pip install -e .` or prefix commands with `PYTHONPATH=.`
- The FastAPI app lazily initializes the embedding model on first query, but Docker pre-populates the model cache during image build
- The current automated suite is close to green, but there is still a stale API test in [`tests/test_api.py`](/Users/sri/Desktop/silly_experiments/Droid_Project/tests/test_api.py) that imports a removed `vector_store` global instead of using the current dependency-injected state
- The corpus is only as good as Wikipedia search relevance and the project’s title-based quantum filtering heuristic

## Example Workflow

Typical local workflow:

1. Create and activate the virtual environment
2. Install dependencies and run `pip install -e .`
3. Download the embedding model
4. Run the full pipeline if you want a fresh corpus
5. Start `uvicorn`
6. Open the chat UI in the browser
7. Ask retrieval questions such as `What is BB84?` or `How does QKD detect eavesdropping?`

## License / Ownership

No explicit license file is currently present in the repository. Add one before publishing or redistributing the project outside your own use.
