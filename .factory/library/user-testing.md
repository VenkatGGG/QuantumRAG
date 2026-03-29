# User Testing

Testing surface, required testing skills/tools, and resource cost classification.

**What belongs here:** How to test the application, what surfaces are available, required tools, resource costs.

---

## Validation Surfaces

### 1. API Surface (curl/HTTP client)

**Testable via:** curl, HTTPie, Python requests, FastAPI TestClient

**Endpoints:**
- GET http://localhost:8000/status - Returns JSON with chunk_count, vector_dimensions
- POST http://localhost:8000/query - Accepts JSON {query: string, k?: number}

**Example curl commands:**
```bash
# Status check
curl -s http://localhost:8000/status | jq

# Query with default k=5
curl -s -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "quantum key distribution"}' | jq

# Query with custom k=3
curl -s -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "quantum entanglement", "k": 3}' | jq
```

**Resource cost:** Low (~50 MB RAM per concurrent test)

### 2. Browser Surface (agent-browser)

**Testable via:** agent-browser skill

**Entry point:** http://localhost:8000/

**Flows to test:**
1. Page loads successfully (index.html served)
2. Query input accepts text
3. Submit button triggers backend call
4. Results display with text and similarity scores
5. Loading state shown during query

**Example agent-browser usage:**
```python
# Navigate to frontend
# Type query in input field
# Click submit button
# Verify results appear with similarity scores
```

**Resource cost:** Medium (~300 MB RAM for browser + ~200 MB for dev server)

## Required Testing Skills

- **curl** - For API endpoint testing
- **agent-browser** - For frontend validation
- **pytest** - For running test suites

## Validation Concurrency

**Machine specs:** 32 GB RAM, 10 CPU cores

**Surface: API (curl)**
- Cost per validator: ~50 MB RAM
- Max concurrent: 5 (conservative, well within budget)

**Surface: Browser (agent-browser)**
- Cost per validator: ~500 MB RAM (browser + server)
- Max concurrent: 5 (conservative, well within budget)

**Note:** Only one surface is typically validated at a time during milestone validation.

## Testing Prerequisites

Before testing:
1. Install Python dependencies: `pip install -r requirements.txt`
2. Run full data pipeline: `python scripts/run_full_pipeline.py`
3. Start backend: `uvicorn src.main:app --host 0.0.0.0 --port 8000`
4. Wait for startup (model loading takes ~10-30s on first run)

## Known Testing Considerations

- **Model loading time:** First startup downloads ~80MB model, takes 1-2 minutes
- **HDF5 persistence:** Test data survives container restarts
- **CORS:** Must be properly configured for frontend to work
- **Chunk count:** Expected 50-200 chunks depending on article sizes
