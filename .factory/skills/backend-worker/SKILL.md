---
name: backend-worker
description: Worker for FastAPI backend and vanilla frontend implementation
---

# Backend Worker

Handles FastAPI backend endpoints, CORS configuration, and vanilla HTML/JS frontend.

## When to Use This Skill

Use for features involving:
- FastAPI route handlers
- API endpoint implementation
- CORS middleware configuration
- Vanilla HTML/JS frontend (no frameworks)
- Static file serving

## Required Skills

- `agent-browser` - For frontend validation via browser automation

## Work Procedure

1. **Write tests first (TDD)**
   - Create test file with pytest using TestClient
   - Test /status endpoint returns correct schema
   - Test /query endpoint with various inputs
   - Test CORS headers are present

2. **Implement FastAPI backend**
   - Create /status endpoint returning {chunk_count, vector_dimensions}
   - Create /query endpoint accepting {query, k?} returning {results: [{text, similarity}]}
   - Configure CORS middleware properly
   - Load vector store from HDF5 on startup

3. **Implement vanilla frontend**
   - Create index.html with chat interface
   - Use vanilla JavaScript (no React/Vue/Svelte)
   - Fetch API for async backend communication
   - Display query results with similarity scores

4. **Manual verification with agent-browser**
   - Start the FastAPI server
   - Use agent-browser to navigate to frontend
   - Test query submission
   - Verify results display correctly

5. **Run validators**
   - `python -m pytest tests/ -v`
   - Check with curl: `curl http://localhost:8000/status`

6. **Commit work**

## Example Handoff

```json
{
  "salientSummary": "Implemented FastAPI backend with /status and /query endpoints, proper CORS configuration, and vanilla HTML/JS chat interface that successfully communicates with the backend.",
  "whatWasImplemented": "Created main.py with FastAPI app, /status endpoint returning chunk count and vector dimensions, /query endpoint that embeds user query and returns top-k similar chunks with similarity scores. Configured CORSMiddleware to allow frontend requests. Created static/index.html with vanilla JS chat interface using Fetch API. Frontend displays query input, results list with similarity scores, and loading states.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {"command": "python -m pytest tests/test_api.py -v", "exitCode": 0, "observation": "8 tests passed including status endpoint, query endpoint, CORS headers, and error handling"},
      {"command": "curl -s http://localhost:8000/status | jq", "exitCode": 0, "observation": "{\"chunk_count\": 127, \"vector_dimensions\": 384}"},
      {"command": "curl -s -X POST http://localhost:8000/query -H 'Content-Type: application/json' -d '{\"query\": \"quantum key distribution\", \"k\": 3}' | jq", "exitCode": 0, "observation": "Returns 3 results with text snippets and similarity scores"}
    ],
    "interactiveChecks": [
      {"action": "Navigate to http://localhost:8000/ with agent-browser, type 'quantum entanglement' in query box, click Submit", "observed": "Results displayed with 3 context snippets and similarity scores (0.89, 0.85, 0.82). No console errors. CORS working correctly."}
    ]
  },
  "tests": {
    "added": [
      {"file": "tests/test_api.py", "cases": [
        {"name": "test_status_endpoint", "verifies": "GET /status returns chunk_count and vector_dimensions"},
        {"name": "test_query_endpoint", "verifies": "POST /query returns k results with text and similarity"},
        {"name": "test_query_default_k", "verifies": "Default k=5 when not specified"},
        {"name": "test_cors_headers", "verifies": "CORS headers present on responses"},
        {"name": "test_empty_query", "verifies": "Empty query returns 400 error"},
        {"name": "test_query_not_found", "verifies": "Query with no matches returns empty results"},
        {"name": "test_frontend_served", "verifies": "GET / serves index.html"},
        {"name": "test_static_files", "verifies": "Static assets served correctly"}
      ]}
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- CORS configuration not working (frontend cannot reach backend)
- Port 8000 is unavailable
- Vector store fails to load on startup
- Frontend assets not being served correctly
