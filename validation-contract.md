# Validation Contract: Localized RAG Application

## Area: Data Ingestion (API Surface)

### VAL-DATA-001: Wikipedia scraper fetches 10 articles
The Wikipedia scraper successfully fetches exactly 10 articles related to "Quantum Cryptography" using the Wikipedia API.
Tool: curl/python script
Evidence: log output showing 10 article titles and URLs

### VAL-DATA-002: Scraper respects rate limiting
The scraper implements proper rate limiting with at least 0.5s delay between requests and includes proper User-Agent header.
Tool: code review
Evidence: source code showing time.sleep() and User-Agent configuration

### VAL-DATA-003: Sentence boundary detection works
The chunking algorithm correctly identifies sentence boundaries and never splits a sentence mid-way.
Tool: pytest
Evidence: test output showing chunks end at sentence boundaries

### VAL-DATA-004: Chunk size is at most 500 tokens
Each chunk contains at most 500 tokens (using the model's tokenizer).
Tool: pytest
Evidence: test output verifying token counts

### VAL-DATA-005: Overlap is sentence-aligned closest to 50 tokens
Consecutive chunks share overlap that is sentence-aligned and closest to 50 tokens (target-based, not exact guarantee).
Tool: pytest
Evidence: test output verifying overlap is within reasonable range of 50 tokens

## Area: Embeddings & Vector Store (API Surface)

### VAL-EMBED-001: Mean pooling formula is correct
The mean pooling implementation follows the exact formula: E = sum(T_i * M_i) / max(sum(M_i), epsilon) with epsilon = 1e-9.
Tool: pytest with mocked tensors
Evidence: test output with baseline vectors showing expected output

### VAL-EMBED-002: Attention mask is applied
The attention mask correctly zeros out padding tokens in the pooling calculation.
Tool: pytest
Evidence: test showing masked tokens don't contribute to output

### VAL-EMBED-003: Output dimension is 384
The embedding pipeline produces vectors with exactly 384 dimensions (all-MiniLM-L6-v2 output size).
Tool: pytest
Evidence: shape assertion showing (384,) or (batch, 384)

### VAL-EMBED-004: No external APIs used
The embedding pipeline loads the model locally using transformers.AutoModel without calling external APIs.
Tool: code review
Evidence: source code showing from_pretrained with local model only

### VAL-EMBED-005: Cosine similarity formula is correct
The cosine similarity implementation follows: similarity = dot(A, B) / (||A|| * ||B||).
Tool: pytest with baseline vectors
Evidence: test output showing expected similarity scores

### VAL-EMBED-006: Top-k search returns correct neighbors
The vector store's top-k search returns exactly k results sorted by similarity (highest first).
Tool: pytest
Evidence: test output showing sorted results with correct count

### VAL-EMBED-007: HDF5 persistence works
The vector store can serialize to HDF5 and deserialize back, preserving all vectors and texts.
Tool: pytest
Evidence: test showing save/load roundtrip with data integrity

## Area: API & Frontend (Browser Surface)

### VAL-API-001: Status endpoint returns correct data
GET /status returns JSON with chunk_count and vector_dimensions fields.
Tool: curl
Evidence: curl output showing valid JSON with required fields

### VAL-API-002: Query endpoint accepts and processes requests
POST /query accepts JSON with "query" field and optional "k" field, returns JSON with "results" array.
Tool: curl
Evidence: curl output showing results with text and similarity fields

### VAL-API-003: Query endpoint default k value
When k is not specified, the query endpoint defaults to returning 5 results.
Tool: curl
Evidence: request without k returns 5 results

### VAL-API-004: CORS headers are present
All API responses include proper CORS headers allowing frontend communication.
Tool: curl with Origin header
Evidence: response headers showing Access-Control-Allow-Origin

### VAL-API-005: Frontend loads successfully
The root URL (GET /) serves the chat interface HTML.
Tool: agent-browser
Evidence: screenshot showing chat interface loaded

### VAL-API-006: Frontend displays chat transcript with message history
The chat interface displays a conversation transcript showing user messages and assistant responses in a turn-based format.
Tool: agent-browser
Evidence: screenshot showing message history with user and assistant messages displayed as chat bubbles

### VAL-API-007: Frontend supports user/assistant message turns
The chat interface supports multiple query/response turns, with each user message followed by an assistant response containing retrieved context.
Tool: agent-browser
Evidence: screenshot showing at least 2 complete turns (user asks, assistant responds with context)

### VAL-API-008: Frontend displays similarity scores in assistant responses
The assistant's response includes retrieved context snippets, each showing the text content and its similarity score.
Tool: agent-browser
Evidence: screenshot showing retrieved context with visible similarity percentages (e.g., "95% match")

## Area: Docker & Persistence (CLI Surface)

### VAL-DOCKER-001: Docker image builds successfully
The Dockerfile builds without errors and produces a runnable image.
Tool: docker-compose build
Evidence: build log showing successful completion

### VAL-DOCKER-002: Container starts and serves requests
The container starts successfully and responds to API requests on port 8000.
Tool: docker-compose up + curl
Evidence: curl response from containerized app

### VAL-DOCKER-003: Volume mount persists data
After restarting the container, the HDF5 data persists and is accessible.
Tool: docker-compose down/up + curl
Evidence: status endpoint shows same chunk count before and after restart

## Cross-Area Flows

### VAL-CROSS-001: End-to-end query flow
A user can submit a query through the frontend, the backend embeds it, searches the vector store, and returns relevant context chunks.
Tool: agent-browser
Evidence: screenshot showing complete flow with query and results

### VAL-CROSS-002: Full data pipeline
The system can scrape articles, chunk them, generate embeddings, store in vector store, and retrieve via API.
Tool: python script + curl
Evidence: script output showing each pipeline stage completing successfully

### VAL-CROSS-003: Mathematical correctness
The custom mean pooling and cosine similarity functions produce mathematically correct, deterministic outputs.
Tool: pytest
Evidence: test_suite.py output showing all mathematical tests pass with expected values
