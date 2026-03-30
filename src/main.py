"""FastAPI backend for the RAG application.

This module implements the FastAPI backend with:
- GET /status endpoint (returns chunk_count and vector_dimensions)
- POST /query endpoint (accepts query string, embeds it, searches vector store)
- CORS middleware configuration
- Static file serving for the frontend
- Vector store loading from HDF5 on startup
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from src.vector_store import VectorStore
from src.embedding import EmbeddingPipeline

# Initialize global instances
vector_store: VectorStore = VectorStore(dimension=384)
embedding_pipeline: Optional[EmbeddingPipeline] = None

# Path to HDF5 file
HDF5_PATH = Path("data/vector_store.h5")


def load_vector_store() -> None:
    """Load the vector store from HDF5 file on startup.
    
    If the HDF5 file exists, load vectors and texts into the global
    vector store instance. If it doesn't exist, the store remains empty.
    """
    global vector_store
    
    if HDF5_PATH.exists():
        try:
            vector_store.load_from_hdf5(str(HDF5_PATH))
            print(f"Loaded vector store from {HDF5_PATH}")
            print(f"  - Chunks: {vector_store.size}")
            print(f"  - Dimensions: {vector_store.dimension}")
        except Exception as e:
            print(f"Warning: Failed to load vector store from {HDF5_PATH}: {e}")
            # Continue with empty store
    else:
        print(f"Note: HDF5 file not found at {HDF5_PATH}, starting with empty vector store")


def init_embedding_pipeline() -> None:
    """Initialize the embedding pipeline on first use.
    
    This is done lazily to avoid slow startup times.
    """
    global embedding_pipeline
    
    if embedding_pipeline is None:
        print("Initializing embedding pipeline...")
        embedding_pipeline = EmbeddingPipeline()
        print("Embedding pipeline ready")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup: Load vector store
    load_vector_store()
    yield
    # Shutdown: (optional cleanup could go here)


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="RAG Query API",
    description="Retrieval-Augmented Generation API for querying Wikipedia articles about Quantum Cryptography",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


# Request/Response models
class QueryRequest(BaseModel):
    """Request model for the /query endpoint."""
    query: str
    k: Optional[int] = 5


class QueryResult(BaseModel):
    """Single result from vector store search."""
    text: str
    similarity: float


class QueryResponse(BaseModel):
    """Response model for the /query endpoint."""
    results: List[QueryResult]


class StatusResponse(BaseModel):
    """Response model for the /status endpoint."""
    chunk_count: int
    vector_dimensions: int


# API Endpoints
@app.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    """Get the current status of the vector store.
    
    Returns:
        StatusResponse with chunk_count and vector_dimensions.
    """
    return StatusResponse(
        chunk_count=vector_store.size,
        vector_dimensions=vector_store.dimension
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Query the vector store for similar chunks.
    
    Embeds the query text using the embedding pipeline and searches
    the vector store for the top-k most similar chunks.
    
    Args:
        request: QueryRequest containing the query string and optional k.
    
    Returns:
        QueryResponse with a list of results containing text and similarity.
    
    Raises:
        HTTPException: If query is empty or embedding pipeline fails.
    """
    # Validate query
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Validate k
    k = request.k if request.k is not None else 5
    if k < 1:
        raise HTTPException(status_code=400, detail="k must be at least 1")
    
    # Initialize embedding pipeline if needed
    init_embedding_pipeline()
    
    # Check if vector store has data
    if vector_store.size == 0:
        return QueryResponse(results=[])
    
    try:
        # Embed the query
        query_embedding = embedding_pipeline.embed(request.query)
        
        # Search vector store
        results = vector_store.search(query_embedding, k=k)
        
        # Convert to response model
        query_results = [
            QueryResult(text=r["text"], similarity=r["similarity"])
            for r in results
        ]
        
        return QueryResponse(results=query_results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


# Static file serving
# Try to serve static files from the static directory
static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_frontend():
    """Serve the frontend index.html file."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    else:
        # Return API info if no frontend
        return JSONResponse({
            "message": "RAG Query API",
            "endpoints": {
                "status": "GET /status",
                "query": "POST /query"
            },
            "docs": "/docs"
        })


# Health check endpoint (for Docker)
@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return {"status": "healthy", "chunks": vector_store.size}
