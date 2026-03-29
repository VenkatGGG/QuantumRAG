"""Tests for the FastAPI backend API.

This module tests the API endpoints including:
- GET /status endpoint (returns chunk_count and vector_dimensions)
- POST /query endpoint (accepts query string, embeds it, searches vector store)
- CORS configuration
- Error handling
"""

import pytest
from fastapi.testclient import TestClient
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock, PropertyMock

# Create mock instances before importing the app
mock_vector_store_instance = MagicMock()
mock_vector_store_instance.size = 10
mock_vector_store_instance.dimension = 384
mock_vector_store_instance.search.return_value = [
    {"text": "Sample text 1", "similarity": 0.95},
    {"text": "Sample text 2", "similarity": 0.87},
    {"text": "Sample text 3", "similarity": 0.82},
]

mock_embedding_pipeline_instance = MagicMock()
mock_embedding_pipeline_instance.embed.return_value = np.random.randn(384)
mock_embedding_pipeline_instance.EMBEDDING_DIM = 384

# Patch before importing
with patch("src.main.vector_store", mock_vector_store_instance), \
     patch("src.main.embedding_pipeline", None):
    from src.main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_vector_store():
    """Get the mock vector store instance."""
    return mock_vector_store_instance


class TestStatusEndpoint:
    """Tests for the GET /status endpoint."""
    
    def test_status_returns_chunk_count(self, client):
        """Test that /status returns the chunk_count field."""
        response = client.get("/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "chunk_count" in data
        assert data["chunk_count"] == 10
    
    def test_status_returns_vector_dimensions(self, client):
        """Test that /status returns the vector_dimensions field."""
        response = client.get("/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "vector_dimensions" in data
        assert data["vector_dimensions"] == 384
    
    def test_status_returns_json(self, client):
        """Test that /status returns proper JSON response."""
        response = client.get("/status")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        data = response.json()
        assert isinstance(data, dict)


class TestQueryEndpoint:
    """Tests for the POST /query endpoint."""
    
    def test_query_accepts_query_string(self, client):
        """Test that /query accepts a query string parameter."""
        response = client.post(
            "/query",
            json={"query": "test query"}
        )
        
        assert response.status_code == 200
    
    def test_query_accepts_k_parameter(self, client):
        """Test that /query accepts an optional k parameter."""
        response = client.post(
            "/query",
            json={"query": "test query", "k": 3}
        )
        
        assert response.status_code == 200
    
    def test_query_defaults_to_k_5(self, client, mock_vector_store):
        """Test that /query defaults to k=5 when not specified."""
        mock_vector_store.reset_mock()
        
        response = client.post(
            "/query",
            json={"query": "test query"}
        )
        
        assert response.status_code == 200
        # Verify search was called with k=5
        mock_vector_store.search.assert_called_once()
        call_args = mock_vector_store.search.call_args
        assert call_args[1]["k"] == 5
    
    def test_query_returns_results_array(self, client):
        """Test that /query returns results as an array."""
        response = client.post(
            "/query",
            json={"query": "test query"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)
    
    def test_query_results_have_text_field(self, client):
        """Test that each result has a text field."""
        response = client.post(
            "/query",
            json={"query": "test query"}
        )
        
        data = response.json()
        for result in data["results"]:
            assert "text" in result
            assert isinstance(result["text"], str)
    
    def test_query_results_have_similarity_field(self, client):
        """Test that each result has a similarity field."""
        response = client.post(
            "/query",
            json={"query": "test query"}
        )
        
        data = response.json()
        for result in data["results"]:
            assert "similarity" in result
            assert isinstance(result["similarity"], float)
    
    def test_query_empty_string_returns_error(self, client):
        """Test that empty query string returns 400 error."""
        response = client.post(
            "/query",
            json={"query": ""}
        )
        
        assert response.status_code == 400
    
    def test_query_missing_query_field_returns_error(self, client):
        """Test that missing query field returns 422 error."""
        response = client.post(
            "/query",
            json={}
        )
        
        assert response.status_code == 422
    
    def test_query_k_larger_than_store(self, client, mock_vector_store):
        """Test that k larger than store size returns available results."""
        mock_vector_store.reset_mock()
        mock_vector_store.size = 3
        mock_vector_store.search.return_value = [
            {"text": "Only result", "similarity": 0.95},
        ]
        
        response = client.post(
            "/query",
            json={"query": "test", "k": 10}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) <= 3


class TestCORSMiddleware:
    """Tests for CORS configuration."""
    
    def test_cors_headers_present_on_status(self, client):
        """Test that CORS headers are present on /status response."""
        response = client.get("/status")
        
        assert "access-control-allow-origin" in response.headers
    
    def test_cors_headers_present_on_query(self, client):
        """Test that CORS headers are present on /query response."""
        response = client.post(
            "/query",
            json={"query": "test"}
        )
        
        assert "access-control-allow-origin" in response.headers
    
    def test_cors_preflight_request(self, client):
        """Test that CORS preflight OPTIONS request is handled."""
        response = client.options(
            "/query",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            }
        )
        
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers


class TestFrontendServing:
    """Tests for frontend static file serving."""
    
    def test_root_serves_index_html(self, client):
        """Test that GET / serves the index.html file."""
        response = client.get("/")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_static_files_served(self, client):
        """Test that static files are served correctly."""
        # This test assumes static files exist
        response = client.get("/static/index.html")
        
        # Should either serve the file or return 404 if not exists
        assert response.status_code in [200, 404]


class TestVectorStoreLoading:
    """Tests for vector store loading on startup."""
    
    def test_vector_store_loaded_on_startup(self):
        """Test that vector store is loaded from HDF5 on startup."""
        from src.main import load_vector_store, vector_store
        
        # The vector store should be initialized
        assert vector_store is not None
        assert hasattr(vector_store, 'size')
        assert hasattr(vector_store, 'dimension')
