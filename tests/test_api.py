"""
Tests for the API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_returns_status(self, client):
        """Test that health endpoint returns status."""
        response = client.get("/api/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "services" in data
        assert "document_count" in data
    
    def test_health_includes_services(self, client):
        """Test that health includes service statuses."""
        response = client.get("/api/health")
        data = response.json()
        
        service_names = [s["name"] for s in data["services"]]
        assert "openai" in service_names
        assert "tavily" in service_names
        assert "chromadb" in service_names


class TestSearchEndpoint:
    """Tests for search endpoint."""
    
    def test_search_requires_query(self, client):
        """Test that search requires a query."""
        response = client.post("/api/search", json={})
        assert response.status_code == 422  # Validation error
    
    def test_search_accepts_valid_request(self, client):
        """Test search with valid request."""
        with patch('src.api.routes.run_search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = {
                "final_answer": "Test answer",
                "citations": [],
                "quality_scores": {
                    "relevance": 0.8,
                    "completeness": 0.7,
                    "accuracy": 0.9,
                    "citation_quality": 0.6,
                    "clarity": 0.8,
                },
                "overall_quality": 0.76,
                "query_type": "simple",
                "sub_queries": [],
                "web_results": [],
                "vector_results": [],
                "arxiv_results": [],
                "iteration_count": 1,
            }
            
            response = client.post("/api/search", json={
                "query": "What is RAG?",
                "include_sources": True,
                "max_results": 5,
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["answer"] == "Test answer"


class TestIngestEndpoint:
    """Tests for document ingestion endpoint."""
    
    def test_ingest_requires_content(self, client):
        """Test that ingest requires some content."""
        response = client.post("/api/ingest", json={
            "title": "Test",
        })
        # Should fail because no text, file_path, or directory_path provided
        assert response.status_code in [400, 500]
    
    def test_ingest_accepts_text(self, client):
        """Test ingesting raw text."""
        with patch('src.api.routes.DocumentEmbedder') as mock_embedder:
            mock_instance = mock_embedder.return_value
            mock_instance.ingest_text = AsyncMock(return_value=5)
            mock_instance.get_stats.return_value = {"document_count": 5}
            
            response = client.post("/api/ingest", json={
                "text": "This is test content for ingestion.",
                "title": "Test Document",
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True


class TestStatsEndpoint:
    """Tests for stats endpoint."""
    
    def test_stats_returns_count(self, client):
        """Test that stats returns document count."""
        response = client.get("/api/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "document_count" in data
        assert "status" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
