"""
Pydantic Schemas for API requests and responses.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# === Search Schemas ===

class SearchRequest(BaseModel):
    """Request schema for search endpoint."""
    
    query: str = Field(..., description="The search query", min_length=1, max_length=2000)
    include_sources: bool = Field(default=True, description="Include source citations")
    max_results: int = Field(default=5, description="Maximum results per source", ge=1, le=20)
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What is RAG in AI?",
                    "include_sources": True,
                    "max_results": 5,
                }
            ]
        }
    }


class Citation(BaseModel):
    """A citation for a source."""
    
    number: int
    title: str
    url: Optional[str] = None
    source_type: str
    excerpt: str


class QualityScores(BaseModel):
    """Quality scores from self-reflection."""
    
    relevance: float = Field(ge=0, le=1)
    completeness: float = Field(ge=0, le=1)
    accuracy: float = Field(ge=0, le=1)
    citation_quality: float = Field(ge=0, le=1)
    clarity: float = Field(ge=0, le=1)


class SearchMetadata(BaseModel):
    """Metadata about the search process."""
    
    query_type: str
    sub_queries: List[str]
    sources_searched: List[str]
    total_results: int
    iterations: int
    quality_score: float
    processing_time_ms: float


class SearchResponse(BaseModel):
    """Response schema for search endpoint."""
    
    query: str
    answer: str
    citations: List[Citation] = []
    quality_scores: Optional[QualityScores] = None
    metadata: Optional[SearchMetadata] = None
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What is RAG in AI?",
                    "answer": "RAG (Retrieval-Augmented Generation) is a technique...",
                    "citations": [
                        {
                            "number": 1,
                            "title": "Understanding RAG",
                            "url": "https://example.com/rag",
                            "source_type": "web",
                            "excerpt": "RAG combines retrieval with generation...",
                        }
                    ],
                }
            ]
        }
    }


# === Ingest Schemas ===

class IngestRequest(BaseModel):
    """Request schema for document ingestion."""
    
    text: Optional[str] = Field(None, description="Raw text to ingest")
    file_path: Optional[str] = Field(None, description="Path to file to ingest")
    directory_path: Optional[str] = Field(None, description="Path to directory to ingest")
    title: str = Field(default="Untitled", description="Document title")
    chunk_strategy: str = Field(default="fixed", description="Chunking strategy")
    chunk_size: int = Field(default=1000, description="Chunk size", ge=100, le=5000)
    chunk_overlap: int = Field(default=200, description="Chunk overlap", ge=0, le=1000)
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "This is some text to ingest into the vector store.",
                    "title": "Sample Document",
                    "chunk_strategy": "fixed",
                }
            ]
        }
    }


class IngestResponse(BaseModel):
    """Response schema for document ingestion."""
    
    success: bool
    chunks_ingested: int
    message: str
    document_count: int


# === Health Schemas ===

class ServiceStatus(BaseModel):
    """Status of a service."""
    
    name: str
    status: str  # "ok", "error", "not_configured"
    message: Optional[str] = None


class HealthResponse(BaseModel):
    """Response schema for health check."""
    
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: datetime
    version: str
    services: List[ServiceStatus]
    document_count: int


# === Streaming Schemas ===

class StreamEvent(BaseModel):
    """A streaming event during search."""
    
    event_type: str  # "status", "result", "error", "complete"
    node: Optional[str] = None
    data: Dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=datetime.utcnow)
