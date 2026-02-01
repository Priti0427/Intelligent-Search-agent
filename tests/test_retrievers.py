"""
Tests for the retrievers.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.retrievers.web_search import TavilyRetriever
from src.retrievers.vector_store import VectorStoreRetriever
from src.retrievers.arxiv_search import ArxivRetriever


class TestTavilyRetriever:
    """Tests for Tavily web search retriever."""
    
    def test_initialization(self):
        """Test retriever initialization."""
        retriever = TavilyRetriever(api_key="test_key", max_results=3)
        assert retriever.api_key == "test_key"
        assert retriever.max_results == 3
    
    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        """Test that search returns properly formatted results."""
        retriever = TavilyRetriever(api_key="test_key")
        
        with patch.object(retriever, 'client') as mock_client:
            mock_client.search.return_value = {
                "results": [
                    {
                        "title": "Test Result",
                        "url": "https://example.com",
                        "content": "Test content",
                        "score": 0.95,
                    }
                ]
            }
            
            results = await retriever.search("test query")
            
            assert len(results) == 1
            assert results[0]["title"] == "Test Result"
            assert results[0]["source_type"] == "web"
            assert results[0]["score"] == 0.95


class TestVectorStoreRetriever:
    """Tests for ChromaDB vector store retriever."""
    
    def test_initialization(self):
        """Test retriever initialization."""
        retriever = VectorStoreRetriever(
            collection_name="test_collection",
            max_results=10,
        )
        assert retriever.collection_name == "test_collection"
        assert retriever.max_results == 10
    
    def test_get_document_count_empty(self):
        """Test document count on empty collection."""
        with patch('src.retrievers.vector_store.chromadb') as mock_chroma:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chroma.PersistentClient.return_value = mock_client
            
            retriever = VectorStoreRetriever()
            retriever._client = mock_client
            retriever._collection = mock_collection
            
            count = retriever.get_document_count()
            assert count == 0


class TestArxivRetriever:
    """Tests for arXiv academic paper retriever."""
    
    def test_initialization(self):
        """Test retriever initialization."""
        retriever = ArxivRetriever(max_results=5)
        assert retriever.max_results == 5
    
    @pytest.mark.asyncio
    async def test_search_formats_results(self):
        """Test that search formats results correctly."""
        retriever = ArxivRetriever(max_results=1)
        
        # Create mock paper
        mock_paper = MagicMock()
        mock_paper.title = "Test Paper"
        mock_paper.summary = "Test abstract"
        mock_paper.entry_id = "http://arxiv.org/abs/2301.00001"
        mock_paper.authors = [MagicMock(name="Author One")]
        mock_paper.published = None
        mock_paper.updated = None
        mock_paper.categories = ["cs.IR"]
        mock_paper.pdf_url = "http://arxiv.org/pdf/2301.00001"
        
        with patch.object(retriever, 'client') as mock_client:
            mock_client.results.return_value = [mock_paper]
            
            results = await retriever.search("information retrieval")
            
            assert len(results) == 1
            assert results[0]["source_type"] == "academic"
            assert "Test Paper" in results[0]["content"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
