"""
Tests for the Agentic Search agent.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.agent.state import SearchState, create_initial_state, SubQuery
from src.agent.nodes.query_analyzer import analyze_query_node
from src.agent.nodes.query_decomposer import decompose_query_node


class TestSearchState:
    """Tests for SearchState creation and manipulation."""
    
    def test_create_initial_state(self):
        """Test creating initial state from a query."""
        query = "What is RAG in AI?"
        state = create_initial_state(query)
        
        assert state["original_query"] == query
        assert state["query_type"] == ""
        assert state["sub_queries"] == []
        assert state["iteration_count"] == 0
        assert state["should_continue"] is True
    
    def test_initial_state_has_all_fields(self):
        """Test that initial state has all required fields."""
        state = create_initial_state("test query")
        
        required_fields = [
            "original_query", "messages", "query_type", "topics",
            "requires_web_search", "requires_academic", "requires_documents",
            "sub_queries", "web_results", "vector_results", "arxiv_results",
            "all_results", "draft_answer", "final_answer", "citations",
            "quality_scores", "overall_quality", "reflection_feedback",
            "missing_aspects", "iteration_count", "should_continue", "error"
        ]
        
        for field in required_fields:
            assert field in state, f"Missing field: {field}"


class TestQueryAnalyzer:
    """Tests for query analyzer node."""
    
    @pytest.mark.asyncio
    async def test_analyze_simple_query(self):
        """Test analyzing a simple query."""
        state = create_initial_state("What is Python?")
        
        with patch('src.agent.nodes.query_analyzer.ChatOpenAI') as mock_llm:
            mock_instance = MagicMock()
            mock_instance.ainvoke = AsyncMock(return_value=MagicMock(
                content='{"query_type": "simple", "topics": ["Python"], "requires_web_search": true, "requires_academic": false, "requires_documents": true}'
            ))
            mock_llm.return_value = mock_instance
            
            result = await analyze_query_node(state)
            
            assert result["query_type"] == "simple"
            assert "Python" in result["topics"]
    
    @pytest.mark.asyncio
    async def test_analyze_complex_query(self):
        """Test analyzing a complex query."""
        state = create_initial_state("Compare BM25 and dense retrieval for question answering")
        
        with patch('src.agent.nodes.query_analyzer.ChatOpenAI') as mock_llm:
            mock_instance = MagicMock()
            mock_instance.ainvoke = AsyncMock(return_value=MagicMock(
                content='{"query_type": "complex", "topics": ["BM25", "dense retrieval", "question answering"], "requires_web_search": true, "requires_academic": true, "requires_documents": true}'
            ))
            mock_llm.return_value = mock_instance
            
            result = await analyze_query_node(state)
            
            assert result["query_type"] == "complex"
            assert result["requires_academic"] is True


class TestQueryDecomposer:
    """Tests for query decomposer node."""
    
    @pytest.mark.asyncio
    async def test_decompose_simple_query(self):
        """Test that simple queries are not decomposed."""
        state = create_initial_state("What is RAG?")
        state["query_type"] = "simple"
        
        result = await decompose_query_node(state)
        
        assert len(result["sub_queries"]) == 1
        assert result["sub_queries"][0]["query"] == "What is RAG?"
    
    @pytest.mark.asyncio
    async def test_decompose_complex_query(self):
        """Test decomposing a complex query."""
        state = create_initial_state("Compare BM25 and BERT for search ranking")
        state["query_type"] = "complex"
        
        with patch('src.agent.nodes.query_decomposer.ChatOpenAI') as mock_llm:
            mock_instance = MagicMock()
            mock_instance.ainvoke = AsyncMock(return_value=MagicMock(
                content='{"sub_queries": [{"query": "What is BM25?", "source_hint": "academic"}, {"query": "How does BERT work for search?", "source_hint": "academic"}], "reasoning": "Split into components"}'
            ))
            mock_llm.return_value = mock_instance
            
            result = await decompose_query_node(state)
            
            assert len(result["sub_queries"]) == 2


class TestSubQuery:
    """Tests for SubQuery type."""
    
    def test_subquery_creation(self):
        """Test creating a SubQuery."""
        sq = SubQuery(
            query="What is RAG?",
            source_hint="web",
            sources=["web", "documents"],
        )
        
        assert sq["query"] == "What is RAG?"
        assert sq["source_hint"] == "web"
        assert "web" in sq["sources"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
