"""
Agent State Schema for Agentic Search.

This module defines the state that flows through the LangGraph workflow.
The state is a TypedDict that tracks all information needed across nodes.
"""

from typing import Annotated, Any, List, Optional, TypedDict

from langgraph.graph.message import add_messages


class SubQuery(TypedDict):
    """A sub-query with routing information."""
    query: str
    source_hint: str  # web, documents, academic, any
    sources: List[str]  # Assigned sources after routing


class RetrievalResult(TypedDict):
    """A single retrieval result from any source."""
    content: str
    source_type: str  # web, documents, academic
    title: Optional[str]
    url: Optional[str]
    score: Optional[float]
    metadata: dict


class Citation(TypedDict):
    """A citation for the final answer."""
    number: int
    title: str
    url: Optional[str]
    source_type: str
    excerpt: str


class QualityScores(TypedDict):
    """Quality scores from self-reflection."""
    relevance: float
    completeness: float
    accuracy: float
    citation_quality: float
    clarity: float


class SearchState(TypedDict):
    """
    The state schema for the Agentic Search workflow.
    
    This state flows through all nodes in the LangGraph and accumulates
    information as the search progresses.
    """
    
    # === Input ===
    original_query: str
    messages: Annotated[list, add_messages]
    
    # === Query Analysis ===
    query_type: str  # simple, complex, multi_hop
    topics: List[str]
    requires_web_search: bool
    requires_academic: bool
    requires_documents: bool
    
    # === Query Decomposition ===
    sub_queries: List[SubQuery]
    
    # === Retrieval Results ===
    web_results: List[RetrievalResult]
    vector_results: List[RetrievalResult]
    arxiv_results: List[RetrievalResult]
    all_results: List[RetrievalResult]  # Aggregated and ranked
    
    # === Synthesis ===
    draft_answer: str
    final_answer: str
    citations: List[Citation]
    
    # === Reflection ===
    quality_scores: QualityScores
    overall_quality: float
    reflection_feedback: str
    missing_aspects: List[str]
    iteration_count: int
    
    # === Control Flow ===
    should_continue: bool
    error: Optional[str]


def create_initial_state(query: str) -> SearchState:
    """Create an initial state for a new search query."""
    return SearchState(
        # Input
        original_query=query,
        messages=[],
        
        # Query Analysis
        query_type="",
        topics=[],
        requires_web_search=False,
        requires_academic=False,
        requires_documents=False,
        
        # Query Decomposition
        sub_queries=[],
        
        # Retrieval Results
        web_results=[],
        vector_results=[],
        arxiv_results=[],
        all_results=[],
        
        # Synthesis
        draft_answer="",
        final_answer="",
        citations=[],
        
        # Reflection
        quality_scores={
            "relevance": 0.0,
            "completeness": 0.0,
            "accuracy": 0.0,
            "citation_quality": 0.0,
            "clarity": 0.0,
        },
        overall_quality=0.0,
        reflection_feedback="",
        missing_aspects=[],
        iteration_count=0,
        
        # Control Flow
        should_continue=True,
        error=None,
    )
