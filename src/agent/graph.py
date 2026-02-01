"""
LangGraph Workflow Definition for Agentic Search.

This module defines the main search agent workflow using LangGraph.
The workflow orchestrates query analysis, decomposition, routing,
retrieval, synthesis, and self-reflection.
"""

import logging
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from src.agent.nodes import (
    analyze_query_node,
    decompose_query_node,
    reflect_node,
    retrieve_parallel_node,
    route_queries_node,
    synthesize_answer_node,
)
from src.agent.state import SearchState, create_initial_state

logger = logging.getLogger(__name__)


def should_continue(state: SearchState) -> Literal["continue", "end"]:
    """
    Determine whether to continue the reflection loop or end.
    
    Args:
        state: Current search state
        
    Returns:
        "continue" to retry with feedback, "end" to finalize
    """
    if state.get("should_continue", False):
        logger.info("Continuing with reflection feedback")
        return "continue"
    else:
        logger.info("Ending search workflow")
        return "end"


def create_search_agent() -> StateGraph:
    """
    Create the Agentic Search workflow graph.
    
    The workflow follows this structure:
    
    1. analyze_query: Classify query type and requirements
    2. decompose_query: Break into sub-queries
    3. route_queries: Assign sources to each sub-query
    4. retrieve_parallel: Fetch results from all sources
    5. synthesize_answer: Generate comprehensive answer
    6. reflect: Evaluate quality and decide to continue or end
    
    The reflection node can loop back to decompose_query for improvement.
    
    Returns:
        Compiled LangGraph workflow
    """
    # Create the state graph
    workflow = StateGraph(SearchState)
    
    # Add all nodes
    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("decompose_query", decompose_query_node)
    workflow.add_node("route_queries", route_queries_node)
    workflow.add_node("retrieve_parallel", retrieve_parallel_node)
    workflow.add_node("synthesize_answer", synthesize_answer_node)
    workflow.add_node("reflect", reflect_node)
    
    # Define the workflow edges
    workflow.set_entry_point("analyze_query")
    
    # Linear flow through main pipeline
    workflow.add_edge("analyze_query", "decompose_query")
    workflow.add_edge("decompose_query", "route_queries")
    workflow.add_edge("route_queries", "retrieve_parallel")
    workflow.add_edge("retrieve_parallel", "synthesize_answer")
    workflow.add_edge("synthesize_answer", "reflect")
    
    # Conditional edge for reflection loop
    workflow.add_conditional_edges(
        "reflect",
        should_continue,
        {
            "continue": "decompose_query",  # Retry with feedback
            "end": END,
        },
    )
    
    # Compile the graph
    return workflow.compile()


# Global agent instance
_agent = None


def get_search_agent():
    """Get or create the global search agent instance."""
    global _agent
    if _agent is None:
        _agent = create_search_agent()
    return _agent


async def run_search(query: str) -> dict[str, Any]:
    """
    Run a search query through the agent workflow.
    
    Args:
        query: The user's search query
        
    Returns:
        Final state with answer and metadata
    """
    logger.info(f"Starting search for: {query[:100]}...")
    
    # Create initial state
    initial_state = create_initial_state(query)
    
    # Get agent and run
    agent = get_search_agent()
    
    # Execute the workflow
    final_state = await agent.ainvoke(initial_state)
    
    logger.info("Search completed")
    
    return final_state


async def run_search_stream(query: str):
    """
    Run a search query with streaming updates.
    
    Args:
        query: The user's search query
        
    Yields:
        State updates as the workflow progresses
    """
    logger.info(f"Starting streaming search for: {query[:100]}...")
    
    # Create initial state
    initial_state = create_initial_state(query)
    
    # Get agent
    agent = get_search_agent()
    
    # Stream the workflow execution
    async for event in agent.astream(initial_state):
        yield event
    
    logger.info("Streaming search completed")
