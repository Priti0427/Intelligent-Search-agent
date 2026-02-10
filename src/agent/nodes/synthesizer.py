"""
Answer Synthesizer Node.

This node synthesizes a comprehensive answer from retrieved results,
including proper citations and source attribution.
"""

import json
import logging
from typing import Any, List

from src.agent.state import Citation, RetrievalResult, SearchState
from src.utils.llm import get_llm
from src.utils.prompts import get_prompt

logger = logging.getLogger(__name__)


def format_context(results: List[RetrievalResult]) -> str:
    """Format retrieval results as context for the synthesizer."""
    if not results:
        return "No results found."
    
    context_parts = []
    for i, result in enumerate(results, 1):
        source_type = result.get("source_type", "unknown")
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        content = result.get("content", "")[:1000]  # Limit content length
        
        context_parts.append(
            f"[Source {i}] ({source_type})\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content: {content}\n"
        )
    
    return "\n---\n".join(context_parts)


def extract_citations(results: List[RetrievalResult]) -> List[Citation]:
    """Extract citation information from results."""
    citations = []
    for i, result in enumerate(results, 1):
        citations.append(
            Citation(
                number=i,
                title=result.get("title", "Untitled"),
                url=result.get("url"),
                source_type=result.get("source_type", "unknown"),
                excerpt=result.get("content", "")[:200],
            )
        )
    return citations


async def synthesize_answer_node(state: SearchState) -> dict[str, Any]:
    """
    Synthesize a comprehensive answer from retrieved results.
    
    This node:
    1. Formats all retrieved results as context
    2. Uses LLM to synthesize a coherent answer
    3. Ensures proper citation of sources
    4. Handles cases with no or limited results
    
    Args:
        state: Current search state with retrieval results
        
    Returns:
        Updated state with draft_answer and citations
    """
    query = state["original_query"]
    all_results = state.get("all_results", [])
    
    logger.info(f"Synthesizing answer from {len(all_results)} results")
    
    # Handle no results case
    if not all_results:
        return {
            "draft_answer": (
                "I couldn't find relevant information to answer your query. "
                "Please try rephrasing your question or providing more context."
            ),
            "citations": [],
        }
    
    try:
        # Initialize LLM (uses Groq or OpenAI based on config)
        llm = get_llm(temperature=0.3)
        
        # Format context from results
        context = format_context(all_results)
        
        # Get synthesis prompt
        prompt = get_prompt("synthesizer", query=query, context=context)
        
        # Get LLM response
        response = await llm.ainvoke(prompt)
        answer = response.content
        
        # Extract citations from results
        citations = extract_citations(all_results)
        
        logger.info(f"Generated answer with {len(citations)} citations")
        
        return {
            "draft_answer": answer,
            "citations": citations,
        }
        
    except Exception as e:
        logger.error(f"Answer synthesis failed: {e}")
        
        # Fallback: simple concatenation of results
        fallback_answer = f"Based on the search results for '{query}':\n\n"
        for i, result in enumerate(all_results[:5], 1):
            fallback_answer += f"[{i}] {result.get('content', '')[:300]}...\n\n"
        
        return {
            "draft_answer": fallback_answer,
            "citations": extract_citations(all_results[:5]),
            "error": str(e),
        }
