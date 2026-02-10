"""
Query Decomposer Node.

This node breaks down complex queries into simpler sub-queries that can be
answered independently. This is a key technique for handling multi-hop
reasoning and complex information needs.
"""

import json
import logging
from typing import Any

from src.agent.state import SearchState, SubQuery
from src.utils.llm import get_llm
from src.utils.prompts import get_prompt

logger = logging.getLogger(__name__)


async def decompose_query_node(state: SearchState) -> dict[str, Any]:
    """
    Decompose a complex query into simpler sub-queries.
    
    This node:
    1. Takes the analyzed query type into account
    2. Breaks complex queries into 2-4 focused sub-queries
    3. Provides source hints for each sub-query
    4. Uses feedback from reflection to improve decomposition on retry
    
    Args:
        state: Current search state with query analysis
        
    Returns:
        Updated state with sub_queries list
    """
    query = state["original_query"]
    query_type = state.get("query_type", "simple")
    feedback = state.get("reflection_feedback", "")
    
    logger.info(f"Decomposing query (type: {query_type})")
    
    # For simple queries, just use the original query
    if query_type == "simple" and not feedback:
        return {
            "sub_queries": [
                SubQuery(
                    query=query,
                    source_hint="any",
                    sources=[],
                )
            ]
        }
    
    try:
        # Initialize LLM (uses Groq or OpenAI based on config)
        llm = get_llm(temperature=0.3)  # Slight creativity for decomposition
        
        # Get decomposition prompt
        prompt = get_prompt(
            "query_decomposer",
            query=query,
            query_type=query_type,
            feedback=feedback or "None",
        )
        
        # Get LLM response
        response = await llm.ainvoke(prompt)
        
        # Parse JSON response
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        decomposition = json.loads(content.strip())
        
        # Convert to SubQuery objects
        sub_queries = []
        for sq in decomposition.get("sub_queries", []):
            sub_queries.append(
                SubQuery(
                    query=sq.get("query", query),
                    source_hint=sq.get("source_hint", "any"),
                    sources=[],
                )
            )
        
        # Ensure at least one sub-query
        if not sub_queries:
            sub_queries = [
                SubQuery(query=query, source_hint="any", sources=[])
            ]
        
        logger.info(f"Decomposed into {len(sub_queries)} sub-queries")
        for i, sq in enumerate(sub_queries):
            logger.debug(f"  {i+1}. {sq['query'][:50]}... (hint: {sq['source_hint']})")
        
        return {"sub_queries": sub_queries}
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse decomposition: {e}")
        return {
            "sub_queries": [
                SubQuery(query=query, source_hint="any", sources=[])
            ]
        }
    except Exception as e:
        logger.error(f"Query decomposition failed: {e}")
        return {
            "sub_queries": [
                SubQuery(query=query, source_hint="any", sources=[])
            ],
            "error": str(e),
        }
