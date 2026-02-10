"""
Router Node.

This node routes each sub-query to the appropriate data source(s).
It uses the query analysis and source hints to make intelligent routing decisions.
"""

import json
import logging
from typing import Any

from src.agent.state import SearchState, SubQuery
from src.utils.config import get_settings
from src.utils.llm import get_llm
from src.utils.prompts import get_prompt

logger = logging.getLogger(__name__)


async def route_queries_node(state: SearchState) -> dict[str, Any]:
    """
    Route sub-queries to appropriate data sources.
    
    This node:
    1. Considers the source hints from decomposition
    2. Uses query analysis to determine source needs
    3. Assigns one or more sources to each sub-query
    
    Available sources:
    - web: Tavily web search for current information
    - documents: ChromaDB vector store for domain knowledge
    - academic: arXiv API for research papers
    
    Args:
        state: Current search state with sub_queries
        
    Returns:
        Updated state with routed sub_queries
    """
    settings = get_settings()
    sub_queries = state.get("sub_queries", [])
    requires_web = state.get("requires_web_search", True)
    requires_academic = state.get("requires_academic", False)
    requires_documents = state.get("requires_documents", True)
    
    if not sub_queries:
        logger.warning("No sub-queries to route")
        return {"sub_queries": []}
    
    logger.info(f"Routing {len(sub_queries)} sub-queries")
    
    try:
        # Initialize LLM for intelligent routing (uses Groq or OpenAI based on config)
        llm = get_llm(temperature=0)
        
        # Format sub-queries for prompt
        sq_text = "\n".join([
            f"{i+1}. {sq['query']} (hint: {sq['source_hint']})"
            for i, sq in enumerate(sub_queries)
        ])
        
        # Get routing prompt
        prompt = get_prompt("router", sub_queries=sq_text)
        
        # Get LLM response
        response = await llm.ainvoke(prompt)
        
        # Parse JSON response
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        routing = json.loads(content.strip())
        
        # Update sub-queries with assigned sources
        routed_queries = []
        routing_list = routing.get("routing", [])
        
        for i, sq in enumerate(sub_queries):
            # Find matching routing or use defaults
            sources = ["web", "documents"]  # Default sources
            
            if i < len(routing_list):
                sources = routing_list[i].get("sources", sources)
            elif sq["source_hint"] != "any":
                sources = [sq["source_hint"]]
            
            # Filter based on query analysis
            if not requires_web and "web" in sources:
                sources = [s for s in sources if s != "web"] or ["documents"]
            if not requires_academic and "academic" in sources:
                sources = [s for s in sources if s != "academic"] or ["documents"]
            
            routed_queries.append(
                SubQuery(
                    query=sq["query"],
                    source_hint=sq["source_hint"],
                    sources=sources,
                )
            )
            
            logger.info(f"  Query {i+1} -> {sources}")
        
        return {"sub_queries": routed_queries}
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse routing: {e}")
        # Default routing based on hints
        routed_queries = []
        for sq in sub_queries:
            if sq["source_hint"] == "any":
                sources = ["web", "documents"]
            else:
                sources = [sq["source_hint"]]
            routed_queries.append(
                SubQuery(
                    query=sq["query"],
                    source_hint=sq["source_hint"],
                    sources=sources,
                )
            )
        return {"sub_queries": routed_queries}
        
    except Exception as e:
        logger.error(f"Routing failed: {e}")
        # Fallback: route all to web and documents
        routed_queries = [
            SubQuery(
                query=sq["query"],
                source_hint=sq["source_hint"],
                sources=["web", "documents"],
            )
            for sq in sub_queries
        ]
        return {"sub_queries": routed_queries, "error": str(e)}
