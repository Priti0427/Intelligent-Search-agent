"""
Query Analyzer Node.

This node analyzes the incoming query to determine its type and requirements.
It identifies whether the query is simple, complex, or multi-hop, and what
data sources might be needed.
"""

import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI

from src.agent.state import SearchState
from src.utils.config import get_settings
from src.utils.prompts import get_prompt

logger = logging.getLogger(__name__)


async def analyze_query_node(state: SearchState) -> dict[str, Any]:
    """
    Analyze the user query to determine its type and requirements.
    
    This node:
    1. Classifies the query as simple, complex, or multi-hop
    2. Extracts main topics/entities
    3. Determines which data sources are needed
    
    Args:
        state: Current search state with original_query
        
    Returns:
        Updated state fields for query analysis
    """
    settings = get_settings()
    query = state["original_query"]
    
    logger.info(f"Analyzing query: {query[:100]}...")
    
    try:
        # Initialize LLM
        llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
        
        # Get analysis prompt
        prompt = get_prompt("query_analyzer", query=query)
        
        # Get LLM response
        response = await llm.ainvoke(prompt)
        
        # Parse JSON response
        content = response.content
        # Extract JSON from response (handle markdown code blocks)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        analysis = json.loads(content.strip())
        
        logger.info(f"Query type: {analysis.get('query_type', 'unknown')}")
        logger.info(f"Topics: {analysis.get('topics', [])}")
        
        return {
            "query_type": analysis.get("query_type", "simple"),
            "topics": analysis.get("topics", []),
            "requires_web_search": analysis.get("requires_web_search", True),
            "requires_academic": analysis.get("requires_academic", False),
            "requires_documents": analysis.get("requires_documents", True),
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse query analysis: {e}")
        # Default to treating as complex query requiring all sources
        return {
            "query_type": "complex",
            "topics": [],
            "requires_web_search": True,
            "requires_academic": True,
            "requires_documents": True,
        }
    except Exception as e:
        logger.error(f"Query analysis failed: {e}")
        return {
            "query_type": "simple",
            "topics": [],
            "requires_web_search": True,
            "requires_academic": False,
            "requires_documents": True,
            "error": str(e),
        }
