"""
Retriever Node.

This node performs parallel retrieval from multiple data sources based on
the routing decisions. It aggregates and ranks results from all sources.
"""

import asyncio
import logging
from typing import Any, List

from src.agent.state import RetrievalResult, SearchState
from src.retrievers import get_arxiv_retriever, get_tavily_retriever, get_vector_retriever

logger = logging.getLogger(__name__)


async def retrieve_from_web(query: str) -> List[RetrievalResult]:
    """Retrieve results from web search."""
    try:
        retriever = get_tavily_retriever()
        results = await retriever.search(query)
        return results
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return []


async def retrieve_from_vectors(query: str) -> List[RetrievalResult]:
    """Retrieve results from vector store."""
    try:
        retriever = get_vector_retriever()
        results = await retriever.search(query)
        return results
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []


async def retrieve_from_arxiv(query: str) -> List[RetrievalResult]:
    """Retrieve results from arXiv."""
    try:
        retriever = get_arxiv_retriever()
        results = await retriever.search(query)
        return results
    except Exception as e:
        logger.error(f"arXiv search failed: {e}")
        return []


async def retrieve_parallel_node(state: SearchState) -> dict[str, Any]:
    """
    Perform parallel retrieval from multiple sources.
    
    This node:
    1. Collects all unique source-query pairs from routed sub-queries
    2. Executes retrievals in parallel for efficiency
    3. Aggregates results by source type
    4. Ranks and deduplicates results
    
    Args:
        state: Current search state with routed sub_queries
        
    Returns:
        Updated state with retrieval results
    """
    sub_queries = state.get("sub_queries", [])
    
    if not sub_queries:
        logger.warning("No sub-queries for retrieval")
        return {
            "web_results": [],
            "vector_results": [],
            "arxiv_results": [],
            "all_results": [],
        }
    
    logger.info(f"Starting parallel retrieval for {len(sub_queries)} sub-queries")
    
    # Collect retrieval tasks
    web_tasks = []
    vector_tasks = []
    arxiv_tasks = []
    
    for sq in sub_queries:
        query = sq["query"]
        sources = sq.get("sources", ["web", "documents"])
        
        if "web" in sources:
            web_tasks.append(retrieve_from_web(query))
        if "documents" in sources:
            vector_tasks.append(retrieve_from_vectors(query))
        if "academic" in sources:
            arxiv_tasks.append(retrieve_from_arxiv(query))
    
    # Execute all retrievals in parallel
    all_tasks = []
    task_types = []
    
    for task in web_tasks:
        all_tasks.append(task)
        task_types.append("web")
    for task in vector_tasks:
        all_tasks.append(task)
        task_types.append("vector")
    for task in arxiv_tasks:
        all_tasks.append(task)
        task_types.append("arxiv")
    
    logger.info(f"Executing {len(all_tasks)} retrieval tasks in parallel")
    
    # Gather results
    results = await asyncio.gather(*all_tasks, return_exceptions=True)
    
    # Organize results by type
    web_results = []
    vector_results = []
    arxiv_results = []
    
    for result, task_type in zip(results, task_types):
        if isinstance(result, Exception):
            logger.error(f"Retrieval task failed: {result}")
            continue
        
        if task_type == "web":
            web_results.extend(result)
        elif task_type == "vector":
            vector_results.extend(result)
        elif task_type == "arxiv":
            arxiv_results.extend(result)
    
    # Deduplicate results (by content hash)
    def deduplicate(results: List[RetrievalResult]) -> List[RetrievalResult]:
        seen = set()
        unique = []
        for r in results:
            content_hash = hash(r["content"][:200])  # Hash first 200 chars
            if content_hash not in seen:
                seen.add(content_hash)
                unique.append(r)
        return unique
    
    web_results = deduplicate(web_results)
    vector_results = deduplicate(vector_results)
    arxiv_results = deduplicate(arxiv_results)
    
    # Aggregate all results with source attribution
    all_results = []
    for i, r in enumerate(web_results[:5]):  # Top 5 from each source
        r["source_type"] = "web"
        all_results.append(r)
    for i, r in enumerate(vector_results[:5]):
        r["source_type"] = "documents"
        all_results.append(r)
    for i, r in enumerate(arxiv_results[:5]):
        r["source_type"] = "academic"
        all_results.append(r)
    
    # Sort by score if available
    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    logger.info(
        f"Retrieved: {len(web_results)} web, {len(vector_results)} vector, "
        f"{len(arxiv_results)} arxiv results"
    )
    
    return {
        "web_results": web_results,
        "vector_results": vector_results,
        "arxiv_results": arxiv_results,
        "all_results": all_results,
    }
