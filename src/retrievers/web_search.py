"""
Tavily Web Search Retriever.

This module provides web search capabilities using the Tavily API.
Tavily is optimized for AI applications and provides clean, relevant results.
"""

import logging
from typing import List, Optional

from tavily import TavilyClient

from src.agent.state import RetrievalResult
from src.utils.config import get_settings

logger = logging.getLogger(__name__)


class TavilyRetriever:
    """
    Web search retriever using Tavily API.
    
    Tavily provides:
    - AI-optimized search results
    - Clean content extraction
    - Relevance scoring
    """
    
    def __init__(self, api_key: Optional[str] = None, max_results: int = 5):
        """
        Initialize the Tavily retriever.
        
        Args:
            api_key: Tavily API key (uses settings if not provided)
            max_results: Maximum number of results to return
        """
        settings = get_settings()
        self.api_key = api_key or settings.tavily_api_key
        self.max_results = max_results
        self._client = None
    
    @property
    def client(self) -> TavilyClient:
        """Lazy initialization of Tavily client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError("Tavily API key not configured")
            self._client = TavilyClient(api_key=self.api_key)
        return self._client
    
    async def search(self, query: str) -> List[RetrievalResult]:
        """
        Search the web using Tavily.
        
        Args:
            query: Search query
            
        Returns:
            List of retrieval results
        """
        logger.info(f"Tavily search: {query[:50]}...")
        
        try:
            # Tavily search (synchronous API, but we wrap it)
            response = self.client.search(
                query=query,
                max_results=self.max_results,
                include_answer=False,
                include_raw_content=False,
            )
            
            results = []
            for item in response.get("results", []):
                results.append(
                    RetrievalResult(
                        content=item.get("content", ""),
                        source_type="web",
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        score=item.get("score", 0.0),
                        metadata={
                            "published_date": item.get("published_date"),
                        },
                    )
                )
            
            logger.info(f"Tavily returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return []
    
    def search_sync(self, query: str) -> List[RetrievalResult]:
        """Synchronous version of search."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.search(query))


# Global instance
_tavily_retriever = None


def get_tavily_retriever() -> TavilyRetriever:
    """Get or create the global Tavily retriever instance."""
    global _tavily_retriever
    if _tavily_retriever is None:
        _tavily_retriever = TavilyRetriever()
    return _tavily_retriever
