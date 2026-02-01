"""
arXiv Academic Paper Retriever.

This module provides academic paper search using the arXiv API.
It's useful for retrieving research papers and technical content.
"""

import logging
from typing import List, Optional

import arxiv

from src.agent.state import RetrievalResult

logger = logging.getLogger(__name__)


class ArxivRetriever:
    """
    Academic paper retriever using arXiv API.
    
    Features:
    - Search by query
    - Access to paper abstracts and metadata
    - No API key required
    """
    
    def __init__(self, max_results: int = 5):
        """
        Initialize the arXiv retriever.
        
        Args:
            max_results: Maximum number of results to return
        """
        self.max_results = max_results
        self._client = None
    
    @property
    def client(self) -> arxiv.Client:
        """Lazy initialization of arXiv client."""
        if self._client is None:
            self._client = arxiv.Client()
        return self._client
    
    async def search(self, query: str) -> List[RetrievalResult]:
        """
        Search arXiv for academic papers.
        
        Args:
            query: Search query
            
        Returns:
            List of retrieval results
        """
        logger.info(f"arXiv search: {query[:50]}...")
        
        try:
            # Create search query
            search = arxiv.Search(
                query=query,
                max_results=self.max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            
            results = []
            
            # Fetch results (synchronous API)
            for paper in self.client.results(search):
                # Combine title and abstract for content
                content = f"{paper.title}\n\nAbstract: {paper.summary}"
                
                # Get authors
                authors = ", ".join([author.name for author in paper.authors[:3]])
                if len(paper.authors) > 3:
                    authors += f" et al. ({len(paper.authors)} authors)"
                
                results.append(
                    RetrievalResult(
                        content=content,
                        source_type="academic",
                        title=paper.title,
                        url=paper.entry_id,
                        score=None,  # arXiv doesn't provide relevance scores
                        metadata={
                            "authors": authors,
                            "published": paper.published.isoformat() if paper.published else None,
                            "updated": paper.updated.isoformat() if paper.updated else None,
                            "categories": paper.categories,
                            "pdf_url": paper.pdf_url,
                            "arxiv_id": paper.entry_id.split("/")[-1],
                        },
                    )
                )
            
            logger.info(f"arXiv returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"arXiv search failed: {e}")
            return []
    
    async def get_paper(self, arxiv_id: str) -> Optional[RetrievalResult]:
        """
        Get a specific paper by arXiv ID.
        
        Args:
            arxiv_id: The arXiv paper ID (e.g., "2301.00001")
            
        Returns:
            RetrievalResult for the paper, or None if not found
        """
        logger.info(f"Fetching arXiv paper: {arxiv_id}")
        
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            
            for paper in self.client.results(search):
                content = f"{paper.title}\n\nAbstract: {paper.summary}"
                authors = ", ".join([author.name for author in paper.authors])
                
                return RetrievalResult(
                    content=content,
                    source_type="academic",
                    title=paper.title,
                    url=paper.entry_id,
                    score=None,
                    metadata={
                        "authors": authors,
                        "published": paper.published.isoformat() if paper.published else None,
                        "categories": paper.categories,
                        "pdf_url": paper.pdf_url,
                    },
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to fetch paper {arxiv_id}: {e}")
            return None


# Global instance
_arxiv_retriever = None


def get_arxiv_retriever() -> ArxivRetriever:
    """Get or create the global arXiv retriever instance."""
    global _arxiv_retriever
    if _arxiv_retriever is None:
        _arxiv_retriever = ArxivRetriever()
    return _arxiv_retriever
