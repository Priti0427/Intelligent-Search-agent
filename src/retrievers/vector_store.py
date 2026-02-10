"""
ChromaDB Vector Store Retriever.

This module provides vector-based semantic search using ChromaDB.
It supports document ingestion, embedding generation, and similarity search.
Supports both HuggingFace (free, local) and OpenAI (paid) embeddings.
"""

import logging
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.agent.state import RetrievalResult
from src.utils.config import get_settings

logger = logging.getLogger(__name__)


def get_embeddings():
    """
    Get the configured embeddings model.
    
    Returns HuggingFace embeddings (free, local) or OpenAI embeddings (paid)
    based on the EMBEDDING_PROVIDER setting.
    """
    settings = get_settings()
    
    if settings.embedding_provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        
        logger.info(f"Using HuggingFace embeddings: {settings.huggingface_embedding_model}")
        return HuggingFaceEmbeddings(
            model_name=settings.huggingface_embedding_model,
            model_kwargs={"device": "cpu"},  # Use CPU for compatibility
            encode_kwargs={"normalize_embeddings": True},
        )
    else:
        from langchain_openai import OpenAIEmbeddings
        
        logger.info(f"Using OpenAI embeddings: {settings.openai_embedding_model}")
        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )


class VectorStoreRetriever:
    """
    Vector store retriever using ChromaDB.
    
    Features:
    - Persistent storage
    - HuggingFace or OpenAI embeddings (configurable)
    - Cosine similarity search
    - Metadata filtering
    """
    
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: Optional[str] = None,
        max_results: int = 5,
    ):
        """
        Initialize the vector store retriever.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory for persistent storage
            max_results: Maximum number of results to return
        """
        settings = get_settings()
        self.collection_name = collection_name
        self.persist_directory = persist_directory or settings.chroma_persist_directory
        self.max_results = max_results
        
        self._client = None
        self._collection = None
        self._embeddings = None
    
    @property
    def embeddings(self):
        """Lazy initialization of embeddings model."""
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        return self._embeddings
    
    @property
    def client(self) -> chromadb.Client:
        """Lazy initialization of ChromaDB client."""
        if self._client is None:
            # Ensure directory exists
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            
            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                ),
            )
        return self._client
    
    @property
    def collection(self):
        """Get or create the collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection
    
    async def search(self, query: str) -> List[RetrievalResult]:
        """
        Search the vector store for similar documents.
        
        Args:
            query: Search query
            
        Returns:
            List of retrieval results
        """
        logger.info(f"Vector search: {query[:50]}...")
        
        try:
            # Check if collection has documents
            if self.collection.count() == 0:
                logger.warning("Vector store is empty")
                return []
            
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Search collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(self.max_results, self.collection.count()),
                include=["documents", "metadatas", "distances"],
            )
            
            # Convert to RetrievalResult format
            retrieval_results = []
            
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            
            for doc, meta, dist in zip(documents, metadatas, distances):
                # Convert distance to similarity score (cosine distance to similarity)
                score = 1 - dist if dist is not None else 0.5
                
                retrieval_results.append(
                    RetrievalResult(
                        content=doc,
                        source_type="documents",
                        title=meta.get("title", "Document"),
                        url=meta.get("source", ""),
                        score=score,
                        metadata=meta,
                    )
                )
            
            logger.info(f"Vector search returned {len(retrieval_results)} results")
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> int:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document texts
            metadatas: Optional metadata for each document
            ids: Optional IDs for each document
            
        Returns:
            Number of documents added
        """
        if not documents:
            return 0
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        try:
            # Generate IDs if not provided
            if ids is None:
                existing_count = self.collection.count()
                ids = [f"doc_{existing_count + i}" for i in range(len(documents))]
            
            # Generate default metadata if not provided
            if metadatas is None:
                metadatas = [{"title": f"Document {i}"} for i in range(len(documents))]
            
            # Generate embeddings
            embeddings = self.embeddings.embed_documents(documents)
            
            # Add to collection
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )
            
            logger.info(f"Successfully added {len(documents)} documents")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return 0
    
    def get_document_count(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()
    
    def clear(self):
        """Clear all documents from the collection."""
        self.client.delete_collection(self.collection_name)
        self._collection = None
        logger.info("Vector store cleared")


# Global instance
_vector_retriever = None


def get_vector_retriever() -> VectorStoreRetriever:
    """Get or create the global vector store retriever instance."""
    global _vector_retriever
    if _vector_retriever is None:
        _vector_retriever = VectorStoreRetriever()
    return _vector_retriever
