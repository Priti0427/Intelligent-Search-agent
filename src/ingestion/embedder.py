"""
Document Embedder.

This module handles the full ingestion pipeline: loading, chunking,
embedding, and storing documents in the vector store.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

from src.ingestion.chunker import Chunk, TextChunker, chunk_documents
from src.ingestion.document_loader import Document, DocumentLoader, load_documents
from src.retrievers.vector_store import VectorStoreRetriever, get_vector_retriever

logger = logging.getLogger(__name__)


class DocumentEmbedder:
    """
    Full document ingestion pipeline.
    
    Pipeline:
    1. Load documents from files
    2. Chunk documents into smaller pieces
    3. Generate embeddings
    4. Store in vector database
    """
    
    def __init__(
        self,
        chunk_strategy: str = "fixed",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        vector_store: Optional[VectorStoreRetriever] = None,
    ):
        """
        Initialize the embedder.
        
        Args:
            chunk_strategy: Chunking strategy to use
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks
            vector_store: Vector store to use (creates default if not provided)
        """
        self.loader = DocumentLoader()
        self.chunker = TextChunker(
            strategy=chunk_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.vector_store = vector_store or get_vector_retriever()
    
    async def ingest_file(self, file_path: Union[str, Path]) -> int:
        """
        Ingest a single file into the vector store.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Number of chunks ingested
        """
        logger.info(f"Ingesting file: {file_path}")
        
        # Load document
        documents = self.loader.load_file(file_path)
        if not documents:
            logger.warning(f"No content loaded from {file_path}")
            return 0
        
        # Chunk documents
        chunks = []
        for doc in documents:
            doc_chunks = self.chunker.chunk_document(doc)
            chunks.extend(doc_chunks)
        
        if not chunks:
            logger.warning(f"No chunks created from {file_path}")
            return 0
        
        # Add to vector store
        texts = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [f"{Path(file_path).stem}_{i}" for i in range(len(chunks))]
        
        count = await self.vector_store.add_documents(texts, metadatas, ids)
        
        logger.info(f"Ingested {count} chunks from {file_path}")
        return count
    
    async def ingest_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
    ) -> int:
        """
        Ingest all documents from a directory.
        
        Args:
            directory: Directory path
            recursive: Whether to search subdirectories
            
        Returns:
            Total number of chunks ingested
        """
        logger.info(f"Ingesting directory: {directory}")
        
        # Load all documents
        documents = self.loader.load_directory(directory, recursive=recursive)
        if not documents:
            logger.warning(f"No documents found in {directory}")
            return 0
        
        # Chunk all documents
        chunks = []
        for doc in documents:
            doc_chunks = self.chunker.chunk_document(doc)
            chunks.extend(doc_chunks)
        
        if not chunks:
            logger.warning(f"No chunks created from {directory}")
            return 0
        
        # Add to vector store
        texts = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [f"doc_{i}" for i in range(len(chunks))]
        
        count = await self.vector_store.add_documents(texts, metadatas, ids)
        
        logger.info(f"Ingested {count} chunks from {directory}")
        return count
    
    async def ingest_text(
        self,
        text: str,
        title: str = "Manual Entry",
        source: str = "user_input",
    ) -> int:
        """
        Ingest raw text into the vector store.
        
        Args:
            text: Text content to ingest
            title: Title for the document
            source: Source identifier
            
        Returns:
            Number of chunks ingested
        """
        logger.info(f"Ingesting text: {title}")
        
        # Create document
        document = Document(
            content=text,
            metadata={
                "title": title,
                "source": source,
                "file_type": "text",
            },
        )
        
        # Chunk document
        chunks = self.chunker.chunk_document(document)
        
        if not chunks:
            logger.warning("No chunks created from text")
            return 0
        
        # Add to vector store
        texts = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [f"{source}_{i}" for i in range(len(chunks))]
        
        count = await self.vector_store.add_documents(texts, metadatas, ids)
        
        logger.info(f"Ingested {count} chunks from text")
        return count
    
    def get_stats(self) -> dict:
        """Get statistics about the vector store."""
        return {
            "document_count": self.vector_store.get_document_count(),
            "chunk_strategy": self.chunker.strategy,
            "chunk_size": self.chunker.chunk_size,
            "chunk_overlap": self.chunker.chunk_overlap,
        }


async def ingest_documents(
    path: Union[str, Path],
    chunk_strategy: str = "fixed",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> int:
    """
    Convenience function to ingest documents from a path.
    
    Args:
        path: File or directory path
        chunk_strategy: Chunking strategy
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        
    Returns:
        Number of chunks ingested
    """
    embedder = DocumentEmbedder(
        chunk_strategy=chunk_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    path = Path(path)
    
    if path.is_file():
        return await embedder.ingest_file(path)
    elif path.is_dir():
        return await embedder.ingest_directory(path)
    else:
        logger.error(f"Path not found: {path}")
        return 0
