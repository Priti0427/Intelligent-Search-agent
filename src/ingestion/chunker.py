"""
Text Chunker.

This module handles splitting documents into smaller chunks for embedding.
It implements various chunking strategies relevant to IR course concepts.
"""

import logging
import re
from typing import List, Optional

from src.ingestion.document_loader import Document

logger = logging.getLogger(__name__)


class Chunk:
    """A chunk of text with metadata."""
    
    def __init__(self, content: str, metadata: dict):
        self.content = content
        self.metadata = metadata
    
    def __repr__(self):
        return f"Chunk(len={len(self.content)}, source={self.metadata.get('source', 'unknown')})"


class TextChunker:
    """
    Text chunking with multiple strategies.
    
    Strategies:
    - fixed: Fixed-size chunks with overlap
    - sentence: Split on sentence boundaries
    - paragraph: Split on paragraph boundaries
    - semantic: Split on semantic boundaries (headers, sections)
    
    This demonstrates text preprocessing concepts from Week 2 of the course.
    """
    
    def __init__(
        self,
        strategy: str = "fixed",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize the chunker.
        
        Args:
            strategy: Chunking strategy (fixed, sentence, paragraph, semantic)
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
        """
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Split a document into chunks.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of chunks
        """
        if self.strategy == "fixed":
            return self._chunk_fixed(document)
        elif self.strategy == "sentence":
            return self._chunk_sentence(document)
        elif self.strategy == "paragraph":
            return self._chunk_paragraph(document)
        elif self.strategy == "semantic":
            return self._chunk_semantic(document)
        else:
            logger.warning(f"Unknown strategy: {self.strategy}, using fixed")
            return self._chunk_fixed(document)
    
    def _chunk_fixed(self, document: Document) -> List[Chunk]:
        """Fixed-size chunking with overlap."""
        text = document.content
        chunks = []
        
        start = 0
        chunk_num = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at a sentence boundary
            if end < len(text):
                # Look for sentence end near the chunk boundary
                for sep in [". ", ".\n", "! ", "? ", "\n\n"]:
                    last_sep = text.rfind(sep, start + self.chunk_size // 2, end)
                    if last_sep != -1:
                        end = last_sep + len(sep)
                        break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        metadata={
                            **document.metadata,
                            "chunk_num": chunk_num,
                            "chunk_start": start,
                            "chunk_end": end,
                            "chunking_strategy": "fixed",
                        },
                    )
                )
                chunk_num += 1
            
            start = end - self.chunk_overlap
            if start >= len(text) - self.chunk_overlap:
                break
        
        return chunks
    
    def _chunk_sentence(self, document: Document) -> List[Chunk]:
        """Sentence-based chunking."""
        text = document.content
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_num = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if current_length + len(sentence) > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        metadata={
                            **document.metadata,
                            "chunk_num": chunk_num,
                            "chunking_strategy": "sentence",
                        },
                    )
                )
                chunk_num += 1
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) > 2 else []
                current_chunk = overlap_sentences
                current_length = sum(len(s) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_length += len(sentence)
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(
                Chunk(
                    content=chunk_text,
                    metadata={
                        **document.metadata,
                        "chunk_num": chunk_num,
                        "chunking_strategy": "sentence",
                    },
                )
            )
        
        return chunks
    
    def _chunk_paragraph(self, document: Document) -> List[Chunk]:
        """Paragraph-based chunking."""
        text = document.content
        
        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_num = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if current_length + len(para) > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        metadata={
                            **document.metadata,
                            "chunk_num": chunk_num,
                            "chunking_strategy": "paragraph",
                        },
                    )
                )
                chunk_num += 1
                current_chunk = []
                current_length = 0
            
            current_chunk.append(para)
            current_length += len(para)
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(
                Chunk(
                    content=chunk_text,
                    metadata={
                        **document.metadata,
                        "chunk_num": chunk_num,
                        "chunking_strategy": "paragraph",
                    },
                )
            )
        
        return chunks
    
    def _chunk_semantic(self, document: Document) -> List[Chunk]:
        """Semantic chunking based on headers and sections."""
        text = document.content
        
        # Look for markdown-style headers or section breaks
        header_pattern = r'^(#{1,6}\s+.+|[A-Z][A-Za-z\s]+:?\s*$)'
        
        lines = text.split('\n')
        sections = []
        current_section = []
        current_header = None
        
        for line in lines:
            if re.match(header_pattern, line.strip()):
                if current_section:
                    sections.append((current_header, '\n'.join(current_section)))
                current_header = line.strip()
                current_section = []
            else:
                current_section.append(line)
        
        if current_section:
            sections.append((current_header, '\n'.join(current_section)))
        
        # If no sections found, fall back to paragraph chunking
        if len(sections) <= 1:
            return self._chunk_paragraph(document)
        
        # Create chunks from sections
        chunks = []
        chunk_num = 0
        
        for header, content in sections:
            content = content.strip()
            if not content:
                continue
            
            # If section is too large, sub-chunk it
            if len(content) > self.chunk_size:
                sub_doc = Document(content, document.metadata)
                sub_chunker = TextChunker(
                    strategy="paragraph",
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
                sub_chunks = sub_chunker.chunk_document(sub_doc)
                
                for sub_chunk in sub_chunks:
                    sub_chunk.metadata["section_header"] = header
                    sub_chunk.metadata["chunk_num"] = chunk_num
                    sub_chunk.metadata["chunking_strategy"] = "semantic"
                    chunks.append(sub_chunk)
                    chunk_num += 1
            else:
                chunk_text = f"{header}\n\n{content}" if header else content
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        metadata={
                            **document.metadata,
                            "chunk_num": chunk_num,
                            "section_header": header,
                            "chunking_strategy": "semantic",
                        },
                    )
                )
                chunk_num += 1
        
        return chunks


def chunk_documents(
    documents: List[Document],
    strategy: str = "fixed",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Chunk]:
    """
    Convenience function to chunk multiple documents.
    
    Args:
        documents: List of documents to chunk
        strategy: Chunking strategy
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of all chunks
    """
    chunker = TextChunker(
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)
    
    logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
    return all_chunks
