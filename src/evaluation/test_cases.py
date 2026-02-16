"""
Test Cases for Evaluation.

This module defines test queries with ground truth relevance judgments
for evaluating the Agentic Search system.

Each test case includes:
- query: The search query
- information_need: Description of what the user is looking for
- relevant_keywords: Keywords that indicate relevance
- relevant_sources: Expected source types (web, academic, documents)
- expected_topics: Topics that should be covered in results
- relevance_criteria: How to judge if a result is relevant

Note: Since we're using dynamic retrieval (web, arXiv, vector store),
we use keyword-based relevance matching rather than fixed document IDs.
"""

from typing import List, Dict, Any, Set
from dataclasses import dataclass, field


@dataclass
class TestCase:
    """A test case for evaluation."""
    
    query: str
    information_need: str
    relevant_keywords: List[str]
    expected_topics: List[str]
    relevant_sources: List[str] = field(default_factory=lambda: ["web", "academic", "documents"])
    min_expected_results: int = 3
    
    def is_result_relevant(self, result: Dict[str, Any]) -> bool:
        """
        Determine if a result is relevant based on keyword matching.
        
        Args:
            result: A search result with 'title', 'content', or 'excerpt'
            
        Returns:
            True if the result is relevant
        """
        # Combine all text fields for matching
        text_fields = [
            result.get("title", ""),
            result.get("content", ""),
            result.get("excerpt", ""),
            result.get("snippet", ""),
        ]
        combined_text = " ".join(text_fields).lower()
        
        # Check if any relevant keyword appears in the result
        keyword_matches = sum(
            1 for kw in self.relevant_keywords 
            if kw.lower() in combined_text
        )
        
        # Result is relevant if it matches at least 2 keywords
        # or matches a highly specific keyword
        return keyword_matches >= 2
    
    def get_relevance_score(self, result: Dict[str, Any]) -> float:
        """
        Get a graded relevance score (0-3) for nDCG calculation.
        
        0 = Not relevant
        1 = Marginally relevant
        2 = Relevant
        3 = Highly relevant
        
        Args:
            result: A search result
            
        Returns:
            Relevance score 0-3
        """
        text_fields = [
            result.get("title", ""),
            result.get("content", ""),
            result.get("excerpt", ""),
            result.get("snippet", ""),
        ]
        combined_text = " ".join(text_fields).lower()
        
        # Count keyword matches
        keyword_matches = sum(
            1 for kw in self.relevant_keywords 
            if kw.lower() in combined_text
        )
        
        # Count topic coverage
        topic_matches = sum(
            1 for topic in self.expected_topics
            if topic.lower() in combined_text
        )
        
        total_matches = keyword_matches + topic_matches
        
        if total_matches >= 5:
            return 3.0  # Highly relevant
        elif total_matches >= 3:
            return 2.0  # Relevant
        elif total_matches >= 1:
            return 1.0  # Marginally relevant
        else:
            return 0.0  # Not relevant


# Define test cases for the three example queries in the UI
TEST_CASES: List[TestCase] = [
    TestCase(
        query="What is RAG in AI?",
        information_need="User wants to understand Retrieval-Augmented Generation, "
                        "including what it is, how it works, and why it's used.",
        relevant_keywords=[
            "retrieval-augmented generation",
            "RAG",
            "retrieval",
            "augmented",
            "generation",
            "LLM",
            "large language model",
            "knowledge base",
            "external knowledge",
            "hallucination",
            "grounding",
            "context",
            "embedding",
            "vector",
        ],
        expected_topics=[
            "retrieval mechanism",
            "language model",
            "knowledge retrieval",
            "document retrieval",
            "answer generation",
        ],
        relevant_sources=["web", "academic"],
        min_expected_results=5,
    ),
    
    TestCase(
        query="Compare BM25 and dense retrieval for question answering",
        information_need="User wants to understand the differences between BM25 "
                        "(sparse/lexical retrieval) and dense retrieval methods, "
                        "specifically for question answering tasks.",
        relevant_keywords=[
            "BM25",
            "dense retrieval",
            "sparse retrieval",
            "lexical",
            "semantic",
            "embedding",
            "TF-IDF",
            "term frequency",
            "question answering",
            "QA",
            "neural",
            "BERT",
            "bi-encoder",
            "cross-encoder",
            "vector search",
            "inverted index",
        ],
        expected_topics=[
            "retrieval comparison",
            "lexical matching",
            "semantic matching",
            "ranking",
            "relevance",
        ],
        relevant_sources=["web", "academic"],
        min_expected_results=5,
    ),
    
    TestCase(
        query="How does BERT improve search ranking compared to TF-IDF?",
        information_need="User wants to understand how BERT-based neural ranking "
                        "improves upon traditional TF-IDF methods for search.",
        relevant_keywords=[
            "BERT",
            "TF-IDF",
            "search ranking",
            "neural ranking",
            "transformer",
            "contextual",
            "semantic",
            "relevance",
            "information retrieval",
            "learning to rank",
            "pre-training",
            "fine-tuning",
            "passage ranking",
            "document ranking",
        ],
        expected_topics=[
            "neural models",
            "traditional IR",
            "ranking improvement",
            "semantic understanding",
            "contextual embeddings",
        ],
        relevant_sources=["web", "academic"],
        min_expected_results=5,
    ),
    
    # Additional test cases for comprehensive evaluation
    TestCase(
        query="What are the main components of a search engine?",
        information_need="User wants to learn about the architecture and components "
                        "of a search engine system.",
        relevant_keywords=[
            "search engine",
            "crawler",
            "indexer",
            "index",
            "query processor",
            "ranking",
            "inverted index",
            "web crawler",
            "document processing",
            "tokenization",
            "stemming",
            "relevance",
        ],
        expected_topics=[
            "crawling",
            "indexing",
            "query processing",
            "ranking algorithms",
            "search architecture",
        ],
        relevant_sources=["web", "academic"],
        min_expected_results=5,
    ),
    
    TestCase(
        query="Explain vector embeddings for semantic search",
        information_need="User wants to understand how vector embeddings enable "
                        "semantic search capabilities.",
        relevant_keywords=[
            "vector",
            "embedding",
            "semantic search",
            "dense vector",
            "similarity",
            "cosine similarity",
            "neural network",
            "representation",
            "sentence embedding",
            "word embedding",
            "transformer",
            "BERT",
            "sentence-transformers",
        ],
        expected_topics=[
            "vector representation",
            "similarity search",
            "neural embeddings",
            "semantic matching",
        ],
        relevant_sources=["web", "academic", "documents"],
        min_expected_results=5,
    ),
    
    TestCase(
        query="What is query decomposition in information retrieval?",
        information_need="User wants to understand how complex queries are broken "
                        "down into simpler sub-queries.",
        relevant_keywords=[
            "query decomposition",
            "sub-query",
            "complex query",
            "query understanding",
            "query analysis",
            "multi-hop",
            "question decomposition",
            "query rewriting",
            "query expansion",
        ],
        expected_topics=[
            "query processing",
            "complex questions",
            "sub-questions",
            "query understanding",
        ],
        relevant_sources=["web", "academic"],
        min_expected_results=3,
    ),
]


def get_test_cases() -> List[TestCase]:
    """Get all test cases for evaluation."""
    return TEST_CASES


def get_test_case_by_query(query: str) -> TestCase:
    """Get a specific test case by query string."""
    for tc in TEST_CASES:
        if tc.query.lower() == query.lower():
            return tc
    raise ValueError(f"No test case found for query: {query}")


# Summary of test cases for documentation
TEST_CASE_SUMMARY = """
## Evaluation Test Cases

| # | Query | Information Need | Expected Sources |
|---|-------|------------------|------------------|
| 1 | What is RAG in AI? | Understanding RAG concept | Web, Academic |
| 2 | Compare BM25 and dense retrieval | Comparison of retrieval methods | Web, Academic |
| 3 | How does BERT improve search ranking? | Neural vs traditional ranking | Web, Academic |
| 4 | Main components of a search engine | Search engine architecture | Web, Academic |
| 5 | Vector embeddings for semantic search | Understanding embeddings | Web, Academic, Docs |
| 6 | Query decomposition in IR | Query processing techniques | Web, Academic |

Each test case includes:
- Relevant keywords for matching
- Expected topics to be covered
- Minimum expected results
- Relevance scoring criteria (0-3 scale)
"""
