"""
Test Cases for Comprehensive Evaluation.

This module defines test queries with ground truth relevance judgments
AND expert-written reference answers for evaluating the Agentic Search system.

Each test case includes:
- query: The search query
- information_need: Description of what the user is looking for
- reference_answer: Expert-written gold-standard answer (for BLEU/ROUGE/BERTScore)
- relevant_keywords: Keywords that indicate relevance
- relevant_sources: Expected source types (web, academic, documents)
- expected_topics: Topics that should be covered in results
"""

from typing import List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class TestCase:
    """A test case for evaluation."""

    query: str
    information_need: str
    reference_answer: str
    relevant_keywords: List[str]
    expected_topics: List[str]
    reference_sources: List[str] = field(default_factory=list)
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
        text_fields = [
            result.get("title", ""),
            result.get("content", ""),
            result.get("excerpt", ""),
            result.get("snippet", ""),
        ]
        combined_text = " ".join(text_fields).lower()

        keyword_matches = sum(
            1 for kw in self.relevant_keywords
            if kw.lower() in combined_text
        )

        return keyword_matches >= 2

    def get_relevance_score(self, result: Dict[str, Any]) -> float:
        """
        Get a graded relevance score (0-3) for nDCG calculation.

        0 = Not relevant
        1 = Marginally relevant
        2 = Relevant
        3 = Highly relevant
        """
        text_fields = [
            result.get("title", ""),
            result.get("content", ""),
            result.get("excerpt", ""),
            result.get("snippet", ""),
        ]
        combined_text = " ".join(text_fields).lower()

        keyword_matches = sum(
            1 for kw in self.relevant_keywords
            if kw.lower() in combined_text
        )

        topic_matches = sum(
            1 for topic in self.expected_topics
            if topic.lower() in combined_text
        )

        total_matches = keyword_matches + topic_matches

        if total_matches >= 5:
            return 3.0
        elif total_matches >= 3:
            return 2.0
        elif total_matches >= 1:
            return 1.0
        else:
            return 0.0


TEST_CASES: List[TestCase] = [
    TestCase(
        query="How does combining BM25 with dense retrieval in a hybrid search system improve RAG pipeline accuracy compared to using either method alone?",
        information_need=(
            "User wants to understand hybrid retrieval approaches that combine "
            "sparse (BM25) and dense (neural) retrieval, specifically in the context "
            "of Retrieval-Augmented Generation pipelines, and why the combination "
            "outperforms individual methods."
        ),
        reference_answer=(
            "Hybrid search systems combine BM25 (a sparse, lexical retrieval method based on "
            "term frequency and inverse document frequency) with dense retrieval (which uses "
            "neural network embeddings to capture semantic meaning) to leverage the strengths "
            "of both approaches. BM25 excels at exact keyword matching and handles rare terms "
            "well, while dense retrieval captures semantic similarity and understands synonyms "
            "and paraphrases that lexical methods miss. In a RAG pipeline, the retrieval stage "
            "is critical because the quality of retrieved documents directly impacts the "
            "generated answer. By combining both methods, a hybrid system addresses the "
            "vocabulary mismatch problem (where relevant documents use different terms than "
            "the query) while maintaining precision on exact-match queries. Common fusion "
            "strategies include Reciprocal Rank Fusion (RRF), which merges ranked lists from "
            "both retrievers, and learned score combination. Research has shown that hybrid "
            "approaches consistently outperform either method alone, with improvements of "
            "5-15% on standard benchmarks like MS MARCO and Natural Questions. The dense "
            "component provides recall for semantically related documents, while BM25 "
            "provides precision for keyword-specific queries, making the combined system "
            "more robust across diverse query types."
        ),
        relevant_keywords=[
            "BM25", "dense retrieval", "hybrid", "sparse retrieval", "lexical",
            "semantic", "embedding", "RAG", "retrieval-augmented generation",
            "fusion", "reciprocal rank", "term frequency", "TF-IDF",
            "vocabulary mismatch", "neural", "vector search",
        ],
        expected_topics=[
            "hybrid retrieval",
            "sparse vs dense comparison",
            "RAG pipeline",
            "score fusion",
            "semantic matching",
        ],
        reference_sources=[
            "BM25 original paper (Robertson et al.)",
            "Dense Passage Retrieval (Karpukhin et al. 2020)",
            "RAG paper (Lewis et al. 2020)",
        ],
        relevant_sources=["web", "academic"],
        min_expected_results=5,
    ),

    TestCase(
        query="Compare the effectiveness of query decomposition vs query expansion for handling complex multi-faceted information needs",
        information_need=(
            "User wants a detailed comparison of two query processing strategies: "
            "decomposing complex queries into sub-queries vs expanding queries with "
            "additional terms, evaluating which is more effective for complex information needs."
        ),
        reference_answer=(
            "Query decomposition and query expansion are two complementary strategies for "
            "handling complex, multi-faceted information needs in information retrieval. "
            "Query decomposition breaks a complex query into multiple simpler sub-queries, "
            "each targeting a specific aspect of the information need. For example, the query "
            "'Compare BERT and TF-IDF for search ranking with computational tradeoffs' might "
            "be decomposed into sub-queries about BERT ranking, TF-IDF methods, and "
            "computational costs separately. This approach is particularly effective for "
            "multi-hop questions that require synthesizing information from multiple sources. "
            "Query expansion, on the other hand, enriches the original query with additional "
            "related terms to improve recall. Techniques include pseudo-relevance feedback "
            "(Rocchio algorithm), synonym expansion using thesauri or WordNet, and neural "
            "expansion using language models. The Rocchio algorithm modifies the query vector "
            "by moving it toward relevant documents and away from non-relevant ones. "
            "Query decomposition is generally more effective for complex, multi-part questions "
            "because it preserves the structure of the information need, while query expansion "
            "is better for simple queries with vocabulary mismatch issues. Modern agentic "
            "search systems often combine both: decomposing complex queries into sub-queries "
            "and then expanding each sub-query for better recall. The key tradeoff is that "
            "decomposition adds latency (multiple retrievals) while expansion can introduce "
            "topic drift if not carefully controlled."
        ),
        relevant_keywords=[
            "query decomposition", "query expansion", "sub-query", "complex query",
            "multi-faceted", "pseudo-relevance feedback", "Rocchio", "query rewriting",
            "information need", "multi-hop", "query understanding", "query analysis",
            "relevance feedback", "term expansion", "vocabulary mismatch",
        ],
        expected_topics=[
            "query processing strategies",
            "complex questions",
            "relevance feedback",
            "sub-question generation",
            "query reformulation",
        ],
        reference_sources=[
            "Rocchio algorithm (relevance feedback)",
            "Query decomposition literature",
            "Pseudo-relevance feedback methods",
        ],
        relevant_sources=["web", "academic", "documents"],
        min_expected_results=5,
    ),

    TestCase(
        query="Explain how BERT-based cross-encoders re-rank search results and why they outperform bi-encoder approaches for passage retrieval",
        information_need=(
            "User wants to understand the architectural differences between cross-encoders "
            "and bi-encoders for search re-ranking, and the quality vs efficiency tradeoffs."
        ),
        reference_answer=(
            "BERT-based cross-encoders and bi-encoders represent two fundamentally different "
            "approaches to neural passage retrieval and re-ranking. A bi-encoder encodes the "
            "query and document independently into dense vector representations, then computes "
            "similarity (typically cosine or dot product) between the two vectors. This "
            "architecture is efficient because document embeddings can be pre-computed and "
            "indexed, enabling fast approximate nearest neighbor search at scale. However, "
            "because the query and document are encoded independently, the model cannot capture "
            "fine-grained token-level interactions between them. A cross-encoder, in contrast, "
            "concatenates the query and document as a single input sequence to BERT, with a "
            "[SEP] token separating them. This allows full attention between all query and "
            "document tokens, enabling the model to capture nuanced semantic interactions like "
            "negation, comparison, and context-dependent meaning. The cross-attention mechanism "
            "produces significantly more accurate relevance scores. Research on MS MARCO and "
            "TREC benchmarks shows cross-encoders outperform bi-encoders by 5-10 MRR points. "
            "However, cross-encoders are computationally expensive because they must process "
            "each query-document pair individually at query time, making them impractical for "
            "first-stage retrieval over large collections. The standard practice is a two-stage "
            "pipeline: use a bi-encoder (or BM25) for initial candidate retrieval (top 100-1000), "
            "then re-rank with a cross-encoder for the final top-k results. This combines the "
            "efficiency of bi-encoders with the accuracy of cross-encoders."
        ),
        relevant_keywords=[
            "BERT", "cross-encoder", "bi-encoder", "re-ranking", "neural ranking",
            "passage retrieval", "transformer", "attention", "semantic",
            "dense retrieval", "cosine similarity", "relevance",
            "information retrieval", "MS MARCO", "fine-tuning",
        ],
        expected_topics=[
            "neural ranking models",
            "cross-attention mechanism",
            "two-stage retrieval",
            "efficiency vs accuracy tradeoff",
            "contextual embeddings",
        ],
        reference_sources=[
            "BERT original paper (Devlin et al. 2019)",
            "Sentence-BERT (Reimers & Gurevych 2019)",
            "Cross-encoder re-ranking literature",
        ],
        relevant_sources=["web", "academic"],
        min_expected_results=5,
    ),

    TestCase(
        query="What are the key architectural differences between Agentic RAG and standard RAG, and how does self-reflection improve answer quality?",
        information_need=(
            "User wants to understand how agentic RAG systems differ from standard RAG, "
            "particularly the role of planning, tool use, and self-reflection loops "
            "in improving answer quality."
        ),
        reference_answer=(
            "Standard RAG (Retrieval-Augmented Generation) follows a simple linear pipeline: "
            "given a query, retrieve relevant documents from a knowledge base, then pass them "
            "as context to a large language model to generate an answer. While effective, this "
            "approach has limitations: it treats all queries the same, performs a single "
            "retrieval pass, and has no mechanism to verify or improve its output. Agentic RAG "
            "extends this with autonomous decision-making capabilities inspired by AI agent "
            "architectures. Key architectural differences include: (1) Query Analysis and "
            "Planning - an agentic system first analyzes the query complexity and creates a "
            "plan, deciding whether to decompose it into sub-queries or handle it directly. "
            "(2) Dynamic Source Routing - instead of querying a single knowledge base, the "
            "agent routes different sub-queries to appropriate sources (web search, academic "
            "databases, local documents). (3) Iterative Retrieval - the agent can perform "
            "multiple rounds of retrieval if initial results are insufficient. (4) Self-Reflection "
            "- after generating a draft answer, a reflection module evaluates it on dimensions "
            "like relevance, completeness, accuracy, citation quality, and clarity. If the "
            "quality score falls below a threshold, the agent loops back to improve its query "
            "decomposition and retrieval strategy. This self-reflection mechanism is critical "
            "because it catches hallucinations, identifies missing information, and ensures "
            "proper source attribution. Research shows that self-reflection can improve answer "
            "quality by 15-25% compared to single-pass generation, with the biggest gains on "
            "complex multi-hop questions that require synthesizing information from multiple sources."
        ),
        relevant_keywords=[
            "agentic RAG", "RAG", "retrieval-augmented generation", "self-reflection",
            "agent", "planning", "query decomposition", "iterative retrieval",
            "LLM", "large language model", "answer quality", "hallucination",
            "multi-hop", "source routing", "tool use", "autonomous",
        ],
        expected_topics=[
            "agentic architecture",
            "self-reflection loop",
            "query planning",
            "iterative improvement",
            "answer quality evaluation",
        ],
        reference_sources=[
            "RAG paper (Lewis et al. 2020)",
            "ReAct (Yao et al. 2023)",
            "Self-RAG (Asai et al. 2023)",
        ],
        relevant_sources=["web", "academic"],
        min_expected_results=5,
    ),

    TestCase(
        query="How do vector embeddings and semantic search overcome the vocabulary mismatch problem that traditional TF-IDF methods face?",
        information_need=(
            "User wants to understand the vocabulary mismatch problem in traditional "
            "IR and how semantic search with vector embeddings addresses it."
        ),
        reference_answer=(
            "The vocabulary mismatch problem is a fundamental limitation of traditional "
            "lexical IR methods like TF-IDF and BM25. These methods represent documents and "
            "queries as sparse vectors in a high-dimensional term space, where matching relies "
            "on exact or near-exact term overlap. When a user searches for 'automobile safety "
            "features' but the relevant document uses 'car crash protection systems', lexical "
            "methods fail because there is no direct term overlap despite the semantic "
            "equivalence. TF-IDF specifically weights terms by their frequency in the document "
            "(TF) normalized by their rarity across the corpus (IDF), creating sparse vectors "
            "that only have non-zero values for terms that actually appear in the text. "
            "Vector embeddings solve this by mapping text into a dense, low-dimensional "
            "continuous space (typically 256-1024 dimensions) where semantically similar texts "
            "are placed close together regardless of the specific words used. Models like "
            "Sentence-BERT and all-MiniLM-L6-v2 learn these representations from large text "
            "corpora, encoding semantic relationships, synonymy, and contextual meaning into "
            "the vector space. Similarity is computed using cosine similarity or dot product "
            "between vectors. In this space, 'automobile safety features' and 'car crash "
            "protection systems' would have high cosine similarity because the model has "
            "learned they express similar concepts. This is the foundation of semantic search: "
            "instead of matching words, we match meanings. The tradeoff is that dense retrieval "
            "can miss important exact-match signals (like product codes or proper nouns) that "
            "lexical methods handle well, which is why hybrid approaches combining both are "
            "often optimal."
        ),
        relevant_keywords=[
            "vector embedding", "semantic search", "vocabulary mismatch",
            "TF-IDF", "cosine similarity", "dense vector", "sparse vector",
            "term frequency", "inverse document frequency", "BERT",
            "sentence-transformers", "neural", "word embedding",
            "representation", "similarity search",
        ],
        expected_topics=[
            "vocabulary mismatch problem",
            "dense vs sparse representations",
            "semantic similarity",
            "embedding models",
            "vector space retrieval",
        ],
        reference_sources=[
            "TF-IDF (Salton et al.)",
            "Word2Vec (Mikolov et al. 2013)",
            "Sentence-BERT (Reimers & Gurevych 2019)",
        ],
        relevant_sources=["web", "academic", "documents"],
        min_expected_results=5,
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


TEST_CASE_SUMMARY = """
## Evaluation Test Cases

| # | Query | Type | Expected Sources |
|---|-------|------|------------------|
| 1 | Hybrid BM25 + dense retrieval in RAG | Multi-hop | Web, Academic |
| 2 | Query decomposition vs query expansion | Comparative | Web, Academic, Docs |
| 3 | BERT cross-encoders vs bi-encoders | Technical | Web, Academic |
| 4 | Agentic RAG vs standard RAG + self-reflection | Synthesis | Web, Academic |
| 5 | Vector embeddings vs TF-IDF vocabulary mismatch | Applied | Web, Academic, Docs |

Each test case includes:
- Expert reference answer (200-400 words) for BERTScore/BLEU/ROUGE
- Relevant keywords for retrieval evaluation
- Expected topics to be covered
- Relevance scoring criteria (0-3 scale)
"""
