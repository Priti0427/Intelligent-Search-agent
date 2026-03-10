"""
Robustness & Stress Testing.

Tests the search pipeline's robustness to:
- Query paraphrasing (consistency)
- Adversarial inputs (misspellings, jargon)
- Query drift (sensitivity to small changes)

Inspired by INFO 624 Week 8: "A system that only works on clean
benchmarks is not robust."
"""

import logging
import random
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PARAPHRASE_VARIANTS = {
    "How does combining BM25 with dense retrieval in a hybrid search system improve RAG pipeline accuracy compared to using either method alone?": [
        "What are the benefits of using both BM25 and neural dense retrieval together in a RAG system instead of just one approach?",
        "Why does hybrid search that merges sparse BM25 with dense vector retrieval produce better results in retrieval-augmented generation?",
        "In what ways does a combined BM25 plus dense retrieval approach outperform standalone methods for RAG accuracy?",
    ],
    "Compare the effectiveness of query decomposition vs query expansion for handling complex multi-faceted information needs": [
        "Which works better for complex questions: breaking queries into sub-queries or expanding the original query with more terms?",
        "What are the pros and cons of query decomposition versus query expansion when dealing with multi-part information needs?",
        "How do query decomposition and query expansion differ in their ability to handle complex search requests?",
    ],
    "Explain how BERT-based cross-encoders re-rank search results and why they outperform bi-encoder approaches for passage retrieval": [
        "Why are BERT cross-encoders more accurate than bi-encoders for re-ranking search results in passage retrieval?",
        "How does the cross-encoder architecture in BERT improve passage re-ranking compared to the bi-encoder approach?",
        "What makes cross-encoder models better than bi-encoder models for search result re-ranking?",
    ],
    "What are the key architectural differences between Agentic RAG and standard RAG, and how does self-reflection improve answer quality?": [
        "How does agentic RAG differ from traditional RAG architecturally, and what role does self-reflection play in improving outputs?",
        "What distinguishes an agentic RAG system from a basic RAG pipeline, particularly regarding self-evaluation of answers?",
        "In what ways does the architecture of agentic RAG systems go beyond standard RAG, and how does reflection enhance quality?",
    ],
    "How do vector embeddings and semantic search overcome the vocabulary mismatch problem that traditional TF-IDF methods face?": [
        "Why can semantic search with embeddings find relevant documents that TF-IDF misses due to vocabulary differences?",
        "How do dense vector representations solve the term mismatch limitation of TF-IDF in information retrieval?",
        "What advantage do vector embeddings provide over TF-IDF when queries and documents use different words for the same concepts?",
    ],
}

ADVERSARIAL_VARIANTS = {
    "How does combining BM25 with dense retrieval in a hybrid search system improve RAG pipeline accuracy compared to using either method alone?": [
        "How does combiing BM25 with dnese retreival in a hybrd search systm improve RAG pipleine accuracy?",
        "bm25 + dense retrieval hybrid rag accuracy improvement why",
        "Elucidate the synergistic amelioration of retrieval-augmented generation pipeline fidelity through the concatenation of Okapi BM25 probabilistic ranking with latent semantic dense vector retrieval methodologies",
    ],
    "Compare the effectiveness of query decomposition vs query expansion for handling complex multi-faceted information needs": [
        "Compare the effectivness of qurey decompostion vs qurey expansion for handeling complex information needs",
        "query decomposition vs expansion complex queries which better",
        "Juxtapose the efficacy of interrogative decomposition contra lexical augmentation methodologies vis-a-vis multifaceted informational desiderata resolution",
    ],
    "Explain how BERT-based cross-encoders re-rank search results and why they outperform bi-encoder approaches for passage retrieval": [
        "Explane how BERT crosse-encoders re-rnk search reults and why they outpreform bi-encoders",
        "bert cross encoder vs bi encoder reranking why better",
        "Expound upon the mechanism by which BERT-predicated cross-encoding architectures effectuate search result re-prioritization and their superiority over dual-encoder paradigms",
    ],
    "What are the key architectural differences between Agentic RAG and standard RAG, and how does self-reflection improve answer quality?": [
        "What are the key architecutral differnces between Agentic RAG and standrad RAG and how does self-reflecton improve quality?",
        "agentic rag vs standard rag differences self reflection quality",
        "Delineate the fundamental architectural divergences between autonomous agentic retrieval-augmented generation and conventional RAG paradigms with particular emphasis on metacognitive self-evaluative loops",
    ],
    "How do vector embeddings and semantic search overcome the vocabulary mismatch problem that traditional TF-IDF methods face?": [
        "How do vector embedings and semnatic serch overcome the vocabulry mismatch problme that TF-IDF faces?",
        "vector embeddings semantic search vocabulary mismatch tfidf problem solve how",
        "Elucidate the mechanism by which distributed representational vectors and latent semantic retrieval methodologies transcend the lexical incongruity predicament inherent in term frequency-inverse document frequency approaches",
    ],
}


def get_paraphrase_variants(query: str) -> List[str]:
    """Get paraphrase variants for a query."""
    return PARAPHRASE_VARIANTS.get(query, [])


def get_adversarial_variants(query: str) -> List[str]:
    """Get adversarial variants (misspellings, keyword-only, jargon)."""
    return ADVERSARIAL_VARIANTS.get(query, [])


def get_drift_variants(query: str, num_variants: int = 3) -> List[str]:
    """
    Generate query drift variants by slightly modifying the query.
    Tests whether small changes cause large ranking shifts.
    """
    words = query.split()
    variants = []

    if len(words) > 5:
        modified = words[:len(words)//2]
        variants.append(" ".join(modified))

    addition_terms = ["recently", "in 2024", "for beginners", "with examples", "in practice"]
    for term in addition_terms[:num_variants]:
        variants.append(f"{query} {term}")

    return variants[:num_variants]


async def evaluate_paraphrase_consistency(
    original_query: str,
    run_search_fn,
) -> Dict[str, Any]:
    """
    Test consistency across paraphrased versions of the same query.

    A robust system should produce similar answers for semantically
    equivalent queries.
    """
    variants = get_paraphrase_variants(original_query)
    if not variants:
        return {"skipped": True, "reason": "No paraphrase variants available"}

    try:
        original_result = await run_search_fn(original_query)
        original_answer = original_result.get("final_answer", original_result.get("draft_answer", ""))

        variant_results = []
        for variant in variants:
            result = await run_search_fn(variant)
            variant_answer = result.get("final_answer", result.get("draft_answer", ""))
            variant_results.append({
                "variant_query": variant,
                "answer": variant_answer,
                "answer_length": len(variant_answer),
            })

        try:
            from src.evaluation.generation_metrics import compute_bert_score
            variant_answers = [vr["answer"] for vr in variant_results]
            scores = compute_bert_score(
                variant_answers,
                [original_answer] * len(variant_answers),
            )
            for i, vr in enumerate(variant_results):
                vr["similarity_to_original"] = scores["f1"][i]
            mean_consistency = sum(scores["f1"]) / len(scores["f1"])
        except Exception:
            mean_consistency = 0.0

        return {
            "original_query": original_query,
            "num_variants": len(variants),
            "mean_consistency": mean_consistency,
            "variant_details": variant_results,
        }
    except Exception as e:
        logger.error(f"Paraphrase consistency test failed: {e}")
        return {"error": str(e)}


async def evaluate_adversarial_robustness(
    original_query: str,
    run_search_fn,
) -> Dict[str, Any]:
    """
    Test robustness to adversarial query variants:
    - Misspellings
    - Keyword-only queries
    - Jargon-heavy reformulations
    """
    variants = get_adversarial_variants(original_query)
    if not variants:
        return {"skipped": True, "reason": "No adversarial variants available"}

    variant_types = ["misspelling", "keyword_only", "jargon_heavy"]

    try:
        original_result = await run_search_fn(original_query)
        original_answer = original_result.get("final_answer", original_result.get("draft_answer", ""))

        variant_results = []
        for i, variant in enumerate(variants):
            result = await run_search_fn(variant)
            variant_answer = result.get("final_answer", result.get("draft_answer", ""))
            vtype = variant_types[i] if i < len(variant_types) else "unknown"
            variant_results.append({
                "variant_type": vtype,
                "variant_query": variant,
                "answer": variant_answer,
                "produced_answer": bool(variant_answer.strip()),
                "answer_length": len(variant_answer),
            })

        try:
            from src.evaluation.generation_metrics import compute_bert_score
            variant_answers = [vr["answer"] for vr in variant_results if vr["answer"]]
            if variant_answers:
                scores = compute_bert_score(
                    variant_answers,
                    [original_answer] * len(variant_answers),
                )
                idx = 0
                for vr in variant_results:
                    if vr["answer"]:
                        vr["similarity_to_original"] = scores["f1"][idx]
                        idx += 1
        except Exception:
            pass

        successful = sum(1 for vr in variant_results if vr["produced_answer"])

        return {
            "original_query": original_query,
            "num_variants": len(variants),
            "success_rate": successful / len(variants) if variants else 0,
            "variant_details": variant_results,
        }
    except Exception as e:
        logger.error(f"Adversarial robustness test failed: {e}")
        return {"error": str(e)}


async def evaluate_query_drift(
    original_query: str,
    run_search_fn,
) -> Dict[str, Any]:
    """
    Test sensitivity to small query modifications.
    Checks if minor changes cause disproportionate ranking shifts.
    """
    variants = get_drift_variants(original_query)
    if not variants:
        return {"skipped": True, "reason": "No drift variants generated"}

    try:
        original_result = await run_search_fn(original_query)
        original_answer = original_result.get("final_answer", original_result.get("draft_answer", ""))

        variant_results = []
        for variant in variants:
            result = await run_search_fn(variant)
            variant_answer = result.get("final_answer", result.get("draft_answer", ""))
            variant_results.append({
                "drift_query": variant,
                "answer": variant_answer,
                "answer_length": len(variant_answer),
            })

        return {
            "original_query": original_query,
            "num_variants": len(variants),
            "variant_details": variant_results,
        }
    except Exception as e:
        logger.error(f"Query drift test failed: {e}")
        return {"error": str(e)}


async def run_full_robustness_evaluation(
    test_queries: List[str],
    run_search_fn,
    test_paraphrase: bool = True,
    test_adversarial: bool = True,
    test_drift: bool = True,
) -> Dict[str, Any]:
    """
    Run complete robustness evaluation across all test queries.

    Returns aggregate robustness metrics and per-query details.
    """
    results = {
        "paraphrase": [],
        "adversarial": [],
        "drift": [],
    }

    for query in test_queries:
        logger.info(f"Robustness testing: {query[:50]}...")

        if test_paraphrase:
            pr = await evaluate_paraphrase_consistency(query, run_search_fn)
            results["paraphrase"].append(pr)

        if test_adversarial:
            ar = await evaluate_adversarial_robustness(query, run_search_fn)
            results["adversarial"].append(ar)

        if test_drift:
            dr = await evaluate_query_drift(query, run_search_fn)
            results["drift"].append(dr)

    aggregate = {}

    para_scores = [r.get("mean_consistency", 0) for r in results["paraphrase"]
                   if "mean_consistency" in r]
    if para_scores:
        aggregate["mean_paraphrase_consistency"] = sum(para_scores) / len(para_scores)

    adv_rates = [r.get("success_rate", 0) for r in results["adversarial"]
                 if "success_rate" in r]
    if adv_rates:
        aggregate["mean_adversarial_success_rate"] = sum(adv_rates) / len(adv_rates)

    return {
        "aggregate": aggregate,
        "details": results,
    }
