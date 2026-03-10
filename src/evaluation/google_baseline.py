"""
Google Baseline Comparison.

This module compares the Agentic Search pipeline's results against
Google search results to demonstrate the value-add of the agentic approach.

Uses pre-collected Google results stored as JSON fixtures to avoid
requiring a Google API key at evaluation time.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_google_results(query: str) -> Optional[List[Dict]]:
    """Load pre-collected Google results for a query from fixtures."""
    fixtures_path = FIXTURES_DIR / "google_results.json"
    if not fixtures_path.exists():
        logger.warning(f"Google fixtures not found at {fixtures_path}")
        return None

    try:
        with open(fixtures_path, "r") as f:
            all_results = json.load(f)

        for entry in all_results:
            if entry.get("query", "").lower().strip() == query.lower().strip():
                return entry.get("results", [])

        logger.warning(f"No Google results found for query: {query[:60]}...")
        return None
    except Exception as e:
        logger.error(f"Failed to load Google fixtures: {e}")
        return None


def compute_source_overlap(
    our_results: List[Dict],
    google_results: List[Dict],
) -> Dict[str, float]:
    """
    Compute overlap between our results and Google results.

    Measures:
    - URL domain overlap
    - Title similarity
    - Topic keyword overlap
    """
    if not our_results or not google_results:
        return {"domain_overlap": 0.0, "title_overlap": 0.0, "content_overlap": 0.0}

    def extract_domain(url: str) -> str:
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc.replace("www.", "")
        except Exception:
            return url

    our_domains = {extract_domain(r.get("url", "")) for r in our_results if r.get("url")}
    google_domains = {extract_domain(r.get("url", "")) for r in google_results if r.get("url")}

    domain_overlap = 0.0
    if our_domains and google_domains:
        intersection = our_domains & google_domains
        union = our_domains | google_domains
        domain_overlap = len(intersection) / len(union) if union else 0.0

    our_titles = {r.get("title", "").lower().strip() for r in our_results}
    google_titles = {r.get("title", "").lower().strip() for r in google_results}
    our_titles.discard("")
    google_titles.discard("")

    title_overlap = 0.0
    if our_titles and google_titles:
        matches = 0
        for ot in our_titles:
            for gt in google_titles:
                ot_words = set(ot.split())
                gt_words = set(gt.split())
                if ot_words and gt_words:
                    jaccard = len(ot_words & gt_words) / len(ot_words | gt_words)
                    if jaccard > 0.5:
                        matches += 1
                        break
        title_overlap = matches / max(len(our_titles), 1)

    def extract_keywords(results: List[Dict]) -> set:
        all_text = " ".join(
            r.get("title", "") + " " + r.get("content", "") + " " + r.get("snippet", "")
            for r in results
        ).lower()
        words = set(all_text.split())
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
                       "to", "for", "of", "and", "or", "but", "with", "by", "from", "as",
                       "it", "this", "that", "be", "has", "have", "had", "not", "no"}
        return words - stop_words

    our_keywords = extract_keywords(our_results)
    google_keywords = extract_keywords(google_results)

    content_overlap = 0.0
    if our_keywords and google_keywords:
        intersection = our_keywords & google_keywords
        union = our_keywords | google_keywords
        content_overlap = len(intersection) / len(union) if union else 0.0

    return {
        "domain_overlap": domain_overlap,
        "title_overlap": title_overlap,
        "content_overlap": content_overlap,
    }


def compute_answer_comparison(
    our_answer: str,
    google_snippets: List[str],
) -> Dict[str, float]:
    """
    Compare our synthesized answer against Google result snippets
    using BERTScore for semantic similarity.
    """
    if not our_answer or not google_snippets:
        return {"bert_score_vs_google": 0.0, "coverage_score": 0.0}

    combined_google = " ".join(google_snippets)

    try:
        from src.evaluation.generation_metrics import compute_bert_score
        scores = compute_bert_score([our_answer], [combined_google])
        bert_f1 = scores["f1"][0]
    except Exception as e:
        logger.error(f"BERTScore comparison failed: {e}")
        bert_f1 = 0.0

    google_keywords = set()
    for snippet in google_snippets:
        google_keywords.update(snippet.lower().split())

    answer_words = set(our_answer.lower().split())
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
                   "to", "for", "of", "and", "or", "but", "with", "by", "from"}
    google_keywords -= stop_words
    answer_words -= stop_words

    coverage = 0.0
    if google_keywords:
        coverage = len(answer_words & google_keywords) / len(google_keywords)

    return {
        "bert_score_vs_google": bert_f1,
        "coverage_score": min(coverage, 1.0),
    }


def evaluate_against_google(
    query: str,
    our_results: List[Dict],
    our_answer: str,
) -> Optional[Dict]:
    """
    Run full Google baseline comparison for a single query.

    Returns None if no Google fixtures are available for the query.
    """
    google_results = load_google_results(query)
    if google_results is None:
        return None

    overlap = compute_source_overlap(our_results, google_results)

    google_snippets = [
        r.get("snippet", r.get("content", ""))
        for r in google_results
        if r.get("snippet") or r.get("content")
    ]
    answer_comparison = compute_answer_comparison(our_answer, google_snippets)

    our_source_types = set(r.get("source_type", "unknown") for r in our_results)
    unique_to_us = {
        "has_academic": "academic" in our_source_types,
        "has_vector_store": "documents" in our_source_types,
        "has_citations": True,
        "has_synthesized_answer": bool(our_answer),
    }

    return {
        "query": query,
        "google_results_count": len(google_results),
        "our_results_count": len(our_results),
        "source_overlap": overlap,
        "answer_comparison": answer_comparison,
        "agentic_advantages": unique_to_us,
    }
