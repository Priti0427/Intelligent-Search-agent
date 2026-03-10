"""
Evaluation module for Agentic Search.

This module provides comprehensive evaluation including:
- Layer 1: IR Retrieval Metrics (Precision, Recall, F1, nDCG, MAP, MRR)
- Layer 2: Text Generation Quality (BERTScore, BLEU, ROUGE, Perplexity)
- Layer 3: Google Baseline Comparison
- Layer 4: LLM-as-Judge Evaluation
- Layer 5: RAGAS (Faithfulness, Answer Relevancy, Context Precision/Recall)
- Layer 6: Robustness & Stress Testing
"""

from src.evaluation.metrics import (
    precision,
    recall,
    f1_score,
    dcg_at_k,
    ndcg_at_k,
    average_precision,
    mean_average_precision,
    reciprocal_rank,
    mean_reciprocal_rank,
)
from src.evaluation.evaluator import SearchEvaluator, ComprehensiveEvaluator
from src.evaluation.test_cases import TEST_CASES, get_test_cases

__all__ = [
    "precision",
    "recall",
    "f1_score",
    "dcg_at_k",
    "ndcg_at_k",
    "average_precision",
    "mean_average_precision",
    "reciprocal_rank",
    "mean_reciprocal_rank",
    "SearchEvaluator",
    "ComprehensiveEvaluator",
    "TEST_CASES",
    "get_test_cases",
]
