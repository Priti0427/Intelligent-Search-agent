"""
Evaluation module for Agentic Search.

This module provides formal IR evaluation metrics including:
- Precision
- Recall
- F1 Score
- nDCG (Normalized Discounted Cumulative Gain)
- MAP (Mean Average Precision)
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
from src.evaluation.evaluator import SearchEvaluator
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
    "TEST_CASES",
    "get_test_cases",
]
