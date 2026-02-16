"""
Information Retrieval Evaluation Metrics.

This module implements standard IR evaluation metrics for measuring
the effectiveness of the search system.

Metrics included:
- Precision: Fraction of retrieved documents that are relevant
- Recall: Fraction of relevant documents that are retrieved
- F1 Score: Harmonic mean of precision and recall
- DCG/nDCG: Discounted Cumulative Gain (considers ranking position)
- MAP: Mean Average Precision
- MRR: Mean Reciprocal Rank
"""

import math
from typing import List, Set, Union


def precision(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Calculate precision: the fraction of retrieved documents that are relevant.
    
    Precision = |Relevant ∩ Retrieved| / |Retrieved|
    
    Args:
        retrieved: List of retrieved document IDs (in ranked order)
        relevant: Set of relevant document IDs (ground truth)
        
    Returns:
        Precision score between 0 and 1
        
    Example:
        >>> precision(['doc1', 'doc2', 'doc3'], {'doc1', 'doc3', 'doc5'})
        0.6666666666666666
    """
    if not retrieved:
        return 0.0
    
    relevant_retrieved = len(set(retrieved) & relevant)
    return relevant_retrieved / len(retrieved)


def recall(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Calculate recall: the fraction of relevant documents that are retrieved.
    
    Recall = |Relevant ∩ Retrieved| / |Relevant|
    
    Args:
        retrieved: List of retrieved document IDs
        relevant: Set of relevant document IDs (ground truth)
        
    Returns:
        Recall score between 0 and 1
        
    Example:
        >>> recall(['doc1', 'doc2', 'doc3'], {'doc1', 'doc3', 'doc5'})
        0.6666666666666666
    """
    if not relevant:
        return 0.0
    
    relevant_retrieved = len(set(retrieved) & relevant)
    return relevant_retrieved / len(relevant)


def f1_score(prec: float, rec: float) -> float:
    """
    Calculate F1 score: the harmonic mean of precision and recall.
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    Args:
        prec: Precision score
        rec: Recall score
        
    Returns:
        F1 score between 0 and 1
        
    Example:
        >>> f1_score(0.8, 0.6)
        0.6857142857142857
    """
    if prec + rec == 0:
        return 0.0
    
    return 2 * (prec * rec) / (prec + rec)


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Calculate precision at rank k (P@k).
    
    Args:
        retrieved: List of retrieved document IDs (in ranked order)
        relevant: Set of relevant document IDs
        k: Rank cutoff
        
    Returns:
        Precision at k
    """
    if k <= 0:
        return 0.0
    
    retrieved_at_k = retrieved[:k]
    return precision(retrieved_at_k, relevant)


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Calculate recall at rank k (R@k).
    
    Args:
        retrieved: List of retrieved document IDs (in ranked order)
        relevant: Set of relevant document IDs
        k: Rank cutoff
        
    Returns:
        Recall at k
    """
    if k <= 0:
        return 0.0
    
    retrieved_at_k = retrieved[:k]
    return recall(retrieved_at_k, relevant)


def dcg_at_k(relevance_scores: List[float], k: int) -> float:
    """
    Calculate Discounted Cumulative Gain at rank k.
    
    DCG@k = Σ (2^rel_i - 1) / log2(i + 1) for i = 1 to k
    
    This version uses graded relevance scores (not just binary).
    
    Args:
        relevance_scores: List of relevance scores in ranked order
                         (e.g., [3, 2, 3, 0, 1, 2] where 3=highly relevant)
        k: Rank cutoff
        
    Returns:
        DCG score
        
    Example:
        >>> dcg_at_k([3, 2, 3, 0, 1, 2], 6)
        13.848...
    """
    if k <= 0 or not relevance_scores:
        return 0.0
    
    dcg = 0.0
    for i, rel in enumerate(relevance_scores[:k]):
        # Position is 1-indexed for the formula
        position = i + 1
        # DCG formula: (2^rel - 1) / log2(position + 1)
        dcg += (2 ** rel - 1) / math.log2(position + 1)
    
    return dcg


def ndcg_at_k(relevance_scores: List[float], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at rank k.
    
    nDCG@k = DCG@k / IDCG@k
    
    Where IDCG is the DCG of the ideal ranking (sorted by relevance).
    
    Args:
        relevance_scores: List of relevance scores in ranked order
        k: Rank cutoff
        
    Returns:
        nDCG score between 0 and 1
        
    Example:
        >>> ndcg_at_k([3, 2, 3, 0, 1, 2], 6)
        0.961...
    """
    if k <= 0 or not relevance_scores:
        return 0.0
    
    # Calculate DCG for the actual ranking
    dcg = dcg_at_k(relevance_scores, k)
    
    # Calculate IDCG (ideal DCG) - sort relevance scores in descending order
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = dcg_at_k(ideal_scores, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def binary_ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Calculate nDCG@k using binary relevance (relevant=1, not relevant=0).
    
    This is a convenience function when you only have binary relevance judgments.
    
    Args:
        retrieved: List of retrieved document IDs (in ranked order)
        relevant: Set of relevant document IDs
        k: Rank cutoff
        
    Returns:
        nDCG score between 0 and 1
    """
    # Convert to binary relevance scores
    relevance_scores = [1.0 if doc in relevant else 0.0 for doc in retrieved]
    return ndcg_at_k(relevance_scores, k)


def average_precision(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Calculate Average Precision (AP) for a single query.
    
    AP = (1/|Relevant|) * Σ (Precision@k * rel(k))
    
    Where rel(k) = 1 if document at rank k is relevant, 0 otherwise.
    
    Args:
        retrieved: List of retrieved document IDs (in ranked order)
        relevant: Set of relevant document IDs
        
    Returns:
        Average Precision score between 0 and 1
        
    Example:
        >>> average_precision(['d1', 'd2', 'd3', 'd4'], {'d1', 'd3'})
        0.8333...
    """
    if not relevant or not retrieved:
        return 0.0
    
    ap_sum = 0.0
    relevant_count = 0
    
    for k, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            relevant_count += 1
            # Precision at this position
            precision_at_k = relevant_count / k
            ap_sum += precision_at_k
    
    return ap_sum / len(relevant)


def mean_average_precision(
    retrieved_lists: List[List[str]], 
    relevant_sets: List[Set[str]]
) -> float:
    """
    Calculate Mean Average Precision (MAP) across multiple queries.
    
    MAP = (1/|Q|) * Σ AP(q) for all queries q
    
    Args:
        retrieved_lists: List of retrieved document lists (one per query)
        relevant_sets: List of relevant document sets (one per query)
        
    Returns:
        MAP score between 0 and 1
    """
    if not retrieved_lists or len(retrieved_lists) != len(relevant_sets):
        return 0.0
    
    ap_scores = [
        average_precision(retrieved, relevant)
        for retrieved, relevant in zip(retrieved_lists, relevant_sets)
    ]
    
    return sum(ap_scores) / len(ap_scores)


def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Calculate Reciprocal Rank (RR) for a single query.
    
    RR = 1 / rank of first relevant document
    
    Args:
        retrieved: List of retrieved document IDs (in ranked order)
        relevant: Set of relevant document IDs
        
    Returns:
        Reciprocal Rank score between 0 and 1
        
    Example:
        >>> reciprocal_rank(['d1', 'd2', 'd3'], {'d2', 'd3'})
        0.5
    """
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / rank
    
    return 0.0


def mean_reciprocal_rank(
    retrieved_lists: List[List[str]], 
    relevant_sets: List[Set[str]]
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) across multiple queries.
    
    MRR = (1/|Q|) * Σ RR(q) for all queries q
    
    Args:
        retrieved_lists: List of retrieved document lists (one per query)
        relevant_sets: List of relevant document sets (one per query)
        
    Returns:
        MRR score between 0 and 1
    """
    if not retrieved_lists or len(retrieved_lists) != len(relevant_sets):
        return 0.0
    
    rr_scores = [
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(retrieved_lists, relevant_sets)
    ]
    
    return sum(rr_scores) / len(rr_scores)


# Convenience function for comprehensive evaluation
def evaluate_retrieval(
    retrieved: List[str], 
    relevant: Set[str], 
    k: int = 10
) -> dict:
    """
    Calculate all retrieval metrics for a single query.
    
    Args:
        retrieved: List of retrieved document IDs (in ranked order)
        relevant: Set of relevant document IDs
        k: Rank cutoff for P@k, R@k, nDCG@k
        
    Returns:
        Dictionary with all metric scores
    """
    prec = precision(retrieved, relevant)
    rec = recall(retrieved, relevant)
    
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1_score(prec, rec),
        f"precision_at_{k}": precision_at_k(retrieved, relevant, k),
        f"recall_at_{k}": recall_at_k(retrieved, relevant, k),
        f"ndcg_at_{k}": binary_ndcg_at_k(retrieved, relevant, k),
        "average_precision": average_precision(retrieved, relevant),
        "reciprocal_rank": reciprocal_rank(retrieved, relevant),
        "num_retrieved": len(retrieved),
        "num_relevant": len(relevant),
        "num_relevant_retrieved": len(set(retrieved) & relevant),
    }
