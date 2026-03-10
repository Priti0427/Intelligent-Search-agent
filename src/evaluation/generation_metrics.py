"""
Text Generation Quality Metrics.

This module evaluates the quality of generated answers using:
- BERTScore: Semantic similarity using contextual embeddings
- BLEU: N-gram precision with brevity penalty
- ROUGE: Recall-oriented n-gram overlap
- Perplexity: Language model fluency measurement

These metrics compare the pipeline's synthesized answer against
expert-written reference answers (gold standard).

Course alignment: INFO 624 Week 8 - Evaluation Metrics for AI-Enhanced IR Systems
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def compute_bert_score(
    candidates: List[str],
    references: List[str],
    model_type: str = "distilbert-base-uncased",
    lang: str = "en",
) -> Dict[str, List[float]]:
    """
    Compute BERTScore between candidate and reference texts.

    BERTScore uses contextual embeddings to measure semantic similarity
    at the token level, capturing synonymy and paraphrase that lexical
    metrics like BLEU miss.

    Args:
        candidates: List of generated answers
        references: List of reference answers
        model_type: BERT model to use for embeddings
        lang: Language code

    Returns:
        Dict with 'precision', 'recall', 'f1' lists
    """
    try:
        from bert_score import score as bert_score

        P, R, F1 = bert_score(
            candidates,
            references,
            model_type=model_type,
            lang=lang,
            verbose=False,
        )

        return {
            "precision": [p.item() for p in P],
            "recall": [r.item() for r in R],
            "f1": [f.item() for f in F1],
        }

    except ImportError:
        logger.error("bert-score not installed. Run: pip install bert-score")
        return {"precision": [0.0] * len(candidates), "recall": [0.0] * len(candidates), "f1": [0.0] * len(candidates)}
    except Exception as e:
        logger.error(f"BERTScore computation failed: {e}")
        return {"precision": [0.0] * len(candidates), "recall": [0.0] * len(candidates), "f1": [0.0] * len(candidates)}


def compute_bleu_score(
    candidate: str,
    reference: str,
    max_n: int = 4,
) -> Dict[str, float]:
    """
    Compute BLEU score between a candidate and reference text.

    BLEU measures n-gram precision with a brevity penalty. Originally
    designed for machine translation evaluation.

    Limitation (from Week 8): "BLEU rewards exact wording overlap.
    It punishes paraphrasing."

    Args:
        candidate: Generated answer
        reference: Reference answer
        max_n: Maximum n-gram order

    Returns:
        Dict with BLEU-1 through BLEU-n and overall BLEU score
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()

        if not cand_tokens or not ref_tokens:
            return {f"bleu_{i}": 0.0 for i in range(1, max_n + 1)}

        smoothing = SmoothingFunction().method1
        results = {}

        for n in range(1, max_n + 1):
            weights = tuple([1.0 / n] * n + [0.0] * (max_n - n))
            try:
                score = sentence_bleu(
                    [ref_tokens],
                    cand_tokens,
                    weights=weights[:max_n],
                    smoothing_function=smoothing,
                )
                results[f"bleu_{n}"] = score
            except Exception:
                results[f"bleu_{n}"] = 0.0

        return results

    except ImportError:
        logger.error("nltk not installed. Run: pip install nltk")
        return {f"bleu_{i}": 0.0 for i in range(1, max_n + 1)}
    except Exception as e:
        logger.error(f"BLEU computation failed: {e}")
        return {f"bleu_{i}": 0.0 for i in range(1, max_n + 1)}


def compute_rouge_score(
    candidate: str,
    reference: str,
) -> Dict[str, Dict[str, float]]:
    """
    Compute ROUGE scores between a candidate and reference text.

    ROUGE is recall-oriented, measuring how much of the reference
    content is captured in the candidate. Complements BLEU's
    precision-oriented approach.

    ROUGE-1: Unigram overlap
    ROUGE-2: Bigram overlap
    ROUGE-L: Longest common subsequence

    Args:
        candidate: Generated answer
        reference: Reference answer

    Returns:
        Dict with ROUGE-1, ROUGE-2, ROUGE-L scores (each has precision, recall, fmeasure)
    """
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True,
        )

        scores = scorer.score(reference, candidate)

        return {
            "rouge_1": {
                "precision": scores["rouge1"].precision,
                "recall": scores["rouge1"].recall,
                "fmeasure": scores["rouge1"].fmeasure,
            },
            "rouge_2": {
                "precision": scores["rouge2"].precision,
                "recall": scores["rouge2"].recall,
                "fmeasure": scores["rouge2"].fmeasure,
            },
            "rouge_l": {
                "precision": scores["rougeL"].precision,
                "recall": scores["rougeL"].recall,
                "fmeasure": scores["rougeL"].fmeasure,
            },
        }

    except ImportError:
        logger.error("rouge-score not installed. Run: pip install rouge-score")
        empty = {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}
        return {"rouge_1": empty, "rouge_2": empty, "rouge_l": empty}
    except Exception as e:
        logger.error(f"ROUGE computation failed: {e}")
        empty = {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}
        return {"rouge_1": empty, "rouge_2": empty, "rouge_l": empty}


def compute_perplexity(
    text: str,
    model_name: str = "gpt2",
    stride: int = 512,
) -> float:
    """
    Compute perplexity of text using a pretrained language model.

    Perplexity measures how well a probability model predicts a sequence.
    Lower perplexity = more fluent/natural text.

    From Week 8: "Perplexity evaluates syntax/probability, not truth
    or relevance. Low perplexity != good retrieval."

    Args:
        text: Text to evaluate
        model_name: Pretrained LM to use
        stride: Stride for sliding window

    Returns:
        Perplexity score (lower is better)
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()

        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = encodings.input_ids

        max_length = model.config.n_positions if hasattr(model.config, "n_positions") else 1024
        seq_len = input_ids.size(1)

        nlls = []
        prev_end_loc = 0

        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_chunk = input_ids[:, begin_loc:end_loc]

            target_ids = input_chunk.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_chunk, labels=target_ids)
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc

            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl.item()

    except ImportError:
        logger.warning("transformers/torch not available for perplexity. Using fallback.")
        return _compute_perplexity_fallback(text)
    except Exception as e:
        logger.error(f"Perplexity computation failed: {e}")
        return _compute_perplexity_fallback(text)


def _compute_perplexity_fallback(text: str) -> float:
    """
    Simple statistical perplexity approximation when torch is unavailable.
    Uses character-level entropy as a rough proxy.
    """
    if not text:
        return float("inf")

    from collections import Counter

    tokens = text.lower().split()
    if len(tokens) < 2:
        return float("inf")

    total = len(tokens)
    freq = Counter(tokens)

    entropy = 0.0
    for count in freq.values():
        prob = count / total
        entropy -= prob * math.log2(prob)

    return 2 ** entropy


def compute_all_generation_metrics(
    candidate: str,
    reference: str,
    compute_ppl: bool = False,
) -> Dict[str, any]:
    """
    Compute all text generation quality metrics for a single candidate-reference pair.

    Args:
        candidate: Generated answer from the pipeline
        reference: Expert-written reference answer
        compute_ppl: Whether to compute perplexity (slower, requires torch)

    Returns:
        Dict with all metric scores
    """
    results = {}

    bert_scores = compute_bert_score([candidate], [reference])
    results["bert_score"] = {
        "precision": bert_scores["precision"][0],
        "recall": bert_scores["recall"][0],
        "f1": bert_scores["f1"][0],
    }

    results["bleu"] = compute_bleu_score(candidate, reference)

    results["rouge"] = compute_rouge_score(candidate, reference)

    if compute_ppl:
        results["perplexity"] = compute_perplexity(candidate)
    else:
        results["perplexity"] = None

    return results


def compute_batch_generation_metrics(
    candidates: List[str],
    references: List[str],
    compute_ppl: bool = False,
) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Compute generation metrics for a batch of candidate-reference pairs
    and return both per-query and aggregate results.

    Args:
        candidates: List of generated answers
        references: List of reference answers
        compute_ppl: Whether to compute perplexity

    Returns:
        Tuple of (per_query_results, aggregate_metrics)
    """
    if len(candidates) != len(references):
        raise ValueError("candidates and references must have the same length")

    bert_scores = compute_bert_score(candidates, references)

    per_query_results = []
    for i in range(len(candidates)):
        result = {
            "bert_score": {
                "precision": bert_scores["precision"][i],
                "recall": bert_scores["recall"][i],
                "f1": bert_scores["f1"][i],
            },
            "bleu": compute_bleu_score(candidates[i], references[i]),
            "rouge": compute_rouge_score(candidates[i], references[i]),
        }

        if compute_ppl:
            result["perplexity"] = compute_perplexity(candidates[i])
        else:
            result["perplexity"] = None

        per_query_results.append(result)

    n = len(candidates)
    aggregate = {
        "mean_bert_f1": sum(r["bert_score"]["f1"] for r in per_query_results) / n,
        "mean_bert_precision": sum(r["bert_score"]["precision"] for r in per_query_results) / n,
        "mean_bert_recall": sum(r["bert_score"]["recall"] for r in per_query_results) / n,
        "mean_bleu_1": sum(r["bleu"].get("bleu_1", 0) for r in per_query_results) / n,
        "mean_bleu_2": sum(r["bleu"].get("bleu_2", 0) for r in per_query_results) / n,
        "mean_bleu_4": sum(r["bleu"].get("bleu_4", 0) for r in per_query_results) / n,
        "mean_rouge_1_f": sum(r["rouge"]["rouge_1"]["fmeasure"] for r in per_query_results) / n,
        "mean_rouge_2_f": sum(r["rouge"]["rouge_2"]["fmeasure"] for r in per_query_results) / n,
        "mean_rouge_l_f": sum(r["rouge"]["rouge_l"]["fmeasure"] for r in per_query_results) / n,
    }

    ppls = [r["perplexity"] for r in per_query_results if r["perplexity"] is not None]
    if ppls:
        aggregate["mean_perplexity"] = sum(ppls) / len(ppls)

    return per_query_results, aggregate
