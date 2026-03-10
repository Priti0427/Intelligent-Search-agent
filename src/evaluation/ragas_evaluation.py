"""
RAGAS Evaluation Integration.

Uses the RAGAS (Retrieval Augmented Generation Assessment) framework
to evaluate the Agentic Search pipeline with industry-standard metrics:

- Faithfulness: Are claims grounded in retrieved context?
- Answer Relevancy: Is the answer relevant to the query?
- Context Precision: Is the retrieved context relevant?
- Context Recall: Does the context cover the reference answer?
- Answer Correctness: Is the answer factually correct?
- Answer Similarity: Semantic similarity to reference answer

RAGAS requires an LLM for evaluation. We use the same LLM provider
configured for the pipeline (Groq or OpenAI).
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


async def run_ragas_evaluation(
    queries: List[str],
    answers: List[str],
    contexts_list: List[List[str]],
    reference_answers: Optional[List[str]] = None,
    metrics_to_run: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run RAGAS evaluation on pipeline outputs.

    Args:
        queries: List of user queries
        answers: List of generated answers
        contexts_list: List of lists of retrieved context strings
        reference_answers: Optional list of reference/ground truth answers
        metrics_to_run: Which RAGAS metrics to compute. Defaults to all available.
            Options: 'faithfulness', 'answer_relevancy', 'context_precision',
                     'context_recall', 'answer_correctness', 'answer_similarity'

    Returns:
        Dict with per_query results and aggregate scores
    """
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
        from ragas.metrics import (
            Faithfulness, AnswerRelevancy, LLMContextPrecisionWithoutReference,
            LLMContextRecall, AnswerCorrectness, SemanticSimilarity,
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError as e:
        logger.error(f"RAGAS import failed: {e}")
        return {"error": "RAGAS not installed. Run: pip install ragas", "per_query": [], "aggregate": {}}

    from src.utils.llm import get_llm
    from src.retrievers.vector_store import get_embeddings

    llm = get_llm(temperature=0)
    embeddings = get_embeddings()
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    if metrics_to_run is None:
        metrics_to_run = ["faithfulness", "answer_relevancy", "context_precision",
                          "context_recall", "answer_correctness", "answer_similarity"]

    metric_instances = []
    metric_names = []

    for name in metrics_to_run:
        try:
            if name == "faithfulness":
                m = Faithfulness(llm=ragas_llm)
                metric_instances.append(m)
                metric_names.append("faithfulness")
            elif name == "answer_relevancy":
                m = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
                metric_instances.append(m)
                metric_names.append("answer_relevancy")
            elif name == "context_precision":
                m = LLMContextPrecisionWithoutReference(llm=ragas_llm)
                metric_instances.append(m)
                metric_names.append("context_precision")
            elif name == "context_recall":
                if reference_answers:
                    m = LLMContextRecall(llm=ragas_llm)
                    metric_instances.append(m)
                    metric_names.append("context_recall")
                else:
                    logger.warning("Skipping context_recall: requires reference answers")
            elif name == "answer_correctness":
                if reference_answers:
                    m = AnswerCorrectness(llm=ragas_llm, embeddings=ragas_embeddings)
                    metric_instances.append(m)
                    metric_names.append("answer_correctness")
                else:
                    logger.warning("Skipping answer_correctness: requires reference answers")
            elif name == "answer_similarity":
                if reference_answers:
                    m = SemanticSimilarity(embeddings=ragas_embeddings)
                    metric_instances.append(m)
                    metric_names.append("answer_similarity")
                else:
                    logger.warning("Skipping answer_similarity: requires reference answers")
        except Exception as e:
            logger.warning(f"Failed to initialize metric '{name}': {e}")

    if not metric_instances:
        return {"error": "No metrics could be initialized", "per_query": [], "aggregate": {}}

    samples = []
    for i in range(len(queries)):
        sample_kwargs = {
            "user_input": queries[i],
            "response": answers[i],
            "retrieved_contexts": contexts_list[i] if i < len(contexts_list) else [],
        }
        if reference_answers and i < len(reference_answers):
            sample_kwargs["reference"] = reference_answers[i]

        samples.append(SingleTurnSample(**sample_kwargs))

    dataset = EvaluationDataset(samples=samples)

    try:
        logger.info(f"Running RAGAS evaluation with {len(metric_instances)} metrics on {len(samples)} samples...")
        result = ragas_evaluate(
            dataset=dataset,
            metrics=metric_instances,
        )

        scores_df = result.to_pandas()

        per_query = []
        for i in range(len(queries)):
            query_scores = {"query": queries[i]}
            for col in scores_df.columns:
                if col not in ["user_input", "response", "retrieved_contexts", "reference"]:
                    val = scores_df.iloc[i][col]
                    if hasattr(val, 'item'):
                        val = val.item()
                    query_scores[col] = float(val) if val is not None else None
            per_query.append(query_scores)

        aggregate = {}
        for name in metric_names:
            vals = [pq.get(name, None) for pq in per_query if pq.get(name) is not None]
            if vals:
                aggregate[f"mean_{name}"] = sum(vals) / len(vals)

        return {
            "per_query": per_query,
            "aggregate": aggregate,
            "metrics_used": metric_names,
        }

    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        return {"error": str(e), "per_query": [], "aggregate": {}}
