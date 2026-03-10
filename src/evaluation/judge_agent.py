"""
LLM-as-Judge Evaluator Agent.

An independent LLM evaluator that scores the pipeline's output on
multiple quality dimensions. This provides an external assessment
separate from the pipeline's internal self-reflection (reflector.py).

Evaluation Dimensions (scored 0-1):
1. Relevance - Does the answer address the query?
2. Completeness - Are all aspects covered?
3. Faithfulness - Are claims supported by retrieved sources?
4. Citation Quality - Are citations accurate and attributed?
5. Coherence - Is the answer well-structured?
6. Factual Accuracy - Are facts correct vs reference answer?

Also computes Hallucination Rate (unsupported claims / total claims).
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are an expert evaluator for an information retrieval system.
You must evaluate the quality of a generated answer on multiple dimensions.

QUERY: {query}

INFORMATION NEED: {information_need}

RETRIEVED SOURCES:
{sources}

GENERATED ANSWER:
{answer}

{reference_section}

Evaluate the answer on these dimensions (score each 0.0 to 1.0):

1. RELEVANCE: Does the answer directly address the query and information need?
2. COMPLETENESS: Does the answer cover all important aspects of the query?
3. FAITHFULNESS: Are all claims in the answer supported by the retrieved sources? (Not hallucinated)
4. CITATION_QUALITY: Are sources properly cited and attributed?
5. COHERENCE: Is the answer well-organized, clear, and logically structured?
6. FACTUAL_ACCURACY: Are the stated facts correct? (Use reference answer if provided)

Also perform a HALLUCINATION CHECK:
- List each factual claim in the answer
- For each claim, note whether it is SUPPORTED or UNSUPPORTED by the sources
- Calculate hallucination_rate = unsupported_claims / total_claims

Respond with ONLY valid JSON in this exact format:
{{
  "scores": {{
    "relevance": <float 0-1>,
    "completeness": <float 0-1>,
    "faithfulness": <float 0-1>,
    "citation_quality": <float 0-1>,
    "coherence": <float 0-1>,
    "factual_accuracy": <float 0-1>
  }},
  "overall_score": <float 0-1>,
  "hallucination_check": {{
    "total_claims": <int>,
    "supported_claims": <int>,
    "unsupported_claims": <int>,
    "hallucination_rate": <float 0-1>,
    "unsupported_details": [<list of unsupported claim strings>]
  }},
  "strengths": [<list of strength strings>],
  "weaknesses": [<list of weakness strings>],
  "reasoning": "<brief explanation of scoring>"
}}"""


async def run_judge_evaluation(
    query: str,
    information_need: str,
    answer: str,
    sources: List[Dict[str, Any]],
    reference_answer: Optional[str] = None,
    use_different_provider: bool = False,
) -> Dict[str, Any]:
    """
    Run LLM-as-judge evaluation on a single query-answer pair.

    Args:
        query: The search query
        information_need: Description of the user's information need
        answer: The generated answer to evaluate
        sources: List of retrieved source dicts with title, content, source_type
        reference_answer: Optional expert reference answer for comparison
        use_different_provider: If True, try to use a different LLM than the pipeline

    Returns:
        Dict with scores, hallucination check, strengths, weaknesses, reasoning
    """
    from src.utils.llm import get_llm

    sources_text = ""
    for i, src in enumerate(sources[:10], 1):
        title = src.get("title", "Untitled")
        content = src.get("content", src.get("excerpt", ""))[:500]
        source_type = src.get("source_type", "unknown")
        sources_text += f"[{i}] ({source_type}) {title}\n{content}\n\n"

    if not sources_text.strip():
        sources_text = "No sources available."

    reference_section = ""
    if reference_answer:
        reference_section = f"REFERENCE ANSWER (expert-written gold standard):\n{reference_answer}\n"

    prompt = JUDGE_PROMPT.format(
        query=query,
        information_need=information_need,
        sources=sources_text,
        answer=answer,
        reference_section=reference_section,
    )

    try:
        llm = get_llm(temperature=0)
        response = await llm.ainvoke(prompt)
        content = response.content

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        result = json.loads(content.strip())

        scores = result.get("scores", {})
        for key in ["relevance", "completeness", "faithfulness",
                     "citation_quality", "coherence", "factual_accuracy"]:
            scores[key] = max(0.0, min(1.0, float(scores.get(key, 0.5))))

        result["scores"] = scores
        result["overall_score"] = max(0.0, min(1.0, float(result.get("overall_score", 0.5))))

        hall = result.get("hallucination_check", {})
        hall["total_claims"] = int(hall.get("total_claims", 0))
        hall["supported_claims"] = int(hall.get("supported_claims", 0))
        hall["unsupported_claims"] = int(hall.get("unsupported_claims", 0))
        hall["hallucination_rate"] = max(0.0, min(1.0, float(hall.get("hallucination_rate", 0))))
        result["hallucination_check"] = hall

        return result

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse judge response: {e}")
        return _default_judge_result("Failed to parse LLM judge response")
    except Exception as e:
        logger.error(f"Judge evaluation failed: {e}")
        return _default_judge_result(str(e))


def _default_judge_result(error_msg: str) -> Dict[str, Any]:
    """Return a default result when judge evaluation fails."""
    return {
        "scores": {
            "relevance": 0.0,
            "completeness": 0.0,
            "faithfulness": 0.0,
            "citation_quality": 0.0,
            "coherence": 0.0,
            "factual_accuracy": 0.0,
        },
        "overall_score": 0.0,
        "hallucination_check": {
            "total_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": 0,
            "hallucination_rate": 0.0,
            "unsupported_details": [],
        },
        "strengths": [],
        "weaknesses": [],
        "reasoning": f"Evaluation failed: {error_msg}",
    }


async def run_batch_judge_evaluation(
    queries: List[str],
    information_needs: List[str],
    answers: List[str],
    sources_list: List[List[Dict]],
    reference_answers: Optional[List[str]] = None,
) -> tuple:
    """
    Run judge evaluation on a batch of queries.

    Returns:
        Tuple of (per_query_results, aggregate_scores)
    """
    if reference_answers is None:
        reference_answers = [None] * len(queries)

    per_query_results = []
    for i in range(len(queries)):
        logger.info(f"Judge evaluating query {i+1}/{len(queries)}: {queries[i][:50]}...")
        result = await run_judge_evaluation(
            query=queries[i],
            information_need=information_needs[i],
            answer=answers[i],
            sources=sources_list[i] if i < len(sources_list) else [],
            reference_answer=reference_answers[i],
        )
        per_query_results.append(result)

    n = len(per_query_results)
    if n == 0:
        return [], {}

    dimensions = ["relevance", "completeness", "faithfulness",
                   "citation_quality", "coherence", "factual_accuracy"]

    aggregate = {}
    for dim in dimensions:
        values = [r["scores"].get(dim, 0) for r in per_query_results]
        aggregate[f"mean_{dim}"] = sum(values) / n

    aggregate["mean_overall"] = sum(r.get("overall_score", 0) for r in per_query_results) / n

    hall_rates = [r.get("hallucination_check", {}).get("hallucination_rate", 0)
                  for r in per_query_results]
    aggregate["mean_hallucination_rate"] = sum(hall_rates) / n

    total_claims = sum(r.get("hallucination_check", {}).get("total_claims", 0)
                       for r in per_query_results)
    total_unsupported = sum(r.get("hallucination_check", {}).get("unsupported_claims", 0)
                            for r in per_query_results)
    aggregate["overall_hallucination_rate"] = (
        total_unsupported / total_claims if total_claims > 0 else 0.0
    )

    return per_query_results, aggregate
