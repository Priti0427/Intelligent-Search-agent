"""
Self-Reflection Node.

This node evaluates the quality of the generated answer and determines
whether it meets the quality threshold or needs improvement.
"""

import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI

from src.agent.state import QualityScores, SearchState
from src.utils.config import get_settings
from src.utils.prompts import get_prompt

logger = logging.getLogger(__name__)


async def reflect_node(state: SearchState) -> dict[str, Any]:
    """
    Evaluate answer quality and determine if improvement is needed.
    
    This node:
    1. Evaluates the draft answer on multiple quality dimensions
    2. Computes an overall quality score
    3. Provides specific feedback for improvement
    4. Determines whether to continue or finalize
    
    Quality dimensions:
    - Relevance: Does the answer address the query?
    - Completeness: Are all aspects covered?
    - Accuracy: Is information consistent with sources?
    - Citation Quality: Are sources properly cited?
    - Clarity: Is the answer well-structured?
    
    Args:
        state: Current search state with draft_answer
        
    Returns:
        Updated state with quality scores and continuation decision
    """
    settings = get_settings()
    query = state["original_query"]
    answer = state.get("draft_answer", "")
    citations = state.get("citations", [])
    iteration_count = state.get("iteration_count", 0)
    
    logger.info(f"Reflecting on answer (iteration {iteration_count + 1})")
    
    # Check iteration limit
    if iteration_count >= settings.max_reflection_iterations:
        logger.info("Max iterations reached, finalizing answer")
        return {
            "final_answer": answer,
            "should_continue": False,
            "iteration_count": iteration_count + 1,
        }
    
    # Handle empty answer
    if not answer:
        return {
            "quality_scores": {
                "relevance": 0.0,
                "completeness": 0.0,
                "accuracy": 0.0,
                "citation_quality": 0.0,
                "clarity": 0.0,
            },
            "overall_quality": 0.0,
            "reflection_feedback": "No answer was generated. Need to retry retrieval.",
            "should_continue": True,
            "iteration_count": iteration_count + 1,
        }
    
    try:
        # Initialize LLM
        llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
        
        # Format sources for evaluation
        sources_text = "\n".join([
            f"[{c['number']}] {c['title']} ({c['source_type']})"
            for c in citations
        ])
        
        # Get reflection prompt
        prompt = get_prompt(
            "reflector",
            query=query,
            answer=answer,
            sources=sources_text or "No sources cited",
        )
        
        # Get LLM response
        response = await llm.ainvoke(prompt)
        
        # Parse JSON response
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        reflection = json.loads(content.strip())
        
        # Extract scores
        scores = reflection.get("scores", {})
        quality_scores = QualityScores(
            relevance=float(scores.get("relevance", 0.5)),
            completeness=float(scores.get("completeness", 0.5)),
            accuracy=float(scores.get("accuracy", 0.5)),
            citation_quality=float(scores.get("citation_quality", 0.5)),
            clarity=float(scores.get("clarity", 0.5)),
        )
        
        overall_quality = float(reflection.get("overall_score", 0.5))
        passed = reflection.get("passed", overall_quality >= settings.quality_threshold)
        feedback = reflection.get("feedback", "")
        missing_aspects = reflection.get("missing_aspects", [])
        
        logger.info(f"Quality score: {overall_quality:.2f} (threshold: {settings.quality_threshold})")
        logger.info(f"Passed: {passed}")
        
        if passed:
            return {
                "quality_scores": quality_scores,
                "overall_quality": overall_quality,
                "reflection_feedback": feedback,
                "missing_aspects": missing_aspects,
                "final_answer": answer,
                "should_continue": False,
                "iteration_count": iteration_count + 1,
            }
        else:
            return {
                "quality_scores": quality_scores,
                "overall_quality": overall_quality,
                "reflection_feedback": feedback,
                "missing_aspects": missing_aspects,
                "should_continue": True,
                "iteration_count": iteration_count + 1,
            }
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse reflection: {e}")
        # Default to accepting the answer
        return {
            "quality_scores": {
                "relevance": 0.7,
                "completeness": 0.7,
                "accuracy": 0.7,
                "citation_quality": 0.7,
                "clarity": 0.7,
            },
            "overall_quality": 0.7,
            "final_answer": answer,
            "should_continue": False,
            "iteration_count": iteration_count + 1,
        }
        
    except Exception as e:
        logger.error(f"Reflection failed: {e}")
        return {
            "final_answer": answer,
            "should_continue": False,
            "iteration_count": iteration_count + 1,
            "error": str(e),
        }
