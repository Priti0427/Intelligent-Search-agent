"""
Search Evaluator.

This module runs the evaluation pipeline:
1. Execute test queries through the search system
2. Collect retrieved results
3. Judge relevance using test case criteria
4. Calculate evaluation metrics
5. Generate evaluation report
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict

from src.evaluation.metrics import (
    precision,
    recall,
    f1_score,
    precision_at_k,
    recall_at_k,
    binary_ndcg_at_k,
    ndcg_at_k,
    average_precision,
    mean_average_precision,
    reciprocal_rank,
    mean_reciprocal_rank,
)
from src.evaluation.test_cases import TestCase, get_test_cases

logger = logging.getLogger(__name__)


@dataclass
class QueryEvaluationResult:
    """Evaluation results for a single query."""
    
    query: str
    information_need: str
    num_retrieved: int
    num_relevant: int
    num_relevant_retrieved: int
    precision: float
    recall: float
    f1: float
    precision_at_5: float
    precision_at_10: float
    recall_at_5: float
    recall_at_10: float
    ndcg_at_5: float
    ndcg_at_10: float
    average_precision: float
    reciprocal_rank: float
    sources_searched: List[str]
    processing_time_ms: float
    relevance_judgments: List[Dict[str, Any]]  # Individual result judgments


@dataclass
class EvaluationReport:
    """Complete evaluation report across all test cases."""
    
    timestamp: str
    num_queries: int
    
    # Aggregate metrics
    mean_precision: float
    mean_recall: float
    mean_f1: float
    mean_precision_at_5: float
    mean_precision_at_10: float
    mean_ndcg_at_5: float
    mean_ndcg_at_10: float
    map_score: float  # Mean Average Precision
    mrr_score: float  # Mean Reciprocal Rank
    
    # Per-query results
    query_results: List[QueryEvaluationResult]
    
    # System info
    avg_processing_time_ms: float
    total_results_retrieved: int
    total_relevant_found: int


class SearchEvaluator:
    """
    Evaluator for the Agentic Search system.
    
    This class runs test queries, judges relevance, and calculates metrics.
    """
    
    def __init__(self, test_cases: Optional[List[TestCase]] = None):
        """
        Initialize the evaluator.
        
        Args:
            test_cases: List of test cases to evaluate. If None, uses default cases.
        """
        self.test_cases = test_cases or get_test_cases()
        self.results: List[QueryEvaluationResult] = []
    
    async def run_evaluation(self, max_results: int = 10) -> EvaluationReport:
        """
        Run full evaluation on all test cases.
        
        Args:
            max_results: Maximum results to retrieve per query
            
        Returns:
            Complete evaluation report
        """
        from src.agent import run_search
        
        logger.info(f"Starting evaluation with {len(self.test_cases)} test cases")
        
        self.results = []
        all_retrieved_lists = []
        all_relevant_sets = []
        
        for i, test_case in enumerate(self.test_cases):
            logger.info(f"Evaluating query {i+1}/{len(self.test_cases)}: {test_case.query}")
            
            try:
                result = await self._evaluate_single_query(test_case, max_results)
                self.results.append(result)
                
                # Collect for MAP/MRR calculation
                # Use result titles as document IDs
                retrieved_ids = [
                    j["title"] for j in result.relevance_judgments
                ]
                relevant_ids = {
                    j["title"] for j in result.relevance_judgments 
                    if j["is_relevant"]
                }
                all_retrieved_lists.append(retrieved_ids)
                all_relevant_sets.append(relevant_ids)
                
            except Exception as e:
                logger.error(f"Failed to evaluate query '{test_case.query}': {e}")
                continue
        
        # Calculate aggregate metrics
        report = self._generate_report(all_retrieved_lists, all_relevant_sets)
        
        logger.info("Evaluation complete")
        return report
    
    async def _evaluate_single_query(
        self, 
        test_case: TestCase, 
        max_results: int
    ) -> QueryEvaluationResult:
        """
        Evaluate a single test query.
        
        Args:
            test_case: The test case to evaluate
            max_results: Maximum results to consider
            
        Returns:
            Evaluation result for this query
        """
        from src.agent import run_search
        import time
        
        start_time = time.time()
        
        # Run the search
        search_result = await run_search(test_case.query)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Collect all retrieved results
        all_results = []
        
        # Web results
        for r in search_result.get("web_results", []):
            all_results.append({
                "title": r.get("title", "Unknown"),
                "content": r.get("content", ""),
                "excerpt": r.get("snippet", r.get("content", ""))[:500],
                "url": r.get("url", ""),
                "source_type": "web",
            })
        
        # Vector store results
        for r in search_result.get("vector_results", []):
            all_results.append({
                "title": r.get("title", r.get("metadata", {}).get("title", "Document")),
                "content": r.get("content", r.get("page_content", "")),
                "excerpt": r.get("content", r.get("page_content", ""))[:500],
                "source_type": "documents",
            })
        
        # arXiv results
        for r in search_result.get("arxiv_results", []):
            all_results.append({
                "title": r.get("title", "Unknown"),
                "content": r.get("summary", r.get("content", "")),
                "excerpt": r.get("summary", "")[:500],
                "url": r.get("url", r.get("entry_id", "")),
                "source_type": "academic",
            })
        
        # Limit to max_results
        all_results = all_results[:max_results]
        
        # Judge relevance for each result
        relevance_judgments = []
        relevance_scores = []
        retrieved_ids = []
        relevant_ids = set()
        
        for idx, result in enumerate(all_results):
            is_relevant = test_case.is_result_relevant(result)
            relevance_score = test_case.get_relevance_score(result)
            
            result_id = f"{result['source_type']}_{idx}_{result['title'][:30]}"
            retrieved_ids.append(result_id)
            
            if is_relevant:
                relevant_ids.add(result_id)
            
            relevance_judgments.append({
                "rank": idx + 1,
                "title": result["title"],
                "source_type": result["source_type"],
                "is_relevant": is_relevant,
                "relevance_score": relevance_score,
                "excerpt": result["excerpt"][:200],
            })
            relevance_scores.append(relevance_score)
        
        # Calculate metrics
        prec = precision(retrieved_ids, relevant_ids)
        rec = recall(retrieved_ids, relevant_ids) if relevant_ids else 0.0
        
        # Get sources searched
        sources_searched = list(set(r["source_type"] for r in all_results))
        
        return QueryEvaluationResult(
            query=test_case.query,
            information_need=test_case.information_need,
            num_retrieved=len(all_results),
            num_relevant=len(relevant_ids),
            num_relevant_retrieved=len(set(retrieved_ids) & relevant_ids),
            precision=prec,
            recall=rec,
            f1=f1_score(prec, rec),
            precision_at_5=precision_at_k(retrieved_ids, relevant_ids, 5),
            precision_at_10=precision_at_k(retrieved_ids, relevant_ids, 10),
            recall_at_5=recall_at_k(retrieved_ids, relevant_ids, 5),
            recall_at_10=recall_at_k(retrieved_ids, relevant_ids, 10),
            ndcg_at_5=ndcg_at_k(relevance_scores, 5),
            ndcg_at_10=ndcg_at_k(relevance_scores, 10),
            average_precision=average_precision(retrieved_ids, relevant_ids),
            reciprocal_rank=reciprocal_rank(retrieved_ids, relevant_ids),
            sources_searched=sources_searched,
            processing_time_ms=processing_time,
            relevance_judgments=relevance_judgments,
        )
    
    def _generate_report(
        self,
        all_retrieved_lists: List[List[str]],
        all_relevant_sets: List[set],
    ) -> EvaluationReport:
        """Generate the final evaluation report."""
        
        if not self.results:
            raise ValueError("No evaluation results to report")
        
        # Calculate means
        mean_prec = sum(r.precision for r in self.results) / len(self.results)
        mean_rec = sum(r.recall for r in self.results) / len(self.results)
        mean_f1 = sum(r.f1 for r in self.results) / len(self.results)
        mean_p5 = sum(r.precision_at_5 for r in self.results) / len(self.results)
        mean_p10 = sum(r.precision_at_10 for r in self.results) / len(self.results)
        mean_ndcg5 = sum(r.ndcg_at_5 for r in self.results) / len(self.results)
        mean_ndcg10 = sum(r.ndcg_at_10 for r in self.results) / len(self.results)
        
        # MAP and MRR
        map_score = mean_average_precision(all_retrieved_lists, all_relevant_sets)
        mrr_score = mean_reciprocal_rank(all_retrieved_lists, all_relevant_sets)
        
        # Totals
        total_retrieved = sum(r.num_retrieved for r in self.results)
        total_relevant = sum(r.num_relevant_retrieved for r in self.results)
        avg_time = sum(r.processing_time_ms for r in self.results) / len(self.results)
        
        return EvaluationReport(
            timestamp=datetime.utcnow().isoformat(),
            num_queries=len(self.results),
            mean_precision=mean_prec,
            mean_recall=mean_rec,
            mean_f1=mean_f1,
            mean_precision_at_5=mean_p5,
            mean_precision_at_10=mean_p10,
            mean_ndcg_at_5=mean_ndcg5,
            mean_ndcg_at_10=mean_ndcg10,
            map_score=map_score,
            mrr_score=mrr_score,
            query_results=self.results,
            avg_processing_time_ms=avg_time,
            total_results_retrieved=total_retrieved,
            total_relevant_found=total_relevant,
        )
    
    def export_report_markdown(self, report: EvaluationReport) -> str:
        """
        Export evaluation report as Markdown.
        
        Args:
            report: The evaluation report
            
        Returns:
            Markdown formatted report
        """
        md = f"""# Agentic Search Evaluation Report

**Generated:** {report.timestamp}

**Number of Test Queries:** {report.num_queries}

---

## Summary Metrics

| Metric | Score |
|--------|-------|
| Mean Precision | {report.mean_precision:.4f} |
| Mean Recall | {report.mean_recall:.4f} |
| Mean F1 Score | {report.mean_f1:.4f} |
| Mean P@5 | {report.mean_precision_at_5:.4f} |
| Mean P@10 | {report.mean_precision_at_10:.4f} |
| Mean nDCG@5 | {report.mean_ndcg_at_5:.4f} |
| Mean nDCG@10 | {report.mean_ndcg_at_10:.4f} |
| MAP (Mean Average Precision) | {report.map_score:.4f} |
| MRR (Mean Reciprocal Rank) | {report.mrr_score:.4f} |

---

## System Performance

| Metric | Value |
|--------|-------|
| Average Processing Time | {report.avg_processing_time_ms:.2f} ms |
| Total Results Retrieved | {report.total_results_retrieved} |
| Total Relevant Found | {report.total_relevant_found} |

---

## Per-Query Results

"""
        
        for i, qr in enumerate(report.query_results, 1):
            md += f"""### Query {i}: {qr.query}

**Information Need:** {qr.information_need}

**Sources Searched:** {', '.join(qr.sources_searched)}

| Metric | Score |
|--------|-------|
| Precision | {qr.precision:.4f} |
| Recall | {qr.recall:.4f} |
| F1 | {qr.f1:.4f} |
| P@5 | {qr.precision_at_5:.4f} |
| nDCG@5 | {qr.ndcg_at_5:.4f} |
| nDCG@10 | {qr.ndcg_at_10:.4f} |
| Average Precision | {qr.average_precision:.4f} |
| Reciprocal Rank | {qr.reciprocal_rank:.4f} |
| Processing Time | {qr.processing_time_ms:.2f} ms |

**Retrieved Results:** {qr.num_retrieved} | **Relevant:** {qr.num_relevant_retrieved}

<details>
<summary>Relevance Judgments</summary>

| Rank | Title | Source | Relevant | Score |
|------|-------|--------|----------|-------|
"""
            for j in qr.relevance_judgments[:10]:
                relevant_mark = "✓" if j["is_relevant"] else "✗"
                title_short = j["title"][:40] + "..." if len(j["title"]) > 40 else j["title"]
                md += f"| {j['rank']} | {title_short} | {j['source_type']} | {relevant_mark} | {j['relevance_score']:.1f} |\n"
            
            md += """
</details>

---

"""
        
        md += """## Metric Definitions

- **Precision**: Fraction of retrieved documents that are relevant
- **Recall**: Fraction of relevant documents that are retrieved  
- **F1 Score**: Harmonic mean of precision and recall
- **P@k**: Precision at rank k
- **nDCG@k**: Normalized Discounted Cumulative Gain at rank k (considers ranking quality)
- **MAP**: Mean Average Precision across all queries
- **MRR**: Mean Reciprocal Rank (average of 1/rank of first relevant result)

---

*Report generated by Agentic Search Evaluation Module*
"""
        
        return md
    
    def export_report_json(self, report: EvaluationReport) -> str:
        """Export evaluation report as JSON."""
        
        # Convert dataclasses to dicts
        report_dict = {
            "timestamp": report.timestamp,
            "num_queries": report.num_queries,
            "summary_metrics": {
                "mean_precision": report.mean_precision,
                "mean_recall": report.mean_recall,
                "mean_f1": report.mean_f1,
                "mean_precision_at_5": report.mean_precision_at_5,
                "mean_precision_at_10": report.mean_precision_at_10,
                "mean_ndcg_at_5": report.mean_ndcg_at_5,
                "mean_ndcg_at_10": report.mean_ndcg_at_10,
                "map": report.map_score,
                "mrr": report.mrr_score,
            },
            "system_performance": {
                "avg_processing_time_ms": report.avg_processing_time_ms,
                "total_results_retrieved": report.total_results_retrieved,
                "total_relevant_found": report.total_relevant_found,
            },
            "query_results": [asdict(qr) for qr in report.query_results],
        }
        
        return json.dumps(report_dict, indent=2)


async def run_evaluation_cli():
    """Run evaluation from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Agentic Search")
    parser.add_argument("--output", "-o", default="EVALUATION_RESULTS.md",
                       help="Output file for results")
    parser.add_argument("--format", "-f", choices=["markdown", "json"], 
                       default="markdown", help="Output format")
    parser.add_argument("--max-results", "-m", type=int, default=10,
                       help="Max results per query")
    
    args = parser.parse_args()
    
    evaluator = SearchEvaluator()
    report = await evaluator.run_evaluation(max_results=args.max_results)
    
    if args.format == "markdown":
        output = evaluator.export_report_markdown(report)
    else:
        output = evaluator.export_report_json(report)
    
    with open(args.output, "w") as f:
        f.write(output)
    
    print(f"Evaluation complete. Results saved to {args.output}")
    print(f"\nSummary:")
    print(f"  Mean Precision: {report.mean_precision:.4f}")
    print(f"  Mean Recall: {report.mean_recall:.4f}")
    print(f"  Mean F1: {report.mean_f1:.4f}")
    print(f"  MAP: {report.map_score:.4f}")
    print(f"  MRR: {report.mrr_score:.4f}")


if __name__ == "__main__":
    asyncio.run(run_evaluation_cli())
