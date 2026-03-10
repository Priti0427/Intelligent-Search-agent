"""
Comprehensive Search Evaluator.

Orchestrates all evaluation layers:
1. IR Retrieval Metrics (Precision, Recall, F1, nDCG, MAP, MRR)
2. Text Generation Quality (BERTScore, BLEU, ROUGE, Perplexity)
3. Google Baseline Comparison
4. LLM-as-Judge Evaluation
5. Robustness Testing (optional)

Also supports ablation study: with vs without vector store.
"""

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.evaluation.metrics import (
    precision, recall, f1_score, precision_at_k, recall_at_k,
    binary_ndcg_at_k, ndcg_at_k, average_precision,
    mean_average_precision, reciprocal_rank, mean_reciprocal_rank,
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
    relevance_judgments: List[Dict[str, Any]]


@dataclass
class EvaluationReport:
    """Complete evaluation report across all test cases."""
    timestamp: str
    num_queries: int
    mean_precision: float
    mean_recall: float
    mean_f1: float
    mean_precision_at_5: float
    mean_precision_at_10: float
    mean_ndcg_at_5: float
    mean_ndcg_at_10: float
    map_score: float
    mrr_score: float
    query_results: List[QueryEvaluationResult]
    avg_processing_time_ms: float
    total_results_retrieved: int
    total_relevant_found: int


class SearchEvaluator:
    """Basic IR metrics evaluator (Layer 1). Kept for backward compatibility."""

    def __init__(self, test_cases: Optional[List[TestCase]] = None):
        self.test_cases = test_cases or get_test_cases()
        self.results: List[QueryEvaluationResult] = []

    async def run_evaluation(self, max_results: int = 10) -> EvaluationReport:
        from src.agent import run_search
        self.results = []
        all_retrieved_lists = []
        all_relevant_sets = []

        for i, test_case in enumerate(self.test_cases):
            logger.info(f"Evaluating query {i+1}/{len(self.test_cases)}: {test_case.query}")
            try:
                result = await self._evaluate_single_query(test_case, max_results)
                self.results.append(result)
                retrieved_ids = [j["title"] for j in result.relevance_judgments]
                relevant_ids = {j["title"] for j in result.relevance_judgments if j["is_relevant"]}
                all_retrieved_lists.append(retrieved_ids)
                all_relevant_sets.append(relevant_ids)
            except Exception as e:
                logger.error(f"Failed to evaluate query '{test_case.query}': {e}")

        return self._generate_report(all_retrieved_lists, all_relevant_sets)

    async def _evaluate_single_query(self, test_case, max_results):
        from src.agent import run_search
        start_time = time.time()
        search_result = await run_search(test_case.query)
        processing_time = (time.time() - start_time) * 1000

        all_results = _collect_results(search_result)[:max_results]
        return _compute_ir_metrics(test_case, all_results, processing_time)

    def _generate_report(self, all_retrieved_lists, all_relevant_sets):
        if not self.results:
            raise ValueError("No evaluation results to report")
        n = len(self.results)
        return EvaluationReport(
            timestamp=datetime.utcnow().isoformat(),
            num_queries=n,
            mean_precision=sum(r.precision for r in self.results) / n,
            mean_recall=sum(r.recall for r in self.results) / n,
            mean_f1=sum(r.f1 for r in self.results) / n,
            mean_precision_at_5=sum(r.precision_at_5 for r in self.results) / n,
            mean_precision_at_10=sum(r.precision_at_10 for r in self.results) / n,
            mean_ndcg_at_5=sum(r.ndcg_at_5 for r in self.results) / n,
            mean_ndcg_at_10=sum(r.ndcg_at_10 for r in self.results) / n,
            map_score=mean_average_precision(all_retrieved_lists, all_relevant_sets),
            mrr_score=mean_reciprocal_rank(all_retrieved_lists, all_relevant_sets),
            query_results=self.results,
            avg_processing_time_ms=sum(r.processing_time_ms for r in self.results) / n,
            total_results_retrieved=sum(r.num_retrieved for r in self.results),
            total_relevant_found=sum(r.num_relevant_retrieved for r in self.results),
        )

    def export_report_markdown(self, report: EvaluationReport) -> str:
        md = f"# Agentic Search Evaluation Report\n\n"
        md += f"**Generated:** {report.timestamp}\n\n"
        md += f"**Queries:** {report.num_queries}\n\n---\n\n"
        md += "## Summary Metrics\n\n| Metric | Score |\n|--------|-------|\n"
        for name, val in [
            ("Mean Precision", report.mean_precision), ("Mean Recall", report.mean_recall),
            ("Mean F1", report.mean_f1), ("Mean P@5", report.mean_precision_at_5),
            ("Mean P@10", report.mean_precision_at_10), ("Mean nDCG@5", report.mean_ndcg_at_5),
            ("Mean nDCG@10", report.mean_ndcg_at_10), ("MAP", report.map_score),
            ("MRR", report.mrr_score),
        ]:
            md += f"| {name} | {val:.4f} |\n"
        md += f"\n---\n\n## Performance\n\n"
        md += f"- Avg Processing Time: {report.avg_processing_time_ms:.2f} ms\n"
        md += f"- Total Retrieved: {report.total_results_retrieved}\n"
        md += f"- Total Relevant: {report.total_relevant_found}\n\n"
        for i, qr in enumerate(report.query_results, 1):
            md += f"### Query {i}: {qr.query}\n\n"
            md += f"P={qr.precision:.3f} R={qr.recall:.3f} F1={qr.f1:.3f} "
            md += f"P@5={qr.precision_at_5:.3f} nDCG@5={qr.ndcg_at_5:.3f} "
            md += f"AP={qr.average_precision:.3f} RR={qr.reciprocal_rank:.3f}\n\n"
        return md

    def export_report_json(self, report: EvaluationReport) -> str:
        return json.dumps({
            "timestamp": report.timestamp,
            "num_queries": report.num_queries,
            "summary_metrics": {
                "mean_precision": report.mean_precision,
                "mean_recall": report.mean_recall, "mean_f1": report.mean_f1,
                "map": report.map_score, "mrr": report.mrr_score,
            },
            "query_results": [asdict(qr) for qr in report.query_results],
        }, indent=2)


class ComprehensiveEvaluator:
    """
    Orchestrates all 5 evaluation layers into a unified evaluation pipeline.
    """

    def __init__(self, test_cases: Optional[List[TestCase]] = None):
        self.test_cases = test_cases or get_test_cases()

    async def run_full_evaluation(
        self,
        max_results: int = 10,
        run_generation_metrics: bool = True,
        run_google_comparison: bool = True,
        run_judge: bool = True,
        run_ragas: bool = True,
        run_robustness: bool = False,
        compute_perplexity: bool = False,
    ) -> Dict[str, Any]:
        """Run the complete multi-layer evaluation."""
        from src.agent import run_search

        logger.info(f"Starting comprehensive evaluation with {len(self.test_cases)} queries")
        overall_start = time.time()

        # --- Run all queries through the pipeline ---
        pipeline_results = []
        for i, tc in enumerate(self.test_cases):
            logger.info(f"[{i+1}/{len(self.test_cases)}] Running: {tc.query[:60]}...")
            start = time.time()
            try:
                result = await run_search(tc.query)
            except Exception as e:
                logger.error(f"Pipeline failed for query: {e}")
                result = {}
            elapsed = (time.time() - start) * 1000
            pipeline_results.append({"result": result, "time_ms": elapsed, "test_case": tc})

        # === Layer 1: IR Retrieval Metrics ===
        logger.info("Computing Layer 1: IR Metrics")
        layer1 = self._compute_ir_layer(pipeline_results, max_results)

        # === Layer 2: Generation Quality ===
        layer2 = None
        if run_generation_metrics:
            logger.info("Computing Layer 2: Generation Metrics")
            layer2 = self._compute_generation_layer(pipeline_results, compute_perplexity)

        # === Layer 3: Google Baseline ===
        layer3 = None
        if run_google_comparison:
            logger.info("Computing Layer 3: Google Baseline")
            layer3 = self._compute_google_layer(pipeline_results, max_results)

        # === Layer 4: LLM Judge ===
        layer4 = None
        if run_judge:
            logger.info("Computing Layer 4: LLM Judge")
            layer4 = await self._compute_judge_layer(pipeline_results, max_results)

        # === Layer 5: RAGAS ===
        layer5_ragas = None
        if run_ragas:
            logger.info("Computing Layer 5: RAGAS Metrics")
            layer5_ragas = await self._compute_ragas_layer(pipeline_results, max_results)

        # === Layer 6: Robustness ===
        layer6 = None
        if run_robustness:
            logger.info("Computing Layer 6: Robustness")
            layer6 = await self._compute_robustness_layer()

        total_time = (time.time() - overall_start) * 1000

        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "num_queries": len(self.test_cases),
            "total_evaluation_time_ms": total_time,
            "layer1_ir_metrics": layer1,
            "layer2_generation_metrics": layer2,
            "layer3_google_baseline": layer3,
            "layer4_judge": layer4,
            "layer5_ragas": layer5_ragas,
            "layer6_robustness": layer6,
        }

        logger.info(f"Comprehensive evaluation complete in {total_time:.0f}ms")
        return report

    def _compute_ir_layer(self, pipeline_results, max_results):
        per_query = []
        all_retrieved, all_relevant = [], []
        for pr in pipeline_results:
            tc = pr["test_case"]
            results = _collect_results(pr["result"])[:max_results]
            qr = _compute_ir_metrics(tc, results, pr["time_ms"])
            per_query.append(asdict(qr))
            ids = [j["title"] for j in qr.relevance_judgments]
            rel = {j["title"] for j in qr.relevance_judgments if j["is_relevant"]}
            all_retrieved.append(ids)
            all_relevant.append(rel)

        n = len(per_query) or 1
        return {
            "per_query": per_query,
            "aggregate": {
                "mean_precision": sum(q["precision"] for q in per_query) / n,
                "mean_recall": sum(q["recall"] for q in per_query) / n,
                "mean_f1": sum(q["f1"] for q in per_query) / n,
                "mean_p_at_5": sum(q["precision_at_5"] for q in per_query) / n,
                "mean_ndcg_at_5": sum(q["ndcg_at_5"] for q in per_query) / n,
                "mean_ndcg_at_10": sum(q["ndcg_at_10"] for q in per_query) / n,
                "map": mean_average_precision(all_retrieved, all_relevant),
                "mrr": mean_reciprocal_rank(all_retrieved, all_relevant),
            },
            "per_source_breakdown": self._source_breakdown(pipeline_results),
        }

    def _source_breakdown(self, pipeline_results):
        totals = {"web": 0, "documents": 0, "academic": 0}
        for pr in pipeline_results:
            r = pr["result"]
            totals["web"] += len(r.get("web_results", []))
            totals["documents"] += len(r.get("vector_results", []))
            totals["academic"] += len(r.get("arxiv_results", []))
        grand = sum(totals.values()) or 1
        return {src: {"count": c, "percentage": c / grand * 100} for src, c in totals.items()}

    def _compute_generation_layer(self, pipeline_results, compute_ppl):
        from src.evaluation.generation_metrics import compute_batch_generation_metrics
        candidates, references = [], []
        for pr in pipeline_results:
            r = pr["result"]
            answer = r.get("final_answer", r.get("draft_answer", ""))
            candidates.append(answer)
            references.append(pr["test_case"].reference_answer)
        per_query, aggregate = compute_batch_generation_metrics(candidates, references, compute_ppl)
        return {"per_query": per_query, "aggregate": aggregate}

    def _compute_google_layer(self, pipeline_results, max_results):
        from src.evaluation.google_baseline import evaluate_against_google
        per_query = []
        for pr in pipeline_results:
            r = pr["result"]
            answer = r.get("final_answer", r.get("draft_answer", ""))
            results = _collect_results(r)[:max_results]
            comparison = evaluate_against_google(pr["test_case"].query, results, answer)
            per_query.append(comparison)
        valid = [q for q in per_query if q is not None]
        agg = {}
        if valid:
            n = len(valid)
            agg["mean_domain_overlap"] = sum(q["source_overlap"]["domain_overlap"] for q in valid) / n
            agg["mean_content_overlap"] = sum(q["source_overlap"]["content_overlap"] for q in valid) / n
            agg["mean_bert_vs_google"] = sum(q["answer_comparison"]["bert_score_vs_google"] for q in valid) / n
        return {"per_query": per_query, "aggregate": agg}

    async def _compute_judge_layer(self, pipeline_results, max_results):
        from src.evaluation.judge_agent import run_batch_judge_evaluation
        queries, needs, answers, sources_list, refs = [], [], [], [], []
        for pr in pipeline_results:
            r = pr["result"]
            tc = pr["test_case"]
            queries.append(tc.query)
            needs.append(tc.information_need)
            answers.append(r.get("final_answer", r.get("draft_answer", "")))
            sources_list.append(_collect_results(r)[:max_results])
            refs.append(tc.reference_answer)
        per_query, aggregate = await run_batch_judge_evaluation(queries, needs, answers, sources_list, refs)
        return {"per_query": per_query, "aggregate": aggregate}

    async def _compute_robustness_layer(self):
        from src.agent import run_search
        from src.evaluation.robustness import run_full_robustness_evaluation
        queries = [tc.query for tc in self.test_cases]
        return await run_full_robustness_evaluation(queries, run_search)

    async def _compute_ragas_layer(self, pipeline_results, max_results):
        from src.evaluation.ragas_evaluation import run_ragas_evaluation
        queries, answers, contexts_list, refs = [], [], [], []
        for pr in pipeline_results:
            r = pr["result"]
            tc = pr["test_case"]
            queries.append(tc.query)
            answers.append(r.get("final_answer", r.get("draft_answer", "")))
            all_results = _collect_results(r)[:max_results]
            contexts = [res.get("content", res.get("excerpt", ""))[:1000] for res in all_results]
            contexts_list.append(contexts if contexts else ["No context retrieved."])
            refs.append(tc.reference_answer)
        return await run_ragas_evaluation(queries, answers, contexts_list, refs)

    def export_comprehensive_markdown(self, report: Dict) -> str:
        """Export the full multi-layer report as Markdown."""
        md = "# Comprehensive Evaluation Report\n\n"
        md += f"**Generated:** {report['timestamp']}\n\n"
        md += f"**Queries:** {report['num_queries']} | "
        md += f"**Total Time:** {report['total_evaluation_time_ms']:.0f}ms\n\n---\n\n"

        # Layer 1
        l1 = report.get("layer1_ir_metrics")
        if l1:
            md += "## Layer 1: IR Retrieval Metrics\n\n"
            agg = l1["aggregate"]
            md += "| Metric | Score |\n|--------|-------|\n"
            for k, v in agg.items():
                md += f"| {k} | {v:.4f} |\n"
            bd = l1.get("per_source_breakdown", {})
            if bd:
                md += "\n**Source Breakdown:**\n\n"
                for src, info in bd.items():
                    md += f"- {src}: {info['count']} results ({info['percentage']:.1f}%)\n"
            md += "\n---\n\n"

        # Layer 2
        l2 = report.get("layer2_generation_metrics")
        if l2:
            md += "## Layer 2: Text Generation Quality\n\n"
            agg = l2["aggregate"]
            md += "| Metric | Score |\n|--------|-------|\n"
            for k, v in agg.items():
                md += f"| {k} | {v:.4f} |\n"
            md += "\n---\n\n"

        # Layer 3
        l3 = report.get("layer3_google_baseline")
        if l3:
            md += "## Layer 3: Google Baseline Comparison\n\n"
            agg = l3.get("aggregate", {})
            if agg:
                md += "| Metric | Score |\n|--------|-------|\n"
                for k, v in agg.items():
                    md += f"| {k} | {v:.4f} |\n"
            md += "\n---\n\n"

        # Layer 4
        l4 = report.get("layer4_judge")
        if l4:
            md += "## Layer 4: LLM-as-Judge Evaluation\n\n"
            agg = l4.get("aggregate", {})
            md += "| Dimension | Score |\n|-----------|-------|\n"
            for k, v in agg.items():
                md += f"| {k} | {v:.4f} |\n"
            md += "\n---\n\n"

        # Layer 5 RAGAS
        l5r = report.get("layer5_ragas")
        if l5r and not l5r.get("error"):
            md += "## Layer 5: RAGAS Evaluation\n\n"
            agg = l5r.get("aggregate", {})
            if agg:
                md += "| Metric | Score |\n|--------|-------|\n"
                for k, v in agg.items():
                    md += f"| {k} | {v:.4f} |\n"
            md += "\n---\n\n"

        # Layer 6 Robustness
        l6 = report.get("layer6_robustness")
        if l6:
            md += "## Layer 6: Robustness Testing\n\n"
            agg = l6.get("aggregate", {})
            for k, v in agg.items():
                md += f"- {k}: {v:.4f}\n"
            md += "\n---\n\n"

        md += "*Report generated by Agentic Search Comprehensive Evaluation*\n"
        return md


# === Helper Functions ===

def _collect_results(search_result: Dict) -> List[Dict]:
    """Collect all retrieved results from a search result dict."""
    all_results = []
    for r in search_result.get("web_results", []):
        all_results.append({
            "title": r.get("title", "Unknown"), "content": r.get("content", ""),
            "excerpt": r.get("snippet", r.get("content", ""))[:500],
            "url": r.get("url", ""), "source_type": "web",
        })
    for r in search_result.get("vector_results", []):
        all_results.append({
            "title": r.get("title", r.get("metadata", {}).get("title", "Document")),
            "content": r.get("content", r.get("page_content", "")),
            "excerpt": r.get("content", r.get("page_content", ""))[:500],
            "source_type": "documents",
        })
    for r in search_result.get("arxiv_results", []):
        all_results.append({
            "title": r.get("title", "Unknown"),
            "content": r.get("summary", r.get("content", "")),
            "excerpt": r.get("summary", "")[:500],
            "url": r.get("url", r.get("entry_id", "")), "source_type": "academic",
        })
    return all_results


def _compute_ir_metrics(test_case: TestCase, all_results: List[Dict], processing_time: float) -> QueryEvaluationResult:
    """Compute all IR metrics for a single query."""
    relevance_judgments = []
    relevance_scores = []
    retrieved_ids = []
    relevant_ids = set()

    for idx, result in enumerate(all_results):
        is_relevant = test_case.is_result_relevant(result)
        rel_score = test_case.get_relevance_score(result)
        result_id = f"{result.get('source_type', 'unknown')}_{idx}_{result.get('title', '')[:30]}"
        retrieved_ids.append(result_id)
        if is_relevant:
            relevant_ids.add(result_id)
        relevance_judgments.append({
            "rank": idx + 1, "title": result.get("title", ""),
            "source_type": result.get("source_type", ""), "is_relevant": is_relevant,
            "relevance_score": rel_score, "excerpt": result.get("excerpt", "")[:200],
        })
        relevance_scores.append(rel_score)

    prec = precision(retrieved_ids, relevant_ids)
    rec = recall(retrieved_ids, relevant_ids) if relevant_ids else 0.0
    sources_searched = list(set(r.get("source_type", "unknown") for r in all_results))

    return QueryEvaluationResult(
        query=test_case.query, information_need=test_case.information_need,
        num_retrieved=len(all_results), num_relevant=len(relevant_ids),
        num_relevant_retrieved=len(set(retrieved_ids) & relevant_ids),
        precision=prec, recall=rec, f1=f1_score(prec, rec),
        precision_at_5=precision_at_k(retrieved_ids, relevant_ids, 5),
        precision_at_10=precision_at_k(retrieved_ids, relevant_ids, 10),
        recall_at_5=recall_at_k(retrieved_ids, relevant_ids, 5),
        recall_at_10=recall_at_k(retrieved_ids, relevant_ids, 10),
        ndcg_at_5=ndcg_at_k(relevance_scores, 5),
        ndcg_at_10=ndcg_at_k(relevance_scores, 10),
        average_precision=average_precision(retrieved_ids, relevant_ids),
        reciprocal_rank=reciprocal_rank(retrieved_ids, relevant_ids),
        sources_searched=sources_searched, processing_time_ms=processing_time,
        relevance_judgments=relevance_judgments,
    )


async def run_evaluation_cli():
    """Run evaluation from command line."""
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Agentic Search")
    parser.add_argument("--output", "-o", default="EVALUATION_RESULTS.md")
    parser.add_argument("--comprehensive", "-c", action="store_true")
    parser.add_argument("--max-results", "-m", type=int, default=10)
    args = parser.parse_args()

    if args.comprehensive:
        evaluator = ComprehensiveEvaluator()
        report = await evaluator.run_full_evaluation(max_results=args.max_results)
        output = evaluator.export_comprehensive_markdown(report)
    else:
        evaluator = SearchEvaluator()
        report = await evaluator.run_evaluation(max_results=args.max_results)
        output = evaluator.export_report_markdown(report)

    with open(args.output, "w") as f:
        f.write(output)
    print(f"Evaluation complete. Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(run_evaluation_cli())
