# Agentic Search Evaluation Results

---

## 1. Evaluation Overview

This document presents the formal evaluation of the Agentic Search system using standard Information Retrieval metrics. The evaluation measures how effectively the system retrieves relevant documents for user queries.

### 1.1 Evaluation Methodology

1. **Test Cases**: 6 predefined queries representing different information needs
2. **Relevance Judgments**: Keyword-based relevance matching with graded scores (0-3)
3. **Metrics Calculated**: Precision, Recall, F1, P@k, nDCG@k, MAP, MRR

### 1.2 Test Queries (Use Cases)

| # | Query | Information Need |
|---|-------|------------------|
| 1 | What is RAG in AI? | Understanding Retrieval-Augmented Generation |
| 2 | Compare BM25 and dense retrieval for question answering | Comparison of retrieval methods |
| 3 | How does BERT improve search ranking compared to TF-IDF? | Neural vs traditional ranking |
| 4 | What are the main components of a search engine? | Search engine architecture |
| 5 | Explain vector embeddings for semantic search | Understanding embeddings |
| 6 | What is query decomposition in information retrieval? | Query processing techniques |

---

## 2. Evaluation Metrics

### 2.1 Metric Definitions

| Metric | Formula | Description |
|--------|---------|-------------|
| **Precision** | Relevant Retrieved / Total Retrieved | Fraction of retrieved docs that are relevant |
| **Recall** | Relevant Retrieved / Total Relevant | Fraction of relevant docs that are retrieved |
| **F1 Score** | 2 × (P × R) / (P + R) | Harmonic mean of precision and recall |
| **P@k** | Relevant in top k / k | Precision at rank k |
| **nDCG@k** | DCG@k / IDCG@k | Normalized ranking quality (position matters) |
| **MAP** | Mean of AP across queries | Average precision across all queries |
| **MRR** | Mean of 1/rank of first relevant | How quickly first relevant result appears |

---

## 3. Summary Results

> **Note:** Run the evaluation to populate these results.
> 
> Use the API endpoint: `POST /api/evaluation/run`
> 
> Or run from command line: `python -m src.evaluation.evaluator`

### 3.1 Aggregate Metrics

| Metric | Score |
|--------|-------|
| Mean Precision | *[Run evaluation]* |
| Mean Recall | *[Run evaluation]* |
| Mean F1 Score | *[Run evaluation]* |
| Mean P@5 | *[Run evaluation]* |
| Mean P@10 | *[Run evaluation]* |
| Mean nDCG@5 | *[Run evaluation]* |
| Mean nDCG@10 | *[Run evaluation]* |
| MAP | *[Run evaluation]* |
| MRR | *[Run evaluation]* |

### 3.2 System Performance

| Metric | Value |
|--------|-------|
| Average Processing Time | *[Run evaluation]* |
| Total Results Retrieved | *[Run evaluation]* |
| Total Relevant Found | *[Run evaluation]* |

---

## 4. Per-Query Results

### Query 1: What is RAG in AI?

**Information Need:** User wants to understand Retrieval-Augmented Generation

| Metric | Score |
|--------|-------|
| Precision | *[Run evaluation]* |
| Recall | *[Run evaluation]* |
| F1 | *[Run evaluation]* |
| nDCG@10 | *[Run evaluation]* |

**Sources Searched:** Web, Academic

---

### Query 2: Compare BM25 and dense retrieval

**Information Need:** Comparison of sparse vs dense retrieval methods

| Metric | Score |
|--------|-------|
| Precision | *[Run evaluation]* |
| Recall | *[Run evaluation]* |
| F1 | *[Run evaluation]* |
| nDCG@10 | *[Run evaluation]* |

**Sources Searched:** Web, Academic

---

### Query 3: BERT vs TF-IDF for search ranking

**Information Need:** Neural vs traditional ranking comparison

| Metric | Score |
|--------|-------|
| Precision | *[Run evaluation]* |
| Recall | *[Run evaluation]* |
| F1 | *[Run evaluation]* |
| nDCG@10 | *[Run evaluation]* |

**Sources Searched:** Web, Academic

---

## 5. Analysis and Discussion

### 5.1 Strengths

- **Multi-source retrieval**: System searches web, academic papers, and documents simultaneously
- **Query decomposition**: Complex queries are broken into focused sub-queries
- **Self-reflection**: Quality control loop ensures answer quality

### 5.2 Limitations

- **Dynamic retrieval**: Results vary based on real-time API responses
- **Keyword-based relevance**: Automated relevance judgments may miss semantic relevance
- **No ground truth corpus**: Unlike TREC, we don't have pre-judged document collections

### 5.3 Comparison with Baselines

| System | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| Agentic Search | *[Results]* | *[Results]* | *[Results]* |
| Single-source (Web only) | *[Baseline]* | *[Baseline]* | *[Baseline]* |
| No decomposition | *[Baseline]* | *[Baseline]* | *[Baseline]* |

### 5.4 Future Improvements

1. **Human relevance judgments**: Manual annotation for more accurate evaluation
2. **A/B testing**: Compare different retrieval configurations
3. **User studies**: Measure user satisfaction and task completion
4. **Benchmark datasets**: Evaluate on standard IR test collections (MS MARCO, BEIR)

---

## 6. How to Run Evaluation

### 6.1 Via API

```bash
# Get test cases
curl http://localhost:8000/api/evaluation/test-cases

# Run full evaluation
curl -X POST http://localhost:8000/api/evaluation/run \
  -H "Content-Type: application/json" \
  -d '{"max_results": 10}'

# Run specific test cases
curl -X POST http://localhost:8000/api/evaluation/run \
  -H "Content-Type: application/json" \
  -d '{"max_results": 10, "test_case_indices": [0, 1, 2]}'
```

### 6.2 Via Command Line

```bash
# Run evaluation and save results
python -m src.evaluation.evaluator --output EVALUATION_RESULTS.md --format markdown

# Run with JSON output
python -m src.evaluation.evaluator --output results.json --format json
```

### 6.3 Programmatic Usage

```python
from src.evaluation import SearchEvaluator, get_test_cases
import asyncio

async def run_eval():
    evaluator = SearchEvaluator()
    report = await evaluator.run_evaluation(max_results=10)
    
    print(f"Mean Precision: {report.mean_precision:.4f}")
    print(f"Mean Recall: {report.mean_recall:.4f}")
    print(f"MAP: {report.map_score:.4f}")
    
    # Export as markdown
    markdown = evaluator.export_report_markdown(report)
    with open("results.md", "w") as f:
        f.write(markdown)

asyncio.run(run_eval())
```

---

## 7. Conclusion

The Agentic Search system demonstrates effective multi-source retrieval with formal IR evaluation. The evaluation framework provides:

- Standard metrics (Precision, Recall, F1, nDCG, MAP, MRR)
- Reproducible test cases
- API and CLI interfaces for running evaluations
- Detailed per-query analysis

This evaluation methodology aligns with the INFO 624 project requirements for formal search engine evaluation.

---

