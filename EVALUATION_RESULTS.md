# Comprehensive Evaluation Report

**Generated:** 2026-03-10T02:15:07.582114

**Queries:** 5 | **Total Time:** 467774ms

---

## Layer 1: IR Retrieval Metrics

| Metric | Score |
|--------|-------|
| mean_precision | 0.3200 |
| mean_recall | 0.8000 |
| mean_f1 | 0.4105 |
| mean_p_at_5 | 0.2800 |
| mean_ndcg_at_5 | 0.4334 |
| mean_ndcg_at_10 | 0.6639 |
| map | 2.0616 |
| mrr | 0.4556 |

**Source Breakdown:**

- web: 5 results (3.9%)
- documents: 55 results (43.3%)
- academic: 67 results (52.8%)

---

## Layer 2: Text Generation Quality

| Metric | Score |
|--------|-------|
| mean_bert_f1 | 0.7914 |
| mean_bert_precision | 0.7802 |
| mean_bert_recall | 0.8031 |
| mean_bleu_1 | 0.1689 |
| mean_bleu_2 | 0.0786 |
| mean_bleu_4 | 0.0255 |
| mean_rouge_1_f | 0.3064 |
| mean_rouge_2_f | 0.0832 |
| mean_rouge_l_f | 0.1497 |

---

## Layer 3: Google Baseline Comparison

| Metric | Score |
|--------|-------|
| mean_domain_overlap | 0.0650 |
| mean_content_overlap | 0.0625 |
| mean_bert_vs_google | 0.7842 |

---

## Layer 4: LLM-as-Judge Evaluation

| Dimension | Score |
|-----------|-------|
| mean_relevance | 0.8000 |
| mean_completeness | 0.7000 |
| mean_faithfulness | 0.8400 |
| mean_citation_quality | 0.7400 |
| mean_coherence | 0.8800 |
| mean_factual_accuracy | 0.9000 |
| mean_overall | 0.8300 |
| mean_hallucination_rate | 0.2116 |
| overall_hallucination_rate | 0.2099 |

---

