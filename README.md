# Intelligent Agentic Search

---

An intelligent multi-source search agent built with LangGraph and LangChain

## Project Overview

This project implements a sophisticated search agent that:
1. **Decomposes complex queries** into manageable sub-queries
2. **Routes queries** to appropriate sources (web, vector DB, academic papers)
3. **Retrieves and ranks** results from multiple indexes in parallel
4. **Synthesizes comprehensive answers** with proper citations
5. **Self-reflects** to improve answer quality through iterative refinement

## Problem Statement

Traditional search engines return a list of links, leaving users to manually:
- Read through multiple documents
- Extract relevant information
- Synthesize findings into coherent answers
- Verify accuracy across sources

This project automates the entire research workflow using an agentic AI system.

## Architecture

```
User Query → Query Analyzer → Query Decomposer → Router
                                                    ↓
                              ┌─────────────────────┼─────────────────────┐
                              ↓                     ↓                     ↓
                         Web Search            Vector DB            arXiv API
                         (Tavily)             (ChromaDB)           (Academic)
                              └─────────────────────┼─────────────────────┘
                                                    ↓
                                          Result Aggregator
                                                    ↓
                                          Answer Synthesizer
                                                    ↓
                                    ┌───── Self-Reflection Loop ─────┐
                                    │                                │
                                    │  Quality < 0.7? ──► Retry     │
                                    │  Quality ≥ 0.7? ──► Done      │
                                    └────────────────────────────────┘
                                                    ↓
                                    Final Answer with Citations
```

## Key Features

### Multi-Source Retrieval
- **Web Search (Tavily)**: Real-time information from the internet
- **Academic Papers (arXiv)**: Research papers and technical content
- **Custom Documents (ChromaDB)**: Your own PDFs, docs, and notes

### Self-Reflection Loop
The agent evaluates its own answers on 5 dimensions:
- Relevance: Does it answer the question?
- Completeness: Are all aspects covered?
- Accuracy: Is it consistent with sources?
- Citation Quality: Are sources properly cited?
- Clarity: Is it well-structured?

If quality score < 0.7, the agent iterates with specific feedback to improve.

### Intelligent Query Processing
- Classifies queries as simple, complex, or multi-hop
- Decomposes complex queries into focused sub-queries
- Routes sub-queries to the most appropriate sources

## Tech Stack Used

| Component | Technology |
|-----------|------------|
| Agent Framework | LangGraph |
| LLM Orchestration | LangChain |
| LLM Provider | Groq (Llama 3.1 8B) - Free |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) - Free |
| Vector Database | ChromaDB |
| Web Search | Tavily API |
| Academic Search | arXiv API |
| Backend | FastAPI |
| Frontend | HTML/CSS/JavaScript |

## Setup

### 1. Clone and Install

```bash
# Clone the repository
git clone https://github.com/Priti0427/Intelligent-Search-agent.git
cd Intelligent-Search-agent

# Create virtual environment (use Python 3.11)
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys:
# - GROQ_API_KEY (free from https://console.groq.com)
# - TAVILY_API_KEY (free from https://tavily.com)
```

### 3. Run the Application

```bash
# Start the FastAPI server
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 4. Access the Interface

- **Chat Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## Project Structure

```
agentic-search/
├── src/
│   ├── agent/                # LangGraph agent definition
│   │   ├── graph.py          # Main workflow
│   │   ├── state.py          # State schema
│   │   └── nodes/            # Individual nodes
│   │       ├── query_analyzer.py
│   │       ├── query_decomposer.py
│   │       ├── router.py
│   │       ├── retriever.py
│   │       ├── synthesizer.py
│   │       └── reflector.py
│   ├── retrievers/           # Data source integrations
│   │   ├── web_search.py     # Tavily integration
│   │   ├── vector_store.py   # ChromaDB integration
│   │   └── arxiv_search.py   # arXiv integration
│   ├── ingestion/            # Document processing
│   │   ├── document_loader.py
│   │   ├── chunker.py
│   │   └── embedder.py
│   ├── evaluation/           # Comprehensive evaluation framework
│   │   ├── evaluator.py      # Orchestrates all evaluation layers
│   │   ├── metrics.py        # IR metrics (Precision, Recall, nDCG, MAP, MRR)
│   │   ├── generation_metrics.py  # BERTScore, BLEU, ROUGE
│   │   ├── google_baseline.py     # Google baseline comparison
│   │   ├── judge_agent.py    # LLM-as-Judge evaluation
│   │   ├── ragas_evaluation.py    # RAGAS (Faithfulness, Context Precision/Recall)
│   │   ├── robustness.py     # Robustness & stress testing
│   │   └── test_cases.py     # Evaluation test cases
│   ├── api/                  # FastAPI backend
│   │   ├── main.py
│   │   ├── routes.py
│   │   └── schemas.py
│   └── utils/                # Configuration and prompts
│       ├── config.py
│       ├── llm.py
│       └── prompts.py
├── frontend/                 # Chat interface
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── notebooks/
│   ├── 01_exploration.ipynb  # Component exploration
│   ├── 02_evaluation.ipynb   # Full evaluation runs
│   └── 03_demo.ipynb         # End-to-end demo
├── data/
│   ├── documents/            # Upload your documents here
│   └── chroma_db/            # Vector database storage
└── tests/                    # Unit tests
```

## Usage

### Example Queries

**Simple Query:**
```
What is RAG in AI?
```

**Comparison Query:**
```
Compare BM25 and dense retrieval methods
```

**Multi-hop Query:**
```
How does BERT improve search ranking compared to TF-IDF?
```

### Ingesting Documents

Add your own documents to search:

```bash
# Ingest a directory of documents
curl -X POST http://localhost:8000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"directory_path": "./data/documents/"}'

# Ingest a single file
curl -X POST http://localhost:8000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/document.pdf"}'
```

Supported formats: PDF, DOCX, TXT, Markdown

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Chat interface |
| `/api/health` | GET | System health and status |
| `/api/search` | POST | Execute search query |
| `/api/search/stream` | POST | Streaming search with updates |
| `/api/ingest` | POST | Ingest documents into vector DB |
| `/api/stats` | GET | System statistics |
| `/api/evaluation/run` | POST | Run formal IR evaluation on test cases |
| `/api/evaluation/test-cases` | GET | List available evaluation test cases |
| `/api/evaluation/metrics-info` | GET | Metric definitions and formulas |

## Evaluation Framework

The project includes a six-layer evaluation framework that goes well beyond basic accuracy checks.

### Layer 1: IR Retrieval Metrics
Classical information retrieval measures computed per query and averaged across the test set:
- **Precision / Recall / F1** — fraction of retrieved results that are relevant, and vice versa
- **P@5, P@10** — precision within the top-k results
- **nDCG@5, nDCG@10** — ranking quality via Normalized Discounted Cumulative Gain
- **MAP** — Mean Average Precision across all queries
- **MRR** — Mean Reciprocal Rank of the first relevant result

### Layer 2: Text Generation Quality
Measures how well the synthesized answer reads and aligns with reference text:
- **BERTScore** — semantic similarity using contextual embeddings
- **ROUGE-1 / ROUGE-L** — n-gram overlap with reference answers
- **BLEU** — precision-oriented n-gram overlap

### Layer 3: Google Baseline Comparison
Compares the agent's answers against Google search results to gauge whether the agentic pipeline adds value over a standard search engine.

### Layer 4: LLM-as-Judge
An LLM evaluates each answer on five dimensions (relevance, completeness, accuracy, citation quality, clarity) on a 0–1 scale, providing qualitative feedback alongside numeric scores.

### Layer 5: RAGAS
Uses the RAGAS framework to evaluate RAG-specific qualities:
- **Faithfulness** — is the answer grounded in the retrieved context?
- **Answer Relevancy** — does the answer address the question?
- **Context Precision / Recall** — are the retrieved passages relevant and sufficient?

### Layer 6: Robustness & Stress Testing
Tests pipeline stability under real-world noise:
- **Paraphrase consistency** — do semantically equivalent queries produce consistent answers?
- **Adversarial inputs** — resilience to misspellings, jargon, and malformed queries
- **Query drift** — sensitivity to small wording changes

### Self-Reflection Quality Scores
The agent also scores its own answers during inference on:
- Relevance, Completeness, Accuracy, Citation Quality, Clarity (each 0–1)
- Answers below the quality threshold (default 0.7) trigger automatic re-retrieval and re-synthesis


## Acknowledgments

- INFO 624: Intelligent Search and Language Models(Professor Mat Kelly, Drexel University)

---

**Developed by:** Priti Sagar, Manjiri Prashant Khodke, Omkar Salunkhe
