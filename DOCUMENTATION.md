# Agentic Search: Multi-Source Intelligent Search Engine

**Authors:** Priti Sagar, Manjiri Prashant Khodke, Omkar Salunkhe

**Course:** INFO 624 - Intelligent Search and Language Models

**Institution:** Drexel University

---

## Abstract

Traditional search engines return a list of links, requiring users to manually read through multiple documents, extract relevant information, and synthesize findings. This project presents **Agentic Search**, an intelligent multi-source search agent that automates the entire research workflow. Built using LangGraph and LangChain, the system decomposes complex queries, routes them to appropriate sources (web, academic papers, and custom documents), retrieves and ranks results, synthesizes comprehensive answers with citations, and employs a self-reflection loop to ensure answer quality. The system demonstrates practical applications of vector space models, neural embeddings, retrieval-augmented generation (RAG), and relevance feedback mechanisms.

---

## 1. Introduction

### 1.1 Problem Statement

Information retrieval has evolved significantly, yet users still face challenges when seeking comprehensive answers to complex questions. Traditional search engines:

- Return ranked lists of documents rather than direct answers
- Require manual synthesis of information from multiple sources
- Lack mechanisms for verifying accuracy across sources
- Do not adapt to query complexity

### 1.2 Solution Overview

Agentic Search addresses these limitations by implementing an autonomous search agent that:

1. **Analyzes and decomposes** complex queries into manageable sub-queries
2. **Routes queries** intelligently to the most appropriate data sources
3. **Retrieves and aggregates** results from multiple indexes in parallel
4. **Synthesizes** comprehensive answers with proper citations
5. **Self-reflects** to iteratively improve answer quality

---

## 2. Software and Hardware Settings

### 2.1 Software Requirements

| Component | Technology | Version |
|-----------|------------|---------|
| Programming Language | Python | 3.11 |
| Agent Framework | LangGraph | Latest |
| LLM Orchestration | LangChain | Latest |
| LLM Provider | Groq (Llama 3.1 8B) | Free Tier |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) | Free |
| Vector Database | ChromaDB | Latest |
| Web Search API | Tavily | Free Tier |
| Academic Search | arXiv API | Public |
| Backend Framework | FastAPI | Latest |
| Frontend | HTML/CSS/JavaScript | - |

### 2.2 API Keys Required

- **GROQ_API_KEY**: Obtain free from https://console.groq.com
- **TAVILY_API_KEY**: Obtain free from https://tavily.com

### 2.3 Hardware Requirements

- Minimum 8GB RAM
- 2GB disk space for vector database storage
- Internet connection for API access

---

## 3. System Architecture

### 3.1 LangGraph Workflow

The system implements a directed graph workflow with six primary nodes:

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

### 3.2 Node Descriptions

#### 3.2.1 Query Analyzer
- Classifies incoming queries as simple, complex, or multi-hop
- Extracts key entities and concepts
- Determines required information sources

#### 3.2.2 Query Decomposer
- Breaks complex queries into focused sub-queries
- Maintains logical dependencies between sub-queries
- Optimizes for parallel retrieval where possible

#### 3.2.3 Router
- Routes each sub-query to appropriate data sources
- Considers query type, domain, and recency requirements
- Supports multi-source routing for comprehensive coverage

#### 3.2.4 Retriever
- Executes parallel searches across selected sources
- **Web Search (Tavily)**: Real-time internet information
- **Vector Store (ChromaDB)**: Custom document retrieval using semantic similarity
- **Academic Search (arXiv)**: Research papers and technical content

#### 3.2.5 Synthesizer
- Aggregates results from all sources
- Generates coherent, comprehensive answers
- Includes proper citations with source attribution

#### 3.2.6 Reflector
- Evaluates answer quality on five dimensions:
  - **Relevance**: Does it answer the question?
  - **Completeness**: Are all aspects covered?
  - **Accuracy**: Is it consistent with sources?
  - **Citation Quality**: Are sources properly cited?
  - **Clarity**: Is it well-structured?
- Triggers re-synthesis if quality score < 0.7

---

## 4. Major Features

### 4.1 Multi-Source Retrieval

The system integrates three distinct retrieval sources:

| Source | Use Case | Technology |
|--------|----------|------------|
| Web Search | Current events, general knowledge | Tavily API |
| Academic Papers | Research, technical content | arXiv API |
| Custom Documents | User-uploaded PDFs, docs, notes | ChromaDB + HuggingFace embeddings |

### 4.2 Self-Reflection Loop

The self-reflection mechanism ensures answer quality through iterative refinement:

1. Generate initial answer with citations
2. Evaluate on 5 quality dimensions (0-1 scale each)
3. If average score < 0.7, identify weaknesses and regenerate
4. Maximum 3 iterations to prevent infinite loops
5. Return best answer with quality metrics

### 4.3 Query Decomposition

Complex queries are automatically decomposed:

**Example Input:** "How does BERT improve search ranking compared to TF-IDF?"

**Decomposed Sub-queries:**
1. "What is TF-IDF and how is it used in search ranking?"
2. "What is BERT and how does it work?"
3. "How is BERT applied to search ranking tasks?"
4. "What are the advantages of BERT over TF-IDF for ranking?"

### 4.4 Citation Generation

All answers include properly formatted citations:

- Source type (Web, Academic, Document)
- Title and URL/reference
- Relevance score
- Snippet from source

---

## 5. Course Concept Integration

| Course Concept | Implementation in System |
|----------------|-------------------------|
| Vector Space Models | ChromaDB embeddings with cosine similarity |
| Neural Embeddings | HuggingFace sentence-transformers (all-MiniLM-L6-v2) |
| BM25/Sparse Retrieval | Understanding integrated in query routing decisions |
| Dense Retrieval | Semantic search with dense vector representations |
| Relevance Feedback | Self-reflection loop for quality improvement |
| Query Expansion | Query decomposition into focused sub-queries |
| RAG | Retrieval-Augmented Generation architecture |
| Evaluation Metrics | Precision, Recall, F1, nDCG, MAP, MRR |

---

## 6. Formal Evaluation

### 6.1 Evaluation Metrics

The system includes a formal IR evaluation module with the following metrics:

| Metric | Formula | Description |
|--------|---------|-------------|
| **Precision** | Relevant Retrieved / Total Retrieved | Fraction of retrieved docs that are relevant |
| **Recall** | Relevant Retrieved / Total Relevant | Fraction of relevant docs that are retrieved |
| **F1 Score** | 2 × (P × R) / (P + R) | Harmonic mean of precision and recall |
| **P@k** | Relevant in top k / k | Precision at rank k |
| **nDCG@k** | DCG@k / IDCG@k | Normalized ranking quality |
| **MAP** | Mean of AP across queries | Mean Average Precision |
| **MRR** | Mean of 1/rank of first relevant | Mean Reciprocal Rank |

### 6.2 Test Cases

| # | Query | Information Need |
|---|-------|------------------|
| 1 | What is RAG in AI? | Understanding RAG concept |
| 2 | Compare BM25 and dense retrieval | Retrieval method comparison |
| 3 | BERT vs TF-IDF for search ranking | Neural vs traditional ranking |
| 4 | Main components of a search engine | Search architecture |
| 5 | Vector embeddings for semantic search | Understanding embeddings |
| 6 | Query decomposition in IR | Query processing |

### 6.3 Running Evaluation

```bash
# Via API
curl -X POST http://localhost:8000/api/evaluation/run \
  -H "Content-Type: application/json" \
  -d '{"max_results": 10}'

# Via command line
python -m src.evaluation.evaluator --output EVALUATION_RESULTS.md
```

---

## 7. Screenshots

### 7.1 Main Interface

*[Screenshot: Main chat interface showing the search input and welcome message]*

### 7.2 Search Results

*[Screenshot: Example search results with citations and quality scores]*

### 7.3 System Status Panel

*[Screenshot: Sidebar showing system status and search statistics]*

---

## 8. How to Run

### 8.1 Installation

```bash
# Clone the repository
git clone https://github.com/Priti0427/Intelligent-Search-agent.git
cd Intelligent-Search-agent

# Create virtual environment (Python 3.11 required)
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 8.2 Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys:
# GROQ_API_KEY=your_groq_api_key
# TAVILY_API_KEY=your_tavily_api_key
```

### 8.3 Running the Application

```bash
# Start the FastAPI server
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 8.4 Accessing the Interface

- **Chat Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

---

## 9. API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Chat interface |
| `/api/health` | GET | System health and status |
| `/api/search` | POST | Execute search query |
| `/api/search/stream` | POST | Streaming search with real-time updates |
| `/api/ingest` | POST | Ingest documents into vector database |
| `/api/stats` | GET | System statistics |
| `/api/evaluation/test-cases` | GET | Get evaluation test cases |
| `/api/evaluation/run` | POST | Run formal IR evaluation |
| `/api/evaluation/metrics-info` | GET | Get metric definitions |

---

## 10. Evaluation Results

See [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md) for detailed evaluation results.

### 10.1 Quality Metrics (Automated)

- Relevance Score (0-1)
- Completeness Score (0-1)
- Accuracy Score (0-1)
- Citation Quality Score (0-1)
- Clarity Score (0-1)

### 10.2 System Performance Metrics

- Average query processing time
- Number of reflection iterations per query
- Sources searched per query
- Documents indexed in vector store

---

## 11. GitHub Repository

**Repository URL:** https://github.com/Priti0427/Intelligent-Search-agent

---

## 12. Conclusion

Agentic Search demonstrates the practical application of modern information retrieval concepts in building an intelligent search system. By combining LangGraph's workflow orchestration with multi-source retrieval and self-reflection mechanisms, the system provides comprehensive, cited answers to complex queries while maintaining quality through automated evaluation.

---

## 13. References

1. Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.
2. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL*.
3. Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*.
4. LangChain Documentation. https://python.langchain.com/
5. LangGraph Documentation. https://langchain-ai.github.io/langgraph/

---

**Developed by:** Priti Sagar, Manjiri Prashant Khodke, Omkar Salunkhe

**Course:** INFO 624 - Intelligent Search and Language Models

**Institution:** Drexel University
