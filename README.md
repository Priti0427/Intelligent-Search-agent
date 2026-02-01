# Agentic Search with LangGraph

An intelligent multi-source search agent built with LangGraph and LangChain for INFO 624: Intelligent Search and Language Models.

## Project Overview

This project implements a sophisticated search agent that:
1. **Decomposes complex queries** into manageable sub-queries
2. **Routes queries** to appropriate sources (web, vector DB, academic papers)
3. **Retrieves and ranks** results from multiple indexes in parallel
4. **Synthesizes comprehensive answers** with proper citations
5. **Self-reflects** to improve answer quality through iterative refinement

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
                                          Self-Reflection Loop
                                                    ↓
                                    Final Answer with Citations
```

## Course Concept Integration

| Week | Concept | Implementation |
|------|---------|----------------|
| 2 | Text Preprocessing | Document chunking, tokenization |
| 4 | Vector Space Models | ChromaDB embeddings, cosine similarity |
| 5 | Neural Language Models | GPT-4 for synthesis, embeddings |
| 6 | BM25/Probabilistic | Hybrid search in ChromaDB |
| 7 | Graph Analysis | Query dependency graph |
| 8 | Evaluation Metrics | Precision, recall, answer quality |
| 9 | Relevance Feedback | Self-reflection loop |
| 11 | RAG, Conversational | Full agent architecture |

## Tech Stack

- **Agent Framework**: LangGraph
- **LLM Orchestration**: LangChain
- **LLM Provider**: OpenAI GPT-4o
- **Vector Database**: ChromaDB
- **Web Search**: Tavily API
- **Academic Search**: arXiv API
- **Backend**: FastAPI
- **Frontend**: HTML + TailwindCSS

## Setup

### 1. Clone and Install

```bash
# Clone the repository
git clone <repository-url>
cd INFO-624

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys
# Required: OPENAI_API_KEY, TAVILY_API_KEY
```

### 3. Run the Application

```bash
# Start the FastAPI server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Or use the CLI
python -m src.api.main
```

### 4. Access the Interface

Open your browser and navigate to:
- **Chat Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## Project Structure

```
INFO-624/
├── src/
│   ├── agent/           # LangGraph agent definition
│   │   ├── graph.py     # Main workflow
│   │   ├── state.py     # State schema
│   │   └── nodes/       # Individual nodes
│   ├── retrievers/      # Data source integrations
│   ├── ingestion/       # Document processing
│   ├── api/             # FastAPI backend
│   └── utils/           # Configuration and prompts
├── frontend/            # Chat interface
├── data/                # Documents and ChromaDB
├── notebooks/           # Jupyter notebooks
└── tests/               # Unit tests
```

## Usage Examples

### Simple Query
```
"What is RAG in AI?"
```

### Complex Query
```
"Compare BM25 and dense retrieval for question answering"
```

### Multi-hop Query
```
"How does BERT improve search ranking compared to TF-IDF, and what are the computational tradeoffs?"
```

## Evaluation Metrics

- **Retrieval Quality**: Precision@k, Recall@k, MRR
- **Answer Quality**: Faithfulness, Relevance, Completeness (RAGAS)
- **Citation Accuracy**: Source attribution correctness
- **Agent Efficiency**: Steps to answer, reflection iterations

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Chat interface |
| `/api/search` | POST | Execute search query |
| `/api/search/stream` | POST | Streaming search |
| `/api/ingest` | POST | Ingest documents |
| `/api/health` | GET | Health check |

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- INFO 624: Intelligent Search and Language Models - Drexel University
- Dr. Mat Kelly
