"""
FastAPI Application for Agentic Search.

This is the main entry point for the API server.
"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.api.routes import router
from src.utils.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Agentic Search API",
    description="""
    An intelligent multi-source search agent built with LangGraph and LangChain.
    
    ## Features
    
    - **Query Decomposition**: Breaks complex queries into sub-queries
    - **Multi-Source Retrieval**: Searches web, documents, and academic papers
    - **Answer Synthesis**: Generates comprehensive answers with citations
    - **Self-Reflection**: Evaluates and improves answer quality
    
    ## Course Alignment
    
    This project demonstrates concepts from INFO 624: Intelligent Search and Language Models,
    including vector space models, neural language models, relevance feedback, and RAG systems.
    """,
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")

# Get the frontend directory path
FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"


@app.get("/", tags=["Frontend"])
async def serve_frontend():
    """Serve the chat interface."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    else:
        return {"message": "Welcome to Agentic Search API. Visit /docs for API documentation."}


@app.get("/styles.css", tags=["Frontend"])
async def serve_styles():
    """Serve CSS styles."""
    css_path = FRONTEND_DIR / "styles.css"
    if css_path.exists():
        return FileResponse(css_path, media_type="text/css")
    return {"error": "Styles not found"}


@app.get("/app.js", tags=["Frontend"])
async def serve_js():
    """Serve JavaScript."""
    js_path = FRONTEND_DIR / "app.js"
    if js_path.exists():
        return FileResponse(js_path, media_type="application/javascript")
    return {"error": "JavaScript not found"}


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting Agentic Search API...")
    
    settings = get_settings()
    api_status = settings.validate_api_keys()
    
    # Check LLM provider
    if settings.llm_provider == "groq":
        if not api_status["groq"]:
            logger.warning("Groq API key not configured!")
        else:
            logger.info(f"Using Groq LLM: {settings.groq_model}")
    else:
        if not api_status["openai"]:
            logger.warning("OpenAI API key not configured!")
        else:
            logger.info(f"Using OpenAI LLM: {settings.openai_model}")
    
    if not api_status["tavily"]:
        logger.warning("Tavily API key not configured!")
    
    logger.info(f"Embedding provider: {settings.embedding_provider}")
    
    # Ensure data directories exist
    Path("./data/documents").mkdir(parents=True, exist_ok=True)
    Path("./data/chroma_db").mkdir(parents=True, exist_ok=True)
    
    logger.info("Agentic Search API started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Agentic Search API...")


def main():
    """Run the API server."""
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )


if __name__ == "__main__":
    main()
