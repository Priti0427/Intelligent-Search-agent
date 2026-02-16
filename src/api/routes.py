"""
API Routes for Agentic Search.
"""

import logging
import time
from datetime import datetime
from typing import AsyncGenerator, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json

from src.agent import run_search, run_search_stream
from src.api.schemas import (
    Citation,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QualityScores,
    SearchMetadata,
    SearchRequest,
    SearchResponse,
    ServiceStatus,
    StreamEvent,
    EvaluationRequest,
    EvaluationResponse,
    QueryEvaluationResult as QueryEvalResultSchema,
    TestCaseInfo,
)
from src.ingestion import DocumentEmbedder
from src.retrievers import get_vector_retriever
from src.utils.config import get_settings
from src.utils.llm import get_llm_info

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Check the health of all services.
    
    Returns status of:
    - LLM API connection (Groq or OpenAI)
    - Tavily API connection
    - ChromaDB vector store
    """
    settings = get_settings()
    services = []
    overall_status = "healthy"
    
    # Check LLM (Groq or OpenAI)
    llm_info = get_llm_info()
    if llm_info["has_api_key"]:
        services.append(ServiceStatus(
            name=f"llm ({llm_info['provider']})",
            status="ok",
            message=f"Using {llm_info['model']}",
        ))
    else:
        services.append(ServiceStatus(
            name=f"llm ({llm_info['provider']})",
            status="not_configured",
            message=f"{llm_info['provider'].upper()}_API_KEY not set",
        ))
        overall_status = "degraded"
    
    # Check Tavily
    if settings.tavily_api_key:
        services.append(ServiceStatus(name="tavily", status="ok"))
    else:
        services.append(ServiceStatus(
            name="tavily",
            status="not_configured",
            message="TAVILY_API_KEY not set",
        ))
        overall_status = "degraded"
    
    # Check Embeddings
    services.append(ServiceStatus(
        name=f"embeddings ({settings.embedding_provider})",
        status="ok",
        message=f"Using {settings.huggingface_embedding_model if settings.embedding_provider == 'huggingface' else settings.openai_embedding_model}",
    ))
    
    # Check ChromaDB
    try:
        vector_store = get_vector_retriever()
        doc_count = vector_store.get_document_count()
        services.append(ServiceStatus(
            name="chromadb",
            status="ok",
            message=f"{doc_count} documents indexed",
        ))
    except Exception as e:
        services.append(ServiceStatus(
            name="chromadb",
            status="error",
            message=str(e),
        ))
        overall_status = "degraded"
    
    # Get document count
    try:
        doc_count = get_vector_retriever().get_document_count()
    except Exception:
        doc_count = 0
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        version="0.1.0",
        services=services,
        document_count=doc_count,
    )


@router.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(request: SearchRequest):
    """
    Execute a search query through the agentic search pipeline.
    
    The agent will:
    1. Analyze and decompose the query
    2. Route to appropriate sources (web, documents, academic)
    3. Retrieve and synthesize results
    4. Self-reflect and improve if needed
    """
    logger.info(f"Search request: {request.query[:100]}...")
    
    start_time = time.time()
    
    try:
        # Run the search agent
        result = await run_search(request.query)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Extract answer
        answer = result.get("final_answer") or result.get("draft_answer", "No answer generated.")
        
        # Extract citations
        citations = []
        if request.include_sources:
            for c in result.get("citations", []):
                citations.append(Citation(
                    number=c.get("number", 0),
                    title=c.get("title", "Unknown"),
                    url=c.get("url"),
                    source_type=c.get("source_type", "unknown"),
                    excerpt=c.get("excerpt", "")[:200],
                ))
        
        # Extract quality scores
        quality_scores = None
        if result.get("quality_scores"):
            qs = result["quality_scores"]
            quality_scores = QualityScores(
                relevance=qs.get("relevance", 0),
                completeness=qs.get("completeness", 0),
                accuracy=qs.get("accuracy", 0),
                citation_quality=qs.get("citation_quality", 0),
                clarity=qs.get("clarity", 0),
            )
        
        # Build metadata
        sub_queries = [sq.get("query", "") for sq in result.get("sub_queries", [])]
        sources_searched = set()
        for sq in result.get("sub_queries", []):
            sources_searched.update(sq.get("sources", []))
        
        total_results = (
            len(result.get("web_results", [])) +
            len(result.get("vector_results", [])) +
            len(result.get("arxiv_results", []))
        )
        
        metadata = SearchMetadata(
            query_type=result.get("query_type", "unknown"),
            sub_queries=sub_queries,
            sources_searched=list(sources_searched),
            total_results=total_results,
            iterations=result.get("iteration_count", 1),
            quality_score=result.get("overall_quality", 0),
            processing_time_ms=processing_time,
        )
        
        return SearchResponse(
            query=request.query,
            answer=answer,
            citations=citations,
            quality_scores=quality_scores,
            metadata=metadata,
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/stream", tags=["Search"])
async def search_stream(request: SearchRequest):
    """
    Execute a search query with streaming updates.
    
    Returns Server-Sent Events (SSE) with progress updates
    as the agent processes the query.
    """
    logger.info(f"Streaming search request: {request.query[:100]}...")
    
    async def generate_events() -> AsyncGenerator[str, None]:
        try:
            async for event in run_search_stream(request.query):
                # Determine which node produced this event
                node_name = list(event.keys())[0] if event else "unknown"
                node_data = event.get(node_name, {})
                
                stream_event = StreamEvent(
                    event_type="status",
                    node=node_name,
                    data={"message": f"Processing: {node_name}"},
                )
                
                yield f"data: {stream_event.model_dump_json()}\n\n"
            
            # Send completion event
            complete_event = StreamEvent(
                event_type="complete",
                data={"message": "Search completed"},
            )
            yield f"data: {complete_event.model_dump_json()}\n\n"
            
        except Exception as e:
            error_event = StreamEvent(
                event_type="error",
                data={"error": str(e)},
            )
            yield f"data: {error_event.model_dump_json()}\n\n"
    
    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
    )


@router.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_documents(request: IngestRequest):
    """
    Ingest documents into the vector store.
    
    Supports:
    - Raw text
    - Single file (PDF, TXT, MD, DOCX)
    - Directory of files
    """
    logger.info("Ingest request received")
    
    embedder = DocumentEmbedder(
        chunk_strategy=request.chunk_strategy,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )
    
    chunks_ingested = 0
    
    try:
        if request.text:
            chunks_ingested = await embedder.ingest_text(
                text=request.text,
                title=request.title,
            )
        elif request.file_path:
            chunks_ingested = await embedder.ingest_file(request.file_path)
        elif request.directory_path:
            chunks_ingested = await embedder.ingest_directory(request.directory_path)
        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide text, file_path, or directory_path",
            )
        
        stats = embedder.get_stats()
        
        return IngestResponse(
            success=True,
            chunks_ingested=chunks_ingested,
            message=f"Successfully ingested {chunks_ingested} chunks",
            document_count=stats["document_count"],
        )
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", tags=["System"])
async def get_stats():
    """Get statistics about the system."""
    settings = get_settings()
    llm_info = get_llm_info()
    
    try:
        vector_store = get_vector_retriever()
        return {
            "document_count": vector_store.get_document_count(),
            "llm_provider": llm_info["provider"],
            "llm_model": llm_info["model"],
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.huggingface_embedding_model if settings.embedding_provider == "huggingface" else settings.openai_embedding_model,
            "status": "ok",
        }
    except Exception as e:
        return {
            "document_count": 0,
            "status": "error",
            "error": str(e),
        }


# === Evaluation Endpoints ===

@router.get("/evaluation/test-cases", response_model=List[TestCaseInfo], tags=["Evaluation"])
async def get_test_cases():
    """
    Get all available test cases for evaluation.
    
    Returns the list of test queries with their information needs
    and relevance criteria.
    """
    from src.evaluation.test_cases import get_test_cases as get_cases
    
    test_cases = get_cases()
    return [
        TestCaseInfo(
            index=i,
            query=tc.query,
            information_need=tc.information_need,
            relevant_keywords=tc.relevant_keywords[:10],  # Limit for response size
            expected_topics=tc.expected_topics,
            relevant_sources=tc.relevant_sources,
        )
        for i, tc in enumerate(test_cases)
    ]


@router.post("/evaluation/run", response_model=EvaluationResponse, tags=["Evaluation"])
async def run_evaluation(request: EvaluationRequest):
    """
    Run formal IR evaluation on test cases.
    
    This endpoint:
    1. Executes test queries through the search system
    2. Judges relevance of retrieved results
    3. Calculates precision, recall, F1, nDCG, MAP, MRR
    4. Returns comprehensive evaluation report
    
    Note: This may take several minutes depending on the number of test cases.
    """
    from src.evaluation import SearchEvaluator, get_test_cases as get_cases
    
    logger.info("Starting evaluation run")
    
    try:
        # Get test cases
        all_test_cases = get_cases()
        
        # Filter to specific indices if requested
        if request.test_case_indices:
            test_cases = [
                all_test_cases[i] 
                for i in request.test_case_indices 
                if 0 <= i < len(all_test_cases)
            ]
        else:
            test_cases = all_test_cases
        
        if not test_cases:
            raise HTTPException(status_code=400, detail="No valid test cases selected")
        
        # Run evaluation
        evaluator = SearchEvaluator(test_cases=test_cases)
        report = await evaluator.run_evaluation(max_results=request.max_results)
        
        # Convert to response schema
        query_results = [
            QueryEvalResultSchema(
                query=qr.query,
                information_need=qr.information_need,
                num_retrieved=qr.num_retrieved,
                num_relevant=qr.num_relevant,
                num_relevant_retrieved=qr.num_relevant_retrieved,
                precision=qr.precision,
                recall=qr.recall,
                f1=qr.f1,
                precision_at_5=qr.precision_at_5,
                precision_at_10=qr.precision_at_10,
                ndcg_at_5=qr.ndcg_at_5,
                ndcg_at_10=qr.ndcg_at_10,
                average_precision=qr.average_precision,
                reciprocal_rank=qr.reciprocal_rank,
                sources_searched=qr.sources_searched,
                processing_time_ms=qr.processing_time_ms,
            )
            for qr in report.query_results
        ]
        
        return EvaluationResponse(
            timestamp=report.timestamp,
            num_queries=report.num_queries,
            mean_precision=report.mean_precision,
            mean_recall=report.mean_recall,
            mean_f1=report.mean_f1,
            mean_precision_at_5=report.mean_precision_at_5,
            mean_precision_at_10=report.mean_precision_at_10,
            mean_ndcg_at_5=report.mean_ndcg_at_5,
            mean_ndcg_at_10=report.mean_ndcg_at_10,
            map_score=report.map_score,
            mrr_score=report.mrr_score,
            avg_processing_time_ms=report.avg_processing_time_ms,
            total_results_retrieved=report.total_results_retrieved,
            total_relevant_found=report.total_relevant_found,
            query_results=query_results,
        )
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evaluation/metrics-info", tags=["Evaluation"])
async def get_metrics_info():
    """
    Get information about the evaluation metrics used.
    
    Returns definitions and formulas for each metric.
    """
    return {
        "metrics": [
            {
                "name": "Precision",
                "formula": "Relevant Retrieved / Total Retrieved",
                "description": "Fraction of retrieved documents that are relevant",
                "range": "0 to 1 (higher is better)",
            },
            {
                "name": "Recall",
                "formula": "Relevant Retrieved / Total Relevant",
                "description": "Fraction of relevant documents that are retrieved",
                "range": "0 to 1 (higher is better)",
            },
            {
                "name": "F1 Score",
                "formula": "2 × (Precision × Recall) / (Precision + Recall)",
                "description": "Harmonic mean of precision and recall",
                "range": "0 to 1 (higher is better)",
            },
            {
                "name": "P@k (Precision at k)",
                "formula": "Relevant in top k / k",
                "description": "Precision considering only top k results",
                "range": "0 to 1 (higher is better)",
            },
            {
                "name": "nDCG@k",
                "formula": "DCG@k / IDCG@k",
                "description": "Normalized Discounted Cumulative Gain - measures ranking quality",
                "range": "0 to 1 (higher is better)",
            },
            {
                "name": "MAP (Mean Average Precision)",
                "formula": "Mean of AP across all queries",
                "description": "Average precision at each relevant document, averaged across queries",
                "range": "0 to 1 (higher is better)",
            },
            {
                "name": "MRR (Mean Reciprocal Rank)",
                "formula": "Mean of 1/rank of first relevant result",
                "description": "How quickly the first relevant result appears",
                "range": "0 to 1 (higher is better)",
            },
        ]
    }
