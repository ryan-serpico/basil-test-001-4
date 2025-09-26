"""FastAPI backend for semantic search application"""
import os
import json
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from vector_db import VectorDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="test-001 Search API",
    description="Semantic search API for test-001 document collection",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize vector database
vector_db = VectorDatabase()


class SearchRequest(BaseModel):
    """Search request model"""
    query: str
    limit: int = 10


class SearchResult(BaseModel):
    """Search result model"""
    chunk_text: str
    source_document: str
    context_before: Optional[str] = None
    context_after: Optional[str] = None
    similarity_score: float
    chunk_position: int


class SearchResponse(BaseModel):
    """Search response model"""
    results: List[SearchResult]
    query: str
    total_results: int


class ExportRequest(BaseModel):
    """Export request model"""
    query: str
    format: str = "json"
    limit: int = 100


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting test-001 Search API")

    # Ensure vector database is loaded
    try:
        await vector_db.initialize()
        logger.info("Vector database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize vector database: {e}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "test-001 Search API",
        "version": "1.0.0",
        "vector_db_loaded": vector_db.is_loaded
    }


@app.post("/api/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Perform semantic search across documents

    Args:
        request: Search request with query and limit

    Returns:
        Search results with similarity scores
    """
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        if request.limit < 1 or request.limit > 100:
            raise HTTPException(status_code=400, detail="Limit must be between 1 and 100")

        logger.info(f"Searching for: '{request.query}' (limit: {request.limit})")

        # Perform vector search
        results = await vector_db.search(request.query, request.limit)

        # Format results
        search_results = []
        for result in results:
            search_result = SearchResult(
                chunk_text=result["text"],
                source_document=result["metadata"]["source_document"],
                context_before=result["metadata"].get("context_before"),
                context_after=result["metadata"].get("context_after"),
                similarity_score=result["similarity_score"],
                chunk_position=result["metadata"]["chunk_position"]
            )
            search_results.append(search_result)

        response = SearchResponse(
            results=search_results,
            query=request.query,
            total_results=len(search_results)
        )

        logger.info(f"Found {len(search_results)} results")
        return response

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/api/documents")
async def list_documents():
    """List all available documents"""
    try:
        documents = await vector_db.get_document_list()
        return {
            "documents": documents,
            "total": len(documents)
        }
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to list documents")


@app.get("/api/stats")
async def get_statistics():
    """Get collection statistics"""
    try:
        stats = await vector_db.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")


@app.post("/api/export")
async def export_search_results(request: ExportRequest):
    """
    Export search results in various formats

    Args:
        request: Export request with query, format, and limit

    Returns:
        File download with search results
    """
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        if request.format not in ["json", "csv", "parquet"]:
            raise HTTPException(status_code=400, detail="Format must be 'json', 'csv', or 'parquet'")

        logger.info(f"Exporting search results for: '{request.query}' as {request.format}")

        # Use vector database export functionality
        export_result = await vector_db.export_data(
            format=request.format,
            query=request.query,
            limit=request.limit
        )

        if "error" in export_result:
            raise HTTPException(status_code=500, detail=f"Export failed: {export_result['error']}")

        filename_base = request.query.replace(' ', '_')[:20]

        if request.format == "json":
            filename = f"search_results_{filename_base}.json"
            return JSONResponse(
                content=export_result.get("data", export_result),
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )

        elif request.format == "csv":
            filename = f"search_results_{filename_base}.csv"
            return JSONResponse(
                content={"data": export_result.get("data", "")},
                headers={
                    "Content-Type": "text/csv",
                    "Content-Disposition": f"attachment; filename={filename}"
                }
            )

        else:  # parquet format
            filename = f"search_results_{filename_base}.parquet"
            return JSONResponse(
                content={
                    "message": "Parquet export completed",
                    "info": export_result,
                    "note": "Full dataset is available in native Parquet format"
                },
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )

    except Exception as e:
        logger.error(f"Export error: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint - serve frontend or API info"""
    return {
        "message": "test-001 Search API",
        "version": "1.0.0",
        "endpoints": {
            "search": "/api/search",
            "documents": "/api/documents",
            "statistics": "/api/stats",
            "export": "/api/export",
            "health": "/health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")