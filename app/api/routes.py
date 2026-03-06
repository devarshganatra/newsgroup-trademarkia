"""
FastAPI route definitions for the semantic search system.

Endpoints:
    POST /query       — Semantic search with cache
    GET  /cache/stats — Cache performance metrics
    DELETE /cache     — Flush the cache
"""

from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Optional


router = APIRouter()


class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""
    query: str


class QueryResponse(BaseModel):
    """Response body for the /query endpoint."""
    query: str
    cache_hit: bool
    matched_query: Optional[str] = None
    similarity_score: Optional[float] = None
    result: str
    dominant_cluster: int


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest, raw_request: Request):
    """
    Semantic search endpoint.

    Embeds the query, checks the cluster-aware cache,
    and returns cached or freshly computed results.
    """
    query_service = raw_request.app.state.query_service
    response = query_service.handle_query(request.query)
    return response


@router.get("/cache/stats")
async def get_cache_stats(request: Request):
    """
    Return cache performance metrics including hit rate.
    """
    query_service = request.app.state.query_service
    return query_service.get_cache_stats()


@router.delete("/cache")
async def clear_cache(request: Request):
    """
    Flush the entire cache and reset statistics.
    """
    query_service = request.app.state.query_service
    return query_service.clear_cache()
