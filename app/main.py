"""
FastAPI application entrypoint for the Semantic Search System.

Initializes the VectorDB and QueryService on startup
and attaches them to the application state for use by route handlers.

Start with:
    uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.api.routes import router as api_router
from app.vector_db import VectorDB
from app.services.query_service import QueryService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all ML models and services at startup."""
    print("[Startup] Connecting to Qdrant...")
    vector_db = VectorDB()
    vector_db.connect()

    print("[Startup] Initializing QueryService...")
    app.state.query_service = QueryService(vector_db=vector_db)
    print("[Startup] System ready.")
    yield
    print("[Shutdown] Cleaning up...")


app = FastAPI(
    title="Semantic Search System",
    description=(
        "A production-grade semantic search system with "
        "cluster-aware caching and Qdrant vector search."
    ),
    version="0.3.0",
    lifespan=lifespan,
)

# Include API routes
app.include_router(api_router)


@app.get("/")
async def root():
    """Health check / welcome endpoint."""
    return {
        "message": "Welcome to the Semantic Search System API",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
