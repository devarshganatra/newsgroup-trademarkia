"""
FastAPI application entrypoint for the Semantic Search System.

Initializes the QueryService (which loads all models) on startup
and attaches it to the application state for use by route handlers.

Start with:
    uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.api.routes import router as api_router
from app.services.query_service import QueryService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all ML models and services at startup."""
    print("[Startup] Initializing QueryService...")
    app.state.query_service = QueryService()
    print("[Startup] System ready.")
    yield
    print("[Shutdown] Cleaning up...")


app = FastAPI(
    title="Semantic Search System",
    description=(
        "A production-grade semantic search system with "
        "cluster-aware caching and fuzzy document clustering."
    ),
    version="0.2.0",
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
