from fastapi import FastAPI
from app.api.routes import router as api_router

app = FastAPI(
    title="Semantic Search System",
    description="A production-grade semantic search system with clustering and caching.",
    version="0.1.0"
)

# Include API routes
app.include_router(api_router)

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Semantic Search System API",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
