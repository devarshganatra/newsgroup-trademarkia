# Semantic Search System

A production-grade semantic search system built with Python, FastAPI, and Sentence Transformers.

## Project Overview

This system provides semantic document retrieval, fuzzy clustering, and a custom semantic query cache. It is designed to be modular and scalable, suitable for real production environments.

## System Architecture

The project follows a modular architecture:

- **Embedder**: Converts text into numerical vectors using `sentence-transformers`.
- **Semantic Cache**: Reduces query latency by storing and retrieving similar previous queries.
- **Vector Store**: Efficient similarity search using `FAISS`.
- **Cluster Manager**: Performs fuzzy clustering for document organization.
- **Query Service**: Orchestrates the workflow between embeddings, cache, and search.
- **API (FastAPI)**: RESTful endpoints for querying and cache management.

## Installation

1. Ensure you have Python 3.8+ installed.
2. Run the setup script to create a virtual environment and install dependencies:

```bash
bash setup.sh
```

## Running the Service

After activating the virtual environment, you can start the FastAPI service using `uvicorn`:

```bash
source .venv/bin/activate
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`. You can explore the interactive documentation at `http://127.0.0.1:8000/docs`.

## Upcoming Stages

- **Stage 2**: Implementation of the `Embedder` using `SentenceTransformers`.
- **Stage 3**: Development of the custom `SemanticCache` with similarity thresholds.
- **Stage 4**: Integration of `FAISS` for the `VectorStore`.
- **Stage 5**: Fuzzy clustering implementation for document categorization.
- **Stage 6**: Production deployment and optimization.
