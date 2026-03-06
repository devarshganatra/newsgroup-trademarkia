# Semantic Search System

A production-grade semantic search system built on the 20 Newsgroups dataset. This system implements a multi-stage pipeline including text preprocessing, fuzzy clustering, semantic caching, and vector indexing.

## Architecture Flow

The system processes queries through the following pipeline:

1.  **Request**: User sends a natural language query via the API.
2.  **Embedding**: Query is converted into a 384-dimension vector using the SentenceTransformers `all-MiniLM-L6-v2` model.
3.  **Dimensionality Reduction**: The vector is projected from 384 dimensions to 50 dimensions using a pre-trained PCA model to reduce noise.
4.  **Cluster Assignment**: A Gaussian Mixture Model (GMM) predicts the primary cluster for the query (1 out of 35 semantic clusters).
5.  **Semantic Cache Lookup**: 
    - The system checks a cluster-aware cache for semantically similar previous queries.
    - If a match is found (cosine similarity > 0.80), the cached result is returned instantly.
6.  **Vector Search (on Cache Miss)**:
    - If no cache match exists, the system performs a vector similarity search in Qdrant Cloud.
    - Results are returned to the user and stored in the semantic cache for future hits.

## Key Components

### Data Preprocessing and Ingestion
The raw 20 Newsgroups dataset is cleaned by removing headers, footers, and quotes. Documents are then embedded and indexed into Qdrant Cloud in optimized batches.

### PCA and Gaussian Mixture Clustering
Rather than hard labels, the system uses fuzzy clustering (GMM) to understand document distribution. PCA is used as a preprocessing step for clustering to improve computational efficiency and cluster separation.

### Cluster-Aware Semantic Cache
The cache is partitioned by semantic clusters. This ensures that lookups are performed only within relevant document spaces, increasing accuracy and reducing the comparison overhead as the cache grows.

## Prerequisites

- Python 3.10+
- Docker and Docker Compose (optional for containerized deployment)
- Qdrant Cloud account and API key

## Local Setup

1.  Clone the repository and navigate to the root directory.
2.  Run the setup script to create a virtual environment and install dependencies:
    ```bash
    bash setup.sh
    ```
3.  Configure your environment by copying `.env.example` to `.env` and adding your Qdrant credentials:
    ```bash
    QDRANT_URL=your_url
    QDRANT_API_KEY=your_api_key
    QDRANT_COLLECTION=newsgroups
    ```
4.  Activate the environment:
    ```bash
    source .venv/bin/activate
    ```
5.  Load the document vectors into Qdrant Cloud:
    ```bash
    python scripts/load_vectors_to_qdrant.py
    ```

## Running the System

### FastAPI Server
Start the core API service:
```bash
uvicorn app.main:app --reload
```
The API is available at `http://localhost:8000`. Documentation can be viewed at `http://localhost:8000/docs`.

### Observability Dashboard
Run the Streamlit dashboard to monitor live system performance, hit rates, and cluster distributions:
```bash
streamlit run dashboard/cache_dashboard.py
```

### Cluster Analysis
To analyze the semantic coherence of clusters and generate a 2D visualization:
```bash
python scripts/cluster_analysis.py
```
View the generated plot at `docs/cluster_visualization.png`.

### Performance Benchmarking
Run the stress test script to evaluate cache hit rates and system speedup:
```bash
python tests/cache_benchmark.py
```

## Docker Deployment

The system can be deployed as a microservice using Docker.

1.  Build and start the container:
    ```bash
    docker compose up --build
    ```
2.  Verify the service health:
    ```bash
    curl http://localhost:8000/health
    ```

## Project Structure

- `app/`: Core application logic (API, cache, clustering, vector DB).
- `data/`: Processed dataset storage.
- `models/`: Trained PCA and GMM model artifacts.
- `scripts/`: Maintenance and analysis scripts.
- `dashboard/`: Streamlit observability code.
- `tests/`: Benchmarking and functional tests.
- `Dockerfile` & `docker-compose.yml`: Containerization configuration.
