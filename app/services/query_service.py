"""
Query service orchestrating the semantic search pipeline.

Handles: embedding generation → PCA transform → cluster prediction
→ cache lookup → result computation → cache storage.
"""

import json
import numpy as np
from typing import Dict, Any
from joblib import load as joblib_load

from app.embeddings.embedder import Embedder
from app.cache.semantic_cache import SemanticCache


class QueryService:
    """
    Orchestrates the end-to-end query pipeline including
    embedding, clustering, caching, and result retrieval.
    """

    def __init__(self):
        # Load models
        print("[QueryService] Loading PCA model...")
        self.pca = joblib_load("models/pca_model.joblib")

        print("[QueryService] Loading GMM model...")
        self.gmm = joblib_load("models/gmm_model.joblib")

        # Load document data for result generation
        print("[QueryService] Loading document index...")
        with open("models/doc_index.json", "r", encoding="utf-8") as f:
            self.doc_index = json.load(f)

        print("[QueryService] Loading embeddings...")
        self.doc_embeddings = np.load("models/embeddings.npy")

        print("[QueryService] Loading processed corpus...")
        with open("data/processed_corpus.json", "r", encoding="utf-8") as f:
            self.corpus = json.load(f)

        # Initialize components
        self.embedder = Embedder()
        n_clusters = self.gmm.n_components
        self.cache = SemanticCache(
            n_clusters=n_clusters,
            similarity_threshold=0.80,
            top_k_clusters=3,
            max_entries_per_cluster=100
        )

        print(f"[QueryService] Ready. Clusters: {n_clusters}")

    def handle_query(self, query: str) -> Dict[str, Any]:
        """
        Process a semantic search query through the full pipeline.

        Steps:
            1. Generate normalized query embedding
            2. PCA transform to clustering space
            3. Predict cluster probabilities
            4. Cache lookup (top-3 clusters)
            5. If miss: compute result, store in cache
            6. Return response

        Args:
            query: Natural language search query.

        Returns:
            Response dict with results and cache metadata.
        """
        # Step 1: Embed query
        query_embedding = self.embedder.encode(query)
        print(f"[QueryService] Query embedded: {query[:60]}...")

        # Step 2: PCA transform
        query_reduced = self.pca.transform(query_embedding.reshape(1, -1))

        # Step 3: Cluster prediction
        cluster_probs = self.gmm.predict_proba(query_reduced)[0]
        primary_cluster = int(np.argmax(cluster_probs))
        print(f"[QueryService] Primary cluster: {primary_cluster}")

        # Step 4: Cache lookup
        hit = self.cache.lookup(query_embedding, cluster_probs)

        if hit is not None:
            print(f"[QueryService] Cache HIT (sim={hit['similarity_score']})")
            return {
                "query": query,
                **hit,
            }

        # Step 5: Cache miss — compute result
        print("[QueryService] Cache MISS — computing result...")
        result = self._compute_result(query_embedding)

        # Step 6: Store in cache
        self.cache.store(
            query=query,
            embedding=query_embedding,
            cluster_probs=cluster_probs,
            result=result
        )

        return {
            "query": query,
            "cache_hit": False,
            "matched_query": None,
            "similarity_score": None,
            "result": result,
            "dominant_cluster": primary_cluster,
        }

    def _compute_result(self, query_embedding: np.ndarray, top_k: int = 5) -> str:
        """
        Compute a search result by finding the top-k most similar documents.

        Uses dot product on normalized vectors for cosine similarity.

        Args:
            query_embedding: Normalized query embedding.
            top_k: Number of results to return.

        Returns:
            JSON-formatted string of top-k document matches.
        """
        similarities = self.doc_embeddings @ query_embedding
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            doc_meta = self.doc_index[idx]
            doc_text = self.corpus[idx]["text"][:300]
            results.append({
                "doc_id": doc_meta["doc_id"],
                "newsgroup": doc_meta["newsgroup"],
                "score": round(float(similarities[idx]), 4),
                "snippet": doc_text
            })

        return json.dumps(results)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache performance statistics."""
        return self.cache.stats()

    def clear_cache(self) -> Dict[str, str]:
        """Flush the cache and return confirmation."""
        self.cache.clear()
        return {"message": "Cache cleared successfully."}
