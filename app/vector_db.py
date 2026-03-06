"""
Qdrant vector database client for semantic document search.

Wraps the qdrant-client library to provide collection management,
document ingestion, and similarity search against Qdrant Cloud.
"""

import os
from typing import List, Dict, Any, Optional

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

load_dotenv()


class VectorDB:
    """
    Manages connection to Qdrant Cloud and provides methods
    for collection setup, document upsert, and vector search.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection: Optional[str] = None,
    ):
        self.url = url or os.getenv("QDRANT_URL")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.collection = collection or os.getenv("QDRANT_COLLECTION", "newsgroups")
        self.client: Optional[QdrantClient] = None

    def connect(self) -> None:
        """Establish connection to Qdrant Cloud."""
        print(f"[VectorDB] Connecting to Qdrant at {self.url}...")
        self.client = QdrantClient(url=self.url, api_key=self.api_key)
        print("[VectorDB] Connected to Qdrant.")

    def create_collection(self, vector_size: int = 384) -> None:
        """
        Create the collection if it does not already exist.

        Args:
            vector_size: Dimensionality of the embedding vectors.
        """
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection in collections:
            print(f"[VectorDB] Collection '{self.collection}' already exists.")
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        print(f"[VectorDB] Collection '{self.collection}' created.")

    def upsert_documents(
        self,
        ids: List[int],
        vectors: np.ndarray,
        payloads: List[Dict[str, Any]],
        batch_size: int = 500,
    ) -> None:
        """
        Upload document vectors and payloads to Qdrant in batches.

        Args:
            ids: Document IDs.
            vectors: Embedding matrix (N x D).
            payloads: List of metadata dicts per document.
            batch_size: Number of points per upsert call.
        """
        total = len(ids)
        n_batches = (total + batch_size - 1) // batch_size

        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, total)

            points = [
                PointStruct(
                    id=int(ids[j]),
                    vector=vectors[j].tolist(),
                    payload=payloads[j],
                )
                for j in range(start, end)
            ]

            self.client.upsert(
                collection_name=self.collection,
                points=points,
            )
            print(f"  Uploading batch {i + 1}/{n_batches} ({end}/{total} points)")

        print(f"[VectorDB] Upserted {total} documents.")

    def search(
        self,
        query_vector: np.ndarray,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Perform cosine similarity search against the collection.

        Args:
            query_vector: Normalized query embedding (1-D array).
            limit: Number of results to return.

        Returns:
            List of result dicts with doc_id, newsgroup, score, snippet.
        """
        hits = self.client.query_points(
            collection_name=self.collection,
            query=query_vector.tolist(),
            limit=limit,
            with_payload=True,
        )

        results = []
        for hit in hits.points:
            payload = hit.payload or {}
            results.append({
                "doc_id": hit.id,
                "newsgroup": payload.get("newsgroup", ""),
                "score": round(float(hit.score), 4),
                "snippet": payload.get("text", "")[:300],
            })

        print(f"[VectorDB] Search returned {len(results)} results.")
        return results
