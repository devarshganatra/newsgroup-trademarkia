"""
Cluster-aware semantic cache built from first principles.

Organizes cached queries into cluster buckets based on GMM predictions,
enabling fast semantic lookup by searching only the top-k relevant clusters.
Implements LRU eviction per cluster bucket.
"""

import time
from typing import Optional, Dict, Any, List
import numpy as np

from app.utils.similarity import cosine_similarity


class CacheEntry:
    """Represents a single cached query and its associated data."""

    __slots__ = [
        "query", "embedding", "cluster_probs",
        "primary_cluster", "result", "timestamp"
    ]

    def __init__(
        self,
        query: str,
        embedding: np.ndarray,
        cluster_probs: np.ndarray,
        primary_cluster: int,
        result: str,
        timestamp: float
    ):
        self.query = query
        self.embedding = embedding
        self.cluster_probs = cluster_probs
        self.primary_cluster = primary_cluster
        self.result = result
        self.timestamp = timestamp


class SemanticCache:
    """
    A cluster-aware semantic query cache.

    Instead of a flat list, entries are organized by primary cluster.
    Lookups only search the top-k most probable clusters for the incoming
    query, reducing comparison count by ~K/top_k factor.

    Eviction follows LRU policy per cluster bucket.
    """

    def __init__(
        self,
        n_clusters: int = 35,
        similarity_threshold: float = 0.85,
        top_k_clusters: int = 3,
        max_entries_per_cluster: int = 100
    ):
        """
        Args:
            n_clusters: Total number of cluster buckets.
            similarity_threshold: Minimum cosine similarity for a cache hit.
            top_k_clusters: Number of clusters to search during lookup.
            max_entries_per_cluster: LRU eviction limit per bucket.
        """
        self.n_clusters = n_clusters
        self.similarity_threshold = similarity_threshold
        self.top_k_clusters = top_k_clusters
        self.max_entries_per_cluster = max_entries_per_cluster

        # Cluster-segmented storage
        self.buckets: Dict[int, List[CacheEntry]] = {
            i: [] for i in range(n_clusters)
        }

        # Statistics
        self.hit_count: int = 0
        self.miss_count: int = 0

   
    def lookup(
        self,
        query_embedding: np.ndarray,
        cluster_probs: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """
        Search the cache for a semantically similar query.

        Only searches the top-k clusters by probability, then
        returns the best match above the similarity threshold.

        Args:
            query_embedding: Normalized query embedding (384-dim).
            cluster_probs: GMM posterior probabilities (K-dim).

        Returns:
            Dict with match info on hit, or None on miss.
        """
        top_clusters = np.argsort(cluster_probs)[-self.top_k_clusters:]

        best_entry: Optional[CacheEntry] = None
        best_sim: float = -1.0

        for cluster_id in top_clusters:
            for entry in self.buckets[int(cluster_id)]:
                sim = cosine_similarity(query_embedding, entry.embedding)
                if sim >= self.similarity_threshold and sim > best_sim:
                    best_sim = sim
                    best_entry = entry

        if best_entry:
            self.hits += 1
            self.cluster_usage[best_entry.primary_cluster] += 1
            # LRU: refresh timestamp on access
            best_entry.timestamp = time.time()
            return {
                "cache_hit": True,
                "matched_query": best_entry.query,
                "similarity_score": round(best_sim, 4),
                "result": best_entry.result,
                "dominant_cluster": best_entry.primary_cluster,
            }

        self.misses += 1
        return None

    def store(
        self,
        query: str,
        embedding: np.ndarray,
        cluster_probs: np.ndarray,
        result: str
    ) -> None:
        """
        Insert a new entry into the appropriate cluster bucket.

        If the bucket exceeds its capacity, the least recently
        used entry is evicted.

        Args:
            query: Original query text.
            embedding: Normalized query embedding.
            cluster_probs: GMM posterior probabilities.
            result: Computed search result to cache.
        """
        primary_cluster = int(np.argmax(cluster_probs))

        entry = CacheEntry(
            query=query,
            embedding=embedding,
            cluster_probs=cluster_probs,
            primary_cluster=primary_cluster,
            result=result,
            timestamp=time.time()
        )

        bucket = self.buckets[primary_cluster]
        bucket.append(entry)

        # LRU eviction
        if len(bucket) > self.max_entries_per_cluster:
            bucket.sort(key=lambda e: e.timestamp)
            self.buckets[primary_cluster] = bucket[1:]  # remove oldest

    def clear(self) -> None:
        """Flush all cache entries and reset statistics."""
        for cluster_id in self.buckets:
            self.buckets[cluster_id] = []
        self.hits = 0
        self.misses = 0
        self.cluster_usage = {i: 0 for i in range(self.n_clusters)}  # Track query distribution

    def stats(self) -> Dict[str, Any]:
        """Return cache performance statistics."""
        total = sum(len(b) for b in self.buckets.values())
        total_lookups = self.hits + self.misses
        hit_rate = (
            self.hits / total_lookups if total_lookups > 0 else 0.0
        )
        return {
            "total_entries": total,
            "hit_count": self.hits,
            "miss_count": self.misses,
            "hit_rate": round(hit_rate, 4),
            "cluster_distribution": self.cluster_usage
        }
