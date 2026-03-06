"""
Embedder module wrapping SentenceTransformers for query encoding.

Provides a singleton-style interface that loads the model once
and exposes a simple encode() method returning a normalized vector.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize


class Embedder:
    """
    Wraps the SentenceTransformer model for consistent embedding generation.

    The model is loaded once on initialization and reused for all queries.
    All output embeddings are L2-normalized so that cosine similarity
    equals the dot product.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            model_name: HuggingFace model identifier.
        """
        self.model_name = model_name
        print(f"[Embedder] Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name, device="cpu")
        print("[Embedder] Model loaded.")

    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single text string into a normalized embedding vector.

        Args:
            text: Input text to embed.

        Returns:
            1-D numpy array of shape (384,), L2-normalized.
        """
        embedding = self.model.encode([text], show_progress_bar=False)
        embedding = normalize(embedding, norm="l2", axis=1)
        return embedding[0]
