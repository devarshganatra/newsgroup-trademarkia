"""
Ingestion script: upload document embeddings to Qdrant Cloud.

Reads the pre-computed embeddings, document index, and corpus,
then batch-upserts all vectors with metadata payloads into Qdrant.

Usage:
    python scripts/load_vectors_to_qdrant.py
"""

import json
import sys
import os

import numpy as np

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.vector_db import VectorDB


def main():
    # ---- Load artifacts ----
    print("Loading embeddings...")
    embeddings = np.load("models/embeddings.npy")
    print(f"  Shape: {embeddings.shape}")

    print("Loading document index...")
    with open("models/doc_index.json", "r", encoding="utf-8") as f:
        doc_index = json.load(f)

    print("Loading processed corpus...")
    with open("data/processed_corpus.json", "r", encoding="utf-8") as f:
        corpus = json.load(f)

    # ---- Build payloads ----
    ids = []
    payloads = []
    for i, meta in enumerate(doc_index):
        doc = corpus[i]
        ids.append(meta["doc_id"])
        payloads.append({
            "text": doc["text"],
            "newsgroup": meta["newsgroup"],
            "length": doc.get("token_length", doc.get("char_length", 0)),
        })

    # ---- Upload to Qdrant ----
    vdb = VectorDB()
    vdb.connect()
    vdb.create_collection(vector_size=embeddings.shape[1])

    print(f"Uploading {len(ids)} vectors to Qdrant...")
    vdb.upsert_documents(ids=ids, vectors=embeddings, payloads=payloads)

    print("\n========================================")
    print("Ingestion complete.")
    print(f"  Documents uploaded: {len(ids)}")
    print(f"  Vector dimension:   {embeddings.shape[1]}")
    print("========================================")


if __name__ == "__main__":
    main()
