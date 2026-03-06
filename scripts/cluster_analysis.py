import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os

# Paths
MEMBERSHIP_PATH = "models/doc_cluster_membership.npy"
CORPUS_PATH = "data/processed_corpus.json"
INDEX_PATH = "models/doc_index.json"
OUTPUT_DIR = "docs"
VIS_PATH = os.path.join(OUTPUT_DIR, "cluster_visualization.png")

def analyze_clusters():
    print("Loading clustering data...")
    membership = np.load(MEMBERSHIP_PATH)
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        doc_index = json.load(f)

    n_docs, n_clusters = membership.shape
    print(f"Dataset: {n_docs} documents, {n_clusters} clusters")

    # 1. Primary cluster assignment
    primary_clusters = np.argmax(membership, axis=1)
    cluster_counts = np.bincount(primary_clusters, minlength=n_clusters)

    print("\n--- Cluster Interpretability ---")
    for cid in range(n_clusters):
        count = cluster_counts[cid]
        if count == 0: continue
        
        print(f"\nCluster {cid} ({count} documents)")
        print("-" * 20)
        
        # Get top 5 documents by membership strength for this cluster
        top_indices = np.argsort(membership[:, cid])[-5:][::-1]
        for idx in top_indices:
            doc = corpus[idx]
            meta = doc_index[idx]
            strength = membership[idx, cid]
            print(f"[{strength:.2f}] {meta['newsgroup']}: {doc['text'][:80]}...")

    # 2. Ambiguity Analysis
    print("\n--- Ambiguity Analysis (Top Multi-cluster docs) ---")
    entropy = -np.sum(membership * np.log(membership + 1e-10), axis=1)
    ambiguous_indices = np.argsort(entropy)[-5:][::-1]
    
    for idx in ambiguous_indices:
        meta = doc_index[idx]
        probs = membership[idx]
        top_c = np.argsort(probs)[-2:][::-1]
        print(f"\nDoc {idx} ({meta['newsgroup']})")
        print(f"  Cluster {top_c[0]}: {probs[top_c[0]]:.2f}")
        print(f"  Cluster {top_c[1]}: {probs[top_c[1]]:.2f}")

    # 3. Visualization
    print("\nGenerating cluster visualization...")
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(membership)  # Visualize membership space

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=primary_clusters, cmap='tab20', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Cluster ID')
    plt.title("Semantic Document Clusters (2D Projection of GMM Membership)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(VIS_PATH)
    print(f"Visualization saved to {VIS_PATH}")

if __name__ == "__main__":
    analyze_clusters()
