import os
import json
import numpy as np
from typing import List, Tuple
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from joblib import dump

def load_embeddings(file_path: str) -> np.ndarray:
    """
    Loads the embedding matrix from a .npy file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Embedding file not found: {file_path}")
    embeddings = np.load(file_path)
    print(f"Embeddings loaded: {embeddings.shape}")
    return embeddings

def run_pca(embeddings: np.ndarray, n_components: int = 50) -> Tuple[np.ndarray, PCA]:
    """
    Applies PCA dimensionality reduction to the embedding matrix.
    
    Reduces from 384D to n_components (default 50) to stabilize
    GMM covariance estimation and improve clustering quality.
    """
    print(f"Applying PCA: {embeddings.shape[1]}D -> {n_components}D...")
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings)
    variance_retained = pca.explained_variance_ratio_.sum()
    print(f"Reduced embeddings: {reduced.shape}")
    print(f"PCA variance retained: {variance_retained:.1%}")
    return reduced, pca

def select_cluster_count(
    reduced_embeddings: np.ndarray,
    k_min: int = 15,
    k_max: int = 35
) -> Tuple[int, List[Tuple[int, float]]]:
    """
    Evaluates cluster counts using Bayesian Information Criterion (BIC).
    Returns the optimal k and all BIC scores.
    
    Lower BIC indicates a better model balancing fit and complexity.
    """
    print(f"\nEvaluating clusters from {k_min} to {k_max}...")
    bic_scores: List[Tuple[int, float]] = []
    
    for k in range(k_min, k_max + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            n_init=5,
            random_state=42
        )
        gmm.fit(reduced_embeddings)
        bic = gmm.bic(reduced_embeddings)
        bic_scores.append((k, bic))
        print(f"  k={k:2d}  BIC={bic:,.0f}")
    
    best_k, best_bic = min(bic_scores, key=lambda x: x[1])
    print(f"\nBest cluster count: {best_k} (BIC: {best_bic:,.0f})")
    return best_k, bic_scores

def train_gmm(reduced_embeddings: np.ndarray, n_clusters: int) -> GaussianMixture:
    """
    Trains the final Gaussian Mixture Model with the selected cluster count.
    Uses n_init=10 for the final model to ensure convergence quality.
    """
    print(f"\nTraining final GMM with k={n_clusters}...")
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="diag",
        n_init=10,
        random_state=42
    )
    gmm.fit(reduced_embeddings)
    print("GMM training complete.")
    return gmm

def compute_memberships(gmm: GaussianMixture, reduced_embeddings: np.ndarray) -> np.ndarray:
    """
    Computes the soft cluster membership probability distribution for each document.
    Each row is P(cluster_k | document), summing to 1.
    """
    print("Computing cluster membership probabilities...")
    cluster_probs = gmm.predict_proba(reduced_embeddings)
    print(f"Cluster membership matrix shape: {cluster_probs.shape}")
    return cluster_probs

def save_outputs(
    pca: PCA,
    gmm: GaussianMixture,
    cluster_probs: np.ndarray,
    output_dir: str = "models"
) -> None:
    """
    Saves all clustering artifacts: PCA model, GMM model, centroids, and memberships.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # PCA model
    pca_path = os.path.join(output_dir, "pca_model.joblib")
    print(f"Saving PCA model to {pca_path}...")
    dump(pca, pca_path)
    
    # GMM model
    gmm_path = os.path.join(output_dir, "gmm_model.joblib")
    print(f"Saving GMM model to {gmm_path}...")
    dump(gmm, gmm_path)
    
    # Cluster centroids
    centroids = gmm.means_
    centroids_path = os.path.join(output_dir, "cluster_centroids.npy")
    print(f"Saving cluster centroids to {centroids_path} (shape: {centroids.shape})...")
    np.save(centroids_path, centroids)
    
    # Document cluster membership matrix
    membership_path = os.path.join(output_dir, "doc_cluster_membership.npy")
    print(f"Saving cluster membership matrix to {membership_path}...")
    np.save(membership_path, cluster_probs.astype("float32"))

def validate(cluster_probs: np.ndarray, n_documents: int) -> None:
    """
    Validates the clustering outputs for correctness.
    """
    print("\n" + "=" * 40)
    print("Validation Checks")
    
    # Check 1: Shape
    assert cluster_probs.shape[0] == n_documents, \
        f"Row count mismatch: {cluster_probs.shape[0]} != {n_documents}"
    print(f"  [PASS] Row count matches documents: {n_documents}")
    
    # Check 2: Row sums
    row_sums = cluster_probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0), \
        f"Row sums not ~1: min={row_sums.min()}, max={row_sums.max()}"
    print(f"  [PASS] All rows sum to ~1.0 (min={row_sums.min():.6f}, max={row_sums.max():.6f})")
    
    # Check 3: No NaN
    nan_count = np.isnan(cluster_probs).sum()
    assert nan_count == 0, f"Found {nan_count} NaN values"
    print(f"  [PASS] No NaN values found")
    
    print("=" * 40)

def main():
    embeddings_path = "models/embeddings.npy"
    
    try:
        # 1. Load embeddings
        embeddings = load_embeddings(embeddings_path)
        n_documents = embeddings.shape[0]
        
        # 2. PCA reduction
        reduced, pca = run_pca(embeddings, n_components=50)
        
        # 3. Select optimal cluster count via BIC
        best_k, bic_scores = select_cluster_count(reduced, k_min=15, k_max=35)
        
        # 4. Train final GMM
        gmm = train_gmm(reduced, best_k)
        
        # 5. Compute soft memberships
        cluster_probs = compute_memberships(gmm, reduced)
        
        # 6. Validate
        validate(cluster_probs, n_documents)
        
        # 7. Save all outputs
        save_outputs(pca, gmm, cluster_probs)
        
        # Summary
        print("\n" + "=" * 40)
        print("Clustering Pipeline Complete")
        print(f"  Documents:          {n_documents}")
        print(f"  PCA dimensions:     {reduced.shape[1]}")
        print(f"  Clusters (k):       {best_k}")
        print(f"  Membership shape:   {cluster_probs.shape}")
        print("=" * 40)
        
    except Exception as e:
        print(f"Error during clustering: {e}")
        raise

if __name__ == "__main__":
    main()
