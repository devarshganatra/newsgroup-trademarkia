import os
import json
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

def load_corpus(file_path: str) -> List[Dict[str, Any]]:
    """
    Loads the processed corpus from a JSON file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Corpus file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def truncate_document(text: str, max_tokens: int = 512) -> str:
    """
    Truncates text to a maximum number of whitespace-separated tokens.
    """
    tokens = text.split()
    if len(tokens) > max_tokens:
        return " ".join(tokens[:max_tokens])
    return text

def batch_iterator(data: List[str], batch_size: int = 64):
    """
    Yields batches of data for processing.
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def generate_embeddings(corpus: List[Dict[str, Any]], model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64) -> np.ndarray:
    """
    Generates semantic embeddings for the corpus using SentenceTransformers.
    """
    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name, device="cpu")
    
    texts = [truncate_document(doc['text']) for doc in corpus]
    all_embeddings = []
    
    print(f"Generating embeddings for {len(texts)} documents (Batch size: {batch_size})...")
    
    # We use tqdm for progress tracking
    for batch in tqdm(batch_iterator(texts, batch_size), total=(len(texts) + batch_size - 1) // batch_size):
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.append(batch_embeddings)
        
    return np.vstack(all_embeddings)

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Normalizes embedding vectors to unit length using sklearn.
    """
    print("Normalizing embeddings to unit length...")
    return normalize(embeddings, norm='l2', axis=1)

def save_outputs(embeddings: np.ndarray, corpus: List[Dict[str, Any]], matrix_path: str, index_path: str):
    """
    Saves the embedding matrix and the document index.
    """
    # Save embedding matrix
    print(f"Saving embedding matrix to {matrix_path}...")
    np.save(matrix_path, embeddings.astype('float32'))
    
    # Create and save document index
    print(f"Saving document index to {index_path}...")
    doc_index = []
    for i, doc in enumerate(corpus):
        doc_index.append({
            "row": i,
            "doc_id": doc['id'],
            "newsgroup": doc['newsgroup']
        })
        
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(doc_index, f, ensure_ascii=False, indent=2)

def main():
    corpus_path = "data/processed_corpus.json"
    matrix_output = "models/embeddings.npy"
    index_output = "models/doc_index.json"
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    try:
        # 1. Load Corpus
        corpus = load_corpus(corpus_path)
        print(f"Documents loaded: {len(corpus)}")
        
        # 2. Generate Embeddings
        embeddings = generate_embeddings(corpus)
        
        # 3. Normalize
        embeddings = normalize_embeddings(embeddings)
        
        # 4. Save Outputs
        save_outputs(embeddings, corpus, matrix_output, index_output)
        
        # Verification Statistics
        print("\n" + "="*30)
        print("Embedding Generation Complete")
        print(f"Matrix Shape: {embeddings.shape}")
        print(f"Embedding Dimension: {embeddings.shape[1]}")
        
        # Final sanity check for normalization
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"Average Vector Norm: {norms.mean():.4f} (Expected: ~1.0000)")
        print("="*30)
        
    except Exception as e:
        print(f"Error during embedding generation: {e}")

if __name__ == "__main__":
    main()
