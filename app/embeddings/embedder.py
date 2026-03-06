from typing import List, Union
import numpy as np

class Embedder:
    """
    Handles generation of semantic embeddings for text documents and queries.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name

    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Convert text into a numerical vector representation.
        
        Args:
            text: Single string or list of strings to embed.
            
        Returns:
            np.ndarray: Embedding vector(s).
        """
        pass
