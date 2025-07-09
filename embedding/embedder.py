"""
Embedding interface for auslegalsearchv2.
- Loads embedding model (default: sentence-transformers/all-MiniLM-L6-v2 or user-specified)
- Provides embed(texts) method for batch embeddings
- Extensible for custom, HuggingFace, or local Ollama/Llama4-compatible models
"""

from typing import List
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    DEFAULT_MODEL = "all-MiniLM-L6-v2"
except ImportError:
    SentenceTransformer = None
    DEFAULT_MODEL = None

class Embedder:
    def __init__(self, model_name: str = None):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers not installed.")
        self.model_name = model_name or DEFAULT_MODEL
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts; returns ndarray [batch, dim].
        """
        return np.array(self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True))

    # TODO: Add support for other (esp. Llama4-compatible) embedding models
    # such as local API, HuggingFace, Instructor, etc.

"""
Usage:
from embedding.embedder import Embedder
embedder = Embedder()
vec = embedder.embed(["sample text"])
"""
