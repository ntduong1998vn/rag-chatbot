from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from .embedding_provider_abc import EmbeddingProviderABC


class BGEProvider(EmbeddingProviderABC):
    """BGE-M3 embedding provider implementation"""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = None):
        self._model_name = model_name
        self._device = device
        self.model = SentenceTransformer(model_name, device=device)
        self._dimension = 1024  # bge-m3 default dimension

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts into vectors"""
        emb = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return np.array(emb, dtype="float32")

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text into a vector"""
        return self.embed([text])[0]

    def get_dimension(self) -> int:
        """Get the dimension of the embedding vectors"""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Get the name of the embedding model"""
        return self._model_name