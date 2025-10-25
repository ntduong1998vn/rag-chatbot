from typing import List
import numpy as np
import voyageai

from .embedding_provider_abc import EmbeddingProviderABC


class VoyageProvider(EmbeddingProviderABC):
    """Voyage AI embedding provider implementation"""

    def __init__(self, model_name: str = "voyage-3", api_key: str = None):
        self._model_name = model_name
        self._api_key = api_key
        self.client = voyageai.Client(api_key=api_key)

        # Voyage model dimensions
        self._dimensions = {
            "voyage-3": 1024,
            "voyage-3-lite": 512,
            "voyage-2": 1536,
            "voyage-2-lite": 768
        }

        if model_name not in self._dimensions:
            raise ValueError(f"Unsupported Voyage model: {model_name}. "
                           f"Supported models: {list(self._dimensions.keys())}")

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts into vectors"""
        embeddings = self.client.embed(
            texts, model=self._model_name, input_type="document"
        )
        return np.array(embeddings, dtype="float32")

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text into a vector"""
        embedding = self.client.embed(
            [text], model=self._model_name, input_type="document"
        )[0]
        return np.array(embedding, dtype="float32")

    def get_dimension(self) -> int:
        """Get the dimension of the embedding vectors"""
        return self._dimensions[self._model_name]

    @property
    def model_name(self) -> str:
        """Get the name of the embedding model"""
        return self._model_name