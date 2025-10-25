from abc import ABC, abstractmethod
from typing import List
import numpy as np


class EmbeddingProviderABC(ABC):
    """Abstract base class for embedding providers"""

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts into vectors

        Args:
            texts: List of text strings to embed

        Returns:
            Numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        pass

    @abstractmethod
    def embed_one(self, text: str) -> np.ndarray:
        """
        Embed a single text into a vector

        Args:
            text: Text string to embed

        Returns:
            Numpy array of embedding with shape (embedding_dim,)
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors

        Returns:
            Integer representing the embedding dimension
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Get the name of the embedding model

        Returns:
            String representing the model name
        """
        pass