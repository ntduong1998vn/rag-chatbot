from abc import ABC, abstractmethod
from typing import Any, Dict, List


class RerankerProviderABC(ABC):
    """Abstract base class for reranker providers"""

    @abstractmethod
    def rerank(self, query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank documents based on query relevance

        Args:
            query: Query string
            docs: List of document dictionaries with 'payload' containing 'text'

        Returns:
            List of documents sorted by relevance score, each with 'score' field
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Get the name of the reranker model

        Returns:
            String representing the model name
        """
        pass

    @property
    @abstractmethod
    def is_enabled(self) -> bool:
        """
        Check if the reranker is enabled

        Returns:
            Boolean indicating if reranker is active
        """
        pass