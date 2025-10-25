from abc import ABC, abstractmethod
from typing import Any, Dict, List


class DatabaseProviderABC(ABC):
    """Abstract base class for vector database providers"""

    @abstractmethod
    def upsert(self, vectors, payloads):
        """
        Insert/update vectors and their payloads

        Args:
            vectors: List of vectors to insert
            payloads: List of payload dictionaries corresponding to vectors
        """
        pass

    @abstractmethod
    def search(self, vector, k: int = 8) -> List[Dict[str, Any]]:
        """
        Search for similar vectors

        Args:
            vector: Query vector to search for
            k: Number of top results to return

        Returns:
            List of dictionaries containing 'score' and 'payload' keys
        """
        pass

    @abstractmethod
    def delete_collection(self):
        """
        Delete the entire collection
        """
        pass

    @property
    @abstractmethod
    def collection_name(self) -> str:
        """
        Get the name of the current collection

        Returns:
            String representing the collection name
        """
        pass

    @property
    @abstractmethod
    def is_healthy(self) -> bool:
        """
        Check if the database connection is healthy

        Returns:
            Boolean indicating connection health
        """
        pass