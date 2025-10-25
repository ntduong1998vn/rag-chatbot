import uuid
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .database_provider_abc import DatabaseProviderABC


class QdrantProvider(DatabaseProviderABC):
    """Qdrant vector database provider implementation"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection: str = "docs",
        dim: int = 1024,
        url: str = None,
    ):
        self._collection_name = collection
        self._is_healthy = False

        # Prefer URL if provided, otherwise use host:port
        if url:
            self.client = QdrantClient(url=url)
        else:
            self.client = QdrantClient(host=host, port=port)

        # Create collection if not exists; cosine for normalized vectors
        try:
            self.client.get_collection(collection_name=collection)
            self._is_healthy = True
            # Keep existing dimension from collection
        except Exception:
            self.client.recreate_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            self._is_healthy = True

    def upsert(self, vectors, payloads):
        """Insert/update vectors and their payloads"""
        points = [
            PointStruct(id=str(uuid.uuid4()), vector=v.tolist(), payload=pl)
            for v, pl in zip(vectors, payloads)
        ]
        self.client.upsert(collection_name=self._collection_name, points=points)

    def search(self, vector, k: int = 8) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        try:
            res = self.client.search(
                collection_name=self._collection_name,
                query_vector=vector.tolist(),
                limit=k,
                with_payload=True,
                score_threshold=None,
            )
            return [{"score": x.score, "payload": x.payload} for x in res]
        except Exception as e:
            print(f"Search error: {e}")
            return []

    def delete_collection(self):
        """Delete the entire collection"""
        try:
            self.client.delete_collection(collection_name=self._collection_name)
            self._is_healthy = False
        except Exception as e:
            print(f"Delete collection error: {e}")

    @property
    def collection_name(self) -> str:
        """Get the name of the current collection"""
        return self._collection_name

    @property
    def is_healthy(self) -> bool:
        """Check if the database connection is healthy"""
        return self._is_healthy
