import uuid
from typing import Any, Dict, List

from FlagEmbedding import FlagReranker
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class VectorIndex:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection: str = "docs",
        dim: int = 1024,
        url: str = None,
    ):
        self.collection = collection
        # Prefer URL if provided, otherwise use host:port
        if url:
            self.client = QdrantClient(url=url)
        else:
            self.client = QdrantClient(host=host, port=port)
        # create if not exists; cosine for normalized vectors
        exists = False
        try:
            info = self.client.get_collection(collection_name=collection)
            exists = True
            dim = info.config.params.vectors.size  # keep existing dim
        except Exception:
            pass
        if not exists:
            self.client.recreate_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, vectors, payloads):
        points = [
            PointStruct(id=str(uuid.uuid4()), vector=v.tolist(), payload=pl)
            for v, pl in zip(vectors, payloads)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, vector, k=8) -> List[Dict[str, Any]]:
        res = self.client.search(
            collection_name=self.collection,
            query_vector=vector.tolist(),
            limit=k,
            with_payload=True,
            score_threshold=None,
        )
        return [{"score": x.score, "payload": x.payload} for x in res]


class OptionalReranker:
    def __init__(
        self,
        model_name: str | None = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
    ):
        self.reranker = (
            FlagReranker(
                model_name, use_fp16=True if device != "cpu" else False, device=device
            )
            if model_name
            else None
        )

    def rerank(self, query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.reranker or not docs:
            return docs
        pairs = [[query, d["payload"]["text"]] for d in docs]
        scores = self.reranker.compute_score(pairs, normalize=True)
        for s, d in zip(scores, docs):
            d["score"] = float(s)
        return sorted(docs, key=lambda x: x["score"], reverse=True)
