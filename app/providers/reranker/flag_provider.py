from typing import Any, Dict, List, Optional

from FlagEmbedding import FlagReranker

from .reranker_provider_abc import RerankerProviderABC


class FlagRerankerProvider(RerankerProviderABC):
    """FlagReranker provider implementation"""

    def __init__(
        self,
        model_name: Optional[str] = "BAAI/bge-reranker-v2-m3",
        device: Optional[str] = None,
    ):
        self._model_name = model_name
        self._device = device
        self._is_enabled = model_name is not None

        if self._is_enabled:
            self.reranker = FlagReranker(
                model_name, use_fp16=True if device != "cpu" else False, device=device
            )
        else:
            self.reranker = None

    def rerank(self, query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank documents based on query relevance"""
        if not self._is_enabled or not self.reranker or not docs:
            return docs

        pairs = [[query, d["payload"]["text"]] for d in docs]
        scores = self.reranker.compute_score(pairs, normalize=True)

        for s, d in zip(scores, docs):
            d["score"] = float(s)

        return sorted(docs, key=lambda x: x["score"], reverse=True)

    @property
    def model_name(self) -> str:
        """Get the name of the reranker model"""
        return self._model_name or "disabled"

    @property
    def is_enabled(self) -> bool:
        """Check if the reranker is enabled"""
        return self._is_enabled