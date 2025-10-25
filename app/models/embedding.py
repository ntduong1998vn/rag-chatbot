from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class LocalEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None):
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: List[str]) -> np.ndarray:
        # với BGE/E5 nên normalize
        emb = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return np.array(emb, dtype="float32")

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
