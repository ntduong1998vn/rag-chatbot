import glob
import os
from typing import Dict, List, Tuple

from ..container import container
from ..utils.chunking import split_japanese
from ..utils.loaders import detect_and_extract


class IngestionService:
    """Service for document ingestion and processing"""

    def __init__(self):
        self.embedding_provider = container.get_embedding_provider()
        self.database_provider = container.get_database_provider()

    def ingest_folder(self, folder_path: str) -> Tuple[int, int]:
        """
        Ingest all documents from a folder

        Args:
            folder_path: Path to folder containing documents

        Returns:
            Tuple of (files_processed, chunks_created)
        """
        paths = []
        for ext in [
            "**/*.pdf",
            "**/*.docx",
            "**/*.pptx",
            "**/*.txt",
            "**/*.md",
            "**/*.csv",
            "**/*.json",
            "**/*.html",
        ]:
            paths.extend(glob.glob(os.path.join(folder_path, ext), recursive=True))

        all_payloads: List[Dict] = []
        all_texts: List[str] = []

        for p in sorted(set(paths)):
            text = detect_and_extract(p)
            if not text or not text.strip():
                continue
            chunks = split_japanese(text, max_chars=800, overlap=80)
            for i, ch in enumerate(chunks):
                payload = {
                    "path": p,
                    "chunk_id": f"{os.path.basename(p)}::{i}",
                    "text": ch
                }
                all_payloads.append(payload)
                all_texts.append(ch)

        if not all_texts:
            return 0, 0

        vecs = self.embedding_provider.embed(all_texts)
        self.database_provider.upsert(vecs, all_payloads)
        return len(set(paths)), len(all_texts)

    def get_embedding_info(self) -> dict:
        """Get information about current embedding provider"""
        return {
            "model": self.embedding_provider.model_name,
            "dimension": self.embedding_provider.get_dimension(),
        }

    def get_database_info(self) -> dict:
        """Get information about current database provider"""
        return {
            "collection": self.database_provider.collection_name,
            "healthy": self.database_provider.is_healthy,
        }