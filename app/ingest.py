import glob
import os
from typing import Dict, List, Tuple

from .models.embedding import LocalEmbedder
from .retriever import VectorIndex
from .utils.chunking import split_japanese
from .utils.loaders import detect_and_extract


def ingest_folder(
    folder: str, embedder: LocalEmbedder, index: VectorIndex
) -> Tuple[int, int]:
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
        paths.extend(glob.glob(os.path.join(folder, ext), recursive=True))

    all_payloads: List[Dict] = []
    all_texts: List[str] = []

    for p in sorted(set(paths)):
        text = detect_and_extract(p)
        if not text or not text.strip():
            continue
        chunks = split_japanese(text, max_chars=800, overlap=80)
        for i, ch in enumerate(chunks):
            payload = {"path": p, "chunk_id": f"{os.path.basename(p)}::{i}", "text": ch}
            all_payloads.append(payload)
            all_texts.append(ch)

    if not all_texts:
        return 0, 0

    vecs = embedder.embed(all_texts)
    index.upsert(vecs, all_payloads)
    return len(set(paths)), len(all_texts)
    vecs = embedder.embed(all_texts)
    index.upsert(vecs, all_payloads)
    return len(set(paths)), len(all_texts)
