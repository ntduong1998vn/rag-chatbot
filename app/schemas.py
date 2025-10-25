from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class IngestRequest(BaseModel):
    folder: str


class IngestResult(BaseModel):
    files: int
    chunks: int


class ChatRequest(BaseModel):
    session_id: str
    message: str
    top_k: int = 8


class Source(BaseModel):
    path: str
    chunk_id: str
    score: float
    text_preview: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
    tokens_used: Optional[Dict[str, Any]] = None
    sources: List[Source]
    tokens_used: Optional[Dict[str, Any]] = None
