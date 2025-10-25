from fastapi import FastAPI, HTTPException

from .config import config
from .container import container
from .services.ingestion_service import IngestionService
from .services.chat_service import ChatService
from .schemas import ChatRequest, ChatResponse, IngestRequest, IngestResult

app = FastAPI(title="Local RAG JP Chatbot", version="0.1.0")

# --- Services using dependency injection ---
ingestion_service = IngestionService()
chat_service = ChatService()


@app.post("/ingest", response_model=IngestResult)
def ingest(req: IngestRequest):
    """Ingest documents from a folder"""
    files, chunks = ingestion_service.ingest_folder(req.folder)
    return IngestResult(files=files, chunks=chunks)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Chat with RAG system"""
    answer, sources = chat_service.chat(
        message=req.message,
        session_id=req.session_id,
        top_k=req.top_k
    )
    return ChatResponse(answer=answer, sources=sources)


@app.get("/health")
def health():
    """Health check endpoint"""
    return {"ok": True}


@app.get("/info")
def info():
    """Get information about current providers and configuration"""
    return {
        "embedding": ingestion_service.get_embedding_info(),
        "database": ingestion_service.get_database_info(),
        "reranker": {
            "model": container.get_reranker_provider().model_name,
            "enabled": container.get_reranker_provider().is_enabled
        },
        "llm": {
            "model": container.get_llm_provider().model_name
        }
    }


@app.post("/session/{session_id}/clear")
def clear_session(session_id: str):
    """Clear conversation memory for a session"""
    chat_service.clear_session(session_id)
    return {"message": f"Session {session_id} cleared"}
