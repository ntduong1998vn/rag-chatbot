from fastapi import FastAPI, HTTPException

from .config import config
from .ingest import ingest_folder
from .memory import InMemoryStore
from .models.embedding import LocalEmbedder
from .models.llm import LocalLLM
from .retriever import OptionalReranker, VectorIndex
from .schemas import ChatRequest, ChatResponse, IngestRequest, IngestResult, Source

app = FastAPI(title="Local RAG JP Chatbot", version="0.1.0")

# --- Singletons using centralized configuration ---
EMBEDDER = LocalEmbedder(
    model_name=config.embedding.model_name,
    device=config.embedding.device
)

INDEX = VectorIndex(
    host=config.vector_db.host,
    port=config.vector_db.port,
    collection=config.vector_db.collection_name,
    dim=config.embedding.dimensions,
    url=config.vector_db.url
)

RERANK = OptionalReranker(
    model_name=config.reranker.model_name,
    device=config.reranker.device
)

LLM = LocalLLM(
    model=config.llm.model,
    api_key=config.llm.api_key
)

MEMORY = InMemoryStore(max_turns=config.app.max_memory_turns)

SYSTEM_PROMPT = config.app.system_prompt


@app.post("/ingest", response_model=IngestResult)
def ingest(req: IngestRequest):
    files, chunks = ingest_folder(req.folder, EMBEDDER, INDEX)
    return IngestResult(files=files, chunks=chunks)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # 1) build messages with memory
    history_msgs = MEMORY.get_context(req.session_id)
    user_msg = {"role": "user", "content": req.message}

    # 2) retrieval
    qvec = EMBEDDER.embed_one(req.message)
    hits = INDEX.search(qvec, k=max(config.app.default_top_k, req.top_k))
    hits = RERANK.rerank(req.message, hits)  # no-op if reranker disabled

    # 3) compose context
    context_blocks = []
    sources = []
    for h in hits[: req.top_k]:
        pl = h["payload"]
        context_blocks.append(f"[{pl['chunk_id']} from {pl['path']}]\n{pl['text']}")
        preview = (pl["text"][:160] + "…") if len(pl["text"]) > 160 else pl["text"]
        sources.append(
            Source(
                path=pl["path"],
                chunk_id=pl["chunk_id"],
                score=float(h["score"]),
                text_preview=preview,
            )
        )

    context_text = (
        "\n\n---\n\n".join(context_blocks) if context_blocks else "No context."
    )
    context_msg = {"role": "system", "content": f"Context snippets:\n{context_text}"}

    # 4) call LLM
    msgs = history_msgs + [context_msg, user_msg]
    answer = LLM.chat(
        system=SYSTEM_PROMPT,
        messages=msgs,
        max_tokens=config.llm.max_tokens,
        temperature=config.llm.temperature
    )

    # 5) save memory
    MEMORY.add(req.session_id, req.message, answer)

    return ChatResponse(answer=answer, sources=sources)


@app.get("/health")
def health():
    return {"ok": True}
