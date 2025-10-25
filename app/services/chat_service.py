from typing import List, Dict

from ..container import container
from ..memory import InMemoryStore
from ..schemas import Source


class ChatService:
    """Service for chat functionality with RAG"""

    def __init__(self):
        self.embedding_provider = container.get_embedding_provider()
        self.database_provider = container.get_database_provider()
        self.reranker_provider = container.get_reranker_provider()
        self.llm_provider = container.get_llm_provider()
        self.memory = InMemoryStore(max_turns=12)  # TODO: get from config
        self.system_prompt = (
            "You are a helpful assistant for Retrieval-Augmented Generation. "
            "Answer in the user's language. Use the provided context snippets if relevant. "
            "If the answer is not in the context, say you don't have enough information. "
            "Cite sources by filename and chunk id."
        )

    def retrieve_context(self, query: str, top_k: int = 4) -> tuple:
        """
        Retrieve relevant context for a query

        Args:
            query: User query string
            top_k: Number of top results to retrieve

        Returns:
            Tuple of (context_text, sources_list)
        """
        # Embed the query
        query_vector = self.embedding_provider.embed_one(query)

        # Search for similar documents
        search_k = max(4, top_k)  # Default minimum
        hits = self.database_provider.search(query_vector, k=search_k)

        # Rerank results if available
        if self.reranker_provider.is_enabled:
            hits = self.reranker_provider.rerank(query, hits)

        # Build context and sources
        context_blocks = []
        sources = []
        for h in hits[:top_k]:
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

        return context_text, sources

    def generate_response(self, message: str, session_id: str, context_text: str) -> str:
        """
        Generate LLM response with context

        Args:
            message: User message
            session_id: Session identifier for memory
            context_text: Retrieved context text

        Returns:
            Generated response
        """
        # Get conversation history
        history_msgs = self.memory.get_context(session_id)
        user_msg = {"role": "user", "content": message}

        # Build messages for LLM
        context_msg = {"role": "system", "content": f"Context snippets:\n{context_text}"}
        msgs = history_msgs + [context_msg, user_msg]

        # Generate response
        response = self.llm_provider.chat(
            system=self.system_prompt,
            messages=msgs,
            max_tokens=700,  # TODO: get from config
            temperature=0.3  # TODO: get from config
        )

        # Save to memory
        self.memory.add(session_id, message, response)

        return response

    def chat(self, message: str, session_id: str, top_k: int = 4) -> tuple:
        """
        Complete chat workflow with RAG

        Args:
            message: User message
            session_id: Session identifier
            top_k: Number of top results to retrieve

        Returns:
            Tuple of (response_text, sources_list)
        """
        # Retrieve context
        context_text, sources = self.retrieve_context(message, top_k)

        # Generate response
        response = self.generate_response(message, session_id, context_text)

        return response, sources

    def clear_session(self, session_id: str):
        """Clear conversation memory for a session"""
        self.memory.clear(session_id)