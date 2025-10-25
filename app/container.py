from typing import Dict, Any

from .config import config
from .providers.embedding.bge_provider import BGEProvider
from .providers.llm.gemini_provider import GeminiProvider
from .providers.database.qdrant_provider import QdrantProvider
from .providers.reranker.flag_provider import FlagRerankerProvider

# Import Voyage provider only if available (disabled for now due to compatibility)
VOYAGE_AVAILABLE = False


class Container:
    """
    Dependency injection container for managing component lifecycles
    """

    def __init__(self):
        self._instances: Dict[str, Any] = {}
        self._embedding_provider = None
        self._llm_provider = None
        self._database_provider = None
        self._reranker_provider = None

    def get_embedding_provider(self):
        """Get or create embedding provider based on configuration"""
        if self._embedding_provider is None:
            provider_type = getattr(config.embedding, 'provider_type', 'bge').lower()

            if provider_type == 'voyage' and VOYAGE_AVAILABLE:
                self._embedding_provider = VoyageProvider(
                    model_name=config.embedding.model_name,
                    api_key=getattr(config.embedding, 'voyage_api_key', None)
                )
            elif provider_type == 'voyage' and not VOYAGE_AVAILABLE:
                print("Voyage provider selected but not available, falling back to BGE")
                self._embedding_provider = BGEProvider(
                    model_name=config.embedding.model_name,
                    device=config.embedding.device
                )
            else:  # default to BGE
                self._embedding_provider = BGEProvider(
                    model_name=config.embedding.model_name,
                    device=config.embedding.device
                )

        return self._embedding_provider

    def get_llm_provider(self):
        """Get or create LLM provider based on configuration"""
        if self._llm_provider is None:
            self._llm_provider = GeminiProvider(
                model=config.llm.model,
                api_key=config.llm.api_key
            )

        return self._llm_provider

    def get_database_provider(self):
        """Get or create database provider based on configuration"""
        if self._database_provider is None:
            self._database_provider = QdrantProvider(
                host=config.vector_db.host,
                port=config.vector_db.port,
                collection=config.vector_db.collection_name,
                dim=config.embedding.dimensions,
                url=config.vector_db.url
            )

        return self._database_provider

    def get_reranker_provider(self):
        """Get or create reranker provider based on configuration"""
        if self._reranker_provider is None:
            self._reranker_provider = FlagRerankerProvider(
                model_name=config.reranker.model_name,
                device=config.reranker.device
            )

        return self._reranker_provider

    def register_instance(self, name: str, instance: Any):
        """Register a specific instance for dependency injection"""
        self._instances[name] = instance

    def get_instance(self, name: str):
        """Get a registered instance"""
        return self._instances.get(name)

    def clear_cache(self):
        """Clear all cached provider instances"""
        self._embedding_provider = None
        self._llm_provider = None
        self._database_provider = None
        self._reranker_provider = None


# Global container instance
container = Container()