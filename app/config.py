import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class LLMConfig:
    """Configuration for Language Model"""

    provider_type: str = "gemini"  # gemini, openai, etc.
    model: str = "gemini-2.5-flash-lite"
    api_key: Optional[str] = None
    max_tokens: int = 700
    temperature: float = 0.3

    def __post_init__(self):
        env_provider = os.getenv("LLM_PROVIDER")
        if env_provider:
            self.provider_type = env_provider.lower()

        if self.api_key is None:
            self.api_key = os.getenv("GEMINI_API_KEY")


@dataclass
class EmbeddingConfig:
    """Configuration for Embedding Model"""

    provider_type: str = "bge"  # bge, voyage
    model_name: str = "BAAI/bge-m3"
    device: Optional[str] = None
    dimensions: int = 1024  # bge-m3 default dimension
    voyage_api_key: Optional[str] = None

    def __post_init__(self):
        # Allow override from environment
        env_provider = os.getenv("EMBEDDING_PROVIDER")
        if env_provider:
            self.provider_type = env_provider.lower()

        env_model = os.getenv("EMBEDDING_MODEL")
        if env_model:
            self.model_name = env_model

        env_device = os.getenv("EMBEDDING_DEVICE")
        if env_device:
            self.device = env_device

        env_voyage_key = os.getenv("VOYAGE_API_KEY")
        if env_voyage_key:
            self.voyage_api_key = env_voyage_key


@dataclass
class VectorDBConfig:
    """Configuration for Vector Database"""

    provider_type: str = "qdrant"  # qdrant, pinecone, etc.
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "jp_docs_semantic"
    url: Optional[str] = None

    def __post_init__(self):
        # Allow override from environment
        env_provider = os.getenv("DATABASE_PROVIDER")
        if env_provider:
            self.provider_type = env_provider.lower()

        env_host = os.getenv("QDRANT_HOST")
        if env_host:
            self.host = env_host

        env_port = os.getenv("QDRANT_PORT")
        if env_port:
            self.port = int(env_port)

        env_url = os.getenv("QDRANT_URL")
        if env_url:
            self.url = env_url

        env_collection = os.getenv("QDRANT_COLLECTION")
        if env_collection:
            self.collection_name = env_collection


@dataclass
class RerankerConfig:
    """Configuration for Reranker"""

    provider_type: str = "flag"  # flag, voyage, etc.
    model_name: Optional[str] = None  # Set to "BAAI/bge-reranker-v2-m3" to enable
    device: Optional[str] = None

    def __post_init__(self):
        env_provider = os.getenv("RERANKER_PROVIDER")
        if env_provider:
            self.provider_type = env_provider.lower()

        env_model = os.getenv("RERANKER_MODEL")
        if env_model:
            self.model_name = env_model

        env_device = os.getenv("RERANKER_DEVICE")
        if env_device:
            self.device = env_device


@dataclass
class AppConfig:
    """General application configuration"""

    system_prompt: str = (
        "You are a helpful assistant for Retrieval-Augmented Generation. "
        "Answer in the user's language. Use the provided context snippets if relevant. "
        "If the answer is not in the context, say you don't have enough information. "
        "Cite sources by filename and chunk id."
    )
    max_memory_turns: int = 12
    default_top_k: int = 4
    persist_dir: str = "./storage"

    def __post_init__(self):
        env_persist_dir = os.getenv("PERSIST_DIR")
        if env_persist_dir:
            self.persist_dir = env_persist_dir


class Config:
    """Central configuration class"""

    def __init__(self):
        self.llm = LLMConfig()
        self.embedding = EmbeddingConfig()
        self.vector_db = VectorDBConfig()
        self.reranker = RerankerConfig()
        self.app = AppConfig()

    @classmethod
    def get_instance(cls) -> "Config":
        """Singleton pattern for global access"""
        if not hasattr(cls, "_instance"):
            cls._instance = cls()
        return cls._instance


# Global configuration instance
config = Config.get_instance()
