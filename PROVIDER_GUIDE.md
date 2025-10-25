# Provider System Guide

This document explains the dependency injection system for easily switching between different providers.

## Architecture Overview

The application uses dependency injection with abstract interfaces, allowing easy swapping of:

- **Embedding Providers**: BGE-M3, Voyage AI, etc.
- **LLM Providers**: Gemini, OpenAI, etc.
- **Database Providers**: Qdrant, Pinecone, etc.
- **Reranker Providers**: Flag, Voyage, etc.

## Directory Structure

```
app/
├── providers/
│   ├── embedding/
│   │   ├── embedding_provider_abc.py      # Abstract interface
│   │   ├── bge_provider.py             # BGE-M3 implementation
│   │   └── voyage_provider.py           # Voyage AI implementation
│   ├── llm/
│   │   ├── llm_provider_abc.py           # Abstract interface
│   │   └── gemini_provider.py           # Gemini implementation
│   ├── database/
│   │   ├── database_provider_abc.py     # Abstract interface
│   │   └── qdrant_provider.py           # Qdrant implementation
│   └── reranker/
│       ├── reranker_provider_abc.py     # Abstract interface
│       └── flag_provider.py             # FlagReranker implementation
├── services/
│   ├── ingestion_service.py              # Document ingestion logic
│   ├── chat_service.py                 # Chat orchestration
├── container.py                           # Dependency injection container
└── main.py                              # FastAPI application
```

## Configuration

### Embedding Providers
```env
EMBEDDING_PROVIDER=bge        # Options: bge, voyage
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=cuda          # Options: cpu, cuda, auto
# VOYAGE_API_KEY=your_key    # Required only if using voyage provider
```

### LLM Providers
```env
LLM_PROVIDER=gemini            # Options: gemini, openai
GEMINI_API_KEY=your_api_key
```

### Database Providers
```env
DATABASE_PROVIDER=qdrant         # Options: qdrant, pinecone
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=jp_docs_semantic
```

### Reranker Providers
```env
RERANKER_PROVIDER=flag         # Options: flag, voyage
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_DEVICE=cuda
```

## Adding New Providers

### 1. Create Abstract Interface
```python
# app/providers/[category]/[category]_provider_abc.py
from abc import ABC, abstractmethod
from typing import [required imports]

class [Category]ProviderABC(ABC):
    @abstractmethod
    def required_method(self, params) -> return_type:
        pass

    @property
    @abstractmethod
    def required_property(self) -> str:
        pass
```

### 2. Implement Provider
```python
# app/providers/[category]/new_provider.py
from .[category]_provider_abc import [Category]ProviderABC

class NewProvider([Category]ProviderABC):
    def __init__(self, config_params):
        # Initialize with configuration
        pass

    def required_method(self, params) -> return_type:
        # Implementation
        pass

    @property
    def required_property(self) -> str:
        return "provider_name"
```

### 3. Update Container
```python
# app/container.py
def get_[category]_provider(self):
    if self._[category]_provider is None:
        provider_type = getattr(config.[category], 'provider_type', 'default').lower()

        if provider_type == 'new':
            self._[category]_provider = NewProvider(
                # Config parameters
            )
        else:
            # Fallback to existing providers
            pass

    return self._[category]_provider
```

### 4. Update Configuration
```python
# app/config.py
@dataclass
class [Category]Config:
    provider_type: str = "default"  # default, new
    # Add new configuration fields
```

## Usage Examples

### Using Different Embeddings
```env
# BGE-M3 (default)
EMBEDDING_PROVIDER=bge
EMBEDDING_MODEL=BAAI/bge-m3

# Voyage AI
EMBEDDING_PROVIDER=voyage
EMBEDDING_MODEL=voyage-3
VOYAGE_API_KEY=your_voyage_key
```

### Switching LLMs
```env
# Gemini (default)
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_key

# OpenAI (would need implementation)
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_key
```

## Benefits

- **Easy Swapping**: Change providers via environment variables
- **Testability**: Mock providers for unit testing
- **Extensibility**: Add new providers without changing core logic
- **Maintainability**: Clean separation of concerns
- **Flexibility**: Runtime dependency injection

## API Endpoints

- `GET /info` - Shows current provider configuration
- `GET /health` - Basic health check
- `POST /session/{session_id}/clear` - Clear conversation memory
- `POST /chat` - Chat with RAG (uses injected providers)
- `POST /ingest` - Ingest documents (uses injected providers)