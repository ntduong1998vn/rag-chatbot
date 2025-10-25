# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python RAG (Retrieval-Augmented Generation) chatbot project using FastAPI with a modern dependency injection architecture. The project implements a flexible chatbot system with pluggable providers for embeddings, LLMs, vector databases, and rerankers:

- `app/main.py` - Main FastAPI application with REST API endpoints
- `app/container.py` - Dependency injection container for provider management
- `app/config.py` - Centralized configuration system with environment variable support
- `app/services/` - Business logic layer (ingestion_service, chat_service)
- `app/providers/` - Pluggable provider system:
  - `embedding/` - Embedding providers (BGE-M3, Voyage AI)
  - `llm/` - Language model providers (Gemini, extensible for others)
  - `database/` - Vector database providers (Qdrant, extensible for others)
  - `reranker/` - Reranking providers (FlagReranker, Voyage)
- `app/memory.py` - In-memory conversation history management
- `app/utils/` - Utility functions for chunking and loading documents
- `app/schemas.py` - Pydantic models for API requests/responses
- `pyproject.toml` - Project configuration with comprehensive dependencies
- `docker-compose.yaml` - Qdrant vector database configuration
- `data/` - Directory for document storage
- `storage/` - Persistent application storage directory
- `PROVIDER_GUIDE.md` - Detailed guide for the provider system

## Development Commands

### Installing Dependencies
This project uses `uv` for dependency management:

```bash
uv sync
```

### Prerequisites
The application requires a vector database. Start Qdrant first:

```bash
docker-compose up -d
```

### Running the Application
**Option A: Using uvicorn (recommended for development)**
```bash
uv run uvicorn app.main:app --reload
```

**Option B: Using Python module syntax**
```bash
uv run python -m app.main
```

The application will be available at:
- Main API: `http://localhost:8000`
- Interactive docs: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

## Architecture

### Provider System
The application uses a dependency injection pattern with pluggable providers:

- **Embedding Providers**: BGE-M3 (local), Voyage AI (cloud)
- **LLM Providers**: Gemini (extensible for OpenAI, Claude, etc.)
- **Database Providers**: Qdrant (extensible for Pinecone, etc.)
- **Reranker Providers**: FlagReranker (local), Voyage AI (cloud)

### Configuration
All components are configured via environment variables or `.env` file:

```env
# Embedding Configuration
EMBEDDING_PROVIDER=bge        # Options: bge, voyage
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=cuda          # Options: cpu, cuda, auto

# LLM Configuration
LLM_PROVIDER=gemini            # Options: gemini, openai
GEMINI_API_KEY=your_api_key

# Database Configuration
DATABASE_PROVIDER=qdrant       # Options: qdrant, pinecone
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=jp_docs_semantic

# Reranker Configuration (Optional)
RERANKER_PROVIDER=flag         # Options: flag, voyage
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

### Key Features
- **Modular architecture**: Code organized into logical modules under `app/`
- **Dependency injection**: Runtime provider switching via configuration
- **Python 3.14+ required**: Specified in `pyproject.toml`
- **FastAPI framework**: Web API with endpoints for chat and ingestion
- **RAG capabilities**: Document ingestion, vector search, and context-aware responses
- **Flexible models**: Support for both local and cloud-based models
- **Vector storage**: Qdrant for efficient similarity search
- **Memory management**: Conversation history tracking with session support
- **Easy testing**: Mock providers for unit testing

## API Endpoints

### Core Endpoints
- `POST /ingest` - Ingest documents from a folder into the vector store
- `POST /chat` - Chat with the RAG system, returns responses with sources

### System Endpoints
- `GET /health` - Basic health check endpoint
- `GET /info` - Get information about current providers and configuration
- `POST /session/{session_id}/clear` - Clear conversation memory for a session

### Interactive Documentation
- `GET /docs` - Interactive Swagger UI documentation
- `GET /redoc` - Alternative ReDoc documentation

## Key Information

- The project implements a complete RAG chatbot system with modern dependency injection architecture
- Dependencies include FastAPI, Qdrant, sentence-transformers, Google GenAI, and Voyage AI
- Supports document ingestion from various formats (PDF, text, etc.) using unstructured
- Flexible model support: Local embeddings (BGE-m3), cloud LLMs (Gemini), rerankers (Flag, Voyage)
- Vector database runs on Docker for persistent storage
- Session-based memory management for conversation context
- Source citation and relevance scoring for responses
- Easy provider switching via environment variables
- Centralized configuration system with `.env` file support
- Comprehensive provider documentation available in `PROVIDER_GUIDE.md`

## Getting Started

1. **Clone and setup**:
   ```bash
   git clone <repository>
   cd chatbot
   uv sync
   ```

2. **Configure environment** (create `.env` file):
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   EMBEDDING_PROVIDER=bge
   LLM_PROVIDER=gemini
   ```

3. **Start vector database**:
   ```bash
   docker-compose up -d
   ```

4. **Run the application**:
   ```bash
   uv run uvicorn app.main:app --reload
   ```

5. **Test the system**:
   - Ingest documents: `POST /ingest` with folder path
   - Chat with system: `POST /chat` with message
   - Check configuration: `GET /info`