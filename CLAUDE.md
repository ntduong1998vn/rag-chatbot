# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python RAG (Retrieval-Augmented Generation) chatbot project using FastAPI. The project implements a local chatbot with vector storage and document ingestion capabilities:

- `app/main.py` - Main FastAPI application with chat and ingest endpoints
- `app/ingest.py` - Document ingestion and processing
- `app/memory.py` - In-memory conversation history management
- `app/retriever.py` - Vector search and retrieval functionality
- `app/models/` - LLM and embedding model implementations
- `app/utils/` - Utility functions for chunking and loading documents
- `app/schemas.py` - Pydantic models for API requests/responses
- `pyproject.toml` - Project configuration with comprehensive dependencies
- `docker-compose.yaml` - Qdrant vector database configuration
- `data/` - Directory for document storage
- `qdrant_storage/` - Persistent vector database storage

## Development Commands

### Installing Dependencies
This project uses `uv` for dependency management:

```bash
uv sync
```

### Running with Docker (Recommended)
Start the vector database first:

```bash
docker-compose up -d
```

### Running the Application
Run the FastAPI development server:

```bash
uvicorn app.main:app --reload
```

Or alternatively:
```bash
python -m app.main
```

## Project Structure

- **Modular architecture**: Code organized into logical modules under `app/`
- **Python 3.14+ required**: Specified in `pyproject.toml`
- **FastAPI framework**: Web API with endpoints for chat and ingestion
- **RAG capabilities**: Document ingestion, vector search, and context-aware responses
- **Local models**: Uses BGE-m3 embeddings and Qwen2.5-7B LLM
- **Vector storage**: Qdrant for efficient similarity search
- **Memory management**: Conversation history tracking with session support

## API Endpoints

- `POST /ingest` - Ingest documents from a folder into the vector store
- `POST /chat` - Chat with the RAG system, returns responses with sources
- `GET /healthz` - Health check endpoint

## Key Information

- The project implements a complete RAG chatbot system
- Dependencies include FastAPI, Qdrant, sentence-transformers, and Google GenAI
- Supports document ingestion from various formats (PDF, text, etc.)
- Uses local embedding models (BGE-m3) and LLM (Qwen2.5-7B)
- Vector database runs on Docker for persistent storage
- Session-based memory management for conversation context
- Source citation and relevance scoring for responses