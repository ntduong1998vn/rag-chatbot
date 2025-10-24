# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python chatbot project using FastAPI. The project is minimal with only a few core files:

- `main.py` - Entry point with a simple "Hello from chatbot!" function
- `pyproject.toml` - Project configuration with FastAPI dependency
- `README.md` - Basic project description

## Development Commands

### Running the Application
```bash
python main.py
```

### Installing Dependencies
This project uses `uv` for dependency management (based on the presence of `uv.lock`):

```bash
uv sync
```

### Running with FastAPI
Since the project has FastAPI as a dependency, you can run the development server:

```bash
uvicorn main:app --reload
```

## Project Structure

- **Single file architecture**: All code is currently in `main.py`
- **Python 3.14+ required**: Specified in `pyproject.toml`
- **FastAPI framework**: Ready for web API development
- **Minimal setup**: Basic project structure suitable for starting a chatbot application

## Key Information

- The project uses Python with FastAPI for building web APIs
- Dependencies are managed with `uv` (modern Python package manager)
- The codebase is currently minimal with just a main function
- No existing build scripts, tests, or complex architecture
- No linting or formatting configurations present