# AusLegalSearch v2 – Legal Document Retrieval & Embedding (v0.1.0)

A robust, self-hosted legal search pipeline using chunked ingestion, fast embeddings, PostgreSQL+pgvector, and Streamlit UI for RAG (Retrieval-Augmented Generation) with Llama/LLM and flexible embedding models.

## Features

- **Directory ingestion**: Chunk and embed local legal document files (.txt, .html, .pdf, .docx) into PostgreSQL with fast batch embedding.
- **Live progress**: Track ingestion progress with file- and chunk-level progress bars.
- **Session checkpointing & resume**: Safely resume incomplete/errored ingestion jobs.
- **Hybrid search & RAG**: Fulltext, vector, and hybrid legal search with relevant context for LLM QA/summarization.
- **UI control**: Upload, search, and manage ingestion from a Streamlit GUI.
- **Model flexibility**: Choose embedding and LLM models from sidebar, add new embedding models as needed.
- **Secure**: DB credentials always masked in sidebar.

## Requirements

- Python 3.9+
- PostgreSQL 15+ with `pgvector` extension enabled
- pip, Poetry, or conda for dependency install
- Node.js and Ollama (for local `llama3/llama4` RAG, if using Ollama backend)
- MacOS or Linux (tested); Windows may work via WSL

## Quickstart (Mac/Linux, local database)
