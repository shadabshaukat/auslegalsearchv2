"""
FastAPI backend for AUSLegalSearchv2.
- Brings REST endpoints for all major app, embedding, search, RAG and chat functions.
- All endpoints secured with HTTP Basic Auth; credentials via FASTAPI_API_USER / FASTAPI_API_PASS env vars.
- Enables alternative frontends (e.g. Gradio) and automation.
"""

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from typing import List, Optional, Any
import os
import secrets

from db.store import (
    create_user, get_user_by_email, hash_password, check_password,
    add_document, add_embedding, start_session, get_session, complete_session, fail_session, update_session_progress,
    search_vector, search_bm25, get_chat_session, save_chat_session,
    get_active_sessions, get_resume_sessions,
)
from embedding.embedder import Embedder
from rag.rag_pipeline import RAGPipeline, list_ollama_models
from ingest.loader import walk_legal_files, parse_txt, parse_html, chunk_document

app = FastAPI(
    title="AUSLegalSearchv2 API",
    description="REST API for legal vector search, ingestion, RAG, chat, and all pipeline functionality.",
    version="0.17"
)

security = HTTPBasic()
def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    api_user = os.environ.get("FASTAPI_API_USER", "legal_api")
    api_pass = os.environ.get("FASTAPI_API_PASS", "letmein")
    correct_username = secrets.compare_digest(credentials.username, api_user)
    correct_password = secrets.compare_digest(credentials.password, api_pass)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect API credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

### User endpoints
class UserCreateReq(BaseModel):
    email: str
    password: str
    name: Optional[str] = None

@app.post("/users", tags=["users"])
def api_create_user(user: UserCreateReq, _: str = Depends(get_current_user)):
    db_user = get_user_by_email(user.email)
    if db_user:
        raise HTTPException(400, detail="Email already registered")
    return create_user(email=user.email, password=user.password, name=user.name)

### Ingestion endpoints
class IngestStartReq(BaseModel):
    directory: str
    session_name: str

@app.post("/ingest/start", tags=["ingest"])
def api_ingest_start(req: IngestStartReq, _: str = Depends(get_current_user)):
    # Kicks off a new ingestion session (main process). DOES NOT spawn workers—just metadata.
    session = start_session(session_name=req.session_name, directory=req.directory)
    return {"session_name": session.session_name, "status": session.status, "started_at": session.started_at}

@app.get("/ingest/sessions", tags=["ingest"])
def api_active_ingest_sessions(_: str = Depends(get_current_user)):
    return [s.session_name for s in get_active_sessions()]

@app.post("/ingest/stop", tags=["ingest"])
def api_stop_ingest(session_name: str, _: str = Depends(get_current_user)):
    fail_session(session_name)
    return {"session": session_name, "status": "stopped"}

### Document endpoints
@app.get("/documents", tags=["documents"])
def api_list_documents(_: str = Depends(get_current_user)):
    # TODO: Pagination
    from db.store import SessionLocal, Document
    with SessionLocal() as session:
        docs = session.query(Document).limit(100).all()
        return [{"id": d.id, "source": d.source, "format": d.format} for d in docs]

@app.get("/documents/{doc_id}", tags=["documents"])
def api_get_document(doc_id: int, _: str = Depends(get_current_user)):
    from db.store import SessionLocal, Document
    with SessionLocal() as session:
        d = session.query(Document).filter_by(id=doc_id).first()
        if not d:
            raise HTTPException(404, "Document not found")
        return {"id": d.id, "source": d.source, "content": d.content, "format": d.format}

### Embedding Search endpoints
class SearchReq(BaseModel):
    query: str
    top_k: int = 5

@app.post("/search/vector", tags=["search"])
def api_search_vector(req: SearchReq, _: str = Depends(get_current_user)):
    embedder = Embedder()
    query_vec = embedder.embed([req.query])[0]
    hits = search_vector(query_vec, top_k=req.top_k)
    return hits

@app.post("/search/bm25", tags=["search"])
def api_search_bm25(req: SearchReq, _: str = Depends(get_current_user)):
    return search_bm25(req.query, top_k=req.top_k)

### RAG/Llama QA
class RagReq(BaseModel):
    question: str
    context_chunks: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    custom_prompt: Optional[str] = None
    temperature: float = 0.1
    top_p: float = 0.90
    max_tokens: int = 1024
    repeat_penalty: float = 1.1

@app.post("/search/rag", tags=["search"])
def api_search_rag(req: RagReq, _: str = Depends(get_current_user)):
    rag = RAGPipeline(model="llama3")  # or expose as a param
    return rag.query(
        req.question, context_chunks=req.context_chunks, sources=req.sources,
        custom_prompt=req.custom_prompt, temperature=req.temperature,
        top_p=req.top_p, max_tokens=req.max_tokens, repeat_penalty=req.repeat_penalty
    )

### Chat Session endpoints
class ChatMsg(BaseModel):
    prompt: str

@app.post("/chat/session", tags=["chat"])
def api_chat_session(msg: ChatMsg, _: str = Depends(get_current_user)):
    # Stateless RAG chat. For stateful: persist/return session token.
    rag = RAGPipeline(model="llama3")
    results = search_bm25(msg.prompt, top_k=5)
    context_chunks = [r["text"] for r in results]
    sources = [r["source"] for r in results]
    answer = rag.query(
        msg.prompt, context_chunks=context_chunks, sources=sources
    )["answer"]
    return {
        "answer": answer,
        "sources": sources
    }

### Utility/model endpoints
@app.get("/models/ollama", tags=["models"])
def api_ollama_models(_: str = Depends(get_current_user)):
    return list_ollama_models()

@app.get("/files/ls", tags=["files"])
def api_ls(path: str, _: str = Depends(get_current_user)):
    # Secure: only allow inside data/ or current directory.
    allowed_roots = [os.getcwd(), "/home/ubuntu/data"]
    real = os.path.realpath(path)
    if not any(real.startswith(os.path.realpath(x)) for x in allowed_roots):
        raise HTTPException(403, "Not allowed")
    if os.path.isdir(real):
        return os.listdir(real)
    else:
        return [real]

@app.get("/health", tags=["utility"])
def healthcheck():
    return {"ok": True}
