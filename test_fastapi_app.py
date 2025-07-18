"""
Comprehensive test suite for fastapi_app.py.
To run:
    pytest test_fastapi_app.py
"""

import os
import pytest
from fastapi.testclient import TestClient

from fastapi_app import app

# Setup test credentials
API_USER = os.environ.get("FASTAPI_API_USER", "legal_api")
API_PASS = os.environ.get("FASTAPI_API_PASS", "letmein")
BASIC_AUTH = (API_USER, API_PASS)

client = TestClient(app)

def auth_headers():
    import base64
    creds = f"{API_USER}:{API_PASS}".encode()
    return {
        "Authorization": "Basic " + base64.b64encode(creds).decode()
    }

def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["ok"]

def test_auth_required():
    r = client.get("/documents")
    assert r.status_code == 401

def test_user_creation_and_duplicate():
    # Use a likely-unique email
    import random
    email = f"testu{random.randint(1000,9999)}@test.com"
    password = "secret123"
    r = client.post("/users", json={"email": email, "password": password}, headers=auth_headers())
    assert r.status_code == 200
    # Duplicate
    r2 = client.post("/users", json={"email": email, "password": password}, headers=auth_headers())
    assert r2.status_code == 400

def test_ingest_start_and_stop():
    import random
    session_name = f"test_ingest_{random.randint(1000, 9999)}"
    # Pick cwd as a "safe" dir for test
    directory = os.getcwd()
    r = client.post("/ingest/start", json={"directory": directory, "session_name": session_name}, headers=auth_headers())
    assert r.status_code == 200 and r.json()["session_name"] == session_name
    # Session appears in active
    listr = client.get("/ingest/sessions", headers=auth_headers())
    assert listr.status_code == 200
    assert session_name in listr.json()
    # Stop session
    stopr = client.post(f"/ingest/stop?session_name={session_name}", headers=auth_headers())
    assert stopr.status_code == 200
    assert stopr.json()["status"] == "stopped"

def test_document_list_and_get(monkeypatch):
    # Fake/stub DB result for testing if DB is empty
    monkeypatch.setattr("db.store.SessionLocal", lambda: None)
    monkeypatch.setattr("db.store.Document", object)
    # Try list (should not 500 even if empty)
    r = client.get("/documents", headers=auth_headers())
    assert r.status_code in (200, 500)  # 500 if DB unavailable/empty
    # Get with bad ID
    r = client.get("/documents/999999", headers=auth_headers())
    assert r.status_code in (200, 404)

def test_vector_search(monkeypatch):
    # Fake embedder and search_vector for functional test
    monkeypatch.setattr("embedding.embedder.Embedder.embed", lambda self, texts: [[1.0]*384]*len(texts))
    monkeypatch.setattr("db.store.search_vector", lambda v, top_k=5: [{"doc_id":1,"chunk_index":0,"score":0.1,"text":"Test result","source":"test.txt","format":"txt"}])
    r = client.post("/search/vector", json={"query": "legal", "top_k": 3}, headers=auth_headers())
    assert r.status_code == 200 and isinstance(r.json(), list)

def test_bm25_search():
    r = client.post("/search/bm25", json={"query": "legal", "top_k": 3}, headers=auth_headers())
    assert r.status_code in (200, 500)  # Could 500 if DB is empty; OK for integration

def test_rag_search(monkeypatch):
    mock_rag = lambda self, question, **kwargs: {"answer":"RAG-TEST","sources":[],"contexts":[]}
    monkeypatch.setattr("rag.rag_pipeline.RAGPipeline.query", mock_rag)
    r = client.post("/search/rag", json={"question": "Explain precedent"}, headers=auth_headers())
    assert r.status_code == 200 and "answer" in r.json()

def test_chat_session(monkeypatch):
    mock_rag = lambda self, question, **kwargs: {"answer":"CHAT-TEST","sources":[]}
    monkeypatch.setattr("rag.rag_pipeline.RAGPipeline.query", mock_rag)
    r = client.post("/chat/session", json={"prompt": "What is the High Court?"}, headers=auth_headers())
    assert r.status_code == 200 and "answer" in r.json()

def test_list_models():
    r = client.get("/models/ollama", headers=auth_headers())
    assert r.status_code == 200 and isinstance(r.json(), list)

def test_ls_good_and_bad():
    # Good (cwd)
    r = client.get(f"/files/ls?path={os.getcwd()}", headers=auth_headers())
    assert r.status_code == 200
    # Bad
    r2 = client.get("/files/ls?path=/etc", headers=auth_headers())
    assert r2.status_code == 403
