"""
Vector store interface for auslegalsearchv2.
- Defines the ORM schema for documents and embeddings.
- Provides add/search methods for plain, vector, and hybrid retrieval.
- Adds ingestion checkpointing/resume with embedding_sessions table.
- Now also includes chat sessions for chat history tracking.
- Now includes User authentication table.
"""

from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, Boolean
from sqlalchemy import select, desc, text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from pgvector.sqlalchemy import Vector
from db.connector import engine, SessionLocal
from embedding.embedder import Embedder
from datetime import datetime
import uuid
import json
import numpy as np
import os
import bcrypt

embedder = Embedder()
EMBEDDING_DIM = embedder.dimension

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=True)
    registered_google = Column(Boolean, default=False)
    google_id = Column(String, nullable=True)
    name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, nullable=True)

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    source = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    format = Column(String, nullable=False)

class Embedding(Base):
    __tablename__ = "embeddings"
    id = Column(Integer, primary_key=True)
    doc_id = Column(Integer, ForeignKey('documents.id'), index=True)
    chunk_index = Column(Integer, nullable=False)
    vector = Column(Vector(EMBEDDING_DIM), nullable=False)
    document = relationship("Document", backref="embeddings")

class EmbeddingSession(Base):
    __tablename__ = "embedding_sessions"
    id = Column(Integer, primary_key=True)
    session_name = Column(String, unique=True, nullable=False)
    directory = Column(String, nullable=False)
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    status = Column(String, nullable=False, default="active")  # active, complete, error
    last_file = Column(String, nullable=True)
    last_chunk = Column(Integer, nullable=True)
    total_files = Column(Integer, nullable=True)
    total_chunks = Column(Integer, nullable=True)
    processed_chunks = Column(Integer, nullable=True)

class EmbeddingSessionFile(Base):
    __tablename__ = "embedding_session_files"
    id = Column(Integer, primary_key=True)
    session_name = Column(String, nullable=False, index=True)
    filepath = Column(String, nullable=False)
    status = Column(String, nullable=False, default="pending") # 'pending', 'complete', 'error'
    completed_at = Column(DateTime, nullable=True)

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    ended_at = Column(DateTime, nullable=True)
    chat_history = Column(JSONB, nullable=False)
    llm_params = Column(JSONB, nullable=False)

def create_all_tables():
    Base.metadata.create_all(engine)

# --- User CRUD and Auth logic ---

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password: str, hashval: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashval.encode('utf-8'))

def create_user(email, password=None, name=None, google_id=None, registered_google=False):
    with SessionLocal() as session:
        user = User(
            email=email,
            password_hash=hash_password(password) if password else None,
            name=name,
            google_id=google_id,
            registered_google=registered_google,
            created_at=datetime.utcnow(),
            last_login=datetime.utcnow(),
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        return user

def get_user_by_email(email: str):
    with SessionLocal() as session:
        return session.query(User).filter_by(email=email).first()

def set_last_login(user_id: int):
    with SessionLocal() as session:
        user = session.query(User).filter_by(id=user_id).first()
        if user:
            user.last_login = datetime.utcnow()
            session.commit()

def get_user_by_googleid(google_id: str):
    with SessionLocal() as session:
        return session.query(User).filter_by(google_id=google_id).first()

# -- Chat Session functions --
def save_chat_session(chat_history, llm_params, ended_at=None):
    # chat_history must be serializable (list of dicts), llm_params is flat dict
    with SessionLocal() as session:
        chat_sess = ChatSession(
            chat_history=chat_history, 
            llm_params=llm_params, 
            ended_at=ended_at or datetime.utcnow()
        )
        session.add(chat_sess)
        session.commit()
        session.refresh(chat_sess)
        return chat_sess.id

def get_chat_session(chat_id):
    with SessionLocal() as session:
        return session.query(ChatSession).filter_by(id=chat_id).first()

# --- rest is unchanged (doc/embedding/session logic) ---

def start_session(session_name, directory, total_files=None, total_chunks=None):
    with SessionLocal() as session:
        sess = EmbeddingSession(
            session_name=session_name,
            directory=directory,
            started_at=datetime.utcnow(),
            status="active",
            total_files=total_files,
            total_chunks=total_chunks,
            processed_chunks=0
        )
        session.add(sess)
        session.commit()
        session.refresh(sess)
        return sess

def update_session_progress(session_name, last_file, last_chunk, processed_chunks):
    with SessionLocal() as session:
        sess = session.query(EmbeddingSession).filter_by(session_name=session_name).first()
        if not sess:
            return None
        sess.last_file = last_file
        sess.last_chunk = last_chunk
        sess.processed_chunks = processed_chunks
        session.commit()
        return sess

def complete_session(session_name):
    with SessionLocal() as session:
        sess = session.query(EmbeddingSession).filter_by(session_name=session_name).first()
        if not sess:
            return None
        sess.ended_at = datetime.utcnow()
        sess.status = "complete"
        session.commit()
        return sess

def fail_session(session_name):
    with SessionLocal() as session:
        sess = session.query(EmbeddingSession).filter_by(session_name=session_name).first()
        if not sess:
            return None
        sess.status = "error"
        session.commit()
        return sess

def get_active_sessions():
    with SessionLocal() as session:
        sessions = session.query(EmbeddingSession).filter(EmbeddingSession.status == "active").all()
        return sessions

def get_resume_sessions():
    with SessionLocal() as session:
        # Show all not complete, i.e. status is not "complete"
        return session.query(EmbeddingSession).filter(EmbeddingSession.status != "complete").all()

def get_session(session_name):
    with SessionLocal() as session:
        return session.query(EmbeddingSession).filter_by(session_name=session_name).first()

def add_document(doc: dict) -> int:
    with SessionLocal() as session:
        doc_obj = Document(
            source=doc["source"],
            content=doc["content"],
            format=doc["format"],
        )
        session.add(doc_obj)
        session.commit()
        session.refresh(doc_obj)
        return doc_obj.id

def add_embedding(doc_id: int, chunk_index: int, vector) -> int:
    with SessionLocal() as session:
        embed_obj = Embedding(
            doc_id=doc_id,
            chunk_index=chunk_index,
            vector=vector,
        )
        session.add(embed_obj)
        session.commit()
        session.refresh(embed_obj)
        return embed_obj.id

def search_vector(query_vec, top_k=5):
    with SessionLocal() as session:
        ndim = len(query_vec)
        array_params = []
        param_dict = {}
        for i, v in enumerate(query_vec):
            pname = f"v{i}"
            array_params.append(f":{pname}")
            param_dict[pname] = float(v)
        param_dict['topk'] = top_k
        array_sql = "ARRAY[" + ",".join(array_params) + "]::vector"
        sql = f'''
        SELECT embeddings.doc_id, embeddings.chunk_index, embeddings.vector <#> {array_sql} AS score,
               documents.content, documents.source, documents.format
        FROM embeddings
        JOIN documents ON embeddings.doc_id = documents.id
        ORDER BY embeddings.vector <#> {array_sql} ASC
        LIMIT :topk
        '''
        result = session.execute(text(sql), param_dict)
        hits = []
        for row in result:
            hits.append({
                "doc_id": row[0],
                "chunk_index": row[1],
                "score": row[2],
                "text": row[3],
                "source": row[4],
                "format": row[5],
            })
        return hits

def search_bm25(query, top_k=5):
    # Simple fallback: naive "ILIKE" for fulltext—replace w/pg_fulltext or pg_trgm for real prod
    with SessionLocal() as session:
        q = f"%{query}%"
        res = session.execute(
            text("""
                SELECT id, content, source, format
                FROM documents
                WHERE content ILIKE :q
                LIMIT :topk
            """), {"q": q, "topk": top_k}
        )
        hits = []
        for row in res:
            hits.append({
                "doc_id": row[0],
                "chunk_index": 0,
                "score": 1.0,
                "text": row[1],
                "source": row[2],
                "format": row[3],
            })
        return hits

def get_file_contents(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        try:
            with open(filepath, "r", encoding="latin-1") as f:
                return f.read()
        except Exception:
            return f"Could not read file: {filepath}"

create_all_tables()
