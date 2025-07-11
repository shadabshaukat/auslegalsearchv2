"""
Streamlit front end for auslegalsearchv2.
- Ensures progress bars never crash.
- Resume and start load buttons renamed.
- Sidebar now shows "App", "Chat" menu.
- Below "Open Legal Assistant" add a "Load Data" button (no click action) styled differently, centered.
"""

import streamlit as st
from pathlib import Path
from ingest import loader
from embedding.embedder import Embedder, DEFAULT_MODEL
from db.store import (
    start_session, update_session_progress, complete_session, fail_session,
    get_active_sessions, get_session, add_document, add_embedding, search_vector,
    EmbeddingSessionFile, SessionLocal
)
from rag.rag_pipeline import RAGPipeline, list_ollama_models
import os
from datetime import datetime
import subprocess
import time
import re

def get_num_gpus():
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        try:
            out = subprocess.check_output(['nvidia-smi', '--list-gpus']).decode()
            return len([l for l in out.split('\n') if 'GPU' in l])
        except Exception:
            return 0

def partition(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def get_child_gpu_sessions(parent_session):
    with SessionLocal() as session:
        pattern = f"{parent_session}-gpu%"
        return session.query(type(get_session(parent_session))).filter(type(get_session(parent_session)).session_name.like(pattern)).all()

# AUTH WALL: force login if no session, always at top
import streamlit as st
if "user" not in st.session_state:
    st.warning("You must login to continue.")
    if hasattr(st, "switch_page"):
        st.switch_page("pages/login.py")
    else:
        st.stop()

st.set_page_config(page_title="AUSLegalSearch v2", layout="wide")
st.title("AUSLegalSearch v2 – Legal Document Search, Background Embedding & RAG")

if "directories" not in st.session_state:
    st.session_state["directories"] = set()

EMBEDDING_MODELS = [
    "all-MiniLM-L6-v2",
    # Add more models as needed
]
selected_embedding_model = st.sidebar.selectbox(
    "Embedding Model (for chunk/vectorization)",
    EMBEDDING_MODELS,
    index=0
)
st.sidebar.caption(f"Embedding model in use: `{selected_embedding_model}`")

db_url = os.environ.get("AUSLEGALSEARCH_DB_URL", "")
if db_url:
    db_url_masked = re.sub(r'(://[^:]+:)[^@]+@', r'\1*****@', db_url)
    st.sidebar.markdown(f"**DB URL:** `{db_url_masked}`")
    st.sidebar.code("""
Example DB URL:
postgresql+psycopg2://username:password@host:port/databasename

   |__ protocol/driver    |__ user    |__ pass  |__ host    |__ port |__ db name
postgresql+psycopg2       myuser      secret    localhost   5432     auslegalsearch
    """, language="text")
else:
    db_url = os.environ.get("AUSLEGALSEARCH_DB_HOST", "localhost")
    db_name = os.environ.get("AUSLEGALSEARCH_DB_NAME", "auslegalsearch")
    st.sidebar.markdown(f"**DB Host:** `{db_url}`<br>**DB Name:** `{db_name}`", unsafe_allow_html=True)

model_list = list_ollama_models()
if not model_list:
    st.sidebar.warning("No Ollama models found locally. Is Ollama running?")
    model_list = ["llama3"]
selected_model = st.sidebar.selectbox("RAG LLM model (Ollama)", model_list, index=0)

# ---- Sidebar Navigation "App" and "Chat" ----
st.sidebar.markdown("""
<div style="display:flex;gap:10px;margin-bottom:6px;">
  <a href="/" target="_self" style="font-weight:bold;color:#3475ce;padding:8px 18px 8px 12px;background:#f7fafc;border-radius:7px;margin-right:6px;text-decoration:none;box-shadow:0 2px 4px #e7e7ff22;">App</a>
  <a href="/chat" target="_self" style="font-weight:bold;color:#3ebbaf;padding:8px 20px 8px 12px;background:#f6fff6;border-radius:7px;text-decoration:none;box-shadow:0 2px 4px #b0ffc322;">Chat</a>
</div>
""", unsafe_allow_html=True)

# Open Legal Assistant button and Load Data button (centered, styled)
st.sidebar.markdown("""
<a target="_blank" href="/chat" style="background:#2274a5;color:white;text-decoration:none;display:block;font-size:22px;padding:8px 16px;margin:18px auto 0 auto;border-radius:30px;width:84%;text-align:center;font-weight:bold;">
💬  Open Legal Assistant Chat
</a>
<div style="background:#2ad0b1;color:#fff;text-align:center;font-size:20px;font-weight:bold;border-radius:30px;width:67%;margin:18px auto 8px auto;padding:8px 0;box-shadow:0 2px 10px #2ad0b155;">
📦 Load Data
</div>
""", unsafe_allow_html=True)

if "session_page_state" not in st.session_state:
    st.session_state["session_page_state"] = None

leftcol, rightcol = st.sidebar.columns(2)
start_load_clicked = leftcol.button("Start Data Load", key="start_new_btn")
resume_load_clicked = rightcol.button("Resume Data Load", key="resume_sess_btn")
session_choice_made = False

if start_load_clicked:
    st.session_state["session_page_state"] = "NEW"
    session_choice_made = True
elif resume_load_clicked:
    st.session_state["session_page_state"] = "RESUME"
    session_choice_made = True
elif st.session_state["session_page_state"]:
    session_choice_made = True

def write_partition_file(partition, fname):
    with open(fname, "w") as f:
        for item in partition:
            f.write(item + "\n")

def launch_embedding_worker(session_name, embedding_model, gpu=None, partition_file=None):
    python_exec = os.environ.get("PYTHON_EXEC", "python3")
    script_path = os.path.join(os.getcwd(), "embedding_worker.py")
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    args = [python_exec, script_path, session_name, embedding_model]
    if partition_file:
        args.append("--partition_file")
        args.append(partition_file)
    proc = subprocess.Popen(args, env=env)
    return proc

def get_completed_file_count(session_name):
    sess = get_session(session_name)
    if sess:
        return get_completed_file_count_sessname(session_name)
    total = 0
    for i in range(16):
        sfx = f"{session_name}-gpu{i}"
        child_sess = get_session(sfx)
        if child_sess:
            with SessionLocal() as session:
                count = session.query(EmbeddingSessionFile).filter_by(session_name=sfx, status="complete").count()
                total += count
    return total

def get_completed_file_count_sessname(sessname):
    with SessionLocal() as session:
        count = session.query(EmbeddingSessionFile).filter_by(session_name=sessname, status="complete").count()
        return count

def get_total_files(session_name):
    sess = get_session(session_name)
    if sess:
        return sess.total_files or 0
    total = 0
    for i in range(16):
        sfx = f"{session_name}-gpu{i}"
        child_sess = get_session(sfx)
        if child_sess:
            total += child_sess.total_files or 0
    return total

def get_processed_chunks(session_name):
    sess = get_session(session_name)
    if sess:
        return sess.processed_chunks or 0
    total = 0
    for i in range(16):
        sfx = f"{session_name}-gpu{i}"
        child_sess = get_session(sfx)
        if child_sess and getattr(child_sess, "processed_chunks", None) is not None:
            total += child_sess.processed_chunks
    return total

def poll_session_progress_bars(session_name, file_bar=None, chunk_bar_text=None, stat_line=None):
    completed_files = get_completed_file_count(session_name)
    total_files = get_total_files(session_name)
    processed_chunks = get_processed_chunks(session_name)
    sess = get_session(session_name)
    stat = sess.status if sess else '-'
    if total_files > 0:
        file_ratio = min(1.0, completed_files / total_files)
    else:
        file_ratio = 0
    if stat_line: stat_line.write(f"**Status:** {stat}")
    if file_bar: file_bar.progress(file_ratio)
    if file_bar: file_bar_text.write(f"Files embedded: {completed_files} / {total_files}")
    if chunk_bar_text: chunk_bar_text.write(f"Chunks processed: {processed_chunks}")
    return stat

def stop_current_ingest_sessions(session_name, multi_gpu=False):
    fail_session(session_name)
    if multi_gpu:
        for i in range(16):
            sfx = f"{session_name}-gpu{i}"
            sess = get_session(sfx)
            if sess:
                fail_session(sfx)

if st.session_state["session_page_state"] == "NEW":
    session_name = st.sidebar.text_input("New Session Name", st.session_state.get("session_name", f"sess-{datetime.now().strftime('%Y%m%d-%H%M%S')}"))
    st.session_state["session_name"] = session_name
    st.sidebar.header("Corpus Directories")
    add_dir = st.sidebar.text_input("Add a directory (absolute path)", "")
    if st.sidebar.button("Add Directory"):
        if Path(add_dir).exists():
            st.session_state["directories"].add(str(Path(add_dir).resolve()))
            st.sidebar.success(f"Added: {add_dir}")
        else:
            st.sidebar.error("Directory does not exist.")
    for d in sorted(st.session_state["directories"]):
        st.sidebar.write("📁", d)
    if st.sidebar.button("Clear Directory List"):
        st.session_state["directories"] = set()
    selected_dirs = sorted(st.session_state["directories"])
    run_ingest_triggered = st.sidebar.button("Start Ingestion", disabled=not(session_name and selected_dirs), key="start_ingest_btn")
    stop_ingest_triggered = st.sidebar.button("Stop Ingestion", key="stop_ingest_btn")
elif st.session_state["session_page_state"] == "RESUME":
    from db.store import get_resume_sessions
    sessions = get_resume_sessions()
    session_names = [s.session_name for s in sessions]
    selected_session = st.sidebar.selectbox("Resume Session", session_names)
    st.session_state["selected_session"] = selected_session
    sess = get_session(selected_session) if selected_session else None
    run_ingest_triggered = st.sidebar.button("Resume Session", disabled=not selected_session, key="resume_session_btn2")
    stop_ingest_triggered = st.sidebar.button("Stop Ingestion", key="stop_ingest_btn2")
    if sess:
        st.sidebar.markdown(f"**Directory:** `{sess.directory}`")
        st.sidebar.markdown(f"**Progress:** File: `{sess.last_file or '-'}` | Chunk: `{sess.last_chunk or 0}` | Status: `{sess.status}`")
else:
    run_ingest_triggered = stop_ingest_triggered = False

if 'stop_ingest_triggered' in locals() and stop_ingest_triggered:
    multi_gpu = False
    if st.session_state.get("session_page_state") == "NEW":
        multi_gpu = get_num_gpus() and get_num_gpus() > 1
        stop_current_ingest_sessions(st.session_state["session_name"], multi_gpu=multi_gpu)
    elif st.session_state.get("session_page_state") == "RESUME":
        cur_session = st.session_state.get("selected_session")
        if cur_session and "-gpu" in cur_session:
            multi_gpu = True
        stop_current_ingest_sessions(cur_session, multi_gpu=multi_gpu)
    st.session_state["run_ingest"] = False
    st.session_state["session_page_state"] = None
    st.rerun()

if (st.session_state.get("run_ingest") or run_ingest_triggered) and session_choice_made:
    st.session_state["run_ingest"] = True
    session_type = st.session_state["session_page_state"]
    if session_type == "NEW":
        file_list = list(loader.walk_legal_files(selected_dirs))
        total_files = len(file_list)
        prev = get_session(session_name)
        if prev:
            st.warning("Session already exists. Please pick a new name or resume.")
        else:
            num_gpus = get_num_gpus()
            if num_gpus and num_gpus > 1:
                sublists = partition(file_list, num_gpus)
                sessions = []
                procs = []
                for i in range(num_gpus):
                    sess_name = f"{session_name}-gpu{i}"
                    partition_fname = f".gpu-partition-{sess_name}.txt"
                    write_partition_file(sublists[i], partition_fname)
                    sess = start_session(sess_name, selected_dirs[0], total_files=len(sublists[i]), total_chunks=None)
                    sessions.append(sess)
                    proc = launch_embedding_worker(sess_name, selected_embedding_model, gpu=i, partition_file=partition_fname)
                    procs.append(proc)
                st.success(f"Started {num_gpus} embedding workers in parallel (each on a separate GPU). Sessions: {[s.session_name for s in sessions]}")
            else:
                sess = start_session(session_name, selected_dirs[0], total_files=total_files, total_chunks=None)
                proc = launch_embedding_worker(session_name, selected_embedding_model)
                st.success(f"Started session {session_name} for dir {selected_dirs[0]} (PID {proc.pid}) in the background using embedding model `{selected_embedding_model}`.")
            poll_sec = 1.0
            stat_line = st.empty()
            file_bar = st.empty()
            file_bar_text = st.empty()
            chunk_bar_text = st.empty()
            for _ in range(100000):
                stat = poll_session_progress_bars(session_name, file_bar, chunk_bar_text, stat_line)
                time.sleep(poll_sec)
                if stat in {"complete", "error"}: break
            st.write(f"Session `{session_name}` finished with status {stat}.")
            st.session_state["run_ingest"] = False
            st.session_state["session_page_state"] = None
            st.rerun()
    elif session_type == "RESUME":
        session_name = st.session_state["selected_session"]
        sess = get_session(session_name)
        proc = launch_embedding_worker(session_name, selected_embedding_model)
        st.success(f"Resuming session {session_name} in background (PID {proc.pid}) with embedding model `{selected_embedding_model}`.")
        stat_line = st.empty()
        file_bar = st.empty()
        file_bar_text = st.empty()
        chunk_bar_text = st.empty()
        for _ in range(100000):
            stat = poll_session_progress_bars(session_name, file_bar, chunk_bar_text, stat_line)
            time.sleep(1.0)
            if stat in {"complete", "error"}: break
        st.write(f"Session `{session_name}` finished with status {stat}.")
        st.session_state["run_ingest"] = False
        st.session_state["session_page_state"] = None
        st.rerun()

st.markdown("## Legal Document & Hybrid Search")
query = st.text_input("Enter a legal research query or plain search...")
top_k = st.slider("How many results?", min_value=1, max_value=10, value=5)
if st.button("Search & RAG"):
    st.write("🔎 Searching embeddings and running RAG...")
    embedder = Embedder(model_name=selected_embedding_model)
    query_vec = embedder.embed([query])[0]
    hits = search_vector(query_vec, top_k=top_k)
    if not hits:
        st.warning("No results found for this query.")
    else:
        st.markdown("**Relevant Document Excerpts:**")
        for i, h in enumerate(hits, 1):
            st.info(f"{i}. Source: {h['source']} | Score: {h['score']:.4f}\n\n{h['text'][:800]}{'...' if len(h['text'])>800 else ''}")
        rag = RAGPipeline(model=selected_model)
        context_chunks = [h["text"] for h in hits]
        sources = [h["source"] for h in hits]
        with st.spinner(f"Calling {selected_model} via Ollama (RAG)..."):
            result = rag.query(query, top_k=top_k, context_chunks=context_chunks, sources=sources)
            answer = result.get("answer", "").strip()
            if answer:
                st.markdown(f"**LLM Answer (from {selected_model}, RAG):**")
                st.success(answer)
            else:
                st.warning("No answer returned from Ollama/LLM.")

st.markdown("---")
st.caption("© 2025 Legalsearch Demo | Background-parallel ingestion, extensible legal AI, pgvector & Ollama")
