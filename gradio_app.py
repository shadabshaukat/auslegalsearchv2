"""
Gradio frontend for AUSLegalSearchv2 FastAPI backend.
- Fully modular legal search, ingestion, chat, and RAG app.
- Talks to fastapi_app.py via REST endpoints.
- Session state, login, and clean layout using modern Gradio UX.
"""

import gradio as gr
import requests
import os

API_ROOT = os.environ.get("AUSLEGALSEARCH_API_URL", "http://localhost:8000")

# --- Session/User State ---
class APISession:
    def __init__(self):
        self.user = None
        self.auth = None  # tuple (user, pass)

    def login(self, username, password):
        # Try protected endpoint, store credentials if accepted
        try:
            r = requests.get(f"{API_ROOT}/health", auth=(username, password), timeout=10)
            if r.ok:
                self.auth = (username, password)
                self.user = username
                return True
        except Exception:
            pass
        return False
    def headers(self):
        return {}

SESS = APISession()

# --- Login page ---
def login_fn(username, password):
    if SESS.login(username, password):
        return gr.update(visible=False), gr.update(visible=True), f"Welcome, {username}!", ""
    else:
        return gr.update(visible=True), gr.update(visible=False), "", "Invalid login."

with gr.Blocks(title="LegalSearch Gradio UI") as demo:
    gr.Markdown("# AUSLegalSearch Gradio Frontend\nElegant, modular UI over REST API for legal search, ingestion, and legal RAG/QA.")
    login_box = gr.Row(visible=True)
    with login_box:
        gr.Markdown("## Login to continue")
        username = gr.Textbox(label="Username", value="legal_api")
        password = gr.Textbox(label="Password", type="password")
        login_err = gr.Markdown("")
        login_btn = gr.Button("Login")

    app_panel = gr.TabGroup(visible=False)
    with app_panel:
        with gr.Tab("Ingest"):
            ingest_dir = gr.Textbox(label="Directory to ingest")
            ingest_session = gr.Textbox(label="Session Name")
            ingest_status = gr.Markdown("")
            ingest_btn = gr.Button("Start Ingestion")
            ingest_sessions = gr.Dataframe(headers=["Active Sessions"], interactive=False)
            refresh_ingest = gr.Button("Refresh Sessions")

        with gr.Tab("Documents"):
            doc_list = gr.Dataframe(headers=["ID", "Source", "Format"], interactive=False)
            doc_refresh = gr.Button("Refresh Document List")
            doc_id = gr.Number(label="Doc ID")
            doc_view_btn = gr.Button("View Document")
            doc_out = gr.Textbox(label="Document Content", lines=8, interactive=False)

        with gr.Tab("Search"):
            with gr.Row():
                search_type = gr.Radio(["BM25", "Vector", "RAG (hybrid, LLM)"], value="BM25", label="Search Type")
                search_top_k = gr.Number(value=5, label="Top K Results", precision=0)
            search_query = gr.Textbox(label="Query")
            search_btn = gr.Button("Search")
            search_out = gr.Dataframe(headers=["Score", "Chunk/Page", "Source", "Excerpt"], interactive=False)
            rag_answer = gr.Textbox(label="LLM Answer (RAG)", lines=4, interactive=False)

        with gr.Tab("Chat Assistant"):
            chat_hist = gr.Chatbot(label="Legal Assistant", height=340)
            chat_input = gr.Textbox(label="Type your legal question here")
            chat_btn = gr.Button("Send")
            chat_status = gr.Markdown("")

# --- Logic Wiring ---
def refresh_sessions():
    try:
        r = requests.get(f"{API_ROOT}/ingest/sessions", auth=SESS.auth, timeout=10)
        return [[s] for s in r.json()]
    except Exception as e:
        return [["Error", str(e)]]

def start_ingest_fn(directory, session_name):
    try:
        r = requests.post(f"{API_ROOT}/ingest/start", json={"directory": directory, "session_name": session_name}, auth=SESS.auth, timeout=15)
        if r.ok:
            return f"Session started: {session_name}"
        return f"Failed: {r.text}"
    except Exception as e:
        return f"Error: {str(e)}"

def list_docs_fn():
    try:
        r = requests.get(f"{API_ROOT}/documents", auth=SESS.auth)
        dat = r.json()
        return [[d["id"], d["source"], d["format"]] for d in dat]
    except Exception as e:
        return [["Error", str(e), ""]]

def get_doc_fn(doc_id):
    try:
        r = requests.get(f"{API_ROOT}/documents/{int(doc_id)}", auth=SESS.auth)
        d = r.json()
        return d.get("content", "")
    except Exception as e:
        return f"Error: {str(e)}"

def search_fn(search_type, query, top_k):
    try:
        if search_type == "BM25":
            r = requests.post(f"{API_ROOT}/search/bm25", json={"query": query, "top_k": int(top_k)}, auth=SESS.auth)
        elif search_type == "Vector":
            r = requests.post(f"{API_ROOT}/search/vector", json={"query": query, "top_k": int(top_k)}, auth=SESS.auth)
        else:
            # RAG hybrid
            r = requests.post(f"{API_ROOT}/search/rag", json={"question": query, "top_k": int(top_k)}, auth=SESS.auth)
        dat = r.json()
        if isinstance(dat, list):
            return [[x.get("score", ""), x.get("chunk_index", ""), x.get("source", ""), x.get("text", "")] for x in dat]
        elif isinstance(dat, dict):
            # For RAG, show answer and source docs
            rows = []
            for i, x in enumerate(dat.get("contexts", [])):
                rows.append([i+1, "", dat.get("sources", [""])[i] if i < len(dat.get("sources", [])) else "", x])
            return rows, dat.get("answer", "")
        return [["No results or error", "", "", ""]], ""
    except Exception as e:
        return [["Error", "", "", str(e)]], ""

def chat_fn(history, message):
    try:
        r = requests.post(f"{API_ROOT}/chat/session", json={"prompt": message}, auth=SESS.auth)
        dat = r.json()
        reply = dat.get("answer", "")
        return history + [[message, reply]], ""
    except Exception as e:
        return history, f"Error: {str(e)}"

# --- Wiring Gradio events to logic ---
login_btn.click(login_fn, inputs=[username, password], outputs=[login_box, app_panel, login_err, login_err])
ingest_btn.click(start_ingest_fn, inputs=[ingest_dir, ingest_session], outputs=ingest_status)
refresh_ingest.click(refresh_sessions, outputs=ingest_sessions)

doc_refresh.click(list_docs_fn, outputs=doc_list)
doc_view_btn.click(get_doc_fn, inputs=doc_id, outputs=doc_out)

search_btn.click(
    lambda t, q, k: (search_fn(t, q, k) if t != "RAG (hybrid, LLM)" else search_fn(t, q, k)),
    inputs=[search_type, search_query, search_top_k],
    outputs=[search_out, rag_answer],
)
chat_btn.click(chat_fn, inputs=[chat_hist, chat_input], outputs=[chat_hist, chat_status])

if __name__ == "__main__":
    demo.launch()
