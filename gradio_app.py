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
    debugmsg = ""
    default_models = ["llama3.2", "llama4"]
    if SESS.login(username, password):
        # After login, fetch Ollama models for RAG and Chat LLM dropdowns
        try:
            r = requests.get(f"{API_ROOT}/models/ollama", auth=SESS.auth, timeout=10)
            live_models = r.json() if isinstance(r.json(), list) and r.json() else []
            models = default_models + [m for m in live_models if m not in default_models]
            debugmsg = f"Fetched LLMs: {models}"
        except Exception as e:
            models = default_models
            debugmsg = f"Model fetch err: {e}"
        # Prefer llama3.2 as default if present, else first model
        value = "llama3.2" if "llama3.2" in models else models[0] if models else "llama3.2"
        update_dropdown = gr.update(choices=models, value=value)
        return (
            gr.update(visible=False),    # Hide login
            gr.update(visible=True),     # Show app
            f"Welcome, {username}!",     # Success msg
            "",                          # No login err
            update_dropdown, update_dropdown, debugmsg, debugmsg # RAG/chat debug
        )
    else:
        return gr.update(visible=True), gr.update(visible=False), "", "Invalid login.", gr.update(), gr.update(), "", ""

with gr.Blocks(title="LegalSearch Gradio UI") as demo:
    gr.Markdown("# AUSLegalSearch Gradio Frontend\nElegant, modular UI over REST API for legal search, ingestion, and legal RAG/QA.")
    login_box = gr.Row(visible=True)
    with login_box:
        gr.Markdown("## Login to continue")
        username = gr.Textbox(label="Username", value="legal_api")
        password = gr.Textbox(label="Password", type="password")
        login_err = gr.Markdown("")
        login_btn = gr.Button("Login")

    app_panel = gr.Row(visible=False)
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
            rag_llm_model = gr.Dropdown(choices=[], value=None, label="RAG LLM Model (Ollama)", interactive=True)
            refresh_models = gr.Button("Refresh LLM Models", variant="secondary")
            search_query = gr.Textbox(label="Query")
            search_btn = gr.Button("Search")
            search_out = gr.Dataframe(headers=["Score", "Chunk/Page", "Source", "Excerpt"], interactive=False)
            rag_answer = gr.JSON(label="LLM Answer (RAG)", visible=True)
            search_debug = gr.Markdown("")  # Show current model choice/debug

        with gr.Tab("Chat Assistant"):
            chat_llm_model = gr.Dropdown(choices=[], value=None, label="Chat LLM Model (Ollama)", interactive=True)
            refresh_models_chat = gr.Button("Refresh LLM Models", variant="secondary")
            chat_hist = gr.Chatbot(label="Legal Assistant", height=340, type="messages")
            chat_input = gr.Textbox(label="Type your legal question here")
            chat_btn = gr.Button("Send")
            chat_status = gr.Markdown("")
            chat_debug = gr.Markdown("")

    # --- Logic Wiring --- (MOVE HERE INSIDE gr.Blocks CONTEXT)
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

    # Remove the thread-based dropdown update logic

    def search_fn(search_type, query, top_k, llm_model):
        debugmsg = (
            f"DEBUG: Search POST — dropdown value: '{llm_model}'\n"
            f"Dropdown choices: {getattr(rag_llm_model, 'choices', None)}\n"
            f"Dropdown value: {getattr(rag_llm_model, 'value', None)}\n"
        )
        payload = None
        try:
            if search_type == "BM25":
                payload = {"query": query, "top_k": int(top_k)}
                r = requests.post(f"{API_ROOT}/search/bm25", json=payload, auth=SESS.auth)
            elif search_type == "Vector":
                payload = {"query": query, "top_k": int(top_k)}
                r = requests.post(f"{API_ROOT}/search/vector", json=payload, auth=SESS.auth)
            else:
                # RAG hybrid, send model param
                if not llm_model or llm_model in ["", None]:
                    debugmsg += "[ERROR] model was not selected! Using fallback.\n"
                    llm_model = "llama3"
                payload = {"question": query, "top_k": int(top_k), "model": llm_model}
                r = requests.post(
                    f"{API_ROOT}/search/rag",
                    json=payload,
                    auth=SESS.auth
                )
            debugmsg += f"POST payload: {payload}\n"
            dat = r.json()
            if isinstance(dat, list):
                return [[x.get("score", ""), x.get("chunk_index", ""), x.get("source", ""), x.get("text", "")] for x in dat], {}, debugmsg
            elif isinstance(dat, dict):
                # For RAG, show answer and source docs
                rows = []
                for i, x in enumerate(dat.get("contexts", [])):
                    rows.append([i+1, "", dat.get("sources", [""])[i] if i < len(dat.get("sources", [])) else "", x])
                return rows, (dat.get("answer", "") + "\n" + debugmsg), debugmsg
            return [["No results or error", "", "", ""]], {}, debugmsg
        except Exception as e:
            return [["Error", "", "", str(e)]], {}, debugmsg

    def chat_fn(history, message, llm_model):
        debugmsg = (
            f"DEBUG: Chat POST — dropdown value: '{llm_model}'\n"
            f"Dropdown choices: {getattr(chat_llm_model, 'choices', None)}\n"
            f"Dropdown value: {getattr(chat_llm_model, 'value', None)}\n"
        )
        payload = None
        try:
            if not llm_model or llm_model in ["", None]:
                debugmsg += "[ERROR] model was not selected! Using fallback.\n"
                llm_model = "llama3"
            payload = {"prompt": message, "model": llm_model}
            r = requests.post(
                f"{API_ROOT}/chat/session",
                json=payload,
                auth=SESS.auth
            )
            debugmsg += f"POST payload: {payload}\n"
            dat = r.json()
            reply = dat.get("answer", "")
            if not isinstance(history, list):
                history = []
            # For type='messages' format
            return (
                history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": reply}
                ], "", debugmsg
            )
        except Exception as e:
            return history, f"Error: {str(e)}, {debugmsg}", debugmsg

    login_btn.click(
        login_fn,
        inputs=[username, password],
        outputs=[login_box, app_panel, login_err, login_err, rag_llm_model, chat_llm_model, search_debug, chat_debug]
    )
    ingest_btn.click(start_ingest_fn, inputs=[ingest_dir, ingest_session], outputs=ingest_status)
    refresh_ingest.click(refresh_sessions, outputs=ingest_sessions)

    doc_refresh.click(list_docs_fn, outputs=doc_list)
    doc_view_btn.click(get_doc_fn, inputs=doc_id, outputs=doc_out)

    import json

    def rag_search_wrapper(search_type, query, top_k, llm_model):
        debugmsg = ""
        if search_type != "RAG (hybrid, LLM)":
            return search_fn(search_type, query, top_k, llm_model)
        try:
            # Get context by vector
            vector_payload = {"query": query, "top_k": int(top_k)}
            r_vec = requests.post(f"{API_ROOT}/search/vector", json=vector_payload, auth=SESS.auth)
            context_results = r_vec.json() if r_vec.ok else []
            chunks = [c.get("text", "") for c in context_results]
            sources = [c.get("source", "") for c in context_results]
            debugmsg += f"[RAG] Got {len(chunks)} context chunks from vector search.\n"
            if not chunks:
                return [["No relevant documents found for RAG context. Please ingest or check your DB.", "", "", ""]], \
                    {"error": "No relevant documents found for RAG context. Please ingest or check your DB."}, debugmsg
            rag_payload = {
                "question": query,
                "context_chunks": chunks,
                "sources": sources,
                "top_k": int(top_k),
                "model": llm_model
            }
            r_rag = requests.post(f"{API_ROOT}/search/rag", json=rag_payload, auth=SESS.auth)
            dat = r_rag.json() if r_rag.ok else {}
            rows = []
            for i, txt in enumerate(dat.get("contexts", [])):
                src = dat.get("sources", [""])[i] if i < len(dat.get("sources", [])) else ""
                rows.append([i+1, "", src, txt])
            answer = dat.get("answer", "")
            # If answer looks like JSON (dict/list), display as JSON
            try:
                parsed = answer
                if isinstance(answer, str):
                    parsed = json.loads(answer)
                elif isinstance(answer, (dict, list)):
                    parsed = answer
                else:
                    parsed = {"LLM Answer": answer}
            except Exception:
                parsed = {"LLM Answer": answer}
            debugmsg += f"POST payload: {rag_payload}\n"
            return rows, parsed, debugmsg
        except Exception as e:
            return [["RAG Error", "", "", str(e)]], {"error": str(e)}, debugmsg

    search_btn.click(
        rag_search_wrapper,
        inputs=[search_type, search_query, search_top_k, rag_llm_model],
        outputs=[search_out, rag_answer, search_debug],
    )
    chat_btn.click(chat_fn, inputs=[chat_hist, chat_input, chat_llm_model], outputs=[chat_hist, chat_status, chat_debug])

    def refresh_models_fn():
        default_models = ["llama3.2", "llama4"]
        try:
            r = requests.get(f"{API_ROOT}/models/ollama", auth=SESS.auth, timeout=10)
            live_models = r.json() if isinstance(r.json(), list) and r.json() else []
            models = default_models + [m for m in live_models if m not in default_models]
        except Exception:
            models = default_models
        # Prefer llama3.2 as default if present, else first model
        value = "llama3.2" if "llama3.2" in models else models[0] if models else "llama3.2"
        return gr.update(choices=models, value=value)

    refresh_models.click(refresh_models_fn, outputs=rag_llm_model)
    refresh_models_chat.click(refresh_models_fn, outputs=chat_llm_model)

if __name__ == "__main__":
    # Try default 7866, then 7867-7879 if in use
    for port in range(7866, 7880):
        try:
            demo.launch(server_port=port)
            break
        except OSError as e:
            print(f"Port {port} unavailable ({e}). Trying next...")
    else:
        raise RuntimeError("Could not find an open port for Gradio between 7866-7879")
