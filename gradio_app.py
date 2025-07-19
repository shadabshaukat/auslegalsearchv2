"""
Gradio frontend for AUSLegalSearchv2 with conversation memory-enabled Chat Assistant.
- Chat history is sanitized and passed so the LLM can answer based on prior interaction context (for true conversational QA).
- No other UI or logic is modified.
"""

import gradio as gr
import requests
import os

API_ROOT = os.environ.get("AUSLEGALSEARCH_API_URL", "http://localhost:8000")

class APISession:
    def __init__(self):
        self.user = None
        self.auth = None

    def login(self, username, password):
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

def login_fn(username, password):
    debugmsg = ""
    default_models = ["llama3.2", "llama4"]
    if SESS.login(username, password):
        try:
            r = requests.get(f"{API_ROOT}/models/ollama", auth=SESS.auth, timeout=10)
            live_models = r.json() if isinstance(r.json(), list) and r.json() else []
            models = default_models + [m for m in live_models if m not in default_models]
            debugmsg = f"Fetched LLMs: {models}"
        except Exception as e:
            models = default_models
            debugmsg = f"Model fetch err: {e}"
        value = "llama3.2" if "llama3.2" in models else models[0] if models else "llama3.2"
        update_dropdown = gr.update(choices=models, value=value)
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            f"Welcome, {username}!",
            "",
            update_dropdown, update_dropdown, debugmsg, debugmsg
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
                search_type = gr.Radio(
                    ["Reranker", "Vector", "RAG (hybrid, LLM)"], 
                    value="Reranker", label="Search Type"
                )
                search_top_k = gr.Number(value=5, label="Top K Results", precision=0)
            reranker_model = gr.Dropdown(choices=[], value=None, label="Reranker Model", interactive=True, visible=True)
            reranker_download_name = gr.Textbox(label="New Reranker Name (download)", value="", visible=False)
            reranker_download_repo = gr.Textbox(label="HuggingFace Repo (e.g. mixedbread-ai/mxbai-rerank-xsmall)", value="", visible=False)
            reranker_download_desc = gr.Textbox(label="Model Description (optional)", value="", visible=False)
            reranker_download_btn = gr.Button("Download/Add Reranker Model", visible=False)
            reranker_download_status = gr.Markdown("", visible=False)
            rag_llm_model = gr.Dropdown(choices=[], value=None, label="RAG LLM Model (Ollama)", interactive=True)
            refresh_models = gr.Button("Refresh LLM Models", variant="secondary")
            search_query = gr.Textbox(label="Query")
            search_btn = gr.Button("Search")
            search_out = gr.Dataframe(headers=["Score", "Chunk/Page", "Source", "Excerpt", "Reranker"], interactive=False)
            rag_answer = gr.JSON(label="LLM Answer (RAG)", visible=True)
            search_debug = gr.Markdown("")

        with gr.Tab("Chat Assistant"):
            chat_llm_model = gr.Dropdown(choices=[], value=None, label="Chat LLM Model (Ollama)", interactive=True)
            refresh_models_chat = gr.Button("Refresh LLM Models", variant="secondary")
            chat_hist = gr.Chatbot(label="Legal Assistant", height=340, type="messages")
            chat_input = gr.Textbox(label="Type your legal question here")
            chat_prompt = gr.Textbox(label="Custom System Prompt (optional)", value="", lines=2)
            chat_btn = gr.Button("Send")
            chat_status = gr.Markdown("")
            chat_debug = gr.Markdown("")

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

    def search_fn(search_type, query, top_k, llm_model, reranker_model_name):
        debugmsg = (
            f"DEBUG: Search POST — reranker_model: '{reranker_model_name}', llm_model: '{llm_model}'\n"
            f"Dropdown choices: {getattr(rag_llm_model, 'choices', None)}\n"
            f"Dropdown value: {getattr(rag_llm_model, 'value', None)}\n"
        )
        payload = None
        try:
            if search_type == "Reranker":
                payload = {"query": query, "top_k": int(top_k), "model": reranker_model_name}
                r = requests.post(f"{API_ROOT}/search/rerank", json=payload, auth=SESS.auth)
            elif search_type == "Vector":
                payload = {"query": query, "top_k": int(top_k), "model": reranker_model_name}
                r = requests.post(f"{API_ROOT}/search/vector", json=payload, auth=SESS.auth)
            else:  # For RAG, search_fn shouldn't be called.
                return [["RAG Error: should not route this path", "", "", "", ""]], {}, debugmsg
            debugmsg += f"POST payload: {payload}\n"
            dat = r.json()
            if isinstance(dat, list):
                return [[
                    x.get("score", ""),
                    x.get("chunk_index", ""),
                    x.get("source", ""),
                    x.get("text", ""),
                    x.get("reranker", "") if "reranker" in x else ""
                ] for x in dat], {}, debugmsg
            elif isinstance(dat, dict):
                rows = []
                for i, x in enumerate(dat.get("contexts", [])):
                    rows.append([i+1, "", dat.get("sources", [""])[i] if i < len(dat.get("sources", [])) else "", x])
                return rows, (dat.get("answer", "") + "\n" + debugmsg), debugmsg
            return [["No results or error", "", "", "", ""]], {}, debugmsg
        except Exception as e:
            return [["Error", "", "", "", str(e)]], {}, debugmsg

    def download_reranker_fn(name, repo, desc):
        try:
            payload = {"name": name, "hf_repo": repo, "desc": desc or ""}
            r = requests.post(f"{API_ROOT}/models/reranker/download", json=payload, auth=SESS.auth, timeout=20)
            if r.status_code == 200:
                status = f"Downloaded/started: {name}"
            else:
                status = f"FAILED (status {r.status_code}): {r.text}"
        except Exception as e:
            status = f"Failed to download: {str(e)}"
        return status

    def reranker_model_updater():
        try:
            r = requests.get(f"{API_ROOT}/models/reranker", auth=SESS.auth, timeout=10)
            models = r.json()
            choices = [m["name"] for m in models]
            value = choices[0] if choices else None
            return gr.update(choices=choices, value=value)
        except Exception:
            return gr.update(choices=["mxbai-rerank-xsmall"], value="mxbai-rerank-xsmall")

    def on_search_type_change(selected_type):
        dropdown_update = reranker_model_updater()
        if selected_type == "Reranker":
            return (
                dropdown_update,
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
            )
        else:
            return (
                dropdown_update,
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )

    def rag_search_wrapper(search_type, query, top_k, llm_model, reranker_model_name):
        debugmsg = ""
        if search_type == "RAG (hybrid, LLM)":
            try:
                vector_payload = {"query": query, "top_k": int(top_k), "model": reranker_model_name}
                r_vec = requests.post(f"{API_ROOT}/search/vector", json=vector_payload, auth=SESS.auth)
                context_results = r_vec.json() if r_vec.ok else []
                chunks = [c.get("text", "") for c in context_results]
                sources = [c.get("source", "") for c in context_results]
                debugmsg += f"[RAG] Got {len(chunks)} context chunks from vector search.\n"
                if not chunks:
                    return [["No relevant documents found for RAG context. Please ingest or check your DB.", "", "", "", ""]], \
                        {"error": "No relevant documents found for RAG context. Please ingest or check your DB."}, debugmsg
                rag_payload = {
                    "question": query,
                    "context_chunks": chunks,
                    "sources": sources,
                    "top_k": int(top_k),
                    "model": llm_model,
                    "reranker_model": reranker_model_name,
                }
                r_rag = requests.post(f"{API_ROOT}/search/rag", json=rag_payload, auth=SESS.auth)
                dat = r_rag.json() if r_rag.ok else {}
                rows = []
                for i, txt in enumerate(dat.get("contexts", [])):
                    src = dat.get("sources", [""])[i] if i < len(dat.get("sources", [])) else ""
                    rows.append([i+1, "", src, txt, ""])
                answer = dat.get("answer", "")
                try:
                    parsed = answer
                    if isinstance(answer, str):
                        import json
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
                return [["RAG Error", "", "", "", str(e)]], {"error": str(e)}, debugmsg
        else:
            return search_fn(search_type, query, top_k, llm_model, reranker_model_name)

    def chat_fn(history, message, llm_model, custom_prompt):
        """
        Maintains conversational context by passing all previous message pairs as chat_history.
        Cleans chat history to ensure only valid dicts {"role", "content"} with string values are sent.
        """
        debugmsg = (
            f"DEBUG: Chat→RAG — dropdown value: '{llm_model}'\n"
            f"Dropdown choices: {getattr(chat_llm_model, 'choices', None)}\n"
            f"Dropdown value: {getattr(chat_llm_model, 'value', None)}\n"
        )

        # Clean and format chat history for API
        gradio_history = history if history is not None else []
        cleaned_history = []
        for m in gradio_history:
            if isinstance(m, dict) and "role" in m and "content" in m:
                if m["role"] and m["content"] is not None:
                    cleaned_history.append({"role": str(m["role"]), "content": str(m["content"])})
            elif isinstance(m, (list, tuple)) and len(m) == 2:
                if m[0] is not None:
                    cleaned_history.append({"role": "user", "content": str(m[0])})
                if m[1] is not None:
                    cleaned_history.append({"role": "assistant", "content": str(m[1])})

        try:
            if not llm_model or llm_model in ["", None]:
                debugmsg += "[ERROR] model was not selected! Using fallback.\n"
                llm_model = "llama3"
            vector_payload = {"query": message, "top_k": 5}
            r_vec = requests.post(f"{API_ROOT}/search/vector", json=vector_payload, auth=SESS.auth)
            context_results = r_vec.json() if r_vec.ok else []
            chunks = [c.get("text", "") for c in context_results]
            sources = [c.get("source", "") for c in context_results]
            debugmsg += f"[RAG] Got {len(chunks)} context chunks from vector search.\n"
            if not chunks:
                reply = "No relevant context found in document database for this query."
            else:
                rag_payload = {
                    "question": message,
                    "context_chunks": chunks,
                    "sources": sources,
                    "top_k": 5,
                    "model": llm_model,
                    "chat_history": cleaned_history,
                }
                if custom_prompt.strip():
                    rag_payload["prompt"] = custom_prompt.strip()
                r_rag = requests.post(f"{API_ROOT}/search/rag", json=rag_payload, auth=SESS.auth)
                dat = r_rag.json() if r_rag.ok else {}
                reply = dat.get("answer", "") if isinstance(dat, dict) else "No answer returned from RAG endpoint."
                debugmsg += f"RAG POST payload: {str(rag_payload)[:512]}...\n"
            if not isinstance(history, list):
                history = []
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

    search_type.change(
        on_search_type_change,
        inputs=search_type,
        outputs=[
            reranker_model,
            reranker_download_name,
            reranker_download_repo,
            reranker_download_desc,
            reranker_download_btn,
            reranker_download_status
        ]
    )

    def reranker_model_refresh_on_any_search():
        try:
            r = requests.get(f"{API_ROOT}/models/reranker", auth=SESS.auth, timeout=10)
            models = r.json()
            choices = [m["name"] for m in models]
            value = choices[0] if choices else None
            return gr.update(choices=choices, value=value)
        except Exception:
            return gr.update(choices=["mxbai-rerank-xsmall"], value="mxbai-rerank-xsmall")

    # Always use rag_search_wrapper for all search types
    search_btn.click(
        rag_search_wrapper,
        inputs=[search_type, search_query, search_top_k, rag_llm_model, reranker_model],
        outputs=[search_out, rag_answer, search_debug]
    )
    reranker_download_btn.click(
        lambda name, repo, desc: (download_reranker_fn(name, repo, desc), reranker_model_refresh_on_any_search()),
        inputs=[reranker_download_name, reranker_download_repo, reranker_download_desc],
        outputs=[reranker_download_status, reranker_model]
    )
    chat_btn.click(chat_fn, inputs=[chat_hist, chat_input, chat_llm_model, chat_prompt], outputs=[chat_hist, chat_status, chat_debug])

    def refresh_models_fn():
        default_models = ["llama3.2", "llama4"]
        try:
            r = requests.get(f"{API_ROOT}/models/ollama", auth=SESS.auth, timeout=10)
            live_models = r.json() if isinstance(r.json(), list) and r.json() else []
            models = default_models + [m for m in live_models if m not in default_models]
        except Exception:
            models = default_models
        value = "llama3.2" if "llama3.2" in models else models[0] if models else "llama3.2"
        return gr.update(choices=models, value=value)

    refresh_models.click(refresh_models_fn, outputs=rag_llm_model)
    refresh_models_chat.click(refresh_models_fn, outputs=chat_llm_model)

    # On app load, populate reranker model dropdown first time
    demo.load(reranker_model_updater, None, reranker_model)

if __name__ == "__main__":
    for port in range(7866, 7880):
        try:
            demo.launch(server_port=port)
            break
        except OSError as e:
            print(f"Port {port} unavailable ({e}). Trying next...")
    else:
        raise RuntimeError("Could not find an open port for Gradio between 7866-7879")
