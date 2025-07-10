"""
RAG pipeline for auslegalsearchv2.
- Retrieves relevant documents/chunks from the vector store.
- Sends context, user question, and options to Ollama Llama4 via API.
- Returns model output (QA answer/summary) and relevant document sources.
- Supports user/system prompt injection for custom instructions.
- Now supports temperature, top_p, max_tokens, repeat_penalty, and other Ollama-compatible LLM parameters.
"""

import requests

def list_ollama_models(ollama_url="http://localhost:11434"):
    try:
        resp = requests.get(f"{ollama_url}/api/tags", timeout=10)
        if resp.status_code == 200:
            result = resp.json()
            return [m["name"] for m in result.get("models", [])]
        else:
            return []
    except Exception:
        return []

class RAGPipeline:
    def __init__(self, ollama_url="http://localhost:11434", model="llama3"):
        self.ollama_url = ollama_url
        self.model = model

    def retrieve(self, query: str, k: int = 5):
        # Placeholder
        contexts = ["Relevant chunk 1...", "Relevant chunk 2..."]
        sources = ["source_1.txt", "source_2.html"]
        return contexts, sources

    def llama4_rag(
        self, query: str, context_chunks, custom_prompt=None,
        temperature=0.2, top_p=0.95, max_tokens=1024, repeat_penalty=1.1
    ) -> str:
        if custom_prompt:
            sys_prompt = custom_prompt.strip()
        else:
            sys_prompt = "You are a legal assistant. Answer only from the provided context. Cite sources. Be concise."
        prompt = (
            sys_prompt + "\n\n"
            "Based on the following legal documents/chunks, answer the question or summarize as requested.\n"
            "CONTEXT:\n"
            + "\n---\n".join(context_chunks) +
            f"\n\nQUESTION: {query}\nANSWER:"
        )
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
                "repeat_penalty": repeat_penalty,
            },
        }
        resp = requests.post(
            f"{self.ollama_url}/api/generate", json=payload, timeout=120
        )
        if resp.status_code == 200:
            return resp.json().get("response", "")
        else:
            return f"Error querying Llama4: {resp.status_code} {resp.text}"

    def query(
        self, question: str, top_k: int = 5, context_chunks=None, sources=None, custom_prompt=None,
        temperature=0.2, top_p=0.95, max_tokens=1024, repeat_penalty=1.1
    ) -> dict:
        if context_chunks is not None:
            contexts = context_chunks
        else:
            contexts, sources = self.retrieve(question, k=top_k)
        answer = self.llama4_rag(
            question,
            contexts,
            custom_prompt=custom_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repeat_penalty=repeat_penalty,
        )
        return {
            "answer": answer,
            "sources": sources,
            "contexts": contexts,
        }
