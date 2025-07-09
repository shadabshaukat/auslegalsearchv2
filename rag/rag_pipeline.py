"""
RAG pipeline for auslegalsearchv2.
- Retrieves relevant documents/chunks from the vector store.
- Sends context, user question, and options to Ollama Llama4 via API.
- Returns model output (QA answer/summary) and relevant document sources.
"""

import requests

def list_ollama_models(ollama_url="http://localhost:11434"):
    """
    Query Ollama for available model tags.
    Returns a list of model names (tags).
    """
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
        """
        Retrieve top-k most relevant chunks for the query (hybrid/vector/search).
        TODO: Implement using db/store.py methods.
        """
        # Placeholder
        contexts = ["Relevant chunk 1...", "Relevant chunk 2..."]
        sources = ["source_1.txt", "source_2.html"]
        return contexts, sources

    def llama4_rag(self, query: str, context_chunks, temperature=0.2) -> str:
        """
        Calls Ollama Llama4 model with RAG prompt.
        """
        prompt = (
            "Based on the following legal documents/chunks, answer the question or summarize as requested.\n"
            "CONTEXT:\n"
            + "\n---\n".join(context_chunks) +
            f"\n\nQUESTION: {query}\nANSWER:"
        )
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        resp = requests.post(
            f"{self.ollama_url}/api/generate", json=payload, timeout=120
        )
        if resp.status_code == 200:
            return resp.json().get("response", "")
        else:
            return f"Error querying Llama4: {resp.status_code} {resp.text}"

    def query(self, question: str, top_k: int = 5, context_chunks=None, sources=None) -> dict:
        """
        RAG QA: Accepts context_chunks (retrieved externally) and sends them to the LLM.
        """
        if context_chunks is not None:
            contexts = context_chunks
        else:
            contexts, sources = self.retrieve(question, k=top_k)
        answer = self.llama4_rag(question, contexts)
        return {
            "answer": answer,
            "sources": sources,
            "contexts": contexts,
        }

# TODO: Integrate with db.store for actual retrieval/search
