"""
query.py — Convert a question into an embedding, retrieve top-k chunks from
           FAISS, and hand off to the LLM agent.
"""

import json
import faiss
import numpy as np
from embed import embed_query, load_index

# ── Config ────────────────────────────────────────────────────────────────────
TOP_K = 4   # number of chunks to retrieve


# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(query: str,
             index: faiss.IndexFlatL2,
             chunks: list[str],
             top_k: int = TOP_K) -> list[str]:
    """
    Embed the query and return the top_k most similar text chunks.
    """
    query_vec = embed_query(query)                      # shape (1, dim)
    distances, indices = index.search(query_vec, top_k) # (1, k) each

    results: list[str] = []
    print(f"[query] Top-{top_k} retrieved chunks (L2 distances):")
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
        if idx == -1:          # FAISS returns -1 for empty slots
            continue
        chunk = chunks[idx]
        preview = chunk[:80].replace("\n", " ")
        print(f"  {rank}. [idx={idx}, dist={dist:.4f}] {preview}…")
        results.append(chunk)

    return results


# ── Answer ────────────────────────────────────────────────────────────────────
def answer(query: str,
           index: faiss.IndexFlatL2,
           chunks: list[str],
           top_k: int = TOP_K) -> dict:
    """
    Full query → retrieve → LLM pipeline.

    Returns:
        {
            "answer":  "<LLM answer string>",
            "sources": ["chunk text 1", "chunk text 2", …]
        }
    """
    # 1. Retrieve relevant chunks
    retrieved = retrieve(query, index, chunks, top_k)

    if not retrieved:
        return {"answer": "No relevant context found in the document.",
                "sources": []}

    # 2. Build context string for the LLM
    context = "\n\n---\n\n".join(retrieved)

    # 3. Call llm_agent
    try:
        from llm_agent import ask_llm                   # your existing module
        llm_answer = ask_llm(context=context, question=query)
    except ImportError:
        # Fallback: call Ollama directly so the pipeline still runs
        llm_answer = _ollama_fallback(context, query)

    return {
        "answer":  llm_answer,
        "sources": retrieved,
    }


# ── Ollama fallback (used when llm_agent.py is absent) ───────────────────────
def _ollama_fallback(context: str, question: str) -> str:
    """
    Minimal direct Ollama call — mirrors what llm_agent.py typically does.
    Remove this once you wire up your real llm_agent.py.
    """
    import requests

    prompt = (
        "You are a helpful assistant. Use ONLY the context below to answer "
        "the question. If the answer is not in the context, say so.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )

    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()
