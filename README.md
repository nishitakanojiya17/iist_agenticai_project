# RAG Pipeline — Local Document Q&A with Ollama + FAISS

A minimal, framework-free RAG system: ingest a PDF or text file, then ask
questions about it.  Responses arrive in under 5 seconds on most laptops.

---

## File Structure

```
rag_pipeline/
├── embed.py          # text extraction, chunking, embedding, FAISS index
├── query.py          # query embedding, retrieval, LLM call
├── llm_agent.py      # Ollama wrapper  ← replace with your own if you have one
├── main.py           # CLI entry point
└── requirements.txt
```

**Generated at runtime:**
```
faiss.index           # saved FAISS index (binary)
chunks.pkl            # saved chunk list (pickle)
result.json           # last query output (full text)
```

---

## Prerequisites

| Tool | Install |
|------|---------|
| Python 3.10+ | https://python.org |
| Ollama | https://ollama.com |
| Llama 3 model | `ollama pull llama3` |

---

## Quick Start

### 1 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2 — Start Ollama

```bash
ollama serve          # keep this running in a separate terminal
```

### 3 — Ingest your document

```bash
# PDF
python main.py --ingest my_document.pdf

# Plain text
python main.py --ingest my_document.txt
```

This builds `faiss.index` and `chunks.pkl` in the current directory.
You only need to do this once per document (or whenever the document changes).

### 4 — Ask a question

```bash
python main.py --query "What is the refund policy?"
```

### 5 — Ingest + query in one command

```bash
python main.py --ingest doc.pdf --query "Summarise the key points."
```

### 6 — Control how many chunks are retrieved (default 4)

```bash
python main.py --query "Who are the authors?" --top-k 3
```

---

## Output

Terminal shows a truncated preview.  The full JSON is saved to `result.json`:

```json
{
  "answer": "The refund policy allows returns within 30 days …",
  "sources": [
    "… full text of chunk 1 …",
    "… full text of chunk 2 …"
  ]
}
```

---

## Integrating Your Existing llm_agent.py

`query.py` does:

```python
from llm_agent import ask_llm
llm_answer = ask_llm(context=context, question=query)
```

Your `llm_agent.py` just needs to expose:

```python
def ask_llm(context: str, question: str) -> str:
    ...
```

Replace the provided `llm_agent.py` with your own file — no other changes needed.

---

## Tuning

| Parameter | Location | Default | Effect |
|-----------|----------|---------|--------|
| `CHUNK_WORDS` | `embed.py` | 250 | Words per chunk |
| `OVERLAP_WORDS` | `embed.py` | 50 | Word overlap between chunks |
| `TOP_K` | `query.py` | 4 | Chunks sent to LLM |
| `OLLAMA_MODEL` | `llm_agent.py` | `llama3` | Any model you have pulled |
| `temperature` | `llm_agent.py` | 0.2 | Lower = more factual |

---

## Troubleshooting

**`ConnectionError: Cannot reach Ollama`**
→ Run `ollama serve` in a separate terminal.

**`FileNotFoundError: No saved index found`**
→ Run `--ingest` before `--query`.

**Slow responses**
→ Try a smaller model: change `OLLAMA_MODEL = "llama3:8b"` in `llm_agent.py`.

**Out of memory**
→ Reduce `TOP_K` in `query.py` or `CHUNK_WORDS` in `embed.py`.
