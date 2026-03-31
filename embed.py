"""
embed.py — Document ingestion, chunking, embedding, and FAISS index management.
"""

import os
import re
import pickle
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "all-MiniLM-L6-v2"
CHUNK_WORDS  = 250          # target words per chunk
OVERLAP_WORDS = 50          # overlap between consecutive chunks
INDEX_PATH   = "faiss.index"
CHUNKS_PATH  = "chunks.pkl"

# ── Model (loaded once) ───────────────────────────────────────────────────────
_model: SentenceTransformer | None = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[embed] Loading model '{MODEL_NAME}' …")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


# ── Text extraction ───────────────────────────────────────────────────────────
def extract_text(file_path: str) -> str:
    """Extract raw text from a .pdf or .txt file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")

    elif suffix == ".pdf":
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("Install pdfplumber:  pip install pdfplumber")
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n".join(text_parts)

    else:
        raise ValueError(f"Unsupported file type: {suffix}  (use .pdf or .txt)")


# ── Chunking ─────────────────────────────────────────────────────────────────
def split_into_chunks(text: str,
                      chunk_words: int = CHUNK_WORDS,
                      overlap_words: int = OVERLAP_WORDS) -> list[str]:
    """
    Split text into overlapping word-based chunks.
    Each chunk is ~chunk_words words with overlap_words words of context
    carried over from the previous chunk.
    """
    # Collapse excess whitespace while preserving sentence breaks
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()

    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_words - overlap_words  # slide with overlap

    print(f"[embed] Split into {len(chunks)} chunks "
          f"(~{chunk_words} words, {overlap_words} overlap)")
    return chunks


# ── Embeddings ────────────────────────────────────────────────────────────────
def embed_chunks(chunks: list[str]) -> np.ndarray:
    """Return a float32 numpy array of shape (N, dim)."""
    model = get_model()
    print(f"[embed] Encoding {len(chunks)} chunks …")
    embeddings = model.encode(chunks, show_progress_bar=True,
                              convert_to_numpy=True)
    return embeddings.astype("float32")


def embed_query(query: str) -> np.ndarray:
    """Return a (1, dim) float32 array for a single query string."""
    model = get_model()
    vec = model.encode([query], convert_to_numpy=True)
    return vec.astype("float32")


# ── FAISS index ───────────────────────────────────────────────────────────────
def build_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Build a flat L2 FAISS index from embeddings."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"[embed] FAISS index built — {index.ntotal} vectors, dim={dim}")
    return index


def save_index(index: faiss.IndexFlatL2,
               chunks: list[str],
               index_path: str = INDEX_PATH,
               chunks_path: str = CHUNKS_PATH) -> None:
    faiss.write_index(index, index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"[embed] Saved index → {index_path}, chunks → {chunks_path}")


def load_index(index_path: str = INDEX_PATH,
               chunks_path: str = CHUNKS_PATH
               ) -> tuple[faiss.IndexFlatL2, list[str]]:
    """Reload a previously saved FAISS index and chunk list."""
    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        raise FileNotFoundError(
            "No saved index found. Run main.py with --ingest first.")
    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    print(f"[embed] Loaded index ({index.ntotal} vectors) + {len(chunks)} chunks")
    return index, chunks


# ── High-level ingest ─────────────────────────────────────────────────────────
def ingest(file_path: str) -> tuple[faiss.IndexFlatL2, list[str]]:
    """
    Full pipeline: extract → chunk → embed → build index → save.
    Returns (index, chunks).
    """
    text       = extract_text(file_path)
    chunks     = split_into_chunks(text)
    embeddings = embed_chunks(chunks)
    index      = build_index(embeddings)
    save_index(index, chunks)
    return index, chunks
