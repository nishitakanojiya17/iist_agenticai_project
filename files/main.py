"""
main.py — CLI entry point for the RAG pipeline.

Usage
-----
# 1. Ingest a document (builds + saves FAISS index)
python main.py --ingest path/to/document.pdf

# 2. Ask a question against the saved index
python main.py --query "What is the refund policy?"

# 3. Ingest AND query in one shot
python main.py --ingest doc.pdf --query "Summarise the key points."
"""

import argparse
import json
import time

from embed import ingest, load_index
from query import answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG pipeline — ingest documents and query them with an LLM."
    )
    parser.add_argument(
        "--ingest", metavar="FILE",
        help="Path to a .pdf or .txt file to ingest."
    )
    parser.add_argument(
        "--query", metavar="QUESTION",
        help="Question to ask against the indexed document."
    )
    parser.add_argument(
        "--top-k", type=int, default=4,
        help="Number of chunks to retrieve (default: 4)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.ingest and not args.query:
        print("Nothing to do. Use --ingest <file> and/or --query <question>.")
        print("Run  python main.py --help  for usage details.")
        return

    index, chunks = None, None

    # ── Step 1: Ingest ────────────────────────────────────────────────────────
    if args.ingest:
        t0 = time.perf_counter()
        print(f"\n{'='*60}")
        print(f"  INGESTING: {args.ingest}")
        print(f"{'='*60}")
        index, chunks = ingest(args.ingest)
        print(f"\n✓ Ingestion complete in {time.perf_counter() - t0:.2f}s")

    # ── Step 2: Query ─────────────────────────────────────────────────────────
    if args.query:
        # Load from disk if we didn't just ingest
        if index is None:
            index, chunks = load_index()

        t0 = time.perf_counter()
        print(f"\n{'='*60}")
        print(f"  QUERY: {args.query}")
        print(f"{'='*60}")

        result = answer(
            query=args.query,
            index=index,
            chunks=chunks,
            top_k=args.top_k,
        )

        elapsed = time.perf_counter() - t0
        print(f"\n✓ Query answered in {elapsed:.2f}s")
        print(f"\n{'='*60}")
        print("  RESULT")
        print(f"{'='*60}")
        # Pretty-print without sources body (too long for terminal)
        display = {
            "answer":  result["answer"],
            "sources": [s[:120] + "…" for s in result["sources"]],
        }
        print(json.dumps(display, indent=2, ensure_ascii=False))

        # Full machine-readable output
        output_path = "result.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nFull result saved → {output_path}")


if __name__ == "__main__":
    main()
