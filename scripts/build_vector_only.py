"""
Build only the FAISS vector index from data/raw.

This is a "smoke test" script:
- Loads raw docs
- Normalizes
- Chunks
- Builds FAISS index
- Saves to data/processed/kb/faiss

Run:
  python scripts/build_vector_only.py
"""

from __future__ import annotations

from pathlib import Path

from app.config import get_settings
from app.ingestion.chunking import chunk_documents
from app.ingestion.loaders import load_documents
from app.ingestion.normalize import normalize_documents
from app.retrieval.vector_store import build_faiss_index, save_faiss_index
from app.utils.logging import setup_logging


def main() -> None:
    s = get_settings()
    setup_logging(s.log_level)

    raw_dir = s.raw_data_dir
    out_dir = s.processed_data_dir / "kb" / "faiss"

    load = load_documents(raw_dir)
    norm = normalize_documents(load.documents)
    chunks = chunk_documents(
        norm,
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
    )

    index = build_faiss_index(chunks, s)
    save_faiss_index(index, out_dir)

    print(f"Built FAISS index with {len(chunks)} chunks -> {out_dir}")


if __name__ == "__main__":
    main()