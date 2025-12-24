"""
Build only the BM25 index from data/raw.

Run:
  python scripts/build_bm25_only.py
"""

from __future__ import annotations

from app.config import get_settings
from app.ingestion.chunking import chunk_documents
from app.ingestion.loaders import load_documents
from app.ingestion.normalize import normalize_documents
from app.retrieval.bm25 import BM25Index
from app.utils.logging import setup_logging


def main() -> None:
    s = get_settings()
    setup_logging(s.log_level)

    out_path = s.processed_data_dir / "kb" / "bm25" / "bm25.pkl"

    load = load_documents(s.raw_data_dir)
    norm = normalize_documents(load.documents)
    chunks = chunk_documents(
        norm,
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
    )

    bm25 = BM25Index(chunks)
    bm25.save(out_path)

    print(f"Built BM25 index with {len(chunks)} chunks -> {out_path}")


if __name__ == "__main__":
    main()