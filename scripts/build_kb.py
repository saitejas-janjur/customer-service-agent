"""
Build the knowledge base indexes from data/raw.

This is the main Phase 1 build step:
- ingest -> normalize -> chunk
- build FAISS vector index
- build BM25 index
- write a manifest.json for debugging/traceability

Run:
  python scripts/build_kb.py --clean
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.ingestion.chunking import chunk_documents
from app.ingestion.loaders import iter_document_stats, load_documents
from app.ingestion.normalize import normalize_documents
from app.retrieval.bm25 import BM25Index
from app.retrieval.vector_store import build_faiss_index, save_faiss_index
from app.utils.logging import setup_logging


def main() -> None:
    s = get_settings()
    setup_logging(s.log_level)

    kb_dir = s.processed_data_dir / "kb"
    faiss_dir = kb_dir / "faiss"
    bm25_path = kb_dir / "bm25" / "bm25.pkl"
    manifest_path = kb_dir / "manifest.json"

    kb_dir.mkdir(parents=True, exist_ok=True)

    # Load
    load = load_documents(s.raw_data_dir)
    norm = normalize_documents(load.documents)
    chunks = chunk_documents(
        norm,
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
    )

    # Build BM25
    bm25 = BM25Index(chunks)
    bm25.save(bm25_path)

    # Build Vector
    faiss = build_faiss_index(chunks, s)
    save_faiss_index(faiss, faiss_dir)

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(s.raw_data_dir),
        "processed_dir": str(s.processed_data_dir),
        "chunk_size": s.chunk_size,
        "chunk_overlap": s.chunk_overlap,
        "embedding_model": s.embedding_model,
        "counts": {
            "loaded": iter_document_stats(load.documents),
            "normalized": iter_document_stats(norm),
            "chunks": len(chunks),
            "skipped_files": len(load.skipped_files),
        },
        "skipped_files": load.skipped_files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("KB build complete.")
    print(f"- FAISS:    {faiss_dir}")
    print(f"- BM25:     {bm25_path}")
    print(f"- Manifest: {manifest_path}")
    if load.skipped_files:
        print("Some files were skipped:")
        for sfile in load.skipped_files:
            print(f"  - {sfile}")


if __name__ == "__main__":
    main()