"""
KB loader utility.

Purpose:
- Load the on-disk KB artifacts created in Phase 1 (FAISS + BM25)
- Build a HybridRetriever instance for runtime querying

This avoids duplicating "where is the KB stored?" logic everywhere.
"""

from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.retrieval.bm25 import BM25Index
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.rerank import LLMReranker
from app.retrieval.vector_store import load_faiss_index


def load_hybrid_retriever(settings: Settings) -> HybridRetriever:
    kb_dir = settings.processed_data_dir / "kb"
    faiss_dir = kb_dir / "faiss"
    bm25_path = kb_dir / "bm25" / "bm25.pkl"

    _assert_exists(faiss_dir, hint="Run: python scripts/build_kb.py")
    _assert_exists(bm25_path, hint="Run: python scripts/build_kb.py")

    vector = load_faiss_index(faiss_dir, settings)
    bm25 = BM25Index.load(bm25_path)

    reranker = LLMReranker(settings) if settings.enable_rerank else None

    return HybridRetriever(
        vector_index=vector,
        bm25_index=bm25,
        settings=settings,
        reranker=reranker,
    )


def _assert_exists(path: Path, *, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing KB artifact: {path}. {hint}")