"""
Query the built KB using hybrid retrieval and optionally generate an answer.

Run (retrieval only):
  python scripts/query_kb.py --question "What is the refund window?" --no-rag

Run (retrieval + RAG answer):
  python scripts/query_kb.py --question "What is the refund window?"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.config import get_settings
from app.rag.answer import RAGAnswerer
from app.retrieval.bm25 import BM25Index
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.rerank import LLMReranker
from app.retrieval.vector_store import load_faiss_index
from app.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True, type=str)
    parser.add_argument("--kb-dir", type=str, default=None)
    parser.add_argument("--no-rag", action="store_true")
    args = parser.parse_args()

    s = get_settings()
    setup_logging(s.log_level)

    kb_dir = Path(args.kb_dir) if args.kb_dir else (s.processed_data_dir / "kb")
    faiss_dir = kb_dir / "faiss"
    bm25_path = kb_dir / "bm25" / "bm25.pkl"

    vector = load_faiss_index(faiss_dir, s)
    bm25 = BM25Index.load(bm25_path)

    reranker = LLMReranker(s) if s.enable_rerank else None
    retriever = HybridRetriever(
        vector_index=vector,
        bm25_index=bm25,
        settings=s,
        reranker=reranker,
    )

    hits = retriever.retrieve(args.question)

    print("\nTOP RETRIEVAL RESULTS\n---------------------")
    for i, h in enumerate(hits, start=1):
        citation = str(h.doc.metadata.get("citation", "unknown"))
        preview = (h.doc.page_content or "").strip().replace("\n", " ")
        preview = preview[:180]
        print(
            f"{i}. score={h.score:.3f} "
            f"(vec={h.vector_score:.3f}, bm25={h.bm25_score:.3f}) "
            f"| {citation}"
        )
        print(f"   {preview}")

    if args.no_rag:
        return

    answerer = RAGAnswerer(s)
    result = answerer.answer(args.question, hits)

    print("\nRAG ANSWER\n----------")
    print(result.answer)
    print("\nSOURCES\n-------")
    for src in result.sources:
        print(f"- {src}")


if __name__ == "__main__":
    main()