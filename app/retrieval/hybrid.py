"""
Hybrid retrieval: FAISS (vector) + BM25 (lexical).

We:
1) Retrieve top-K from vector store.
2) Retrieve top-K from BM25.
3) Merge + score normalize.
4) (Optional) rerank with LLM.
5) Return final top-K Documents with stable metadata.

Scoring:
- Vector scores are distances; we convert to similarity.
- BM25 scores are already higher=better.
- We min-max normalize each signal and combine:
    combined = alpha * vector_norm + (1-alpha) * bm25_norm
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import Settings
from app.retrieval.bm25 import BM25Index
from app.retrieval.rerank import LLMReranker, RerankCandidate


@dataclass(frozen=True)
class RetrievedChunk:
    doc: Document
    score: float
    vector_score: float
    bm25_score: float


class HybridRetriever:
    def __init__(
        self,
        *,
        vector_index: FAISS,
        bm25_index: BM25Index,
        settings: Settings,
        reranker: LLMReranker | None = None,
    ) -> None:
        self._vector = vector_index
        self._bm25 = bm25_index
        self._s = settings
        self._reranker = reranker

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        vec_k = self._s.vector_top_k
        bm_k = self._s.bm25_top_k
        final_k = self._s.final_top_k

        vec_hits = self._vector.similarity_search_with_score(query, k=vec_k)
        bm_hits = self._bm25.search(query, k=bm_k)

        # Convert FAISS distances to similarity scores.
        vec_docs: list[Document] = []
        vec_scores: list[float] = []
        for doc, dist in vec_hits:
            vec_docs.append(doc)
            vec_scores.append(_distance_to_similarity(float(dist)))

        bm_docs: list[Document] = [r.doc for r in bm_hits]
        bm_scores: list[float] = [float(r.score) for r in bm_hits]

        merged = _merge_dedup(vec_docs, vec_scores, bm_docs, bm_scores)

        # Normalize each signal.
        vec_norm = _minmax_normalize([m["vector_score"] for m in merged])
        bm_norm = _minmax_normalize([m["bm25_score"] for m in merged])

        alpha = float(self._s.hybrid_alpha)
        results: list[RetrievedChunk] = []
        for i, m in enumerate(merged):
            score = alpha * vec_norm[i] + (1.0 - alpha) * bm_norm[i]
            results.append(
                RetrievedChunk(
                    doc=m["doc"],
                    score=score,
                    vector_score=m["vector_score"],
                    bm25_score=m["bm25_score"],
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        results = results[: max(final_k, 1)]

        # Optional rerank on the top-N (only if enabled and reranker exists).
        if self._s.enable_rerank and self._reranker:
            top_n = min(self._s.rerank_top_n, len(results))
            candidates = [
                RerankCandidate(
                    candidate_id=i,
                    doc=results[i].doc,
                    score=results[i].score,
                )
                for i in range(top_n)
            ]
            reranked = self._reranker.rerank(query, candidates)
            new_order = [results[c.candidate_id] for c in reranked]
            results = new_order + results[top_n:]

        return results


def _distance_to_similarity(distance: float) -> float:
    # Safe conversion for distance >= 0. Larger similarity => better.
    return 1.0 / (1.0 + max(distance, 0.0))


def _minmax_normalize(xs: Iterable[float]) -> list[float]:
    vals = list(xs)
    if not vals:
        return []
    mn = min(vals)
    mx = max(vals)
    if mx - mn < 1e-9:
        return [1.0 for _ in vals]
    return [(v - mn) / (mx - mn) for v in vals]


def _merge_dedup(
    vec_docs: list[Document],
    vec_scores: list[float],
    bm_docs: list[Document],
    bm_scores: list[float],
) -> list[dict]:
    """
    Merge two ranked lists of Documents, deduplicating by chunk_id if present,
    else by citation, else by raw text hash.
    """
    merged: dict[str, dict] = {}

    def key_for(doc: Document) -> str:
        md = doc.metadata or {}
        if "chunk_id" in md:
            return f"chunk_id:{md['chunk_id']}"
        if "citation" in md:
            return f"citation:{md['citation']}"
        return f"text:{hash(doc.page_content or '')}"

    for doc, score in zip(vec_docs, vec_scores):
        k = key_for(doc)
        merged.setdefault(
            k,
            {"doc": doc, "vector_score": 0.0, "bm25_score": 0.0},
        )
        merged[k]["vector_score"] = max(merged[k]["vector_score"], float(score))

    for doc, score in zip(bm_docs, bm_scores):
        k = key_for(doc)
        merged.setdefault(
            k,
            {"doc": doc, "vector_score": 0.0, "bm25_score": 0.0},
        )
        merged[k]["bm25_score"] = max(merged[k]["bm25_score"], float(score))

    return list(merged.values())