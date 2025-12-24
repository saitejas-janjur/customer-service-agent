"""
BM25 lexical index.

Why BM25?
- Vector search is great for semantics, but BM25 boosts exact matches
  (order numbers, specific policy phrasing, SKU codes, etc.).

We implement our own small wrapper so we can:
- persist it
- control tokenization
- return Documents with scores
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


@dataclass
class BM25SearchResult:
    doc: Document
    score: float


class BM25Index:
    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents
        self._tokenized = [_tokenize(d.page_content or "") for d in documents]
        self._bm25 = BM25Okapi(self._tokenized)

    @property
    def documents(self) -> list[Document]:
        return self._documents

    def search(self, query: str, k: int) -> list[BM25SearchResult]:
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        ranked = sorted(
            enumerate(scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        out: list[BM25SearchResult] = []
        for idx, score in ranked[:k]:
            out.append(BM25SearchResult(doc=self._documents[idx], score=float(score)))
        return out

    def save(self, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"documents": self._documents}
        out_path.write_bytes(pickle.dumps(payload))

    @staticmethod
    def load(path: Path) -> "BM25Index":
        payload = pickle.loads(path.read_bytes())
        documents = payload["documents"]
        return BM25Index(documents=documents)