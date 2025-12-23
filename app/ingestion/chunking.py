"""
Chunking stage (token-based).

We chunk by tokens (not characters) to match LLM limits and embedding costs.

Design notes:
- We use tiktoken to approximate OpenAI tokenization.
- Each chunk becomes a new Document, inheriting metadata from its parent.
- We attach stable chunk identifiers for citations and debugging.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

import tiktoken
from langchain_core.documents import Document


def chunk_documents(
    docs: Iterable[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
    encoding_name: str = "cl100k_base",
) -> list[Document]:
    """
    Chunk a collection of Documents into token-sized chunks.

    Args:
        docs: input Documents (already normalized).
        chunk_size: max tokens per chunk (e.g. 512).
        chunk_overlap: overlap tokens between chunks (e.g. 64).
        encoding_name: tiktoken encoding to use.

    Returns:
        List of chunk Documents.
    """
    enc = tiktoken.get_encoding(encoding_name)
    out: list[Document] = []

    step = max(1, chunk_size - chunk_overlap)

    for doc in docs:
        text = doc.page_content or ""
        if not text.strip():
            continue

        tokens = enc.encode(text)
        if not tokens:
            continue

        source = str(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page")

        for i, start in enumerate(range(0, len(tokens), step)):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = enc.decode(chunk_tokens).strip()
            if not chunk_text:
                continue

            metadata = dict(doc.metadata)
            metadata["chunk_index"] = i
            metadata["chunk_start_token"] = start
            metadata["chunk_end_token"] = end

            chunk_id = _stable_chunk_id(
                source=source,
                page=page,
                chunk_index=i,
                start=start,
                end=end,
            )
            metadata["chunk_id"] = chunk_id
            metadata["citation"] = _citation_string(
                source=source, page=page, chunk_index=i
            )

            out.append(Document(page_content=chunk_text, metadata=metadata))

            if end >= len(tokens):
                break

    return out


def _citation_string(source: str, page: int | None, chunk_index: int) -> str:
    if page is None:
        return f"{source}#chunk={chunk_index}"
    return f"{source}#page={page}#chunk={chunk_index}"


def _stable_chunk_id(
    *,
    source: str,
    page: int | None,
    chunk_index: int,
    start: int,
    end: int,
) -> str:
    """
    Produce a stable identifier to join citations / debugging across systems.
    """
    raw = f"{source}|{page}|{chunk_index}|{start}|{end}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]