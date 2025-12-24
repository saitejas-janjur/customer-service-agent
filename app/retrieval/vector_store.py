"""
Vector store layer (FAISS).

Responsibilities:
- Build FAISS index from chunk Documents.
- Save/load index locally for fast iteration.
- Keep embeddings configuration centralized.

Security note:
- FAISS.load_local uses pickle under the hood for docstore metadata.
  Only load indexes you built yourself.
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.config import Settings


def build_faiss_index(chunks: list[Document], settings: Settings) -> FAISS:
    embeddings = OpenAIEmbeddings(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
    )
    return FAISS.from_documents(chunks, embedding=embeddings)


def save_faiss_index(index: FAISS, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    index.save_local(str(out_dir))


def load_faiss_index(out_dir: Path, settings: Settings) -> FAISS:
    embeddings = OpenAIEmbeddings(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
    )
    return FAISS.load_local(
        str(out_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )