"""
Document loaders for the RAG pipeline.

Responsibilities:
- Discover supported files under a root directory.
- Load each file into LangChain `Document` objects.
- Preserve useful metadata (source path, file type, page number).

We keep loading separate from normalization and chunking so each stage can
be tested and evolved independently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from pypdf import PdfReader

logger = logging.getLogger(__name__)


SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".md", ".txt"}


@dataclass(frozen=True)
class LoadResult:
    """
    Output of a load operation.

    `documents` contains one or more Documents (PDF -> per-page Documents).
    `skipped_files` includes files we ignored (unsupported or errors).
    """

    documents: list[Document]
    skipped_files: list[str]


def discover_files(root: Path) -> list[Path]:
    """
    Recursively discover supported files under `root`.

    Args:
        root: Directory containing raw documents.

    Returns:
        Sorted list of file paths.
    """
    if not root.exists():
        raise FileNotFoundError(f"Raw data directory not found: {root}")

    paths = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            paths.append(p)

    return sorted(paths)


def load_documents(root: Path) -> LoadResult:
    """
    Load all supported documents found under `root`.

    Returns:
        LoadResult with loaded Documents and any skipped file paths.
    """
    files = discover_files(root)
    return load_documents_from_paths(files, root=root)


def load_documents_from_paths(
    paths: Sequence[Path],
    *,
    root: Path | None = None,
) -> LoadResult:
    """
    Load documents from explicit file paths.

    Args:
        paths: A list of file paths to load.
        root: Optional root used to make "source" metadata relative and stable.

    Returns:
        LoadResult
    """
    documents: list[Document] = []
    skipped: list[str] = []

    for path in paths:
        try:
            documents.extend(_load_single_file(path, root=root))
        except Exception as e:  # noqa: BLE001 (we want robust batch ingest)
            logger.exception("Failed to load file: %s", path)
            skipped.append(f"{path} ({type(e).__name__}: {e})")

    return LoadResult(documents=documents, skipped_files=skipped)


def _load_single_file(path: Path, *, root: Path | None) -> list[Document]:
    """
    Load a single file into one or more Documents.
    """
    ext = path.suffix.lower()

    source = (
        str(path.relative_to(root)) if root and path.is_relative_to(root)
        else str(path)
    )

    if ext == ".pdf":
        return _load_pdf(path, source=source)

    if ext in {".html", ".htm"}:
        return [_load_html(path, source=source)]

    if ext in {".md", ".txt"}:
        return [_load_text(path, source=source)]

    raise ValueError(f"Unsupported file extension: {ext}")


def _load_pdf(path: Path, *, source: str) -> list[Document]:
    """
    Load a PDF into one Document per page.

    Notes:
    - PDF text extraction quality depends heavily on the source PDF.
    - Scanned PDFs (images) will yield little/no text unless OCR is used
      (we are not doing OCR in Phase 1).
    """
    reader = PdfReader(str(path))
    docs: list[Document] = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        metadata = {
            "source": source,
            "file_ext": ".pdf",
            "page": i + 1,
        }
        docs.append(Document(page_content=text, metadata=metadata))

    return docs


def _load_html(path: Path, *, source: str) -> Document:
    """
    Load HTML and extract visible text.
    """
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    # Remove common non-content tags.
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    metadata = {"source": source, "file_ext": path.suffix.lower()}
    return Document(page_content=text, metadata=metadata)


def _load_text(path: Path, *, source: str) -> Document:
    """
    Load Markdown or plain text.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    metadata = {"source": source, "file_ext": path.suffix.lower()}
    return Document(page_content=text, metadata=metadata)


def iter_document_stats(docs: Iterable[Document]) -> dict[str, int]:
    """
    Small helper for debug/telemetry.
    """
    total_docs = 0
    total_chars = 0
    for d in docs:
        total_docs += 1
        total_chars += len(d.page_content or "")
    return {"docs": total_docs, "chars": total_chars}